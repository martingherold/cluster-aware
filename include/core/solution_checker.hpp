#pragma once

#include "instance.hpp"
#include "solution.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cluster_aware {

struct SolutionCheckResult {
    bool feasible;
    bool objective_matches;
    double recomputed_objective;
    std::string message;

    [[nodiscard]] bool valid() const noexcept
    {
        return feasible && objective_matches;
    }
};

struct ObjectiveTolerance {
    double absolute = 1.0e-9;
    double relative = 1.0e-9;
};

template <ValidNorm InnerNorm, ValidNorm OuterNorm>
[[nodiscard]] SolutionCheckResult check_solution(const Instance<InnerNorm, OuterNorm>& instance,
                                                 const Solution& solution,
                                                 ObjectiveTolerance tolerance = {})
{
    if (!std::isfinite(tolerance.absolute) || tolerance.absolute < 0.0 ||
        !std::isfinite(tolerance.relative) || tolerance.relative < 0.0) {
        throw std::invalid_argument("solution-check tolerances must be finite and nonnegative");
    }

    const auto& distances = instance.distances();
    const auto infeasible = [](std::string message) {
        return SolutionCheckResult{false, false, 0.0, std::move(message)};
    };

    if (solution.open_centers.size() != instance.number_of_clusters()) {
        return infeasible("number of open centers does not equal the requested cluster count");
    }

    std::vector<bool> is_open(distances.center_count(), false);
    for (const auto center : solution.open_centers) {
        if (center >= distances.center_count()) {
            return infeasible("open-center index is out of range");
        }
        if (is_open[center]) {
            return infeasible("open-center list contains a duplicate");
        }
        is_open[center] = true;
    }

    if (solution.assignments.size() != distances.client_count()) {
        return infeasible("number of assignments does not equal the number of clients");
    }

    std::vector<std::vector<double>> cluster_distances(distances.center_count());
    for (std::size_t client = 0; client < distances.client_count(); ++client) {
        const auto center = solution.assignments[client];
        if (center >= distances.center_count()) {
            return infeasible("assignment index is out of range");
        }
        if (!is_open[center]) {
            return infeasible("client is assigned to a closed center");
        }
        cluster_distances[center].push_back(distances(client, center));
    }

    std::vector<double> cluster_costs(distances.center_count(), 0.0);
    for (std::size_t center = 0; center < distances.center_count(); ++center) {
        if (!cluster_distances[center].empty()) {
            cluster_costs[center] = instance.inner_norm()(cluster_distances[center]);
        }
    }

    const double recomputed_objective = instance.outer_norm()(cluster_costs);
    const double scale = std::max(std::abs(solution.objective), std::abs(recomputed_objective));
    const bool objective_matches = std::isfinite(solution.objective) &&
                                   std::isfinite(recomputed_objective) &&
                                   std::abs(solution.objective - recomputed_objective) <=
                                       tolerance.absolute + tolerance.relative * scale;

    return {true, objective_matches, recomputed_objective,
            objective_matches
                ? std::string{}
                : std::string{"reported objective does not match recomputed objective"}};
}

template <ValidNorm InnerNorm, ValidNorm OuterNorm>
[[nodiscard]] SolutionCheckResult check_solution(const Instance<InnerNorm, OuterNorm>& instance,
                                                 const Solution& solution, double tolerance)
{
    return check_solution(instance, solution, ObjectiveTolerance{tolerance, tolerance});
}

} // namespace cluster_aware
