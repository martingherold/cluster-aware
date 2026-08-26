#include "core/initial_solution.hpp"
#include "core/solution_checker.hpp"
#include "scip/compact_cluster_aware_solver.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

constexpr double tolerance = 1.0e-9;

bool run_case(std::string_view name, cluster_aware::DistanceMatrix distances,
              std::size_t number_of_clusters, double expected_objective,
              bool check_single_use = false)
{
    cluster_aware::Instance instance{std::move(distances), number_of_clusters,
                                     cluster_aware::L1Norm{}, cluster_aware::L1Norm{}};

    cluster_aware::CompactClusterAwareSolver solver{instance};

    const auto initial_solution = cluster_aware::construct_greedy_initial_solution(instance);
    const auto result = solver.solve(initial_solution);
    if (!result.proven_optimal() || !result.solution.has_value() ||
        !result.model_objective.has_value()) {
        std::cerr << name << ": expected a proven optimal solution\n";
        return false;
    }
    const auto& solution = *result.solution;
    const auto check = cluster_aware::check_solution(instance, solution, tolerance);
    if (!check.valid()) {
        std::cerr << name << ": invalid extracted solution: " << check.message << '\n';
        return false;
    }

    if (std::abs(solution.objective - expected_objective) > tolerance) {
        std::cerr << name << ": expected objective " << expected_objective << ", got "
                  << solution.objective << '\n';
        return false;
    }
    if (std::abs(*result.model_objective - solution.objective) > tolerance) {
        std::cerr << name << ": model and domain objectives disagree\n";
        return false;
    }

    if (check_single_use) {
        try {
            static_cast<void>(solver.solve());
        } catch (const std::logic_error&) {
            return true;
        }

        std::cerr << name << ": second solve call did not throw\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    bool passed = true;

    // Distances are center-major. Opening either center and assigning both
    // clients to it has cost 0 + 2, so the optimum for k = 1 is 2.
    passed &= run_case("square instance",
                       cluster_aware::DistanceMatrix{2,
                                                     2,
                                                     {
                                                         0.0,
                                                         2.0,
                                                         2.0,
                                                         0.0,
                                                     }},
                       1, 2.0, true);

    // Three candidate centers and two clients exercise rectangular indexing.
    // Center 1 serves both clients at total L1 cost 2 + 2 = 4.
    passed &= run_case("rectangular instance",
                       cluster_aware::DistanceMatrix{2,
                                                     3,
                                                     {
                                                         0.0,
                                                         5.0,
                                                         2.0,
                                                         2.0,
                                                         5.0,
                                                         0.0,
                                                     }},
                       1, 4.0);

    return passed ? 0 : 1;
}
