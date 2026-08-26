#include "core/initial_solution.hpp"
#include "core/solution_checker.hpp"
#include "scip/compact_cluster_aware_solver.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr double tolerance = 1.0e-9;

double assignment_objective(const cluster_aware::DistanceMatrix& distances,
                            const std::vector<std::size_t>& assignment, std::size_t ell)
{
    std::vector<std::vector<double>> cluster_distances(distances.center_count());

    for (std::size_t client = 0; client < distances.client_count(); ++client) {
        const auto center = assignment[client];
        cluster_distances[center].push_back(distances(client, center));
    }

    double objective = 0.0;
    for (auto& values : cluster_distances) {
        std::ranges::sort(values, std::greater<>{});
        const auto count = std::min(ell, values.size());
        for (std::size_t index = 0; index < count; ++index) {
            objective += values[index];
        }
    }

    return objective;
}

double exhaustive_optimum(const cluster_aware::DistanceMatrix& distances,
                          std::size_t maximum_center_count, std::size_t ell)
{
    if (maximum_center_count == 0 || ell == 0) {
        throw std::invalid_argument("the exhaustive oracle requires k, ell >= 1");
    }
    if (distances.center_count() >= std::numeric_limits<std::uint64_t>::digits) {
        throw std::invalid_argument("too many centers for the exhaustive oracle");
    }

    double best = std::numeric_limits<double>::infinity();
    const std::uint64_t subset_count = std::uint64_t{1} << distances.center_count();

    for (std::uint64_t mask = 1; mask < subset_count; ++mask) {
        if (static_cast<std::size_t>(std::popcount(mask)) > maximum_center_count) {
            continue;
        }

        std::vector<std::size_t> open_centers;
        for (std::size_t center = 0; center < distances.center_count(); ++center) {
            if ((mask & (std::uint64_t{1} << center)) != 0) {
                open_centers.push_back(center);
            }
        }

        std::vector<std::size_t> assignment(distances.client_count());
        const auto enumerate_assignments = [&](auto&& self, std::size_t client) -> void {
            if (client == distances.client_count()) {
                best = std::min(best, assignment_objective(distances, assignment, ell));
                return;
            }

            for (const auto center : open_centers) {
                assignment[client] = center;
                self(self, client + 1);
            }
        };

        enumerate_assignments(enumerate_assignments, 0);
    }

    return best;
}

bool run_case(std::string_view name, cluster_aware::DistanceMatrix distances,
              std::size_t maximum_center_count, std::size_t ell,
              std::optional<double> known_expected = std::nullopt)
{
    const double enumerated = exhaustive_optimum(distances, maximum_center_count, ell);
    if (known_expected.has_value() && std::abs(enumerated - *known_expected) > tolerance) {
        std::cerr << name << ": exhaustive oracle expected " << *known_expected << ", got "
                  << enumerated << '\n';
        return false;
    }

    cluster_aware::Instance instance{std::move(distances), maximum_center_count,
                                     cluster_aware::TopLNorm{ell}, cluster_aware::L1Norm{}};

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

    if (std::abs(solution.objective - enumerated) > tolerance) {
        std::cerr << name << ": exhaustive optimum " << enumerated << ", SCIP returned "
                  << solution.objective << '\n';
        return false;
    }

    return true;
}

cluster_aware::DistanceMatrix generate_one_dimensional_instance(std::size_t client_count,
                                                                std::size_t center_count,
                                                                std::mt19937& generator)
{
    std::uniform_int_distribution<int> coordinate_distribution{0, 80};
    std::vector<double> client_positions(client_count);
    std::vector<double> center_positions(center_count);

    for (auto& position : client_positions) {
        position = static_cast<double>(coordinate_distribution(generator)) / 4.0;
    }
    for (auto& position : center_positions) {
        position = static_cast<double>(coordinate_distribution(generator)) / 4.0;
    }

    std::vector<double> distances;
    distances.reserve(client_count * center_count);
    for (const double center_position : center_positions) {
        for (const double client_position : client_positions) {
            distances.push_back(std::abs(client_position - center_position));
        }
    }

    return {client_count, center_count, std::move(distances)};
}

bool run_generated_cases()
{
    constexpr std::size_t case_count = 12;
    std::mt19937 generator{0x5EED};
    bool passed = true;

    for (std::size_t case_index = 0; case_index < case_count; ++case_index) {
        const std::size_t client_count = 2 + case_index % 4;
        const std::size_t center_count = 2 + case_index / 4;
        const std::size_t number_of_clusters = 1 + case_index % center_count;
        const std::size_t ell = 1 + (2 * case_index + 1) % client_count;
        const std::string name = "generated instance " + std::to_string(case_index);

        passed &=
            run_case(name, generate_one_dimensional_instance(client_count, center_count, generator),
                     number_of_clusters, ell);
    }

    return passed;
}

bool run_generated_edge_cases()
{
    struct CaseSpec {
        std::size_t client_count;
        std::size_t center_count;
        std::size_t number_of_clusters;
        std::size_t ell;
    };

    constexpr std::array cases{
        CaseSpec{1, 1, 1, 1}, CaseSpec{1, 3, 1, 1}, CaseSpec{1, 3, 3, 4}, CaseSpec{4, 1, 1, 1},
        CaseSpec{4, 1, 1, 7}, CaseSpec{2, 4, 4, 2}, CaseSpec{5, 2, 2, 8}, CaseSpec{3, 5, 5, 6},
    };

    std::mt19937 generator{0xC0FFEE};
    bool passed = true;

    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const auto& spec = cases[case_index];
        const std::string name = "generated edge instance " + std::to_string(case_index);
        passed &= run_case(
            name,
            generate_one_dimensional_instance(spec.client_count, spec.center_count, generator),
            spec.number_of_clusters, spec.ell);
    }

    return passed;
}

} // namespace

int main()
{
    bool passed = true;

    // Square two-point instance. For ell = 1, either single-center solution
    // has Top-1 cost 2.
    passed &= run_case("two clients",
                       cluster_aware::DistanceMatrix{2,
                                                     2,
                                                     {
                                                         0.0,
                                                         2.0,
                                                         2.0,
                                                         0.0,
                                                     }},
                       1, 1, 2.0);

    // Rectangular instance with fractional distances. Center 0 has distances
    // [0, 0.4, 2.5], whose Top-2 cost is 2.9 and beats center 1.
    passed &= run_case("rectangular fractional instance",
                       cluster_aware::DistanceMatrix{3,
                                                     2,
                                                     {
                                                         0.0,
                                                         0.4,
                                                         2.5,
                                                         3.0,
                                                         0.0,
                                                         0.5,
                                                     }},
                       1, 2, 2.9);

    // A larger one-dimensional-style instance. The exhaustive oracle checks
    // every assignment to every subset of at most two of the four centers.
    passed &= run_case(
        "seven clients and four centers",
        cluster_aware::DistanceMatrix{7,
                                      4,
                                      {
                                          0.0,  1.0,  2.0,  8.0,  9.0,  10.0, 20.0, 2.0, 1.0, 0.0,
                                          6.0,  7.0,  8.0,  18.0, 9.0,  8.0,  7.0,  1.0, 0.0, 1.0,
                                          11.0, 20.0, 19.0, 18.0, 12.0, 11.0, 10.0, 0.0,
                                      }},
        2, 3, 16.0);

    passed &= run_generated_cases();
    passed &= run_generated_edge_cases();

    return passed ? 0 : 1;
}
