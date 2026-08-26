#include "core/initial_solution.hpp"
#include "core/solution_checker.hpp"
#include "scip/compact_cluster_aware_solver.hpp"
#include "scip/dynamic/dynamic_cluster_aware_solver.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string_view>
#include <utility>

namespace {

constexpr double tolerance = 1.0e-9;

bool run_case(std::string_view name, cluster_aware::DistanceMatrix distances,
              std::size_t number_of_clusters, std::size_t ell, double expected_objective)
{
    using TopLInstance = cluster_aware::Instance<cluster_aware::TopLNorm, cluster_aware::L1Norm>;

    auto instance =
        std::make_shared<const TopLInstance>(std::move(distances), number_of_clusters,
                                             cluster_aware::TopLNorm{ell}, cluster_aware::L1Norm{});

    const auto initial_solution = cluster_aware::construct_greedy_initial_solution(*instance);
    cluster_aware::DynamicClusterAwareSolver dynamic_solver{instance};
    const auto dynamic_result = dynamic_solver.solve(initial_solution);
    if (!dynamic_result.proven_optimal() || !dynamic_result.solution.has_value()) {
        std::cerr << name << ": dynamic solver did not prove an optimum\n";
        return false;
    }
    const auto& dynamic_solution = *dynamic_result.solution;
    const auto dynamic_check =
        cluster_aware::check_solution(*instance, dynamic_solution, tolerance);

    if (!dynamic_check.valid()) {
        std::cerr << name << ": invalid dynamic solution: " << dynamic_check.message << '\n';
        return false;
    }

    cluster_aware::CompactClusterAwareSolver compact_solver{instance};
    const auto compact_result = compact_solver.solve(initial_solution);
    if (!compact_result.proven_optimal() || !compact_result.solution.has_value()) {
        std::cerr << name << ": compact solver did not prove an optimum\n";
        return false;
    }
    const auto& compact_solution = *compact_result.solution;

    if (std::abs(dynamic_solution.objective - compact_solution.objective) > tolerance) {
        std::cerr << name << ": dynamic objective " << dynamic_solution.objective
                  << ", compact objective " << compact_solution.objective << '\n';
        return false;
    }

    if (std::abs(dynamic_solution.objective - expected_objective) > tolerance) {
        std::cerr << name << ": expected objective " << expected_objective << ", got "
                  << dynamic_solution.objective << '\n';
        return false;
    }

    return true;
}

} // namespace

int main()
{
    bool passed = true;

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

    return passed ? 0 : 1;
}
