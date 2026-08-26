#include "core/initial_solution.hpp"
#include "core/solution_checker.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr double tolerance = 1.0e-9;

template <typename InstanceType>
bool expect_solution(const char* name, const InstanceType& instance,
                     const std::vector<std::size_t>& expected_centers,
                     const std::vector<std::size_t>& expected_assignments,
                     double expected_objective)
{
    const auto first = cluster_aware::construct_greedy_initial_solution(instance);
    const auto second = cluster_aware::construct_greedy_initial_solution(instance);

    if (first.open_centers != expected_centers || first.assignments != expected_assignments ||
        second.open_centers != first.open_centers || second.assignments != first.assignments) {
        std::cerr << name << ": initial solution is incorrect or nondeterministic\n";
        return false;
    }
    if (std::abs(first.objective - expected_objective) > tolerance) {
        std::cerr << name << ": expected objective " << expected_objective << ", got "
                  << first.objective << '\n';
        return false;
    }

    const auto check = cluster_aware::check_solution(instance, first, tolerance);
    if (!check.valid()) {
        std::cerr << name << ": constructed solution is invalid: " << check.message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    // Center-major rows are [0,1,8,9], [9,8,1,0], and [4,4,4,4].
    // Center 2 is selected first; centers 0 and 1 then tie, so index 0 wins.
    const std::vector<double> values{
        0.0, 1.0, 8.0, 9.0, 9.0, 8.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0,
    };

    const cluster_aware::Instance top_l_instance{cluster_aware::DistanceMatrix{4, 3, values}, 2,
                                                 cluster_aware::TopLNorm{1},
                                                 cluster_aware::L1Norm{}};
    bool passed = expect_solution("Top-l", top_l_instance, {0, 2}, {0, 0, 2, 2}, 5.0);

    const cluster_aware::Instance l1_instance{cluster_aware::DistanceMatrix{4, 3, values}, 2,
                                              cluster_aware::L1Norm{}, cluster_aware::L1Norm{}};
    passed &= expect_solution("L1", l1_instance, {0, 2}, {0, 0, 2, 2}, 9.0);

    const cluster_aware::Instance all_centers_instance{cluster_aware::DistanceMatrix{4, 3, values},
                                                       3, cluster_aware::TopLNorm{5},
                                                       cluster_aware::L1Norm{}};
    const auto all_centers = cluster_aware::construct_greedy_initial_solution(all_centers_instance);
    if (all_centers.open_centers != std::vector<std::size_t>{0, 1, 2} ||
        !cluster_aware::check_solution(all_centers_instance, all_centers, tolerance).valid()) {
        std::cerr << "all-centers case produced an invalid solution\n";
        passed = false;
    }

    return passed ? 0 : 1;
}
