#include "core/solution_checker.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

constexpr double tolerance = 1.0e-9;

bool expect_valid(std::string_view name, const cluster_aware::SolutionCheckResult& result,
                  double expected_objective)
{
    if (!result.valid()) {
        std::cerr << name << ": expected a valid solution, got: " << result.message << '\n';
        return false;
    }
    if (std::abs(result.recomputed_objective - expected_objective) > tolerance) {
        std::cerr << name << ": expected objective " << expected_objective << ", got "
                  << result.recomputed_objective << '\n';
        return false;
    }
    return true;
}

bool expect_infeasible(std::string_view name, const cluster_aware::SolutionCheckResult& result)
{
    if (result.feasible) {
        std::cerr << name << ": expected an infeasible solution\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    const cluster_aware::Instance top_l_instance{cluster_aware::DistanceMatrix{3,
                                                                               3,
                                                                               {
                                                                                   0.0,
                                                                                   2.0,
                                                                                   5.0,
                                                                                   1.0,
                                                                                   1.0,
                                                                                   1.0,
                                                                                   5.0,
                                                                                   2.0,
                                                                                   0.0,
                                                                               }},
                                                 2, cluster_aware::TopLNorm{1},
                                                 cluster_aware::L1Norm{}};

    const cluster_aware::Solution valid_solution{{0, 2}, {0, 0, 2}, 2.0};

    bool passed = expect_valid("valid Top-l solution",
                               cluster_aware::check_solution(top_l_instance, valid_solution), 2.0);

    auto invalid_solution = valid_solution;
    invalid_solution.open_centers.pop_back();
    passed &= expect_infeasible("incorrect center count",
                                cluster_aware::check_solution(top_l_instance, invalid_solution));

    invalid_solution = valid_solution;
    invalid_solution.open_centers = {0, 0};
    passed &= expect_infeasible("duplicate center",
                                cluster_aware::check_solution(top_l_instance, invalid_solution));

    invalid_solution = valid_solution;
    invalid_solution.open_centers = {0, 3};
    passed &= expect_infeasible("out-of-range center",
                                cluster_aware::check_solution(top_l_instance, invalid_solution));

    invalid_solution = valid_solution;
    invalid_solution.assignments.pop_back();
    passed &= expect_infeasible("incorrect assignment count",
                                cluster_aware::check_solution(top_l_instance, invalid_solution));

    invalid_solution = valid_solution;
    invalid_solution.assignments[0] = 3;
    passed &= expect_infeasible("out-of-range assignment",
                                cluster_aware::check_solution(top_l_instance, invalid_solution));

    invalid_solution = valid_solution;
    invalid_solution.assignments[1] = 1;
    passed &= expect_infeasible("assignment to closed center",
                                cluster_aware::check_solution(top_l_instance, invalid_solution));

    invalid_solution = valid_solution;
    invalid_solution.objective = 3.0;
    const auto objective_mismatch = cluster_aware::check_solution(top_l_instance, invalid_solution);
    if (!objective_mismatch.feasible || objective_mismatch.objective_matches ||
        objective_mismatch.valid()) {
        std::cerr << "objective mismatch was not detected\n";
        passed = false;
    }

    const cluster_aware::Instance l_inf_instance{cluster_aware::DistanceMatrix{3,
                                                                               3,
                                                                               {
                                                                                   0.0,
                                                                                   2.0,
                                                                                   5.0,
                                                                                   1.0,
                                                                                   1.0,
                                                                                   1.0,
                                                                                   5.0,
                                                                                   2.0,
                                                                                   0.0,
                                                                               }},
                                                 2, cluster_aware::LInfNorm{},
                                                 cluster_aware::L1Norm{}};
    const cluster_aware::Solution solution_with_unused_center{{0, 2}, {0, 0, 0}, 5.0};
    passed &= expect_valid(
        "unused open center",
        cluster_aware::check_solution(l_inf_instance, solution_with_unused_center), 5.0);

    auto slightly_inexact_solution = valid_solution;
    slightly_inexact_solution.objective += 5.0e-4;
    const auto absolute_tolerance_check = cluster_aware::check_solution(
        top_l_instance, slightly_inexact_solution,
        cluster_aware::ObjectiveTolerance{.absolute = 1.0e-3, .relative = 0.0});
    if (!absolute_tolerance_check.valid()) {
        std::cerr << "independent absolute tolerance was not applied\n";
        passed = false;
    }

    const auto strict_relative_check = cluster_aware::check_solution(
        top_l_instance, slightly_inexact_solution,
        cluster_aware::ObjectiveTolerance{.absolute = 0.0, .relative = 1.0e-6});
    if (strict_relative_check.objective_matches) {
        std::cerr << "strict relative tolerance accepted an objective mismatch\n";
        passed = false;
    }

    try {
        static_cast<void>(cluster_aware::check_solution(
            top_l_instance, valid_solution,
            cluster_aware::ObjectiveTolerance{.absolute = -1.0, .relative = 0.0}));
        std::cerr << "negative solution-check tolerance was accepted\n";
        passed = false;
    } catch (const std::invalid_argument&) {
    }

    return passed ? 0 : 1;
}
