#include "scip/scip_wrapper.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<cluster_aware::ScipWrapper>);
static_assert(!std::is_copy_assignable_v<cluster_aware::ScipWrapper>);
static_assert(!std::is_move_constructible_v<cluster_aware::ScipWrapper>);
static_assert(!std::is_move_assignable_v<cluster_aware::ScipWrapper>);

int main()
{
    using cluster_aware::ScipWrapper;

    ScipWrapper scip{"scip_demo", ScipWrapper::ObjectiveSense::maximize};
    const auto x = scip.create_variable({
        .name = "x",
        .lower_bound = 0.0,
        .upper_bound = std::nullopt,
        .objective = 3.0,
        .type = ScipWrapper::VariableType::integer,
    });
    const auto y = scip.create_variable({
        .name = "y",
        .lower_bound = 0.0,
        .upper_bound = std::nullopt,
        .objective = 2.0,
        .type = ScipWrapper::VariableType::integer,
    });

    {
        const std::array terms{
            ScipWrapper::LinearTerm{x, 2.0},
            ScipWrapper::LinearTerm{y, 1.0},
        };
        scip.add_linear_constraint(
            {
                .name = "first_constraint",
                .lower_bound = std::nullopt,
                .upper_bound = 4.0,
            },
            terms);
    }

    {
        const std::array terms{
            ScipWrapper::LinearTerm{x, 1.0},
            ScipWrapper::LinearTerm{y, 2.0},
        };
        scip.add_linear_constraint(
            {
                .name = "second_constraint",
                .lower_bound = std::nullopt,
                .upper_bound = 5.0,
            },
            terms);
    }

    const std::array initial_values{
        ScipWrapper::SolutionValue{x, 0.0},
        ScipWrapper::SolutionValue{y, 2.0},
    };
    bool passed = scip.add_initial_solution(initial_values);
    if (!passed) {
        std::cerr << "SCIP rejected a feasible initial solution\n";
    }

    const auto& statistics = scip.solve();

    constexpr double tolerance = 1.0e-9;
    const double x_value = scip.value(x);
    const double y_value = scip.value(y);
    const double objective = scip.objective_value();
    passed &= statistics.proven_optimal() && statistics.has_primal_solution &&
              statistics.primal_bound.has_value() && statistics.dual_bound.has_value() &&
              statistics.relative_gap.has_value() &&
              std::abs(*statistics.relative_gap) <= tolerance &&
              std::abs(x_value - 1.0) <= tolerance && std::abs(y_value - 2.0) <= tolerance &&
              std::abs(objective - 7.0) <= tolerance;

    if (!passed) {
        std::cerr << "Unexpected SCIP demo solution: x=" << x_value << ", y=" << y_value
                  << ", objective=" << objective << '\n';
    }

    ScipWrapper infeasible{"infeasible_demo"};
    const auto infeasible_x = infeasible.create_variable({
        .name = "x",
        .lower_bound = 0.0,
        .upper_bound = 0.0,
        .objective = 0.0,
        .type = ScipWrapper::VariableType::binary,
    });
    const std::array infeasible_terms{
        ScipWrapper::LinearTerm{infeasible_x, 1.0},
    };
    infeasible.add_linear_constraint(
        {
            .name = "force_x",
            .lower_bound = 1.0,
            .upper_bound = 1.0,
        },
        infeasible_terms);
    const std::array infeasible_initial_values{
        ScipWrapper::SolutionValue{infeasible_x, 0.0},
    };
    if (infeasible.add_initial_solution(infeasible_initial_values)) {
        std::cerr << "SCIP accepted an infeasible initial solution\n";
        passed = false;
    }
    const auto& infeasible_statistics = infeasible.solve();
    if (infeasible_statistics.status != cluster_aware::SolveStatus::infeasible ||
        infeasible_statistics.has_primal_solution ||
        infeasible_statistics.primal_bound.has_value()) {
        std::cerr << "Infeasible SCIP status was not represented correctly\n";
        passed = false;
    }

    return passed ? 0 : 1;
}
