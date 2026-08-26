#pragma once

#include "core/solution.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

namespace cluster_aware {

enum class SolveStatus {
    unknown,
    optimal,
    infeasible,
    unbounded,
    infeasible_or_unbounded,
    user_interrupt,
    termination_signal,
    node_limit,
    total_node_limit,
    stall_node_limit,
    time_limit,
    memory_limit,
    gap_limit,
    primal_limit,
    dual_limit,
    solution_limit,
    best_solution_limit,
    restart_limit,
};

[[nodiscard]] std::string_view to_string(SolveStatus status) noexcept;

struct SolveStatistics {
    SolveStatus status = SolveStatus::unknown;
    bool has_primal_solution = false;
    std::optional<double> primal_bound;
    std::optional<double> dual_bound;
    std::optional<double> relative_gap;
    double solving_time_seconds = 0.0;
    std::uint64_t node_count = 0;

    [[nodiscard]] bool proven_optimal() const noexcept
    {
        return status == SolveStatus::optimal;
    }
};

struct SolveResult {
    std::optional<Solution> solution;
    SolveStatistics statistics;
    std::optional<double> model_objective;

    [[nodiscard]] bool has_solution() const noexcept
    {
        return solution.has_value();
    }

    [[nodiscard]] bool proven_optimal() const noexcept
    {
        return statistics.proven_optimal();
    }
};

} // namespace cluster_aware
