#include "scip/solve_result.hpp"

namespace cluster_aware {

std::string_view to_string(SolveStatus status) noexcept
{
    switch (status) {
    case SolveStatus::unknown:
        return "unknown";
    case SolveStatus::optimal:
        return "optimal";
    case SolveStatus::infeasible:
        return "infeasible";
    case SolveStatus::unbounded:
        return "unbounded";
    case SolveStatus::infeasible_or_unbounded:
        return "infeasible-or-unbounded";
    case SolveStatus::user_interrupt:
        return "user-interrupt";
    case SolveStatus::termination_signal:
        return "termination-signal";
    case SolveStatus::node_limit:
        return "node-limit";
    case SolveStatus::total_node_limit:
        return "total-node-limit";
    case SolveStatus::stall_node_limit:
        return "stall-node-limit";
    case SolveStatus::time_limit:
        return "time-limit";
    case SolveStatus::memory_limit:
        return "memory-limit";
    case SolveStatus::gap_limit:
        return "gap-limit";
    case SolveStatus::primal_limit:
        return "primal-limit";
    case SolveStatus::dual_limit:
        return "dual-limit";
    case SolveStatus::solution_limit:
        return "solution-limit";
    case SolveStatus::best_solution_limit:
        return "best-solution-limit";
    case SolveStatus::restart_limit:
        return "restart-limit";
    }
    return "unknown";
}

} // namespace cluster_aware
