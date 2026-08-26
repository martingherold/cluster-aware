#pragma once

#include "core/instance.hpp"
#include "core/solution.hpp"
#include "scip/scip_wrapper.hpp"
#include "scip/solve_result.hpp"

#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

namespace cluster_aware {

namespace detail {
class DiscreteClusteringModel;
}

class CompactClusterAwareSolver {
  public:
    using L1Instance = Instance<L1Norm, L1Norm>;
    using TopLInstance = Instance<TopLNorm, L1Norm>;

    explicit CompactClusterAwareSolver(std::shared_ptr<const L1Instance> instance);
    explicit CompactClusterAwareSolver(std::shared_ptr<const TopLInstance> instance);
    explicit CompactClusterAwareSolver(const L1Instance& instance);
    explicit CompactClusterAwareSolver(const TopLInstance& instance);
    ~CompactClusterAwareSolver();

    [[nodiscard]] SolveResult solve();
    [[nodiscard]] SolveResult solve(const Solution& initial_solution);

  private:
    using VariableHandle = ScipWrapper::VariableHandle;

    void create_top_l_excess_variables();
    void create_top_l_auxiliary_variables();
    void add_cluster_cost_constraints(const TopLInstance& instance);
    void add_compact_top_l_constraints(const TopLInstance& instance);
    void add_initial_solution(const Solution& initial_solution);
    void append_top_l_solution_values(const TopLInstance& instance,
                                      const Solution& initial_solution,
                                      std::vector<ScipWrapper::SolutionValue>& values) const;
    [[nodiscard]] SolveResult solve_impl(const Solution* initial_solution);

    std::variant<std::shared_ptr<const L1Instance>, std::shared_ptr<const TopLInstance>> instance_;
    ScipWrapper scip_;
    std::unique_ptr<detail::DiscreteClusteringModel> base_model_;
    std::vector<VariableHandle> threshold_vars_;
    std::vector<VariableHandle> excess_vars_;
    std::vector<VariableHandle> cluster_cost_vars_;
};

} // namespace cluster_aware
