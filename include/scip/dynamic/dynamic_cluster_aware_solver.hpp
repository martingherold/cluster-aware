#pragma once

#include "core/instance.hpp"
#include "core/solution.hpp"
#include "scip/scip_wrapper.hpp"
#include "scip/solve_result.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace cluster_aware {

namespace detail {
class DiscreteClusteringModel;
}

class DynamicClusterAwareSolver {
  public:
    using TopLInstance = Instance<TopLNorm, L1Norm>;
    explicit DynamicClusterAwareSolver(std::shared_ptr<const TopLInstance> instance);
    ~DynamicClusterAwareSolver();

    [[nodiscard]] SolveResult solve();
    [[nodiscard]] SolveResult solve(const Solution& initial_solution);

  private:
    using VariableHandle = ScipWrapper::VariableHandle;

    void create_cluster_cost_variables();
    void add_top_l_constraints();

    void add_vars_to_prob_data();
    void add_initial_solution(const Solution& initial_solution);
    [[nodiscard]] SolveResult solve_impl(const Solution* initial_solution);

    std::shared_ptr<const TopLInstance> instance_;
    ScipWrapper scip_;
    std::unique_ptr<detail::DiscreteClusteringModel> base_model_;
    std::vector<VariableHandle> cluster_cost_vars_;
};

} // namespace cluster_aware
