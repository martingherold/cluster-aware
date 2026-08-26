#pragma once

#include "core/distance_matrix.hpp"
#include "core/solution.hpp"
#include "scip/scip_wrapper.hpp"

#include <cstddef>
#include <vector>

namespace cluster_aware::detail {

enum class AssignmentObjective {
    zero,
    distance,
};

class DiscreteClusteringModel {
  public:
    using VariableHandle = ScipWrapper::VariableHandle;

    DiscreteClusteringModel(ScipWrapper& scip, const DistanceMatrix& distances,
                            std::size_t number_of_clusters,
                            AssignmentObjective assignment_objective);

    [[nodiscard]] std::size_t client_count() const noexcept;
    [[nodiscard]] std::size_t center_count() const noexcept;
    [[nodiscard]] std::size_t assignment_index(std::size_t client, std::size_t center) const;
    [[nodiscard]] VariableHandle assignment_variable(std::size_t client, std::size_t center) const;
    [[nodiscard]] VariableHandle center_variable(std::size_t center) const;
    void append_solution_values(const Solution& solution,
                                std::vector<ScipWrapper::SolutionValue>& values) const;
    [[nodiscard]] Solution extract_solution() const;

  private:
    void create_assignment_variables(const DistanceMatrix& distances,
                                     AssignmentObjective assignment_objective);
    void create_center_variables();
    void add_center_count_constraint(std::size_t number_of_clusters);
    void add_assignment_constraints();
    void add_open_center_constraints();

    ScipWrapper& scip_;
    std::size_t client_count_;
    std::size_t center_count_;
    std::size_t number_of_clusters_;
    std::vector<VariableHandle> assignment_variables_;
    std::vector<VariableHandle> center_variables_;
};

} // namespace cluster_aware::detail
