#include "scip/discrete_clustering_model.hpp"

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cluster_aware::detail {

DiscreteClusteringModel::DiscreteClusteringModel(ScipWrapper& scip, const DistanceMatrix& distances,
                                                 std::size_t number_of_clusters,
                                                 AssignmentObjective assignment_objective)
    : scip_(scip), client_count_(distances.client_count()), center_count_(distances.center_count()),
      number_of_clusters_(number_of_clusters)
{
    if (number_of_clusters == 0 || number_of_clusters > center_count_) {
        throw std::invalid_argument("number of clusters must be between one and the center count");
    }

    create_assignment_variables(distances, assignment_objective);
    create_center_variables();
    add_center_count_constraint(number_of_clusters);
    add_assignment_constraints();
    add_open_center_constraints();
}

std::size_t DiscreteClusteringModel::client_count() const noexcept
{
    return client_count_;
}

std::size_t DiscreteClusteringModel::center_count() const noexcept
{
    return center_count_;
}

std::size_t DiscreteClusteringModel::assignment_index(std::size_t client, std::size_t center) const
{
    if (client >= client_count_) {
        throw std::out_of_range("assignment client index is out of range");
    }
    if (center >= center_count_) {
        throw std::out_of_range("assignment center index is out of range");
    }
    return center * client_count_ + client;
}

DiscreteClusteringModel::VariableHandle
DiscreteClusteringModel::assignment_variable(std::size_t client, std::size_t center) const
{
    return assignment_variables_.at(assignment_index(client, center));
}

DiscreteClusteringModel::VariableHandle
DiscreteClusteringModel::center_variable(std::size_t center) const
{
    return center_variables_.at(center);
}

void DiscreteClusteringModel::create_assignment_variables(const DistanceMatrix& distances,
                                                          AssignmentObjective assignment_objective)
{
    assignment_variables_.reserve(distances.values().size());

    for (std::size_t center = 0; center < center_count_; ++center) {
        const auto distances_to_center = distances.distances_to_center(center);
        for (std::size_t client = 0; client < client_count_; ++client) {
            const double objective = assignment_objective == AssignmentObjective::distance
                                         ? distances_to_center[client]
                                         : 0.0;
            assignment_variables_.push_back(scip_.create_variable({
                .name = "x_" + std::to_string(client) + "," + std::to_string(center),
                .lower_bound = 0.0,
                .upper_bound = 1.0,
                .objective = objective,
                .type = ScipWrapper::VariableType::binary,
            }));
        }
    }
}

void DiscreteClusteringModel::create_center_variables()
{
    center_variables_.reserve(center_count_);
    for (std::size_t center = 0; center < center_count_; ++center) {
        center_variables_.push_back(scip_.create_variable({
            .name = "y_" + std::to_string(center),
            .lower_bound = 0.0,
            .upper_bound = 1.0,
            .objective = 0.0,
            .type = ScipWrapper::VariableType::binary,
        }));
    }
}

void DiscreteClusteringModel::add_center_count_constraint(std::size_t number_of_clusters)
{
    const double required_center_count = static_cast<double>(number_of_clusters);
    std::vector<ScipWrapper::LinearTerm> terms;
    terms.reserve(center_variables_.size());
    for (const auto variable : center_variables_) {
        terms.push_back({variable, 1.0});
    }

    scip_.add_linear_constraint(
        {
            .name = "select_k_centers",
            .lower_bound = required_center_count,
            .upper_bound = required_center_count,
        },
        terms);
}

void DiscreteClusteringModel::add_assignment_constraints()
{
    std::vector<ScipWrapper::LinearTerm> terms;
    terms.reserve(center_count_);
    for (std::size_t client = 0; client < client_count_; ++client) {
        terms.clear();
        for (std::size_t center = 0; center < center_count_; ++center) {
            terms.push_back({assignment_variable(client, center), 1.0});
        }
        scip_.add_linear_constraint(
            {
                .name = "assign_client_" + std::to_string(client),
                .lower_bound = 1.0,
                .upper_bound = 1.0,
            },
            terms);
    }
}

void DiscreteClusteringModel::add_open_center_constraints()
{
    for (std::size_t center = 0; center < center_count_; ++center) {
        for (std::size_t client = 0; client < client_count_; ++client) {
            const std::array terms{
                ScipWrapper::LinearTerm{assignment_variable(client, center), 1.0},
                ScipWrapper::LinearTerm{center_variable(center), -1.0},
            };
            scip_.add_linear_constraint(
                {
                    .name = "link_client_" + std::to_string(client) + "_center_" +
                            std::to_string(center),
                    .lower_bound = std::nullopt,
                    .upper_bound = 0.0,
                },
                terms);
        }
    }
}

void DiscreteClusteringModel::append_solution_values(
    const Solution& solution, std::vector<ScipWrapper::SolutionValue>& values) const
{
    if (solution.open_centers.size() != number_of_clusters_) {
        throw std::invalid_argument(
            "initial solution center count does not match the model's cluster count");
    }
    if (solution.assignments.size() != client_count_) {
        throw std::invalid_argument(
            "initial solution assignment count does not match the model's client count");
    }

    std::vector<bool> is_open(center_count_, false);
    values.reserve(values.size() + solution.open_centers.size() + solution.assignments.size());
    for (const std::size_t center : solution.open_centers) {
        if (center >= center_count_) {
            throw std::invalid_argument("initial solution contains an out-of-range center");
        }
        if (is_open[center]) {
            throw std::invalid_argument("initial solution contains a duplicate center");
        }
        is_open[center] = true;
        values.push_back({center_variable(center), 1.0});
    }

    for (std::size_t client = 0; client < client_count_; ++client) {
        const std::size_t center = solution.assignments[client];
        if (center >= center_count_ || !is_open[center]) {
            throw std::invalid_argument(
                "initial solution assigns a client to a center that is not open");
        }
        values.push_back({assignment_variable(client, center), 1.0});
    }
}

Solution DiscreteClusteringModel::extract_solution() const
{
    Solution solution;
    solution.open_centers.reserve(center_count_);
    solution.assignments.resize(client_count_);

    for (std::size_t center = 0; center < center_count_; ++center) {
        if (scip_.value(center_variable(center)) > 0.5) {
            solution.open_centers.push_back(center);
        }
    }

    for (std::size_t client = 0; client < client_count_; ++client) {
        bool assignment_found = false;
        for (std::size_t center = 0; center < center_count_; ++center) {
            if (scip_.value(assignment_variable(client, center)) <= 0.5) {
                continue;
            }
            if (assignment_found) {
                throw std::runtime_error("SCIP solution assigns a client to multiple centers");
            }
            solution.assignments[client] = center;
            assignment_found = true;
        }
        if (!assignment_found) {
            throw std::runtime_error("SCIP solution contains an unassigned client");
        }
    }

    solution.objective = scip_.objective_value();
    return solution;
}

} // namespace cluster_aware::detail
