#include "scip/dynamic/dynamic_cluster_aware_prob_data.hpp"

#include "scip/scip_callback.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace cluster_aware {

ClusterAwareProbData::ClusterAwareProbData(std::shared_ptr<const TopLInstance> instance)
    : instance_(std::move(instance))
{
    if (instance_ == nullptr) {
        throw std::invalid_argument("problem data requires an instance");
    }

    const auto& distances = instance_->distances();
    assignment_vars_.resize(distances.values().size(), nullptr);
    center_vars_.resize(distances.center_count(), nullptr);
    cluster_cost_vars_.resize(distances.center_count(), nullptr);
}

const ClusterAwareProbData::TopLInstance& ClusterAwareProbData::instance() const noexcept
{
    return *instance_;
}

const std::shared_ptr<const ClusterAwareProbData::TopLInstance>&
ClusterAwareProbData::shared_instance() const noexcept
{
    return instance_;
}

SCIP_VAR* ClusterAwareProbData::assignment_var(std::size_t client, std::size_t center) const
{
    return assignment_vars_[connection_index(client, center)];
}

void ClusterAwareProbData::set_assignment_var(std::size_t client, std::size_t center,
                                              SCIP_VAR* variable)
{
    if (variable == nullptr) {
        throw std::invalid_argument("assignment variable must not be null");
    }
    assignment_vars_[connection_index(client, center)] = variable;
}

SCIP_VAR* ClusterAwareProbData::center_var(std::size_t center) const
{
    if (center >= instance_->distances().center_count()) {
        throw std::out_of_range("Expected center < n_centers");
    }
    return center_vars_[center];
}

void ClusterAwareProbData::set_center_var(std::size_t center, SCIP_VAR* variable)
{
    if (center >= instance_->distances().center_count()) {
        throw std::out_of_range("Expected center < n_centers");
    }
    if (variable == nullptr) {
        throw std::invalid_argument("center variable must not be null");
    }
    center_vars_[center] = variable;
}

SCIP_VAR* ClusterAwareProbData::cluster_cost_var(std::size_t center) const
{
    if (center >= instance_->distances().center_count()) {
        throw std::out_of_range("Expected center < n_centers");
    }
    return cluster_cost_vars_[center];
}

void ClusterAwareProbData::set_cluster_cost_var(std::size_t center, SCIP_VAR* variable)
{
    if (center >= instance_->distances().center_count()) {
        throw std::out_of_range("Expected center < n_centers");
    }
    if (variable == nullptr) {
        throw std::invalid_argument("cluster-cost variable must not be null");
    }
    cluster_cost_vars_[center] = variable;
}

std::size_t ClusterAwareProbData::connection_index(std::size_t client, std::size_t center) const
{
    if (client >= instance_->distances().client_count()) {
        throw std::out_of_range("Expected 0 <= client < n_clients");
    }
    if (center >= instance_->distances().center_count()) {
        throw std::out_of_range("Expected 0 <= center < n_centers");
    }
    return center * instance_->distances().client_count() + client;
}

SCIP_RETCODE ClusterAwareProbData::scip_trans(SCIP* scip, scip::ObjProbData** object_problem_data,
                                              SCIP_Bool* delete_object)
{
    return detail::invoke_scip_callback("cluster-aware problem transformation", [&] {
        if (object_problem_data == nullptr || delete_object == nullptr) {
            return SCIP_INVALIDDATA;
        }

        auto transformed = std::make_unique<ClusterAwareProbData>(instance_);
        const auto& distances = instance_->distances();

        for (std::size_t center = 0; center < distances.center_count(); ++center) {
            for (std::size_t client = 0; client < distances.client_count(); ++client) {
                SCIP_VAR* transformed_assignment = nullptr;
                SCIP_CALL(SCIPgetTransformedVar(scip, assignment_var(client, center),
                                                &transformed_assignment));
                transformed->set_assignment_var(client, center, transformed_assignment);
            }

            SCIP_VAR* transformed_center = nullptr;
            SCIP_CALL(SCIPgetTransformedVar(scip, center_var(center), &transformed_center));
            transformed->set_center_var(center, transformed_center);

            SCIP_VAR* transformed_cluster_cost = nullptr;
            SCIP_CALL(
                SCIPgetTransformedVar(scip, cluster_cost_var(center), &transformed_cluster_cost));
            transformed->set_cluster_cost_var(center, transformed_cluster_cost);
        }

        *object_problem_data = transformed.release();
        *delete_object = TRUE;
        return SCIP_OKAY;
    });
}

} // namespace cluster_aware
