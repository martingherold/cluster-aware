#include "scip/compact_cluster_aware_solver.hpp"

#include "core/solution_checker.hpp"
#include "scip/discrete_clustering_model.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

template <typename InstanceType>
std::shared_ptr<const InstanceType> require_instance(std::shared_ptr<const InstanceType> instance)
{
    if (instance == nullptr) {
        throw std::invalid_argument("compact solver requires an instance");
    }
    return instance;
}

template <typename InstanceType>
void require_valid_initial_solution(const InstanceType& instance,
                                    const cluster_aware::Solution& solution)
{
    const auto check = cluster_aware::check_solution(instance, solution);
    if (!check.valid()) {
        throw std::invalid_argument("invalid initial solution: " + check.message);
    }
}

} // namespace

namespace cluster_aware {

CompactClusterAwareSolver::CompactClusterAwareSolver(std::shared_ptr<const L1Instance> instance)
    : instance_(require_instance(std::move(instance))), scip_("compact_l1_l1_clustering")
{
    const auto& retained_instance = *std::get<std::shared_ptr<const L1Instance>>(instance_);
    base_model_ = std::make_unique<detail::DiscreteClusteringModel>(
        scip_, retained_instance.distances(), retained_instance.number_of_clusters(),
        detail::AssignmentObjective::distance);
}

CompactClusterAwareSolver::CompactClusterAwareSolver(std::shared_ptr<const TopLInstance> instance)
    : instance_(require_instance(std::move(instance))), scip_("compact_top_l_l1_clustering")
{
    const auto& retained_instance = *std::get<std::shared_ptr<const TopLInstance>>(instance_);
    base_model_ = std::make_unique<detail::DiscreteClusteringModel>(
        scip_, retained_instance.distances(), retained_instance.number_of_clusters(),
        detail::AssignmentObjective::zero);

    create_top_l_excess_variables();
    create_top_l_auxiliary_variables();
    add_cluster_cost_constraints(retained_instance);
    add_compact_top_l_constraints(retained_instance);
}

CompactClusterAwareSolver::CompactClusterAwareSolver(const L1Instance& instance)
    : CompactClusterAwareSolver(std::make_shared<const L1Instance>(instance))
{}

CompactClusterAwareSolver::CompactClusterAwareSolver(const TopLInstance& instance)
    : CompactClusterAwareSolver(std::make_shared<const TopLInstance>(instance))
{}

CompactClusterAwareSolver::~CompactClusterAwareSolver() = default;

void CompactClusterAwareSolver::create_top_l_excess_variables()
{
    excess_vars_.reserve(base_model_->center_count() * base_model_->client_count());
    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        for (std::size_t client = 0; client < base_model_->client_count(); ++client) {
            excess_vars_.push_back(scip_.create_variable({
                .name = "w_" + std::to_string(client) + "," + std::to_string(center),
                .lower_bound = 0.0,
                .upper_bound = std::nullopt,
                .objective = 0.0,
                .type = ScipWrapper::VariableType::continuous,
            }));
        }
    }
}

void CompactClusterAwareSolver::create_top_l_auxiliary_variables()
{
    threshold_vars_.reserve(base_model_->center_count());
    cluster_cost_vars_.reserve(base_model_->center_count());

    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        threshold_vars_.push_back(scip_.create_variable({
            .name = "t_" + std::to_string(center),
            .lower_bound = 0.0,
            .upper_bound = std::nullopt,
            .objective = 0.0,
            .type = ScipWrapper::VariableType::continuous,
        }));

        cluster_cost_vars_.push_back(scip_.create_variable({
            .name = "z_" + std::to_string(center),
            .lower_bound = 0.0,
            .upper_bound = std::nullopt,
            .objective = 1.0,
            .type = ScipWrapper::VariableType::continuous,
        }));
    }
}

void CompactClusterAwareSolver::add_cluster_cost_constraints(const TopLInstance& instance)
{
    std::vector<ScipWrapper::LinearTerm> terms;
    terms.reserve(base_model_->client_count() + 2);
    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        terms.clear();
        for (std::size_t client = 0; client < base_model_->client_count(); ++client) {
            terms.push_back({excess_vars_[base_model_->assignment_index(client, center)], 1.0});
        }
        terms.push_back({threshold_vars_[center], static_cast<double>(instance.inner_norm().l())});
        terms.push_back({cluster_cost_vars_[center], -1.0});
        scip_.add_linear_constraint(
            {
                .name = "define_cluster_cost_" + std::to_string(center),
                .lower_bound = std::nullopt,
                .upper_bound = 0.0,
            },
            terms);
    }
}

void CompactClusterAwareSolver::add_compact_top_l_constraints(const TopLInstance& instance)
{
    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        for (std::size_t client = 0; client < base_model_->client_count(); ++client) {
            const std::array terms{
                ScipWrapper::LinearTerm{base_model_->assignment_variable(client, center),
                                        instance.distances().at(client, center)},
                ScipWrapper::LinearTerm{threshold_vars_[center], -1.0},
                ScipWrapper::LinearTerm{excess_vars_[base_model_->assignment_index(client, center)],
                                        -1.0},
            };
            scip_.add_linear_constraint(
                {
                    .name = "define_excess_client_" + std::to_string(client) + "_center_" +
                            std::to_string(center),
                    .lower_bound = std::nullopt,
                    .upper_bound = 0.0,
                },
                terms);
        }
    }
}

SolveResult CompactClusterAwareSolver::solve()
{
    return solve_impl(nullptr);
}

SolveResult CompactClusterAwareSolver::solve(const Solution& initial_solution)
{
    return solve_impl(&initial_solution);
}

void CompactClusterAwareSolver::add_initial_solution(const Solution& initial_solution)
{
    std::vector<ScipWrapper::SolutionValue> values;
    base_model_->append_solution_values(initial_solution, values);

    std::visit(
        [&](const auto& instance) {
            require_valid_initial_solution(*instance, initial_solution);
            using InstanceType = std::remove_cvref_t<decltype(*instance)>;
            if constexpr (std::is_same_v<InstanceType, TopLInstance>) {
                append_top_l_solution_values(*instance, initial_solution, values);
            }
        },
        instance_);

    if (!scip_.add_initial_solution(values)) {
        throw std::runtime_error("SCIP rejected the compact model's initial solution");
    }
}

void CompactClusterAwareSolver::append_top_l_solution_values(
    const TopLInstance& instance, const Solution& initial_solution,
    std::vector<ScipWrapper::SolutionValue>& values) const
{
    const auto& distances = instance.distances();
    const std::size_t ell = instance.inner_norm().l();
    std::vector<std::vector<double>> cluster_distances(base_model_->center_count());
    for (std::size_t client = 0; client < base_model_->client_count(); ++client) {
        const std::size_t center = initial_solution.assignments[client];
        cluster_distances[center].push_back(distances(client, center));
    }

    values.reserve(values.size() + base_model_->center_count() * 2 + base_model_->client_count());
    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        auto sorted_distances = cluster_distances[center];
        std::ranges::sort(sorted_distances, std::greater<>{});
        const double threshold = sorted_distances.size() >= ell ? sorted_distances[ell - 1] : 0.0;
        if (threshold != 0.0) {
            values.push_back({threshold_vars_[center], threshold});
        }

        double cluster_cost = static_cast<double>(ell) * threshold;
        for (std::size_t client = 0; client < base_model_->client_count(); ++client) {
            if (initial_solution.assignments[client] != center) {
                continue;
            }
            const double excess = std::max(distances(client, center) - threshold, 0.0);
            if (excess != 0.0) {
                values.push_back(
                    {excess_vars_[base_model_->assignment_index(client, center)], excess});
            }
            cluster_cost += excess;
        }
        if (cluster_cost != 0.0) {
            values.push_back({cluster_cost_vars_[center], cluster_cost});
        }
    }
}

SolveResult CompactClusterAwareSolver::solve_impl(const Solution* initial_solution)
{
    if (initial_solution != nullptr) {
        add_initial_solution(*initial_solution);
    }

    SolveResult result;
    result.statistics = scip_.solve();
    if (!result.statistics.has_primal_solution) {
        return result;
    }

    Solution solution = base_model_->extract_solution();
    result.model_objective = solution.objective;

    const double feasibility_tolerance = scip_.feasibility_tolerance();
    const ObjectiveTolerance objective_tolerance{
        feasibility_tolerance * static_cast<double>(std::max<std::size_t>(
                                    1, base_model_->client_count() + base_model_->center_count())),
        feasibility_tolerance,
    };
    const SolutionCheckResult check = std::visit(
        [&](const auto& instance) {
            return check_solution(*instance, solution, objective_tolerance);
        },
        instance_);

    if (!check.feasible) {
        throw std::runtime_error("SCIP returned a structurally invalid solution: " + check.message);
    }
    if (!check.objective_matches) {
        throw std::runtime_error(
            "SCIP model objective disagrees with the independently recomputed objective");
    }

    solution.objective = check.recomputed_objective;
    result.solution = std::move(solution);
    return result;
}

} // namespace cluster_aware
