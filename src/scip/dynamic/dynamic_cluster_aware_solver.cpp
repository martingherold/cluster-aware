#include "scip/dynamic/dynamic_cluster_aware_solver.hpp"

#include "core/solution_checker.hpp"
#include "core/top_l_selection.hpp"
#include "scip/discrete_clustering_model.hpp"
#include "scip/dynamic/dynamic_cluster_aware_constraint_handler.hpp"
#include "scip/dynamic/dynamic_cluster_aware_prob_data.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

template <typename InstanceType>
std::shared_ptr<const InstanceType> require_instance(std::shared_ptr<const InstanceType> instance)
{
    if (instance == nullptr) {
        throw std::invalid_argument("dynamic solver requires an instance");
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

DynamicClusterAwareSolver::DynamicClusterAwareSolver(std::shared_ptr<const TopLInstance> instance)
    : instance_(require_instance(std::move(instance))),
      scip_("dynamic_top_l_clustering", std::make_unique<ClusterAwareProbData>(instance_))
{
    scip_.include_constraint_handler(
        std::make_unique<ConshndlrClusterAwareTopL>(scip_.native_scip()));
    base_model_ = std::make_unique<detail::DiscreteClusteringModel>(
        scip_, instance_->distances(), instance_->number_of_clusters(),
        detail::AssignmentObjective::zero);

    create_cluster_cost_variables();
    add_vars_to_prob_data();
    add_top_l_constraints();
}

DynamicClusterAwareSolver::~DynamicClusterAwareSolver() = default;

void DynamicClusterAwareSolver::create_cluster_cost_variables()
{
    cluster_cost_vars_.reserve(base_model_->center_count());
    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        cluster_cost_vars_.push_back(scip_.create_variable({
            .name = "z_" + std::to_string(center),
            .lower_bound = 0.0,
            .upper_bound = std::nullopt,
            .objective = 1.0,
            .type = ScipWrapper::VariableType::continuous,
        }));
    }
}

void DynamicClusterAwareSolver::add_top_l_constraints()
{
    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        scip_.add_constraint("top_l_" + std::to_string(center),
                             [center](SCIP* scip, SCIP_CONS** constraint, const char* name) {
                                 return SCIP_create_cons_cluster_aware_top_l(scip, constraint, name,
                                                                             center);
                             });
    }
}

void DynamicClusterAwareSolver::add_vars_to_prob_data()
{
    auto& problem_data = dynamic_cast<ClusterAwareProbData&>(scip_.object_problem_data());
    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        problem_data.set_center_var(center,
                                    scip_.raw_variable(base_model_->center_variable(center)));
        problem_data.set_cluster_cost_var(center, scip_.raw_variable(cluster_cost_vars_[center]));
        for (std::size_t client = 0; client < base_model_->client_count(); ++client) {
            problem_data.set_assignment_var(
                client, center,
                scip_.raw_variable(base_model_->assignment_variable(client, center)));
        }
    }
}

SolveResult DynamicClusterAwareSolver::solve()
{
    return solve_impl(nullptr);
}

SolveResult DynamicClusterAwareSolver::solve(const Solution& initial_solution)
{
    return solve_impl(&initial_solution);
}

void DynamicClusterAwareSolver::add_initial_solution(const Solution& initial_solution)
{
    require_valid_initial_solution(*instance_, initial_solution);

    std::vector<ScipWrapper::SolutionValue> values;
    base_model_->append_solution_values(initial_solution, values);
    values.reserve(values.size() + base_model_->center_count());

    const auto& distances = instance_->distances();
    for (std::size_t center = 0; center < base_model_->center_count(); ++center) {
        std::vector<double> entries(base_model_->client_count(), 0.0);
        for (std::size_t client = 0; client < base_model_->client_count(); ++client) {
            if (initial_solution.assignments[client] == center) {
                entries[client] = distances(client, center);
            }
        }

        const double cluster_cost = select_top_l_values(entries, instance_->inner_norm().l()).value;
        if (cluster_cost != 0.0) {
            values.push_back({cluster_cost_vars_[center], cluster_cost});
        }
    }

    if (!scip_.add_initial_solution(values)) {
        throw std::runtime_error("SCIP rejected the dynamic model's initial solution");
    }
}

SolveResult DynamicClusterAwareSolver::solve_impl(const Solution* initial_solution)
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
    const SolutionCheckResult check = check_solution(*instance_, solution, objective_tolerance);
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
