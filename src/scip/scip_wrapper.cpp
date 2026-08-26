#include "scip/scip_wrapper.hpp"

#include <scip/cons_linear.h>
#include <scip/scip.h>
#include <scip/scip_numerics.h>
#include <scip/scip_sol.h>
#include <scip/scip_solvingstats.h>
#include <scip/scip_timing.h>
#include <scip/scipdefplugins.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void check_scip(SCIP_RETCODE return_code, const char* operation)
{
    if (return_code != SCIP_OKAY) {
        throw std::runtime_error(std::string{operation} + " failed with SCIP return code " +
                                 std::to_string(static_cast<int>(return_code)));
    }
}

SCIP_VARTYPE to_scip_variable_type(cluster_aware::ScipWrapper::VariableType type)
{
    using VariableType = cluster_aware::ScipWrapper::VariableType;

    switch (type) {
    case VariableType::binary:
        return SCIP_VARTYPE_BINARY;
    case VariableType::integer:
        return SCIP_VARTYPE_INTEGER;
    case VariableType::continuous:
        return SCIP_VARTYPE_CONTINUOUS;
    }
    throw std::invalid_argument("unknown variable type");
}

cluster_aware::SolveStatus to_solve_status(SCIP_STATUS status)
{
    using cluster_aware::SolveStatus;

    switch (status) {
    case SCIP_STATUS_UNKNOWN:
        return SolveStatus::unknown;
    case SCIP_STATUS_OPTIMAL:
        return SolveStatus::optimal;
    case SCIP_STATUS_INFEASIBLE:
        return SolveStatus::infeasible;
    case SCIP_STATUS_UNBOUNDED:
        return SolveStatus::unbounded;
    case SCIP_STATUS_INFORUNBD:
        return SolveStatus::infeasible_or_unbounded;
    case SCIP_STATUS_USERINTERRUPT:
        return SolveStatus::user_interrupt;
    case SCIP_STATUS_TERMINATE:
        return SolveStatus::termination_signal;
    case SCIP_STATUS_NODELIMIT:
        return SolveStatus::node_limit;
    case SCIP_STATUS_TOTALNODELIMIT:
        return SolveStatus::total_node_limit;
    case SCIP_STATUS_STALLNODELIMIT:
        return SolveStatus::stall_node_limit;
    case SCIP_STATUS_TIMELIMIT:
        return SolveStatus::time_limit;
    case SCIP_STATUS_MEMLIMIT:
        return SolveStatus::memory_limit;
    case SCIP_STATUS_GAPLIMIT:
        return SolveStatus::gap_limit;
    case SCIP_STATUS_PRIMALLIMIT:
        return SolveStatus::primal_limit;
    case SCIP_STATUS_DUALLIMIT:
        return SolveStatus::dual_limit;
    case SCIP_STATUS_SOLLIMIT:
        return SolveStatus::solution_limit;
    case SCIP_STATUS_BESTSOLLIMIT:
        return SolveStatus::best_solution_limit;
    case SCIP_STATUS_RESTARTLIMIT:
        return SolveStatus::restart_limit;
    }
    return SolveStatus::unknown;
}

} // namespace

namespace cluster_aware {

void ScipWrapper::ScipDeleter::operator()(SCIP* scip) const noexcept
{
    if (scip != nullptr) {
        static_cast<void>(SCIPfree(&scip));
    }
}

void ScipWrapper::VariableDeleter::operator()(SCIP_VAR* variable) const noexcept
{
    if (scip != nullptr && variable != nullptr) {
        static_cast<void>(SCIPreleaseVar(scip, &variable));
    }
}

void ScipWrapper::ConstraintDeleter::operator()(SCIP_CONS* constraint) const noexcept
{
    if (scip != nullptr && constraint != nullptr) {
        static_cast<void>(SCIPreleaseCons(scip, &constraint));
    }
}

void ScipWrapper::SolutionDeleter::operator()(SCIP_SOL* solution) const noexcept
{
    if (scip != nullptr && solution != nullptr) {
        static_cast<void>(SCIPfreeSol(scip, &solution));
    }
}

ScipWrapper::ScipWrapper(std::string problem_name, ObjectiveSense objective_sense)
    : ScipWrapper(std::move(problem_name), objective_sense, nullptr)
{}

ScipWrapper::ScipWrapper(std::string problem_name, std::unique_ptr<scip::ObjProbData> problem_data,
                         ObjectiveSense objective_sense)
    : ScipWrapper(std::move(problem_name), objective_sense, std::move(problem_data))
{}

ScipWrapper::ScipWrapper(std::string problem_name, ObjectiveSense objective_sense,
                         std::unique_ptr<scip::ObjProbData> problem_data)
{
    SCIP* raw_scip = nullptr;
    check_scip(SCIPcreate(&raw_scip), "SCIPcreate");
    scip_.reset(raw_scip);

    check_scip(SCIPincludeDefaultPlugins(scip_.get()), "SCIPincludeDefaultPlugins");

    const SCIP_OBJSENSE scip_objective_sense = objective_sense == ObjectiveSense::minimize
                                                   ? SCIP_OBJSENSE_MINIMIZE
                                                   : SCIP_OBJSENSE_MAXIMIZE;
    if (problem_data != nullptr) {
        original_problem_data_ = problem_data.get();

        check_scip(SCIPcreateObjProb(scip_.get(), problem_name.c_str(), problem_data.get(), TRUE),
                   "SCIPcreateObjProb");

        problem_data.release();
    } else {
        check_scip(SCIPcreateProbBasic(scip_.get(), problem_name.c_str()), "SCIPcreateProbBasic");
    }
    check_scip(SCIPsetObjsense(scip_.get(), scip_objective_sense), "SCIPsetObjsense");
    // check_scip(SCIPsetIntParam(scip_.get(), "display/verblevel", 0), "SCIPsetIntParam");
}

void ScipWrapper::include_constraint_handler(std::unique_ptr<scip::ObjConshdlr> handler)
{
    assert(handler != nullptr);
    check_scip(SCIPincludeObjConshdlr(scip_.get(), handler.get(), TRUE), "SCIPincludeObjConshndlr");
    handler.release();
}

ScipWrapper::VariableHandle ScipWrapper::create_variable(const VariableSpec& spec)
{
    require_unsolved("create_variable");

    const double lower_bound = spec.lower_bound.value_or(-SCIPinfinity(scip_.get()));
    const double upper_bound = spec.upper_bound.value_or(SCIPinfinity(scip_.get()));
    const auto create_operation = "SCIPcreateVarBasic(" + spec.name + ")";
    const auto add_operation = "SCIPaddVar(" + spec.name + ")";

    SCIP_VAR* raw_variable = nullptr;
    check_scip(SCIPcreateVarBasic(scip_.get(), &raw_variable, spec.name.c_str(), lower_bound,
                                  upper_bound, spec.objective, to_scip_variable_type(spec.type)),
               create_operation.c_str());

    VariablePtr variable{raw_variable, VariableDeleter{scip_.get()}};
    check_scip(SCIPaddVar(scip_.get(), variable.get()), add_operation.c_str());

    const auto index = variables_.size();
    variables_.push_back(std::move(variable));
    return VariableHandle{index};
}

void ScipWrapper::add_linear_constraint(const LinearConstraintSpec& spec,
                                        std::span<const LinearTerm> terms)
{
    require_unsolved("add_linear_constraint");
    if (terms.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("linear constraint has too many terms for SCIP");
    }

    std::vector<SCIP_VAR*> variables;
    std::vector<SCIP_Real> coefficients;
    variables.reserve(terms.size());
    coefficients.reserve(terms.size());

    for (const auto& term : terms) {
        variables.push_back(raw_variable(term.variable));
        coefficients.push_back(term.coefficient);
    }

    const double lower_bound = spec.lower_bound.value_or(-SCIPinfinity(scip_.get()));
    const double upper_bound = spec.upper_bound.value_or(SCIPinfinity(scip_.get()));
    const auto create_operation = "SCIPcreateConsBasicLinear(" + spec.name + ")";
    const auto add_operation = "SCIPaddCons(" + spec.name + ")";

    SCIP_CONS* raw_constraint = nullptr;
    check_scip(SCIPcreateConsBasicLinear(scip_.get(), &raw_constraint, spec.name.c_str(),
                                         static_cast<int>(variables.size()), variables.data(),
                                         coefficients.data(), lower_bound, upper_bound),
               create_operation.c_str());

    ConstraintPtr constraint{raw_constraint, ConstraintDeleter{scip_.get()}};
    check_scip(SCIPaddCons(scip_.get(), constraint.get()), add_operation.c_str());
}

void ScipWrapper::add_constraint(std::string name, ConstraintCreator creator)
{
    require_unsolved("add_constraint");

    if (!creator) {
        throw std::invalid_argument("constraint creator must not be empty");
    }

    SCIP_CONS* raw_constraint = nullptr;
    const auto create_operation = "create constraint (" + name + ")";
    const SCIP_RETCODE create_result = creator(scip_.get(), &raw_constraint, name.c_str());

    ConstraintPtr constraint{raw_constraint, ConstraintDeleter{scip_.get()}};
    check_scip(create_result, create_operation.c_str());

    if (constraint == nullptr) {
        throw std::runtime_error("constraint creator returned success but no constraint");
    }

    const auto add_operation = "SCIPaddCons(" + name + ")";
    check_scip(SCIPaddCons(scip_.get(), constraint.get()), add_operation.c_str());
}

bool ScipWrapper::add_initial_solution(std::span<const SolutionValue> values)
{
    require_unsolved("add_initial_solution");

    SCIP_SOL* raw_solution = nullptr;
    check_scip(SCIPcreateSol(scip_.get(), &raw_solution, nullptr), "SCIPcreateSol");
    SolutionPtr solution{raw_solution, SolutionDeleter{scip_.get()}};

    std::vector<bool> value_was_set(variables_.size(), false);
    for (const auto& [variable, value] : values) {
        if (variable.index_ >= variables_.size()) {
            throw std::out_of_range("initial solution variable index is out of range");
        }
        if (value_was_set[variable.index_]) {
            throw std::invalid_argument("initial solution sets a variable more than once");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("initial solution values must be finite");
        }

        value_was_set[variable.index_] = true;
        check_scip(SCIPsetSolVal(scip_.get(), solution.get(), raw_variable(variable), value),
                   "SCIPsetSolVal");
    }

    SCIP_Bool feasible = FALSE;
    check_scip(SCIPcheckSolOrig(scip_.get(), solution.get(), &feasible, FALSE, TRUE),
               "SCIPcheckSolOrig");
    if (!feasible) {
        return false;
    }

    SCIP_Bool stored = FALSE;
    raw_solution = solution.release();
    const SCIP_RETCODE add_result = SCIPaddSolFree(scip_.get(), &raw_solution, &stored);
    if (add_result != SCIP_OKAY) {
        solution.reset(raw_solution);
        check_scip(add_result, "SCIPaddSolFree");
    }
    return stored == TRUE;
}

const SolveStatistics& ScipWrapper::solve()
{
    require_unsolved("solve");
    solved_ = true;

    check_scip(SCIPsolve(scip_.get()), "SCIPsolve");
    best_solution_ = SCIPgetBestSol(scip_.get());

    statistics_.status = to_solve_status(SCIPgetStatus(scip_.get()));
    statistics_.has_primal_solution = best_solution_ != nullptr;
    statistics_.solving_time_seconds = SCIPgetSolvingTime(scip_.get());
    const SCIP_Longint node_count = SCIPgetNNodes(scip_.get());
    statistics_.node_count = static_cast<std::uint64_t>(std::max<SCIP_Longint>(node_count, 0));

    if (statistics_.has_primal_solution) {
        statistics_.primal_bound = SCIPgetPrimalbound(scip_.get());
        statistics_.relative_gap = SCIPgetGap(scip_.get());
    }
    if (statistics_.status != SolveStatus::unknown) {
        statistics_.dual_bound = SCIPgetDualbound(scip_.get());
    }

    return statistics_;
}

scip::ObjProbData& ScipWrapper::object_problem_data() const
{
    if (original_problem_data_ == nullptr) {
        throw std::logic_error("ScipWrapper has no object problem data");
    }
    return *original_problem_data_;
}

double ScipWrapper::value(VariableHandle variable) const
{
    require_solution("value");
    return SCIPgetSolVal(scip_.get(), best_solution_, raw_variable(variable));
}

double ScipWrapper::objective_value() const
{
    require_solution("objective_value");
    return SCIPgetSolOrigObj(scip_.get(), best_solution_);
}

const SolveStatistics& ScipWrapper::statistics() const
{
    if (!solved_) {
        throw std::logic_error("ScipWrapper::statistics requires solve to have been called");
    }
    return statistics_;
}

double ScipWrapper::feasibility_tolerance() const noexcept
{
    return SCIPfeastol(scip_.get());
}

SCIP_VAR* ScipWrapper::raw_variable(VariableHandle variable) const
{
    return variables_.at(variable.index_).get();
}

SCIP* ScipWrapper::native_scip() const noexcept
{
    return scip_.get();
}

void ScipWrapper::require_unsolved(const char* operation) const
{
    if (solved_) {
        throw std::logic_error(std::string{"ScipWrapper::"} + operation +
                               " cannot be called after solve");
    }
}

void ScipWrapper::require_solution(const char* operation) const
{
    if (!solved_ || best_solution_ == nullptr) {
        throw std::logic_error(std::string{"ScipWrapper::"} + operation +
                               " requires a successfully solved problem");
    }
}

} // namespace cluster_aware
