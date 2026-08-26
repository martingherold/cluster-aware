#pragma once

#include "scip/solve_result.hpp"

#include <objscip/objprobdata.h>
#include <objscip/objscip.h>
#include <scip/scip.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cluster_aware {

class ScipWrapper {
  public:
    enum class ObjectiveSense {
        minimize,
        maximize,
    };

    enum class VariableType {
        binary,
        integer,
        continuous,
    };

    class VariableHandle {
      public:
        VariableHandle(const VariableHandle&) = default;
        VariableHandle& operator=(const VariableHandle&) = default;

      private:
        friend class ScipWrapper;

        explicit VariableHandle(std::size_t index) : index_(index)
        {}

        std::size_t index_;
    };

    struct VariableSpec {
        std::string name;
        std::optional<double> lower_bound;
        std::optional<double> upper_bound;
        double objective;
        VariableType type;
    };

    struct LinearTerm {
        VariableHandle variable;
        double coefficient;
    };

    struct SolutionValue {
        VariableHandle variable;
        double value;
    };

    struct LinearConstraintSpec {
        std::string name;
        std::optional<double> lower_bound;
        std::optional<double> upper_bound;
    };

    using ConstraintCreator = std::function<SCIP_RETCODE(SCIP*, SCIP_CONS**, const char*)>;

    explicit ScipWrapper(std::string problem_name,
                         ObjectiveSense objective_sense = ObjectiveSense::minimize);

    explicit ScipWrapper(std::string problem_name, std::unique_ptr<scip::ObjProbData> problem_data,
                         ObjectiveSense objective_sense = ObjectiveSense::minimize);

    ScipWrapper(const ScipWrapper&) = delete;
    ScipWrapper& operator=(const ScipWrapper&) = delete;
    ScipWrapper(ScipWrapper&&) = delete;
    ScipWrapper& operator=(ScipWrapper&&) = delete;

    void include_constraint_handler(std::unique_ptr<scip::ObjConshdlr> handler);

    [[nodiscard]] VariableHandle create_variable(const VariableSpec& spec);
    void add_linear_constraint(const LinearConstraintSpec& spec, std::span<const LinearTerm> terms);
    void add_constraint(std::string name, ConstraintCreator creator);
    [[nodiscard]] bool add_initial_solution(std::span<const SolutionValue> values);
    [[nodiscard]] const SolveStatistics& solve();

    [[nodiscard]] double value(VariableHandle variable) const;
    [[nodiscard]] double objective_value() const;
    [[nodiscard]] const SolveStatistics& statistics() const;
    [[nodiscard]] double feasibility_tolerance() const noexcept;

    [[nodiscard]] scip::ObjProbData& object_problem_data() const;
    [[nodiscard]] SCIP* native_scip() const noexcept;
    [[nodiscard]] SCIP_VAR* raw_variable(VariableHandle variable) const;

  private:
    explicit ScipWrapper(std::string problem_name, ObjectiveSense objective_sense,
                         std::unique_ptr<scip::ObjProbData> problem_data);

    struct ScipDeleter {
        void operator()(SCIP* scip) const noexcept;
    };

    struct VariableDeleter {
        SCIP* scip;

        void operator()(SCIP_VAR* variable) const noexcept;
    };

    struct ConstraintDeleter {
        SCIP* scip;

        void operator()(SCIP_CONS* constraint) const noexcept;
    };

    struct SolutionDeleter {
        SCIP* scip;

        void operator()(SCIP_SOL* solution) const noexcept;
    };

    using ScipPtr = std::unique_ptr<SCIP, ScipDeleter>;
    using VariablePtr = std::unique_ptr<SCIP_VAR, VariableDeleter>;
    using ConstraintPtr = std::unique_ptr<SCIP_CONS, ConstraintDeleter>;
    using SolutionPtr = std::unique_ptr<SCIP_SOL, SolutionDeleter>;

    void require_unsolved(const char* operation) const;
    void require_solution(const char* operation) const;

    ScipPtr scip_;
    std::vector<VariablePtr> variables_;
    SCIP_SOL* best_solution_ = nullptr;
    SolveStatistics statistics_;
    bool solved_ = false;
    scip::ObjProbData* original_problem_data_ = nullptr;
};

} // namespace cluster_aware
