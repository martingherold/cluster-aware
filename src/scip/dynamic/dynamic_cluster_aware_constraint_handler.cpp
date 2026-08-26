#include "scip/dynamic/dynamic_cluster_aware_constraint_handler.hpp"

#include "core/top_l_selection.hpp"
#include "scip/dynamic/dynamic_cluster_aware_prob_data.hpp"
#include "scip/scip_callback.hpp"

#include <scip/scip.h>
#include <scip/scip_lp.h>
#include <scip/scip_mem.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

struct SCIP_ConsData {
    std::size_t center;
};

namespace {

class RowGuard {
  public:
    explicit RowGuard(SCIP* scip) : scip_(scip)
    {}

    RowGuard(const RowGuard&) = delete;
    RowGuard& operator=(const RowGuard&) = delete;

    ~RowGuard()
    {
        if (row_ != nullptr) {
            static_cast<void>(SCIPreleaseRow(scip_, &row_));
        }
    }

    [[nodiscard]] SCIP_ROW** output() noexcept
    {
        return &row_;
    }

    [[nodiscard]] SCIP_ROW* get() const noexcept
    {
        return row_;
    }

  private:
    SCIP* scip_;
    SCIP_ROW* row_ = nullptr;
};

const cluster_aware::ClusterAwareProbData* problem_data(SCIP* scip)
{
    return dynamic_cast<const cluster_aware::ClusterAwareProbData*>(SCIPgetObjProbData(scip));
}

SCIP_RETCODE constraint_center(const cluster_aware::ClusterAwareProbData& data,
                               SCIP_CONS* constraint, std::size_t* center)
{
    if (constraint == nullptr || center == nullptr) {
        SCIPerrorMessage("Top-l callback received a null constraint or output pointer\n");
        return SCIP_INVALIDDATA;
    }

    const SCIP_CONSDATA* constraint_data = SCIPconsGetData(constraint);
    if (constraint_data == nullptr ||
        constraint_data->center >= data.instance().distances().center_count()) {
        SCIPerrorMessage("Top-l constraint has missing or invalid data\n");
        return SCIP_INVALIDDATA;
    }

    *center = constraint_data->center;
    return SCIP_OKAY;
}

std::vector<double> cluster_cost_entries(SCIP* scip,
                                         const cluster_aware::ClusterAwareProbData& data,
                                         std::size_t center, SCIP_SOL* solution)
{
    const auto distances = data.instance().distances().distances_to_center(center);
    std::vector<double> entries;
    entries.reserve(distances.size());

    for (std::size_t client = 0; client < distances.size(); ++client) {
        SCIP_VAR* assignment = data.assignment_var(client, center);
        if (assignment == nullptr) {
            throw std::logic_error("Top-l assignment variable is missing");
        }
        entries.push_back(SCIPgetSolVal(scip, solution, assignment) * distances[client]);
    }
    return entries;
}

SCIP_RETCODE separate_top_l_cluster_cost(SCIP* scip, SCIP_CONSHDLR* constraint_handler,
                                         SCIP_CONS** constraints, int constraint_count,
                                         SCIP_SOL* solution, SCIP_Bool enforce, SCIP_RESULT* result)
{
    if (result == nullptr || constraints == nullptr) {
        SCIPerrorMessage("Top-l separation received a null input\n");
        return SCIP_INVALIDDATA;
    }
    *result = SCIP_DIDNOTFIND;

    const auto* data = problem_data(scip);
    if (data == nullptr) {
        SCIPerrorMessage("ClusterAwareProbData is missing\n");
        return SCIP_INVALIDDATA;
    }

    for (int index = 0; index < constraint_count && *result != SCIP_CUTOFF; ++index) {
        std::size_t center = 0;
        SCIP_CALL(constraint_center(*data, constraints[index], &center));

        const auto distances = data->instance().distances().distances_to_center(center);
        const auto entries = cluster_cost_entries(scip, *data, center, solution);
        const auto selection =
            cluster_aware::select_top_l_values(entries, data->instance().inner_norm().l());

        SCIP_VAR* cluster_cost = data->cluster_cost_var(center);
        if (cluster_cost == nullptr) {
            SCIPerrorMessage("Top-l cluster-cost variable is missing\n");
            return SCIP_INVALIDDATA;
        }
        const SCIP_Real cluster_cost_value = SCIPgetSolVal(scip, solution, cluster_cost);
        if (!SCIPisFeasGT(scip, selection.value, cluster_cost_value)) {
            continue;
        }

        RowGuard row{scip};
        const std::string row_name = "cluster_cost_top_l_" + std::to_string(center);
        SCIP_CALL(SCIPcreateEmptyRowConshdlr(scip, row.output(), constraint_handler,
                                             row_name.c_str(), -SCIPinfinity(scip), 0.0, FALSE,
                                             FALSE, TRUE));
        SCIP_CALL(SCIPcacheRowExtensions(scip, row.get()));
        SCIP_CALL(SCIPaddVarToRow(scip, row.get(), cluster_cost, -1.0));
        for (const std::size_t client : selection.active_indices) {
            SCIP_CALL(SCIPaddVarToRow(scip, row.get(), data->assignment_var(client, center),
                                      distances[client]));
        }
        SCIP_CALL(SCIPflushRowExtensions(scip, row.get()));

        if (enforce || SCIPisCutEfficacious(scip, solution, row.get())) {
            SCIP_Bool infeasible = FALSE;
            SCIP_CALL(SCIPaddRow(scip, row.get(), FALSE, &infeasible));
            *result = infeasible ? SCIP_CUTOFF : SCIP_SEPARATED;
        }
    }

    return SCIP_OKAY;
}

SCIP_RETCODE has_top_l_violation(SCIP* scip, SCIP_CONS** constraints, int constraint_count,
                                 SCIP_SOL* solution, SCIP_Bool* violated)
{
    if (violated == nullptr || constraints == nullptr) {
        SCIPerrorMessage("Top-l feasibility check received a null input\n");
        return SCIP_INVALIDDATA;
    }
    *violated = FALSE;

    const auto* data = problem_data(scip);
    if (data == nullptr) {
        SCIPerrorMessage("ClusterAwareProbData is missing\n");
        return SCIP_INVALIDDATA;
    }

    for (int index = 0; index < constraint_count; ++index) {
        std::size_t center = 0;
        SCIP_CALL(constraint_center(*data, constraints[index], &center));
        const auto entries = cluster_cost_entries(scip, *data, center, solution);
        const auto selection =
            cluster_aware::select_top_l_values(entries, data->instance().inner_norm().l());

        SCIP_VAR* cluster_cost = data->cluster_cost_var(center);
        if (cluster_cost == nullptr) {
            SCIPerrorMessage("Top-l cluster-cost variable is missing\n");
            return SCIP_INVALIDDATA;
        }
        if (SCIPisFeasGT(scip, selection.value, SCIPgetSolVal(scip, solution, cluster_cost))) {
            *violated = TRUE;
            return SCIP_OKAY;
        }
    }

    return SCIP_OKAY;
}

} // namespace

namespace cluster_aware {

SCIP_RETCODE SCIP_create_cons_cluster_aware_top_l(SCIP* scip, SCIP_CONS** constraint,
                                                  const char* name, std::size_t center)
{
    SCIP_CONSHDLR* handler = SCIPfindConshdlr(scip, "topL");
    if (handler == nullptr) {
        return SCIP_PLUGINNOTFOUND;
    }

    SCIP_CONSDATA* constraint_data = nullptr;
    SCIP_CALL(SCIPallocBlockMemory(scip, &constraint_data));
    constraint_data->center = center;

    const SCIP_RETCODE return_code =
        SCIPcreateCons(scip, constraint, name, handler, constraint_data, FALSE, TRUE, TRUE, TRUE,
                       FALSE, FALSE, FALSE, FALSE, TRUE, FALSE);
    if (return_code != SCIP_OKAY) {
        SCIPfreeBlockMemory(scip, &constraint_data);
    }
    return return_code;
}

SCIP_DECL_CONSDELETE(ConshndlrClusterAwareTopL::scip_delete)
{
    static_cast<void>(conshdlr);
    static_cast<void>(cons);
    if (consdata == nullptr || *consdata == nullptr) {
        SCIPerrorMessage("Top-l constraint deletion received no data\n");
        return SCIP_INVALIDDATA;
    }
    SCIPfreeBlockMemory(scip, consdata);
    return SCIP_OKAY;
}

SCIP_DECL_CONSTRANS(ConshndlrClusterAwareTopL::scip_trans)
{
    return detail::invoke_scip_callback("Top-l constraint transformation", [&] {
        if (sourcecons == nullptr || targetcons == nullptr) {
            SCIPerrorMessage("Top-l transformation received a null constraint\n");
            return SCIP_INVALIDDATA;
        }

        const SCIP_CONSDATA* source_data = SCIPconsGetData(sourcecons);
        if (source_data == nullptr) {
            SCIPerrorMessage("Top-l source constraint data is missing\n");
            return SCIP_INVALIDDATA;
        }

        SCIP_CONSDATA* target_data = nullptr;
        SCIP_CALL(SCIPallocBlockMemory(scip, &target_data));
        target_data->center = source_data->center;

        const SCIP_RETCODE return_code =
            SCIPcreateCons(scip, targetcons, SCIPconsGetName(sourcecons), conshdlr, target_data,
                           SCIPconsIsInitial(sourcecons), SCIPconsIsSeparated(sourcecons),
                           SCIPconsIsEnforced(sourcecons), SCIPconsIsChecked(sourcecons),
                           SCIPconsIsPropagated(sourcecons), SCIPconsIsLocal(sourcecons),
                           SCIPconsIsModifiable(sourcecons), SCIPconsIsDynamic(sourcecons),
                           SCIPconsIsRemovable(sourcecons), SCIPconsIsStickingAtNode(sourcecons));
        if (return_code != SCIP_OKAY) {
            SCIPfreeBlockMemory(scip, &target_data);
        }
        return return_code;
    });
}

SCIP_DECL_CONSSEPALP(ConshndlrClusterAwareTopL::scip_sepalp)
{
    return detail::invoke_scip_callback("Top-l LP separation", [&] {
        // Every center is cheap to inspect and exact separation must not skip
        // an obsolete-tagged constraint merely because nusefulconss is smaller.
        static_cast<void>(nusefulconss);
        return separate_top_l_cluster_cost(scip, conshdlr, conss, nconss, nullptr, FALSE, result);
    });
}

SCIP_DECL_CONSENFOLP(ConshndlrClusterAwareTopL::scip_enfolp)
{
    return detail::invoke_scip_callback("Top-l LP enforcement", [&] {
        static_cast<void>(nusefulconss);
        static_cast<void>(solinfeasible);
        SCIP_CALL(
            separate_top_l_cluster_cost(scip, conshdlr, conss, nconss, nullptr, TRUE, result));
        if (*result == SCIP_DIDNOTFIND) {
            *result = SCIP_FEASIBLE;
        }
        return SCIP_OKAY;
    });
}

SCIP_DECL_CONSENFOPS(ConshndlrClusterAwareTopL::scip_enfops)
{
    return detail::invoke_scip_callback("Top-l pseudo-solution enforcement", [&] {
        static_cast<void>(conshdlr);
        static_cast<void>(nusefulconss);
        static_cast<void>(solinfeasible);
        static_cast<void>(objinfeasible);
        if (result == nullptr) {
            return SCIP_INVALIDDATA;
        }

        SCIP_Bool violated = FALSE;
        SCIP_CALL(has_top_l_violation(scip, conss, nconss, nullptr, &violated));
        *result = violated ? SCIP_SOLVELP : SCIP_FEASIBLE;
        return SCIP_OKAY;
    });
}

SCIP_DECL_CONSCHECK(ConshndlrClusterAwareTopL::scip_check)
{
    return detail::invoke_scip_callback("Top-l solution check", [&] {
        static_cast<void>(conshdlr);
        static_cast<void>(checkintegrality);
        static_cast<void>(checklprows);
        if (result == nullptr || conss == nullptr) {
            return SCIP_INVALIDDATA;
        }
        *result = SCIP_FEASIBLE;

        const auto* data = problem_data(scip);
        if (data == nullptr) {
            SCIPerrorMessage("ClusterAwareProbData is missing\n");
            return SCIP_INVALIDDATA;
        }

        for (int index = 0; index < nconss; ++index) {
            std::size_t center = 0;
            SCIP_CALL(constraint_center(*data, conss[index], &center));
            const auto entries = cluster_cost_entries(scip, *data, center, sol);
            const auto selection = select_top_l_values(entries, data->instance().inner_norm().l());

            SCIP_VAR* cluster_cost = data->cluster_cost_var(center);
            if (cluster_cost == nullptr) {
                return SCIP_INVALIDDATA;
            }
            if (!SCIPisFeasGT(scip, selection.value, SCIPgetSolVal(scip, sol, cluster_cost))) {
                continue;
            }

            *result = SCIP_INFEASIBLE;
            if (printreason) {
                SCIP_CALL(SCIPprintCons(scip, conss[index], nullptr));
                SCIPinfoMessage(scip, nullptr,
                                "violation: actual cluster cost exceeds its model variable\n");
            }
            if (!completely) {
                return SCIP_OKAY;
            }
        }
        return SCIP_OKAY;
    });
}

SCIP_DECL_CONSLOCK(ConshndlrClusterAwareTopL::scip_lock)
{
    return detail::invoke_scip_callback("Top-l variable locking", [&] {
        static_cast<void>(conshdlr);
        const auto* data = problem_data(scip);
        if (data == nullptr) {
            SCIPerrorMessage("ClusterAwareProbData is missing\n");
            return SCIP_INVALIDDATA;
        }

        std::size_t center = 0;
        SCIP_CALL(constraint_center(*data, cons, &center));
        const auto& distances = data->instance().distances();
        for (std::size_t client = 0; client < distances.client_count(); ++client) {
            SCIP_VAR* assignment = data->assignment_var(client, center);
            if (assignment == nullptr) {
                SCIPerrorMessage("Top-l assignment variable is missing\n");
                return SCIP_INVALIDDATA;
            }
            SCIP_CALL(SCIPaddVarLocksType(scip, assignment, locktype, nlocksneg, nlockspos));
        }

        SCIP_VAR* cluster_cost = data->cluster_cost_var(center);
        if (cluster_cost == nullptr) {
            SCIPerrorMessage("Top-l cluster-cost variable is missing\n");
            return SCIP_INVALIDDATA;
        }
        SCIP_CALL(SCIPaddVarLocksType(scip, cluster_cost, locktype, nlockspos, nlocksneg));
        return SCIP_OKAY;
    });
}

} // namespace cluster_aware
