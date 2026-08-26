#pragma once

#include <objscip/objscip.h>

#include <cstddef>

namespace cluster_aware {

class ConshndlrClusterAwareTopL final : public scip::ObjConshdlr {
  public:
    explicit ConshndlrClusterAwareTopL(SCIP* scip)
        : ObjConshdlr(scip, "topL", "cluster-aware Top-l constraints", separation_priority,
                      enforcement_priority, check_priority, separation_frequency,
                      propagation_frequency, eager_frequency, maximum_presolving_rounds, FALSE,
                      FALSE, TRUE, SCIP_PROPTIMING_BEFORELP, SCIP_PRESOLTIMING_FAST)
    {}

    ~ConshndlrClusterAwareTopL() override = default;

    SCIP_DECL_CONSDELETE(scip_delete) override;
    SCIP_DECL_CONSTRANS(scip_trans) override;
    SCIP_DECL_CONSSEPALP(scip_sepalp) override;
    SCIP_DECL_CONSENFOLP(scip_enfolp) override;
    SCIP_DECL_CONSENFOPS(scip_enfops) override;
    SCIP_DECL_CONSCHECK(scip_check) override;
    SCIP_DECL_CONSLOCK(scip_lock) override;

  private:
    // Match SCIP's nonlinear/TSP-style ordering: separate early, enforce and
    // check after handlers that can resolve simpler linear violations.
    static constexpr int separation_priority = 1'000'000;
    static constexpr int enforcement_priority = -2'000'000;
    static constexpr int check_priority = -2'000'000;
    static constexpr int separation_frequency = 1;
    static constexpr int propagation_frequency = -1;
    static constexpr int eager_frequency = 1;
    static constexpr int maximum_presolving_rounds = 0;
};

SCIP_RETCODE SCIP_create_cons_cluster_aware_top_l(SCIP* scip, SCIP_CONS** constraint,
                                                  const char* name, std::size_t center);

} // namespace cluster_aware
