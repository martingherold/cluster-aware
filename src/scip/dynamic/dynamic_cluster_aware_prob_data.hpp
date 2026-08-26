#pragma once

#include "core/instance.hpp"

#include <objscip/objprobdata.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace cluster_aware {

class ClusterAwareProbData final : public scip::ObjProbData {
  public:
    using TopLInstance = Instance<TopLNorm, L1Norm>;

    explicit ClusterAwareProbData(std::shared_ptr<const TopLInstance> instance);

    [[nodiscard]] const TopLInstance& instance() const noexcept;
    [[nodiscard]] const std::shared_ptr<const TopLInstance>& shared_instance() const noexcept;

    [[nodiscard]] SCIP_VAR* assignment_var(std::size_t client, std::size_t center) const;
    void set_assignment_var(std::size_t client, std::size_t center, SCIP_VAR* variable);

    [[nodiscard]] SCIP_VAR* center_var(std::size_t center) const;
    void set_center_var(std::size_t center, SCIP_VAR* variable);

    [[nodiscard]] SCIP_VAR* cluster_cost_var(std::size_t center) const;
    void set_cluster_cost_var(std::size_t center, SCIP_VAR* variable);

    SCIP_RETCODE scip_trans(SCIP* scip, scip::ObjProbData** object_problem_data,
                            SCIP_Bool* delete_object) override;

  private:
    [[nodiscard]] std::size_t connection_index(std::size_t client, std::size_t center) const;

    std::shared_ptr<const TopLInstance> instance_;
    std::vector<SCIP_VAR*> assignment_vars_;
    std::vector<SCIP_VAR*> center_vars_;
    std::vector<SCIP_VAR*> cluster_cost_vars_;
};

} // namespace cluster_aware
