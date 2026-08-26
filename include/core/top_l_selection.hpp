#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace cluster_aware {

// Describes the maximum sum over subsets containing min(l, values.size())
// entries. For nonnegative values, this is the Top-l norm and supplies the
// exact subset used by the dynamic formulation's separation oracle.
struct TopLSelection {
    double value;
    std::vector<std::size_t> active_indices;
};

[[nodiscard]] TopLSelection select_top_l_values(std::span<const double> values, std::size_t l);

} // namespace cluster_aware
