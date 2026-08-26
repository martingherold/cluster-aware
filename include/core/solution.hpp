#pragma once

#include <cstddef>
#include <vector>

namespace cluster_aware {
struct Solution {
    std::vector<std::size_t> open_centers;
    std::vector<std::size_t> assignments;
    double objective = 0.0;
};
} // namespace cluster_aware
