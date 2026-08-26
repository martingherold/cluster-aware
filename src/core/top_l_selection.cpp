#include "core/top_l_selection.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace cluster_aware {

TopLSelection select_top_l_values(std::span<const double> values, std::size_t l)
{
    if (l == 0) {
        throw std::invalid_argument("Top-l selection requires l >= 1");
    }
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Top-l selection values must be finite");
        }
    }

    const std::size_t selected_count = std::min(l, values.size());
    std::vector<std::size_t> indices;
    indices.reserve(selected_count);

    if (selected_count == values.size()) {
        indices.resize(values.size());
        std::iota(indices.begin(), indices.end(), std::size_t{0});
    } else {
        using Entry = std::pair<double, std::size_t>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<>> heap;

        for (std::size_t index = 0; index < values.size(); ++index) {
            const Entry candidate{values[index], index};
            if (heap.size() < selected_count) {
                heap.push(candidate);
            } else if (candidate > heap.top()) {
                heap.pop();
                heap.push(candidate);
            }
        }

        while (!heap.empty()) {
            indices.push_back(heap.top().second);
            heap.pop();
        }
    }

    double value = 0.0;
    for (const std::size_t index : indices) {
        value += values[index];
    }
    return {value, std::move(indices)};
}

} // namespace cluster_aware
