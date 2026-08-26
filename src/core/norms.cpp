#include "core/norms.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cluster_aware {

std::vector<std::size_t> l_largest_magnitudes(std::span<const double> values, std::size_t l)
{
    using Entry = std::pair<double, std::size_t>;

    if (l >= values.size()) {
        std::vector<std::size_t> result(values.size());
        std::iota(result.begin(), result.end(), std::size_t{0});
        return result;
    }

    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> heap;

    for (std::size_t index = 0; index < values.size(); ++index) {
        const Entry candidate{std::abs(values[index]), index};

        if (heap.size() < l) {
            heap.push(candidate);
        } else if (candidate > heap.top()) {
            heap.pop();
            heap.push(candidate);
        }
    }
    std::vector<std::size_t> result;
    result.reserve(l);
    while (!heap.empty()) {
        const auto [value, index] = heap.top();
        static_cast<void>(value);
        heap.pop();

        result.push_back(index);
    }
    return result;
}

TopLNorm::TopLNorm() : l_(1)
{}

TopLNorm::TopLNorm(std::size_t l) : l_(l)
{
    if (l < 1) {
        throw std::invalid_argument("top l norm requires l >= 1");
    }
}

double TopLNorm::operator()(std::span<const double> x) const
{
    const std::vector<std::size_t> largest = l_largest_magnitudes(x, l_);
    double result = 0;
    for (const auto index : largest) {
        result += std::abs(x[index]);
    }
    return result;
}

std::size_t TopLNorm::l() const noexcept
{
    return l_;
}

double L1Norm::operator()(std::span<const double> vec) const
{
    return std::accumulate(vec.begin(), vec.end(), 0.0,
                           [](double sum, double val) { return sum + std::abs(val); });
}

double LInfNorm::operator()(std::span<const double> vec) const
{
    if (vec.empty()) {
        throw std::invalid_argument("L_Inf can not be evaluated over an empty vector");
    }

    auto it = std::ranges::max_element(vec, {}, [](double val) { return std::abs(val); });

    return std::abs(*it);
}

} // namespace cluster_aware
