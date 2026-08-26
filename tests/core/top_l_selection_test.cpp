#include "core/top_l_selection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr double tolerance = 1.0e-12;

double sorted_top_l(std::vector<double> values, std::size_t l)
{
    std::ranges::sort(values, std::greater<>{});
    const auto count = std::min(l, values.size());
    double result = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        result += values[index];
    }
    return result;
}

bool check_selection(std::string_view name, const std::vector<double>& values, std::size_t l)
{
    const auto selection = cluster_aware::select_top_l_values(values, l);
    if (selection.active_indices.size() != std::min(l, values.size())) {
        std::cerr << name << ": incorrect selection size\n";
        return false;
    }

    std::vector<bool> selected(values.size(), false);
    double selected_sum = 0.0;
    for (const auto index : selection.active_indices) {
        if (index >= values.size() || selected[index]) {
            std::cerr << name << ": selection contains an invalid or duplicate index\n";
            return false;
        }
        selected[index] = true;
        selected_sum += values[index];
    }

    const double expected = sorted_top_l(values, l);
    if (std::abs(selection.value - expected) > tolerance ||
        std::abs(selection.value - selected_sum) > tolerance) {
        std::cerr << name << ": expected " << expected << ", got " << selection.value << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    bool passed = true;
    passed &= check_selection("fractional values", {0.1, 0.9, 0.4, 0.8}, 2);
    passed &= check_selection("ties", {0.5, 0.5, 0.5, 0.1}, 2);
    passed &= check_selection("l larger than dimension", {1.0, 2.0}, 5);
    passed &= check_selection("empty vector", {}, 3);
    passed &= check_selection("signed numerical values", {-1.0e-12, 0.0, 0.4}, 2);

    std::mt19937 generator{0x51EC7};
    std::uniform_real_distribution<double> distribution{-0.01, 1.0};
    for (std::size_t case_index = 0; case_index < 200; ++case_index) {
        const std::size_t value_count = case_index % 17;
        const std::size_t l = 1 + (case_index * 7) % 23;
        std::vector<double> values(value_count);
        std::ranges::generate(values, [&] { return distribution(generator); });
        passed &= check_selection("generated values", values, l);
    }

    try {
        static_cast<void>(cluster_aware::select_top_l_values(std::array{1.0}, 0));
        std::cerr << "zero l was accepted\n";
        passed = false;
    } catch (const std::invalid_argument&) {
    }

    try {
        static_cast<void>(cluster_aware::select_top_l_values(
            std::array{std::numeric_limits<double>::infinity()}, 1));
        std::cerr << "non-finite value was accepted\n";
        passed = false;
    } catch (const std::invalid_argument&) {
    }

    return passed ? 0 : 1;
}
