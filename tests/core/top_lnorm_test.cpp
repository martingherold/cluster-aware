#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "core/norms.hpp"

namespace {

static_assert(cluster_aware::ValidNorm<cluster_aware::TopLNorm>);
static_assert(std::is_copy_assignable_v<cluster_aware::TopLNorm>);
static_assert(!std::is_convertible_v<std::size_t, cluster_aware::TopLNorm>);

bool expect(bool condition, std::string_view message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
    }

    return condition;
}
} // namespace

int main()
{

    using cluster_aware::TopLNorm;

    std::vector<double> input{0.0, 2.0, -1000.0, 9.0, 7.0};
    std::vector<double> fractional_input{1.1, -1.9, 0.4};

    bool passed = true;

    constexpr double tolerance = 1.0e-12;
    passed &= expect(std::abs((TopLNorm(1))(input)-1000.0) < tolerance,
                     "The Top-1 Norm should be 1000.0");
    passed &= expect(std::abs((TopLNorm(2))(input)-1009.0) < tolerance,
                     "The Top-2 Norm should be 1009.0");
    passed &= expect(std::abs((TopLNorm(3))(input)-1016.0) < tolerance,
                     "The Top-3 Norm should be 10016.0");
    passed &= expect(std::abs((TopLNorm(1))(fractional_input)-1.9) < tolerance,
                     "The Top-1 Norm should preserve fractional values");
    passed &= expect(std::abs((TopLNorm(2))(fractional_input)-3.0) < tolerance,
                     "The Top-2 Norm should preserve fractional values");

    bool parameter_error_thrown = false;
    try {
        static_cast<void>(TopLNorm(0));
    } catch (const std::invalid_argument&) {
        parameter_error_thrown = true;
    }

    passed &= expect(parameter_error_thrown, "invalid l argument needs to throw invaild_argument");

    return passed ? 0 : 1;
}
