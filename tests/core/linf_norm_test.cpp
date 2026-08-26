#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "core/norms.hpp"

namespace {

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

    using cluster_aware::LInfNorm;

    std::vector<double> input{111.0, 7.0, -1.0, 19.0, 23.0};

    std::vector<double> input2{0.0, 2.0, -1000.0, 9.0, 7.0};

    bool passed = true;

    constexpr double tolerance = 1.0e-12;
    passed &=
        expect(std::abs((LInfNorm{})(input)-111.0) < tolerance, "The L_INF Norm should be 111.0");
    passed &= expect(std::abs((LInfNorm{})(input2)-1000.0) < tolerance,
                     "The L_INF Norm should be 1000.0");

    bool parameter_error_thrown = false;
    try {
        static_cast<void>(static_cast<void>((LInfNorm{})(std::vector<double>(0))));
    } catch (const std::invalid_argument&) {
        parameter_error_thrown = true;
    }

    passed &= expect(parameter_error_thrown,
                     "Evaluating L_INF on an empty vector needs to throw invalid_argument");

    return passed ? 0 : 1;
}
