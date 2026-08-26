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

    using cluster_aware::L1Norm;

    std::vector<double> input{0.0, 2.0, -1000.0, 9.0, 7.0};
    std::vector<double> input2{-111.0, 7.0, -1.0, 19.0, 23.0};

    bool passed = true;

    constexpr double tolerance = 1.0e-12;
    passed &=
        expect(std::abs((L1Norm{})(input)-1018.0) < tolerance, "The L_1 Norm should be 1018.0");
    passed &=
        expect(std::abs((L1Norm{})(input2)-161.0) < tolerance, "The L_1 Norm should be 161.0");

    return passed ? 0 : 1;
}
