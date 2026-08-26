#include "core/distance_matrix.hpp"
#include "core/instance.hpp"
#include "core/norms.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, std::string_view message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
    }

    return condition;
}

template <typename Exception, typename Function>
bool expect_throws(Function&& function, std::string_view message)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
    }

    std::cerr << "FAILED: " << message << '\n';
    return false;
}

} // namespace

int main()
{
    using cluster_aware::DistanceMatrix;

    bool passed = true;

    // Center 0 has distances [0, 1, 2]; center 1 has [10, 11, 12].
    DistanceMatrix distances{3, 2, {0.0, 1.0, 2.0, 10.0, 11.0, 12.0}};

    passed &= expect(distances.client_count() == 3, "client count should be 3");
    passed &= expect(distances.center_count() == 2, "center count should be 2");
    passed &= expect(distances(2, 1) == 12.0, "unchecked lookup should use client-center order");
    passed &= expect(distances.at(1, 0) == 1.0, "checked lookup should return the stored distance");

    const auto center_distances = distances.distances_to_center(1);
    passed &= expect(center_distances.size() == 3, "a center span should contain every client");
    passed &= expect(center_distances[0] == 10.0 && center_distances[2] == 12.0,
                     "center distances should be contiguous and center-major");

    cluster_aware::Instance instance{std::move(distances), 1, cluster_aware::TopLNorm{2},
                                     cluster_aware::L1Norm{}};
    passed &=
        expect(instance.distances().at(2, 1) == 12.0, "an instance should own its distance matrix");

    passed &= expect_throws<std::out_of_range>(
        [&instance] { static_cast<void>(instance.distances().at(3, 0)); },
        "checked lookup should reject an invalid client");
    passed &= expect_throws<std::invalid_argument>(
        [] {
            DistanceMatrix invalid{2, 2, {0.0, 1.0}};
        },
        "construction should reject the wrong number of values");
    passed &= expect_throws<std::invalid_argument>(
        [] {
            DistanceMatrix invalid{1, 1, {-1.0}};
        },
        "construction should reject negative distances");
    passed &= expect_throws<std::invalid_argument>(
        [] {
            DistanceMatrix invalid{1, 1, {std::numeric_limits<double>::infinity()}};
        },
        "construction should reject non-finite distances");
    passed &= expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(cluster_aware::Instance{DistanceMatrix{1, 1, {0.0}}, 0,
                                                      cluster_aware::TopLNorm{1},
                                                      cluster_aware::L1Norm{}});
        },
        "an instance should reject zero clusters");
    passed &= expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(cluster_aware::Instance{DistanceMatrix{1, 1, {0.0}}, 2,
                                                      cluster_aware::TopLNorm{1},
                                                      cluster_aware::L1Norm{}});
        },
        "an instance should reject more clusters than centers");

    return passed ? 0 : 1;
}
