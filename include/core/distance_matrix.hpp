#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace cluster_aware {

class DistanceMatrix {
  public:
    // Values are stored center-major: all client distances for one center
    // form one contiguous block.
    DistanceMatrix(std::size_t client_count, std::size_t center_count, std::vector<double> values);

    [[nodiscard]] std::size_t client_count() const noexcept;
    [[nodiscard]] std::size_t center_count() const noexcept;

    // Fast unchecked access for performance-critical solver loops.
    [[nodiscard]] double operator()(std::size_t client, std::size_t center) const noexcept;

    // Checked access for input handling, tests, and other boundary code.
    [[nodiscard]] double at(std::size_t client, std::size_t center) const;

    [[nodiscard]] std::span<const double> distances_to_center(std::size_t center) const;

    [[nodiscard]] std::span<const double> values() const noexcept;

  private:
    [[nodiscard]] std::size_t index(std::size_t client, std::size_t center) const noexcept;

    std::size_t client_count_;
    std::size_t center_count_;
    std::vector<double> values_;
};

} // namespace cluster_aware
