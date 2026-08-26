#include "core/distance_matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cluster_aware {

DistanceMatrix::DistanceMatrix(std::size_t client_count, std::size_t center_count,
                               std::vector<double> values)
    : client_count_(client_count), center_count_(center_count), values_(std::move(values))
{
    if (client_count_ == 0 || center_count_ == 0) {
        throw std::invalid_argument(
            "a distance matrix requires at least one client and one center");
    }

    if (client_count_ > std::numeric_limits<std::size_t>::max() / center_count_) {
        throw std::length_error("distance matrix dimensions overflow size_t");
    }

    const auto expected_size = client_count_ * center_count_;
    if (values_.size() != expected_size) {
        throw std::invalid_argument("distance matrix value count does not match its dimensions");
    }

    const auto invalid_distance = std::ranges::find_if(
        values_, [](double distance) { return !std::isfinite(distance) || distance < 0.0; });
    if (invalid_distance != values_.end()) {
        throw std::invalid_argument("distance matrix values must be finite and nonnegative");
    }
}

std::size_t DistanceMatrix::client_count() const noexcept
{
    return client_count_;
}

std::size_t DistanceMatrix::center_count() const noexcept
{
    return center_count_;
}

double DistanceMatrix::operator()(std::size_t client, std::size_t center) const noexcept
{
    assert(client < client_count_);
    assert(center < center_count_);
    return values_[index(client, center)];
}

double DistanceMatrix::at(std::size_t client, std::size_t center) const
{
    if (client >= client_count_ || center >= center_count_) {
        throw std::out_of_range("distance matrix index is out of range");
    }

    return values_[index(client, center)];
}

std::span<const double> DistanceMatrix::distances_to_center(std::size_t center) const
{
    if (center >= center_count_) {
        throw std::out_of_range("distance matrix center index is out of range");
    }

    return std::span<const double>{values_}.subspan(center * client_count_, client_count_);
}

std::span<const double> DistanceMatrix::values() const noexcept
{
    return values_;
}

std::size_t DistanceMatrix::index(std::size_t client, std::size_t center) const noexcept
{
    return center * client_count_ + client;
}

} // namespace cluster_aware
