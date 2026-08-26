#pragma once

#include "distance_matrix.hpp"
#include "norms.hpp"

#include <stdexcept>
#include <utility>

namespace cluster_aware {

template <ValidNorm InnerNorm, ValidNorm OuterNorm> class Instance {
  public:
    Instance(DistanceMatrix distances, std::size_t number_of_clusters, InnerNorm inner_norm,
             OuterNorm outer_norm)
        : distances_(std::move(distances)), inner_norm_(std::move(inner_norm)),
          outer_norm_(std::move(outer_norm)), number_of_clusters_(number_of_clusters)
    {
        if (number_of_clusters_ > distances_.center_count() || number_of_clusters_ < 1) {
            throw std::invalid_argument(
                "number_of clusters has to be in [1,number_of_potential_centers]");
        }
    }

    [[nodiscard]] const DistanceMatrix& distances() const noexcept
    {
        return distances_;
    }

    [[nodiscard]] const InnerNorm& inner_norm() const noexcept
    {
        return inner_norm_;
    }

    [[nodiscard]] const OuterNorm& outer_norm() const noexcept
    {
        return outer_norm_;
    }

    [[nodiscard]] std::size_t number_of_clusters() const noexcept
    {
        return number_of_clusters_;
    }

  private:
    DistanceMatrix distances_;
    [[no_unique_address]] InnerNorm inner_norm_;
    [[no_unique_address]] OuterNorm outer_norm_;
    std::size_t number_of_clusters_;
};

} // namespace cluster_aware
