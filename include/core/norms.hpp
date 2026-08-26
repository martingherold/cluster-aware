#pragma once

#include <concepts>
#include <cstddef>
#include <span>

namespace cluster_aware {

template <typename T>
concept ValidNorm = requires(const T& norm, std::span<const double> values)
{
    {
        norm(values)
        } -> std::same_as<double>;
};

class TopLNorm {
  public:
    TopLNorm();
    explicit TopLNorm(std::size_t l);
    [[nodiscard]] double operator()(std::span<const double> x) const;
    [[nodiscard]] std::size_t l() const noexcept;

  private:
    std::size_t l_;
};

struct L1Norm {
    [[nodiscard]] double operator()(std::span<const double> vec) const;
};

struct LInfNorm {
    [[nodiscard]] double operator()(std::span<const double> vec) const;
};
} // namespace cluster_aware
