#pragma once

#include "core/instance.hpp"
#include "core/solution.hpp"

#include <concepts>
#include <cstddef>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

namespace cluster_aware {

class JsonFormatError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

template <typename Norm>
concept JsonSerializableNorm =
    std::same_as<Norm, L1Norm> || std::same_as<Norm, LInfNorm> || std::same_as<Norm, TopLNorm>;

using JsonInstance =
    std::variant<Instance<L1Norm, L1Norm>, Instance<L1Norm, LInfNorm>, Instance<L1Norm, TopLNorm>,
                 Instance<LInfNorm, L1Norm>, Instance<LInfNorm, LInfNorm>,
                 Instance<LInfNorm, TopLNorm>, Instance<TopLNorm, L1Norm>,
                 Instance<TopLNorm, LInfNorm>, Instance<TopLNorm, TopLNorm>>;

struct ParsedInstance {
    std::string id;
    JsonInstance instance;
};

struct ParsedSolution {
    std::string instance_id;
    Solution solution;
};

[[nodiscard]] ParsedInstance parse_instance_json(std::istream& input);
[[nodiscard]] ParsedSolution parse_solution_json(std::istream& input);

void write_instance_json(std::ostream& output, const ParsedInstance& parsed_instance);
void write_solution_json(std::ostream& output, std::string_view instance_id,
                         const Solution& solution);
void write_solution_json(std::ostream& output, const ParsedSolution& parsed_solution);

namespace detail {

enum class JsonNormType {
    l1,
    l_inf,
    top_l,
};

struct JsonNormSpec {
    JsonNormType type;
    std::optional<std::size_t> l;
};

void write_instance_json_impl(std::ostream& output, std::string_view id,
                              const DistanceMatrix& distances, std::size_t number_of_clusters,
                              const JsonNormSpec& inner_norm, const JsonNormSpec& outer_norm);

template <JsonSerializableNorm Norm> [[nodiscard]] JsonNormSpec describe_norm(const Norm& norm)
{
    if constexpr (std::same_as<Norm, L1Norm>) {
        return {JsonNormType::l1, std::nullopt};
    } else if constexpr (std::same_as<Norm, LInfNorm>) {
        return {JsonNormType::l_inf, std::nullopt};
    } else {
        return {JsonNormType::top_l, norm.l()};
    }
}

} // namespace detail

template <JsonSerializableNorm InnerNorm, JsonSerializableNorm OuterNorm>
void write_instance_json(std::ostream& output, std::string_view id,
                         const Instance<InnerNorm, OuterNorm>& instance)
{
    detail::write_instance_json_impl(
        output, id, instance.distances(), instance.number_of_clusters(),
        detail::describe_norm(instance.inner_norm()), detail::describe_norm(instance.outer_norm()));
}

} // namespace cluster_aware
