#include "io/json_io.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using cluster_aware::JsonFormatError;
using cluster_aware::detail::JsonNormSpec;
using cluster_aware::detail::JsonNormType;

constexpr std::uint64_t format_version = 1;

void require_exact_keys(const Json& object, std::initializer_list<std::string_view> keys,
                        std::string_view path)
{
    if (!object.is_object()) {
        throw JsonFormatError(std::string{path} + " must be an object");
    }

    for (const auto key : keys) {
        if (!object.contains(std::string{key})) {
            throw JsonFormatError(std::string{path} + " is missing field '" + std::string{key} +
                                  "'");
        }
    }

    if (object.size() != keys.size()) {
        for (const auto& [key, value] : object.items()) {
            static_cast<void>(value);
            const bool known = std::ranges::any_of(
                keys, [&key](std::string_view expected) { return key == expected; });
            if (!known) {
                throw JsonFormatError(std::string{path} + " contains unknown field '" + key + "'");
            }
        }
    }
}

std::string require_string(const Json& value, std::string_view path)
{
    if (!value.is_string()) {
        throw JsonFormatError(std::string{path} + " must be a string");
    }
    return value.get<std::string>();
}

std::size_t require_size(const Json& value, std::string_view path, bool require_positive = false)
{
    std::uint64_t parsed = 0;
    if (value.is_number_unsigned()) {
        parsed = value.get<std::uint64_t>();
    } else if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value < 0) {
            throw JsonFormatError(std::string{path} + " must be nonnegative");
        }
        parsed = static_cast<std::uint64_t>(signed_value);
    } else {
        throw JsonFormatError(std::string{path} + " must be an integer");
    }

    if (parsed > std::numeric_limits<std::size_t>::max()) {
        throw JsonFormatError(std::string{path} + " does not fit in size_t");
    }
    if (require_positive && parsed == 0) {
        throw JsonFormatError(std::string{path} + " must be positive");
    }
    return static_cast<std::size_t>(parsed);
}

double require_number(const Json& value, std::string_view path)
{
    if (!value.is_number()) {
        throw JsonFormatError(std::string{path} + " must be a number");
    }

    const double parsed = value.get<double>();
    if (!std::isfinite(parsed)) {
        throw JsonFormatError(std::string{path} + " must be finite");
    }
    return parsed;
}

void require_format_header(const Json& root, std::string_view expected_format)
{
    const std::string format = require_string(root.at("format"), "format");
    if (format != expected_format) {
        throw JsonFormatError("unexpected JSON format '" + format + "'");
    }

    const auto version = require_size(root.at("version"), "version");
    if (version != format_version) {
        throw JsonFormatError("unsupported JSON format version " + std::to_string(version));
    }
}

JsonNormSpec parse_norm(const Json& value, std::string_view path)
{
    require_exact_keys(value, {"type", "parameters"}, path);
    const std::string type = require_string(value.at("type"), std::string{path} + ".type");
    const auto& parameters = value.at("parameters");
    if (!parameters.is_object()) {
        throw JsonFormatError(std::string{path} + ".parameters must be an object");
    }

    if (type == "l1") {
        require_exact_keys(parameters, {}, std::string{path} + ".parameters");
        return {JsonNormType::l1, std::nullopt};
    }
    if (type == "l_inf") {
        require_exact_keys(parameters, {}, std::string{path} + ".parameters");
        return {JsonNormType::l_inf, std::nullopt};
    }
    if (type == "top_l") {
        require_exact_keys(parameters, {"l"}, std::string{path} + ".parameters");
        return {JsonNormType::top_l,
                require_size(parameters.at("l"), std::string{path} + ".parameters.l", true)};
    }

    throw JsonFormatError(std::string{path} + " has unsupported norm type '" + type + "'");
}

Json norm_to_json(const JsonNormSpec& norm)
{
    switch (norm.type) {
    case JsonNormType::l1:
        return {{"type", "l1"}, {"parameters", Json::object()}};
    case JsonNormType::l_inf:
        return {{"type", "l_inf"}, {"parameters", Json::object()}};
    case JsonNormType::top_l:
        if (!norm.l.has_value()) {
            throw std::logic_error("Top-l JSON norm is missing l");
        }
        return {{"type", "top_l"}, {"parameters", {{"l", *norm.l}}}};
    }

    throw std::logic_error("unknown JSON norm type");
}

template <typename InnerNorm, typename OuterNorm>
cluster_aware::JsonInstance make_instance(cluster_aware::DistanceMatrix distances,
                                          std::size_t number_of_clusters, const JsonNormSpec& inner,
                                          const JsonNormSpec& outer)
{
    const auto make_norm = []<typename Norm>(const JsonNormSpec& spec) {
        if constexpr (std::same_as<Norm, cluster_aware::TopLNorm>) {
            return Norm{*spec.l};
        } else {
            return Norm{};
        }
    };

    return cluster_aware::Instance<InnerNorm, OuterNorm>{
        std::move(distances), number_of_clusters, make_norm.template operator()<InnerNorm>(inner),
        make_norm.template operator()<OuterNorm>(outer)};
}

template <typename InnerNorm>
cluster_aware::JsonInstance make_instance_with_outer_norm(cluster_aware::DistanceMatrix distances,
                                                          std::size_t number_of_clusters,
                                                          const JsonNormSpec& inner,
                                                          const JsonNormSpec& outer)
{
    switch (outer.type) {
    case JsonNormType::l1:
        return make_instance<InnerNorm, cluster_aware::L1Norm>(std::move(distances),
                                                               number_of_clusters, inner, outer);
    case JsonNormType::l_inf:
        return make_instance<InnerNorm, cluster_aware::LInfNorm>(std::move(distances),
                                                                 number_of_clusters, inner, outer);
    case JsonNormType::top_l:
        return make_instance<InnerNorm, cluster_aware::TopLNorm>(std::move(distances),
                                                                 number_of_clusters, inner, outer);
    }

    throw std::logic_error("unknown outer norm type");
}

cluster_aware::JsonInstance make_instance(cluster_aware::DistanceMatrix distances,
                                          std::size_t number_of_clusters, const JsonNormSpec& inner,
                                          const JsonNormSpec& outer)
{
    switch (inner.type) {
    case JsonNormType::l1:
        return make_instance_with_outer_norm<cluster_aware::L1Norm>(
            std::move(distances), number_of_clusters, inner, outer);
    case JsonNormType::l_inf:
        return make_instance_with_outer_norm<cluster_aware::LInfNorm>(
            std::move(distances), number_of_clusters, inner, outer);
    case JsonNormType::top_l:
        return make_instance_with_outer_norm<cluster_aware::TopLNorm>(
            std::move(distances), number_of_clusters, inner, outer);
    }

    throw std::logic_error("unknown inner norm type");
}

Json parse_json(std::istream& input)
{
    try {
        Json document;
        input >> document;
        return document;
    } catch (const nlohmann::json::exception& error) {
        throw JsonFormatError(std::string{"invalid JSON: "} + error.what());
    }
}

void write_json(std::ostream& output, const Json& document)
{
    output << std::setw(2) << document << '\n';
    if (!output) {
        throw std::runtime_error("failed to write JSON");
    }
}

} // namespace

namespace cluster_aware {

ParsedInstance parse_instance_json(std::istream& input)
{
    try {
        const Json root = parse_json(input);
        require_exact_keys(root,
                           {"format", "version", "id", "number_of_clusters", "inner_norm",
                            "outer_norm", "distance_matrix"},
                           "instance");
        require_format_header(root, "cluster-aware-instance");

        const std::string id = require_string(root.at("id"), "id");
        if (id.empty()) {
            throw JsonFormatError("id must not be empty");
        }

        const auto number_of_clusters =
            require_size(root.at("number_of_clusters"), "number_of_clusters", true);
        const auto inner_norm = parse_norm(root.at("inner_norm"), "inner_norm");
        const auto outer_norm = parse_norm(root.at("outer_norm"), "outer_norm");

        const auto& matrix = root.at("distance_matrix");
        require_exact_keys(matrix, {"client_count", "center_count", "layout", "values"},
                           "distance_matrix");
        const auto client_count =
            require_size(matrix.at("client_count"), "distance_matrix.client_count", true);
        const auto center_count =
            require_size(matrix.at("center_count"), "distance_matrix.center_count", true);
        const std::string layout = require_string(matrix.at("layout"), "distance_matrix.layout");
        if (layout != "center_major") {
            throw JsonFormatError("distance_matrix.layout must be 'center_major'");
        }

        const auto& rows = matrix.at("values");
        if (!rows.is_array() || rows.size() != center_count) {
            throw JsonFormatError("distance_matrix.values must contain one row per center");
        }

        std::vector<double> distances;
        distances.reserve(client_count * center_count);
        for (std::size_t center = 0; center < center_count; ++center) {
            const auto& row = rows[center];
            if (!row.is_array() || row.size() != client_count) {
                throw JsonFormatError("each distance row must contain one value per client");
            }
            for (std::size_t client = 0; client < client_count; ++client) {
                const double distance =
                    require_number(row[client], "distance_matrix.values[" + std::to_string(center) +
                                                    "][" + std::to_string(client) + "]");
                if (distance < 0.0) {
                    throw JsonFormatError("distance values must be nonnegative");
                }
                distances.push_back(distance);
            }
        }

        return {id, make_instance(DistanceMatrix{client_count, center_count, std::move(distances)},
                                  number_of_clusters, inner_norm, outer_norm)};
    } catch (const JsonFormatError&) {
        throw;
    } catch (const std::exception& error) {
        throw JsonFormatError(std::string{"invalid cluster-aware instance: "} + error.what());
    }
}

ParsedSolution parse_solution_json(std::istream& input)
{
    const Json root = parse_json(input);
    require_exact_keys(
        root, {"format", "version", "instance_id", "open_centers", "assignments", "objective"},
        "solution");
    require_format_header(root, "cluster-aware-solution");

    const std::string instance_id = require_string(root.at("instance_id"), "instance_id");
    if (instance_id.empty()) {
        throw JsonFormatError("instance_id must not be empty");
    }

    const auto parse_indices = [](const Json& values, std::string_view path) {
        if (!values.is_array()) {
            throw JsonFormatError(std::string{path} + " must be an array");
        }

        std::vector<std::size_t> indices;
        indices.reserve(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            indices.push_back(
                require_size(values[index], std::string{path} + "[" + std::to_string(index) + "]"));
        }
        return indices;
    };

    Solution solution{
        parse_indices(root.at("open_centers"), "open_centers"),
        parse_indices(root.at("assignments"), "assignments"),
        require_number(root.at("objective"), "objective"),
    };

    if (!std::ranges::is_sorted(solution.open_centers) ||
        std::ranges::adjacent_find(solution.open_centers) != solution.open_centers.end()) {
        throw JsonFormatError("open_centers must be sorted and unique");
    }

    return {instance_id, std::move(solution)};
}

void detail::write_instance_json_impl(std::ostream& output, std::string_view id,
                                      const DistanceMatrix& distances,
                                      std::size_t number_of_clusters,
                                      const JsonNormSpec& inner_norm,
                                      const JsonNormSpec& outer_norm)
{
    if (id.empty()) {
        throw std::invalid_argument("instance id must not be empty");
    }

    Json rows = Json::array();
    for (std::size_t center = 0; center < distances.center_count(); ++center) {
        Json row = Json::array();
        for (const double distance : distances.distances_to_center(center)) {
            row.push_back(distance);
        }
        rows.push_back(std::move(row));
    }

    const Json root{
        {"format", "cluster-aware-instance"},
        {"version", format_version},
        {"id", id},
        {"number_of_clusters", number_of_clusters},
        {"inner_norm", norm_to_json(inner_norm)},
        {"outer_norm", norm_to_json(outer_norm)},
        {"distance_matrix",
         {
             {"client_count", distances.client_count()},
             {"center_count", distances.center_count()},
             {"layout", "center_major"},
             {"values", std::move(rows)},
         }},
    };
    write_json(output, root);
}

void write_instance_json(std::ostream& output, const ParsedInstance& parsed_instance)
{
    std::visit(
        [&](const auto& instance) { write_instance_json(output, parsed_instance.id, instance); },
        parsed_instance.instance);
}

void write_solution_json(std::ostream& output, std::string_view instance_id,
                         const Solution& solution)
{
    if (instance_id.empty()) {
        throw std::invalid_argument("instance id must not be empty");
    }
    if (!std::isfinite(solution.objective)) {
        throw std::invalid_argument("solution objective must be finite");
    }

    auto open_centers = solution.open_centers;
    std::ranges::sort(open_centers);
    if (std::ranges::adjacent_find(open_centers) != open_centers.end()) {
        throw std::invalid_argument("open centers must be unique");
    }

    const Json root{
        {"format", "cluster-aware-solution"},  {"version", format_version},
        {"instance_id", instance_id},          {"open_centers", std::move(open_centers)},
        {"assignments", solution.assignments}, {"objective", solution.objective},
    };
    write_json(output, root);
}

void write_solution_json(std::ostream& output, const ParsedSolution& parsed_solution)
{
    write_solution_json(output, parsed_solution.instance_id, parsed_solution.solution);
}

} // namespace cluster_aware
