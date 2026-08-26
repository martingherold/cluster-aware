#include "io/json_io.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr double tolerance = 1.0e-9;

bool expect_instance_error(std::string_view name, std::string_view json)
{
    std::istringstream input{std::string{json}};
    try {
        static_cast<void>(cluster_aware::parse_instance_json(input));
    } catch (const cluster_aware::JsonFormatError&) {
        return true;
    }

    std::cerr << name << ": malformed instance was accepted\n";
    return false;
}

bool expect_solution_error(std::string_view name, std::string_view json)
{
    std::istringstream input{std::string{json}};
    try {
        static_cast<void>(cluster_aware::parse_solution_json(input));
    } catch (const cluster_aware::JsonFormatError&) {
        return true;
    }

    std::cerr << name << ": malformed solution was accepted\n";
    return false;
}

bool test_instance_round_trip()
{
    const cluster_aware::Instance instance{cluster_aware::DistanceMatrix{3,
                                                                         2,
                                                                         {
                                                                             0.0,
                                                                             0.4,
                                                                             2.5,
                                                                             3.0,
                                                                             0.0,
                                                                             0.5,
                                                                         }},
                                           1, cluster_aware::TopLNorm{2}, cluster_aware::L1Norm{}};

    std::ostringstream output;
    cluster_aware::write_instance_json(output, "rectangular-fractional", instance);
    if (output.str().find("\"client_count\": 3") == std::string::npos) {
        std::cerr << "instance writer did not emit a numeric client count\n";
        return false;
    }

    std::istringstream input{output.str()};
    const auto parsed = cluster_aware::parse_instance_json(input);
    if (parsed.id != "rectangular-fractional") {
        std::cerr << "instance id did not survive round trip\n";
        return false;
    }

    const auto* parsed_instance =
        std::get_if<cluster_aware::Instance<cluster_aware::TopLNorm, cluster_aware::L1Norm>>(
            &parsed.instance);
    if (parsed_instance == nullptr) {
        std::cerr << "instance norm types did not survive round trip\n";
        return false;
    }

    const std::vector<double> expected_distances{0.0, 0.4, 2.5, 3.0, 0.0, 0.5};
    if (parsed_instance->number_of_clusters() != 1 || parsed_instance->inner_norm().l() != 2 ||
        !std::ranges::equal(parsed_instance->distances().values(), expected_distances)) {
        std::cerr << "instance data did not survive round trip\n";
        return false;
    }

    std::ostringstream rewritten_output;
    cluster_aware::write_instance_json(rewritten_output, parsed);
    std::istringstream rewritten_input{rewritten_output.str()};
    if (cluster_aware::parse_instance_json(rewritten_input).id != "rectangular-fractional") {
        std::cerr << "parsed-instance writer did not preserve the id\n";
        return false;
    }

    return true;
}

bool test_all_norm_positions()
{
    const cluster_aware::Instance instance{cluster_aware::DistanceMatrix{2, 1, {1.0, 3.0}}, 1,
                                           cluster_aware::LInfNorm{}, cluster_aware::TopLNorm{1}};

    std::ostringstream output;
    cluster_aware::write_instance_json(output, "all-norm-positions", instance);
    std::istringstream input{output.str()};
    const auto parsed = cluster_aware::parse_instance_json(input);

    const auto* parsed_instance =
        std::get_if<cluster_aware::Instance<cluster_aware::LInfNorm, cluster_aware::TopLNorm>>(
            &parsed.instance);
    return parsed_instance != nullptr && parsed_instance->outer_norm().l() == 1;
}

bool test_solution_round_trip()
{
    const cluster_aware::Solution solution{{2, 0}, {0, 0, 2}, 2.9};
    std::ostringstream output;
    cluster_aware::write_solution_json(output, "rectangular-fractional", solution);
    if (output.str().find("\"objective\": 2.9") == std::string::npos) {
        std::cerr << "solution writer did not emit a numeric objective\n";
        return false;
    }

    std::istringstream input{output.str()};
    const auto parsed = cluster_aware::parse_solution_json(input);
    const std::vector<std::size_t> expected_centers{0, 2};
    const std::vector<std::size_t> expected_assignments{0, 0, 2};
    if (parsed.instance_id != "rectangular-fractional" ||
        parsed.solution.open_centers != expected_centers ||
        parsed.solution.assignments != expected_assignments ||
        std::abs(parsed.solution.objective - 2.9) > tolerance) {
        std::cerr << "solution data did not survive round trip\n";
        return false;
    }

    std::ostringstream rewritten_output;
    cluster_aware::write_solution_json(rewritten_output, parsed);
    std::istringstream rewritten_input{rewritten_output.str()};
    if (cluster_aware::parse_solution_json(rewritten_input).instance_id !=
        "rectangular-fractional") {
        std::cerr << "parsed-solution writer did not preserve the instance id\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    bool passed = test_instance_round_trip();
    passed &= test_all_norm_positions();
    passed &= test_solution_round_trip();

    passed &= expect_instance_error("quoted dimension",
                                    R"({
            "format": "cluster-aware-instance",
            "version": 1,
            "id": "bad",
            "number_of_clusters": 1,
            "inner_norm": {"type": "l1", "parameters": {}},
            "outer_norm": {"type": "l1", "parameters": {}},
            "distance_matrix": {
                "client_count": "2",
                "center_count": 1,
                "layout": "center_major",
                "values": [[1.0, 2.0]]
            }
        })");

    passed &= expect_instance_error("incorrect row length",
                                    R"({
            "format": "cluster-aware-instance",
            "version": 1,
            "id": "bad",
            "number_of_clusters": 1,
            "inner_norm": {"type": "top_l", "parameters": {"l": 1}},
            "outer_norm": {"type": "l1", "parameters": {}},
            "distance_matrix": {
                "client_count": 2,
                "center_count": 1,
                "layout": "center_major",
                "values": [[1.0]]
            }
        })");

    passed &= expect_instance_error("unknown norm parameter",
                                    R"({
            "format": "cluster-aware-instance",
            "version": 1,
            "id": "bad",
            "number_of_clusters": 1,
            "inner_norm": {"type": "l1", "parameters": {"l": 1}},
            "outer_norm": {"type": "l1", "parameters": {}},
            "distance_matrix": {
                "client_count": 1,
                "center_count": 1,
                "layout": "center_major",
                "values": [[0.0]]
            }
        })");

    passed &= expect_solution_error("unsorted centers",
                                    R"({
            "format": "cluster-aware-solution",
            "version": 1,
            "instance_id": "bad",
            "open_centers": [2, 0],
            "assignments": [0, 2],
            "objective": 1.0
        })");

    try {
        std::ostringstream output;
        cluster_aware::write_solution_json(
            output, "bad",
            cluster_aware::Solution{{0}, {0}, std::numeric_limits<double>::infinity()});
        std::cerr << "non-finite solution objective was written\n";
        passed = false;
    } catch (const std::invalid_argument&) {
    }

    return passed ? 0 : 1;
}
