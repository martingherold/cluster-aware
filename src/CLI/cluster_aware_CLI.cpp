#include "core/initial_solution.hpp"
#include "core/solution_checker.hpp"
#include "io/json_io.hpp"
#include "scip/compact_cluster_aware_solver.hpp"
#include "scip/dynamic/dynamic_cluster_aware_solver.hpp"
#include "scip/solve_result.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace {

enum class Formulation {
    compact,
    dynamic,
};

struct Options {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    Formulation formulation = Formulation::compact;
    bool show_help = false;
};

class CommandLineError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

void print_usage(std::ostream& output, std::string_view program)
{
    output << "Usage: " << program
           << " --input FILE --output FILE [--formulation compact|dynamic]\n\n"
           << "Solve one cluster-aware clustering instance and write a verified JSON "
              "solution.\n"
           << "The norms, cluster count, and Top-l parameter are read from the instance.\n\n"
           << "  --input FILE                    Read the instance from FILE\n"
           << "  --output FILE                   Write the solution to FILE\n"
           << "  --formulation compact|dynamic  Select the exact formulation; default: "
              "compact\n"
           << "  -h, --help                      Show this help\n";
}

[[nodiscard]] std::string_view require_option_value(int argc, char** argv, int& index,
                                                    std::string_view option)
{
    if (index + 1 >= argc) {
        throw CommandLineError{std::string{option} + " expects one value"};
    }
    return argv[++index];
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
    Options options;
    bool formulation_seen = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "-h" || argument == "--help") {
            options.show_help = true;
            continue;
        }
        if (argument == "--input") {
            if (!options.input_path.empty()) {
                throw CommandLineError{"--input may only be specified once"};
            }
            options.input_path = require_option_value(argc, argv, index, argument);
            continue;
        }
        if (argument == "--output") {
            if (!options.output_path.empty()) {
                throw CommandLineError{"--output may only be specified once"};
            }
            options.output_path = require_option_value(argc, argv, index, argument);
            continue;
        }
        if (argument == "--formulation") {
            if (formulation_seen) {
                throw CommandLineError{"--formulation may only be specified once"};
            }

            const auto value = require_option_value(argc, argv, index, argument);
            if (value == "compact") {
                options.formulation = Formulation::compact;
            } else if (value == "dynamic") {
                options.formulation = Formulation::dynamic;
            } else {
                throw CommandLineError{"--formulation expects either 'compact' or 'dynamic'"};
            }
            formulation_seen = true;
            continue;
        }
        throw CommandLineError{"unknown option: " + std::string{argument}};
    }

    if (options.show_help) {
        return options;
    }
    if (options.input_path.empty()) {
        throw CommandLineError{"--input is required"};
    }
    if (options.output_path.empty()) {
        throw CommandLineError{"--output is required"};
    }
    return options;
}

[[nodiscard]] cluster_aware::ParsedInstance read_instance(const std::filesystem::path& path)
{
    std::ifstream input{path};
    if (!input) {
        throw std::runtime_error{"could not open input file '" + path.string() + "'"};
    }
    return cluster_aware::parse_instance_json(input);
}

[[nodiscard]] cluster_aware::SolveResult solve_instance(const cluster_aware::JsonInstance& instance,
                                                        Formulation formulation)
{
    using L1Instance = cluster_aware::Instance<cluster_aware::L1Norm, cluster_aware::L1Norm>;
    using TopLInstance = cluster_aware::Instance<cluster_aware::TopLNorm, cluster_aware::L1Norm>;

    return std::visit(
        [formulation](const auto& concrete_instance) -> cluster_aware::SolveResult {
            using InstanceType = std::remove_cvref_t<decltype(concrete_instance)>;

            if constexpr (std::is_same_v<InstanceType, L1Instance>) {
                if (formulation == Formulation::dynamic) {
                    throw std::runtime_error{
                        "the dynamic formulation only supports Top-l/L1 instances"};
                }
                auto shared_instance = std::make_shared<const L1Instance>(concrete_instance);
                const auto initial_solution =
                    cluster_aware::construct_greedy_initial_solution(*shared_instance);
                std::cout << "Initial heuristic objective: " << initial_solution.objective << '\n';
                cluster_aware::CompactClusterAwareSolver solver{std::move(shared_instance)};
                return solver.solve(initial_solution);
            } else if constexpr (std::is_same_v<InstanceType, TopLInstance>) {
                auto shared_instance = std::make_shared<const TopLInstance>(concrete_instance);
                const auto initial_solution =
                    cluster_aware::construct_greedy_initial_solution(*shared_instance);
                std::cout << "Initial heuristic objective: " << initial_solution.objective << '\n';
                if (formulation == Formulation::compact) {
                    cluster_aware::CompactClusterAwareSolver solver{std::move(shared_instance)};
                    return solver.solve(initial_solution);
                }

                cluster_aware::DynamicClusterAwareSolver solver{std::move(shared_instance)};
                return solver.solve(initial_solution);
            } else {
                throw std::runtime_error{
                    "no exact solver is implemented for this norm combination; supported "
                    "instances are L1/L1 and Top-l/L1"};
            }
        },
        instance);
}

void print_solve_summary(const cluster_aware::SolveResult& result)
{
    const auto& statistics = result.statistics;
    std::cout << "SCIP status: " << cluster_aware::to_string(statistics.status)
              << ", nodes: " << statistics.node_count
              << ", time: " << statistics.solving_time_seconds << " s";
    if (statistics.dual_bound.has_value()) {
        std::cout << ", dual bound: " << *statistics.dual_bound;
    }
    if (statistics.relative_gap.has_value()) {
        std::cout << ", relative gap: " << *statistics.relative_gap;
    }
    std::cout << '\n';
}

void verify_solution(const cluster_aware::JsonInstance& instance,
                     const cluster_aware::Solution& solution)
{
    const auto check = std::visit(
        [&solution](const auto& concrete_instance) {
            return cluster_aware::check_solution(concrete_instance, solution);
        },
        instance);

    if (!check.valid()) {
        throw std::runtime_error{"solver produced an invalid solution: " + check.message};
    }
}

void write_solution(const std::filesystem::path& path, std::string_view instance_id,
                    const cluster_aware::Solution& solution)
{
    std::ofstream output{path};
    if (!output) {
        throw std::runtime_error{"could not open output file '" + path.string() + "'"};
    }
    cluster_aware::write_solution_json(output, instance_id, solution);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse_options(argc, argv);
        if (options.show_help) {
            print_usage(std::cout, argv[0]);
            return 0;
        }

        const auto parsed_instance = read_instance(options.input_path);
        const auto result = solve_instance(parsed_instance.instance, options.formulation);
        print_solve_summary(result);
        if (!result.solution.has_value()) {
            std::cerr << "No feasible incumbent is available; no solution was written.\n";
            return 3;
        }

        verify_solution(parsed_instance.instance, *result.solution);
        write_solution(options.output_path, parsed_instance.id, *result.solution);

        std::cout << "Wrote verified solution for '" << parsed_instance.id << "' to '"
                  << options.output_path.string() << "'.\n";
        if (!result.proven_optimal()) {
            std::cerr << "The written incumbent has not been proven optimal.\n";
            return 3;
        }
        return 0;
    } catch (const CommandLineError& error) {
        std::cerr << "Error: " << error.what() << "\n\n";
        print_usage(std::cerr, argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
