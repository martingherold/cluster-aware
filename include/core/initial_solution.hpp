#pragma once

#include "instance.hpp"
#include "solution.hpp"
#include "solution_checker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cluster_aware {

// Greedily adds the center that most reduces the sum of each client's nearest
// distance. The construction is deterministic and works for rectangular
// client-by-center distance matrices.
template <ValidNorm InnerNorm, ValidNorm OuterNorm>
[[nodiscard]] Solution
construct_greedy_initial_solution(const Instance<InnerNorm, OuterNorm>& instance)
{
    const auto& distances = instance.distances();
    const auto client_count = distances.client_count();
    const auto center_count = distances.center_count();

    std::vector<bool> is_selected(center_count, false);
    std::vector<double> nearest_distances(client_count, std::numeric_limits<double>::infinity());
    std::vector<std::size_t> assignments(client_count, 0);
    std::vector<std::size_t> open_centers;
    open_centers.reserve(instance.number_of_clusters());

    while (open_centers.size() < instance.number_of_clusters()) {
        std::size_t best_center = center_count;
        double best_score = std::numeric_limits<double>::infinity();

        for (std::size_t candidate = 0; candidate < center_count; ++candidate) {
            if (is_selected[candidate]) {
                continue;
            }

            double score = 0.0;
            for (std::size_t client = 0; client < client_count; ++client) {
                score += std::min(nearest_distances[client], distances(client, candidate));
            }

            // Iterating by index supplies a stable lower-index tie break.
            if (best_center == center_count || score < best_score) {
                best_center = candidate;
                best_score = score;
            }
        }

        if (best_center == center_count) {
            throw std::logic_error("initial-solution heuristic could not select a center");
        }

        is_selected[best_center] = true;
        open_centers.push_back(best_center);
        for (std::size_t client = 0; client < client_count; ++client) {
            const double candidate_distance = distances(client, best_center);
            if (candidate_distance < nearest_distances[client] ||
                (candidate_distance == nearest_distances[client] &&
                 best_center < assignments[client])) {
                nearest_distances[client] = candidate_distance;
                assignments[client] = best_center;
            }
        }
    }

    std::ranges::sort(open_centers);
    Solution solution{std::move(open_centers), std::move(assignments), 0.0};
    const auto evaluation = check_solution(instance, solution);
    if (!evaluation.feasible || !std::isfinite(evaluation.recomputed_objective)) {
        throw std::logic_error("initial-solution heuristic produced an invalid solution");
    }
    solution.objective = evaluation.recomputed_objective;
    return solution;
}

} // namespace cluster_aware
