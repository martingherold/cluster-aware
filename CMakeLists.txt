cmake_minimum_required(VERSION 3.20)

project(
    cluster_aware
    VERSION 0.1.0
    DESCRIPTION "Exact and approximation algorithms for Cluster-Aware Clustering."
    LANGUAGES CXX
)

option(CLUSTER_AWARE_BUILD_SCIP "Build the exact SCIP solvers and CLI" ON)

find_package(nlohmann_json 3.11 REQUIRED CONFIG)

add_library(
    cluster_aware_core
    src/core/distance_matrix.cpp
    src/core/norms.cpp
    src/core/top_l_selection.cpp
    src/io/json_io.cpp
)
add_library(cluster_aware::core ALIAS cluster_aware_core)

target_compile_features(cluster_aware_core PUBLIC cxx_std_23)
target_link_libraries(cluster_aware_core PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(
    cluster_aware_core
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

if(CLUSTER_AWARE_BUILD_SCIP)
    find_package(SCIP 10.0 REQUIRED CONFIG)

    add_library(
        cluster_aware_scip
        src/scip/compact_cluster_aware_solver.cpp
        src/scip/discrete_clustering_model.cpp
        src/scip/dynamic/dynamic_cluster_aware_solver.cpp
        src/scip/dynamic/dynamic_cluster_aware_constraint_handler.cpp
        src/scip/dynamic/dynamic_cluster_aware_prob_data.cpp
        src/scip/scip_wrapper.cpp
        src/scip/solve_result.cpp
    )
    add_library(cluster_aware::scip ALIAS cluster_aware_scip)

    target_compile_features(cluster_aware_scip PUBLIC cxx_std_23)
    target_link_libraries(cluster_aware_scip PUBLIC cluster_aware_core libscip)
    target_include_directories(
        cluster_aware_scip
        PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    add_executable(cluster_aware_cli src/CLI/cluster_aware_CLI.cpp)
    set_target_properties(cluster_aware_cli PROPERTIES OUTPUT_NAME cluster-aware)
    target_compile_features(cluster_aware_cli PRIVATE cxx_std_23)
    target_link_libraries(cluster_aware_cli PRIVATE cluster_aware_scip)
endif()

# Compatibility target for current consumers. New code can choose the narrow
# cluster_aware::core or cluster_aware::scip component explicitly.
add_library(cluster_aware INTERFACE)
if(CLUSTER_AWARE_BUILD_SCIP)
    target_link_libraries(cluster_aware INTERFACE cluster_aware_scip)
else()
    target_link_libraries(cluster_aware INTERFACE cluster_aware_core)
endif()
add_library(cluster_aware::cluster_aware ALIAS cluster_aware)

include(CTest)

if(BUILD_TESTING)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

    function(add_cluster_aware_test target source test_name library)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE ${library})
        target_compile_features(${target} PRIVATE cxx_std_23)
        add_test(NAME ${test_name} COMMAND ${target})
    endfunction()

    add_cluster_aware_test(
        top_lnorm_test tests/core/top_lnorm_test.cpp top_lnorm cluster_aware_core
    )
    add_cluster_aware_test(
        l1_norm_test tests/core/l1_norm_test.cpp l1_norm cluster_aware_core
    )
    add_cluster_aware_test(
        linf_norm_test tests/core/linf_norm_test.cpp linf_norm cluster_aware_core
    )
    add_cluster_aware_test(
        distance_matrix_test tests/core/distance_matrix_test.cpp distance_matrix
        cluster_aware_core
    )
    add_cluster_aware_test(
        solution_checker_test tests/core/solution_checker_test.cpp solution_checker
        cluster_aware_core
    )
    add_cluster_aware_test(
        initial_solution_test tests/core/initial_solution_test.cpp initial_solution
        cluster_aware_core
    )
    add_cluster_aware_test(
        top_l_selection_test tests/core/top_l_selection_test.cpp top_l_selection
        cluster_aware_core
    )
    add_cluster_aware_test(
        json_io_test tests/io/json_io_test.cpp json_io cluster_aware_core
    )
    add_test(
        NAME prepare_uci_iris
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/prepare_uci_iris_test.py
    )

    if(CLUSTER_AWARE_BUILD_SCIP)
        add_cluster_aware_test(
            scip_smoke_test tests/scip/scip_smoke_test.cpp scip_smoke
            cluster_aware_scip
        )
        add_cluster_aware_test(
            compact_cluster_aware_solver_test
            tests/scip/compact_cluster_aware_solver_test.cpp
            compact_cluster_aware_solver cluster_aware_scip
        )
        add_cluster_aware_test(
            scip_compact_top_l_test tests/scip/scip_compact_top_l_test.cpp
            scip_compact_top_l cluster_aware_scip
        )
        add_cluster_aware_test(
            scip_dynamic_top_l_test tests/scip/scip_dynamic_top_l_test.cpp
            scip_dynamic_top_l cluster_aware_scip
        )

        add_test(
            NAME cluster_aware_cli_compact_l1
            COMMAND
                $<TARGET_FILE:cluster_aware_cli>
                --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/l1_l1.json
                --output ${CMAKE_CURRENT_BINARY_DIR}/cli_compact_l1_solution.json
        )
        add_test(
            NAME cluster_aware_cli_compact_top_l
            COMMAND
                $<TARGET_FILE:cluster_aware_cli>
                --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/top_l_l1.json
                --output ${CMAKE_CURRENT_BINARY_DIR}/cli_compact_top_l_solution.json
                --formulation compact
        )
        add_test(
            NAME cluster_aware_cli_dynamic_top_l
            COMMAND
                $<TARGET_FILE:cluster_aware_cli>
                --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/top_l_l1.json
                --output ${CMAKE_CURRENT_BINARY_DIR}/cli_dynamic_top_l_solution.json
                --formulation dynamic
        )
        add_test(
            NAME cluster_aware_cli_rejects_dynamic_l1
            COMMAND
                $<TARGET_FILE:cluster_aware_cli>
                --input ${CMAKE_CURRENT_SOURCE_DIR}/examples/l1_l1.json
                --output ${CMAKE_CURRENT_BINARY_DIR}/cli_invalid_solution.json
                --formulation dynamic
        )
        set_tests_properties(
            cluster_aware_cli_rejects_dynamic_l1 PROPERTIES WILL_FAIL TRUE
        )
        set_tests_properties(
            cluster_aware_cli_compact_l1 cluster_aware_cli_compact_top_l
            cluster_aware_cli_dynamic_top_l
            PROPERTIES PASS_REGULAR_EXPRESSION "Initial heuristic objective:"
        )
    endif()
endif()
