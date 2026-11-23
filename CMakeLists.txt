cmake_minimum_required(VERSION 3.20)

project(gymj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GYMJ_BUILD_PYTHON "Build Python bindings" OFF)
option(GYMJ_BUILD_BENCHMARKS "Build standalone benchmarks" OFF)

add_library(gymj_libgy
    libgy/src/algo/agari.cpp
    libgy/src/algo/effective_draws.cpp
    libgy/src/algo/flush_progress.cpp
    libgy/src/algo/guiyang_mahjong.cpp
    libgy/src/algo/shanten.cpp
    libgy/src/arena/arena.cpp
    libgy/src/arena/search.cpp
    libgy/src/arena/bean_scoring.cpp
    libgy/src/arena/chicken.cpp
    libgy/src/arena/chicken_attribution.cpp
    libgy/src/arena/game.cpp
    libgy/src/arena/guiyang_rules.cpp
    libgy/src/arena/guiyang_scoring.cpp
    libgy/src/arena/round_settlement.cpp
    libgy/src/arena/score_ledger.cpp
    libgy/src/arena/settlement_snapshot.cpp
    libgy/src/arena/hu_evaluator.cpp
    libgy/src/arena/wall.cpp
    libgy/src/log/text_log_writer.cpp
    libgy/src/rl/action_encoder.cpp
    libgy/src/rl/action_feature_encoder.cpp
    libgy/src/rl/decision_feature_encoder.cpp
    libgy/src/rl/observation_encoder.cpp
    libgy/src/rl/pattern_feature_encoder.cpp
    libgy/src/rl/privileged_pattern_encoder.cpp
    libgy/src/rl/privileged_state_encoder.cpp
)

add_executable(gymj_chicken_attribution_tests tests/chicken_attribution_test.cpp)
target_link_libraries(gymj_chicken_attribution_tests PRIVATE gymj_libgy)
add_test(NAME gymj_chicken_attribution_tests COMMAND gymj_chicken_attribution_tests)

set_target_properties(gymj_libgy PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(gymj_libgy
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/libgy/src/algo
        ${CMAKE_CURRENT_SOURCE_DIR}/libgy/src/arena
        ${CMAKE_CURRENT_SOURCE_DIR}/libgy/src/log
        ${CMAKE_CURRENT_SOURCE_DIR}/libgy/src/rl
)

if(MSVC)
    target_compile_options(gymj_libgy PUBLIC /EHsc)
endif()

add_executable(gymj_basic_tests
    tests/basic_agari_test.cpp
)

target_link_libraries(gymj_basic_tests PRIVATE gymj_libgy)

add_executable(gymj_state_driver_tests
    tests/state_driver_test.cpp
)

target_link_libraries(gymj_state_driver_tests PRIVATE gymj_libgy)

add_executable(gymj_shanten_tests
    tests/shanten_test.cpp
)

target_link_libraries(gymj_shanten_tests PRIVATE gymj_libgy)

add_executable(gymj_effective_draws_tests
    tests/effective_draws_test.cpp
)

target_link_libraries(gymj_effective_draws_tests PRIVATE gymj_libgy)

add_executable(gymj_flush_progress_tests
    tests/flush_progress_test.cpp
)

target_link_libraries(gymj_flush_progress_tests PRIVATE gymj_libgy)

add_executable(gymj_wall_tests
    tests/wall_test.cpp
)

target_link_libraries(gymj_wall_tests PRIVATE gymj_libgy)

add_executable(gymj_arena_tests
    tests/arena_test.cpp
)

target_link_libraries(gymj_arena_tests PRIVATE gymj_libgy)

add_executable(gymj_action_encoder_tests
    tests/action_encoder_test.cpp
)

target_link_libraries(gymj_action_encoder_tests PRIVATE gymj_libgy)

add_executable(gymj_action_feature_encoder_tests
    tests/action_feature_encoder_test.cpp
)

target_link_libraries(gymj_action_feature_encoder_tests PRIVATE gymj_libgy)

add_executable(gymj_observation_encoder_tests
    tests/observation_encoder_test.cpp
)

target_link_libraries(gymj_observation_encoder_tests PRIVATE gymj_libgy)

add_executable(gymj_privileged_state_encoder_tests
    tests/privileged_state_encoder_test.cpp
)

target_link_libraries(gymj_privileged_state_encoder_tests PRIVATE gymj_libgy)

add_executable(gymj_pattern_feature_encoder_tests
    tests/pattern_feature_encoder_test.cpp
)

target_link_libraries(gymj_pattern_feature_encoder_tests PRIVATE gymj_libgy)

add_executable(gymj_decision_feature_encoder_tests
    tests/decision_feature_encoder_test.cpp
)

target_link_libraries(gymj_decision_feature_encoder_tests PRIVATE gymj_libgy)

add_executable(gymj_privileged_pattern_encoder_tests
    tests/privileged_pattern_encoder_test.cpp
)

target_link_libraries(gymj_privileged_pattern_encoder_tests PRIVATE gymj_libgy)

add_executable(gymj_guiyang_scoring_tests
    tests/guiyang_scoring_test.cpp
)

target_link_libraries(gymj_guiyang_scoring_tests PRIVATE gymj_libgy)

add_executable(gymj_hu_evaluator_tests
    tests/hu_evaluator_test.cpp
)

target_link_libraries(gymj_hu_evaluator_tests PRIVATE gymj_libgy)

add_executable(gymj_settlement_snapshot_tests
    tests/settlement_snapshot_test.cpp
)

target_link_libraries(gymj_settlement_snapshot_tests PRIVATE gymj_libgy)

add_executable(gymj_chicken_tests
    tests/chicken_test.cpp
)

target_link_libraries(gymj_chicken_tests PRIVATE gymj_libgy)

add_executable(gymj_bean_scoring_tests
    tests/bean_scoring_test.cpp
)

target_link_libraries(gymj_bean_scoring_tests PRIVATE gymj_libgy)

add_executable(gymj_round_settlement_tests
    tests/round_settlement_test.cpp
)

target_link_libraries(gymj_round_settlement_tests PRIVATE gymj_libgy)

add_executable(gymj_game_tests
    tests/game_test.cpp
)

target_link_libraries(gymj_game_tests PRIVATE gymj_libgy)

add_executable(gymj_text_log_tests
    tests/text_log_test.cpp
)

target_link_libraries(gymj_text_log_tests PRIVATE gymj_libgy)

enable_testing()
add_test(NAME gymj_basic_tests COMMAND gymj_basic_tests)
add_test(NAME gymj_state_driver_tests COMMAND gymj_state_driver_tests)
add_test(NAME gymj_shanten_tests COMMAND gymj_shanten_tests)
add_test(NAME gymj_effective_draws_tests COMMAND gymj_effective_draws_tests)
add_test(NAME gymj_flush_progress_tests COMMAND gymj_flush_progress_tests)
add_test(NAME gymj_wall_tests COMMAND gymj_wall_tests)
add_test(NAME gymj_arena_tests COMMAND gymj_arena_tests)
add_test(NAME gymj_action_encoder_tests COMMAND gymj_action_encoder_tests)
add_test(NAME gymj_action_feature_encoder_tests COMMAND gymj_action_feature_encoder_tests)
add_test(NAME gymj_observation_encoder_tests COMMAND gymj_observation_encoder_tests)
add_test(NAME gymj_privileged_state_encoder_tests COMMAND gymj_privileged_state_encoder_tests)
add_test(NAME gymj_pattern_feature_encoder_tests COMMAND gymj_pattern_feature_encoder_tests)
add_test(NAME gymj_decision_feature_encoder_tests COMMAND gymj_decision_feature_encoder_tests)
add_test(NAME gymj_privileged_pattern_encoder_tests COMMAND gymj_privileged_pattern_encoder_tests)
add_test(NAME gymj_guiyang_scoring_tests COMMAND gymj_guiyang_scoring_tests)
add_test(NAME gymj_hu_evaluator_tests COMMAND gymj_hu_evaluator_tests)
add_test(NAME gymj_settlement_snapshot_tests COMMAND gymj_settlement_snapshot_tests)
add_test(NAME gymj_chicken_tests COMMAND gymj_chicken_tests)
add_test(NAME gymj_bean_scoring_tests COMMAND gymj_bean_scoring_tests)
add_test(NAME gymj_round_settlement_tests COMMAND gymj_round_settlement_tests)
add_test(NAME gymj_game_tests COMMAND gymj_game_tests)
add_test(NAME gymj_text_log_tests COMMAND gymj_text_log_tests)

if(GYMJ_BUILD_BENCHMARKS)
    add_executable(gymj_effective_draws_benchmark
        benchmarks/effective_draws_benchmark.cpp
    )
    target_link_libraries(gymj_effective_draws_benchmark PRIVATE gymj_libgy)
endif()

if(GYMJ_BUILD_PYTHON)
    set(GYMJ_PYTHON_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/gymj"
        CACHE PATH "Directory for the Python extension (can isolate experiments)")
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_gymj
        libgy/src/python/bindings_module.cpp
        libgy/src/python/bindings_algo.cpp
        libgy/src/python/bindings_enums_chicken.cpp
        libgy/src/python/bindings_scoring.cpp
        libgy/src/python/bindings_core.cpp
        libgy/src/python/bindings_action_pattern.cpp
        libgy/src/python/bindings_decision.cpp
        libgy/src/python/bindings_observation.cpp
        libgy/src/python/bindings_arena.cpp
    )

    target_link_libraries(_gymj PRIVATE gymj_libgy)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(_gymj PRIVATE -O0)
    endif()

    set_target_properties(_gymj PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${GYMJ_PYTHON_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${GYMJ_PYTHON_OUTPUT_DIRECTORY}
    )

    add_test(
        NAME gymj_python_arena_tests
        COMMAND
            ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python
            ${Python_EXECUTABLE} -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/python
            -v
    )
endif()
