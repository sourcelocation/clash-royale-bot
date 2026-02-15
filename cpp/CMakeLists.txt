cmake_minimum_required(VERSION 3.20)
project(knockoff_cr_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(knockoff_cr_cpp MODULE
    engine/src/knockoff_cr_batch.cpp
    engine/src/knockoff_cr_actions.cpp
    engine/src/knockoff_cr_api.cpp
    engine/src/knockoff_cr_cards.cpp
    engine/src/knockoff_cr_combat.cpp
    engine/src/knockoff_cr_geometry.cpp
    engine/src/knockoff_cr_module.cpp
    engine/src/knockoff_cr_state.cpp
    engine/src/knockoff_cr_tick.cpp
)

target_include_directories(knockoff_cr_cpp PRIVATE
    engine/include
)

set_target_properties(knockoff_cr_cpp PROPERTIES
    OUTPUT_NAME "knockoff_cr_cpp"
)
