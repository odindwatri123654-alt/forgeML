cmake_minimum_required(VERSION 3.20)
project(ForgeML VERSION 1.0 LANGUAGES CXX)

# C++17 строго, без расширений компилятора
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Для одноконфигурационных генераторов (Ninja, Make) по умолчанию — Release
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FORGE_BUILD_TESTS    "Собирать тесты"                 ON)
option(FORGE_BUILD_EXAMPLES "Собирать примеры"               ON)
option(FORGE_USE_OPENMP     "Параллелить matmul через OpenMP" ON)
option(FORGE_BUILD_QA       "Собирать QA-набор (тесты чёрного ящика)" ON)

# MSVC: читать все исходники как UTF-8 (в коде есть русские комментарии)
if(MSVC)
    add_compile_options(/utf-8)
endif()

# --- Библиотека forge ---
add_library(forge
    src/tensor.cpp
    src/autograd.cpp
    src/ops.cpp
    src/nn.cpp
    src/optim.cpp
    src/data.cpp
    src/console.cpp
)
# Заголовки лежат в include/ — и это нужно знать всем, кто использует forge
target_include_directories(forge PUBLIC include)

# Максимум предупреждений: они ловят баги с size_t/int
if(MSVC)
    target_compile_options(forge PRIVATE /W4 /permissive-)
else()
    target_compile_options(forge PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(FORGE_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(forge PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "ForgeML: OpenMP enabled")
    else()
        message(STATUS "ForgeML: OpenMP not found, matmul is single-threaded")
    endif()
endif()

# --- Примеры ---
if(FORGE_BUILD_EXAMPLES)
    foreach(example playground autograd_demo xor regression mnist)
        add_executable(${example} examples/${example}.cpp)
        target_link_libraries(${example} PRIVATE forge)
    endforeach()
    # mnist по умолчанию ищет данные в <корень проекта>/data
    target_compile_definitions(mnist PRIVATE FORGE_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
endif()

# --- Тесты: запуск через `ctest` или напрямую forge_tests ---
if(FORGE_BUILD_TESTS)
    enable_testing()
    add_executable(forge_tests
        tests/test_main.cpp
        tests/test_tensor.cpp
        tests/test_ops.cpp
        tests/test_autograd.cpp
        tests/test_nn.cpp
    )
    target_link_libraries(forge_tests PRIVATE forge)
    add_test(NAME forge_tests COMMAND forge_tests)
endif()

# --- QA: тесты "чёрного ящика", каждый в своём процессе (запуск: forge_qa) ---
if(FORGE_BUILD_QA)
    find_package(Threads REQUIRED)
    add_executable(forge_qa
        qa/qa_main.cpp
        qa/qa_basics.cpp
        qa/qa_autograd.cpp
        qa/qa_training.cpp
        qa/qa_stress.cpp
    )
    target_link_libraries(forge_qa PRIVATE forge Threads::Threads)
endif()
