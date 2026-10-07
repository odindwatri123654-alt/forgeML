cmake_minimum_required(VERSION 3.20)
project(ForgeML VERSION 0.1 LANGUAGES CXX)

# C++17 строго, без расширений компилятора
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC: читать все исходники как UTF-8 (в коде есть русские комментарии)
if(MSVC)
    add_compile_options(/utf-8)
endif()

# --- Библиотека forge: все .cpp из src/ ---
add_library(forge
    src/tensor.cpp
    src/ops.cpp
)
# Заголовки лежат в include/ — и это нужно знать всем, кто использует forge
target_include_directories(forge PUBLIC include)

# Максимум предупреждений: они ловят баги с size_t/int
if(MSVC)
    target_compile_options(forge PRIVATE /W4 /permissive-)
else()
    target_compile_options(forge PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Программа-песочница, использует библиотеку forge ---
add_executable(playground examples/playground.cpp)
target_link_libraries(playground PRIVATE forge)
