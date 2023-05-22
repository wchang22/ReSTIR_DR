
cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0015 NEW)

project(reservoir_sampling LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} reservoir_sampling.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "75"
)
target_compile_options(${PROJECT_NAME} PRIVATE "--keep")