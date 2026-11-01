enable_language(CUDA)
find_package(Eigen3 REQUIRED)
file(GLOB SOURCE "*.cpp" "*.h" "*.cu")

add_executable(muda_test ${SOURCE})

target_link_libraries(muda_test PRIVATE muda Eigen3::Eigen)

set_target_properties(muda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75
    CUDA_INCLUDE_WHAT_YOU_USE ON
)

target_include_directories(muda_test PRIVATE "${PROJECT_SOURCE_DIR}/src/backends/cuda")


