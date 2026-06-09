cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(AutoBalancedTrainer VERSION 1.0 LANGUAGES CXX)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_PREFIX_PATH "C:/libtorch")
find_package(Torch REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
add_executable(${PROJECT_NAME} main_template.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE "include")
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} cppzmq)
set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
