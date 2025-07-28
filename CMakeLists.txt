cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Segmenter)

# Find PCL package
find_package(PCL REQUIRED)

# Include header directory
include_directories(inc)

# Create executable with all source files
add_executable(Segmenter 
    src/Segmenter.cpp
    src/PartLoader.cpp
)

# Link PCL libraries
target_link_libraries(Segmenter ${PCL_LIBRARIES})
target_include_directories(Segmenter PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(Segmenter PRIVATE ${PCL_DEFINITIONS})