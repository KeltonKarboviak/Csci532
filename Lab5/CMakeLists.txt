cmake_minimum_required (VERSION 2.6)
project (OPENCL_MATRIX_ADD)

SET (PLATFORM 64)

#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -stdlib=libstdc++")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -fsanitize=address -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -g -Wall -O1 -funroll-loops -msse3 -fsanitize=address -fno-omit-frame-pointer -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -DNAN_CHECKS")
SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -funroll-loops -DNDEBUG")

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

find_package(OpenCL)

message(STATUS "opencl libraries: ${OpenCL_LIBRARIES}")
message(STATUS "opencl include_dirs: ${OpenCL_INCLUDE_DIRS}")

message(STATUS "ADDING __OPENCL__ DEFINE")
add_definitions( -D__OPENCL__ )

include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(matrix_mul matrix_mul opencl_utils)
target_link_libraries(matrix_mul ${OpenCL_LIBRARIES})
