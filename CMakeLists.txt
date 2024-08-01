cmake_minimum_required (VERSION 3.18)

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

project ( Kokkos-FFTX
          VERSION 0.0.1
          DESCRIPTION "Integrating FFTX into Kokkos application"
          LANGUAGES C CXX )

##  Prevent building directly into the source tre
string (COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" DISALLOW_COMPILE_INPLACE)
if ( DISALLOW_COMPILE_INPLACE )
    message (FATAL_ERROR "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
                 rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
                 mkdir <build>
                 cd <build>
                 cmake ..
                 make | make install")
endif ()

# Define a custom path variable with a default value
set(HELPER_CUDA_LOC $ENV{HOME} CACHE PATH "location of cuda_helper.h")

# Print the variable value to verify it's being set correctly
message(STATUS "HELPER_CUDA_LOC is set to ${HELPER_CUDA_LOC}")

##  Set C standard 17
set ( CMAKE_C_STANDARD 17 )
set ( CMAKE_CXX_STANDARD 17 )

find_package(Kokkos REQUIRED)

##  FFTX_HOME must be defined in the environment or on the command line
if ( DEFINED ENV{FFTX_HOME} )
    message ( STATUS "FFTX_HOME = $ENV{FFTX_HOME}" )
    set ( FFTX_SOURCE_DIR $ENV{FFTX_HOME} )
else ()
    if ( "x${FFTX_HOME}" STREQUAL "x" )
        message ( FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( FFTX_SOURCE_DIR ${FFTX_HOME} )
endif ()

##  Include FFTX CMake functions
include ( "${FFTX_SOURCE_DIR}/CMakeIncludes/FFTXCmakeFunctions.cmake" )

add_subdirectory ( examples )