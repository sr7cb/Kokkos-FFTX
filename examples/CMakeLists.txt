cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )
project ( examples )

set(MDDFT_TEST kokkos_mddft)

add_executable (${MDDFT_TEST} kokkos_mddft.cpp)
target_compile_options(${MDDFT_TEST} PRIVATE -DFFTX_CUDA -I$ENV{CUDA_HOME}/include -I${HELPER_CUDA_LOC})
target_link_libraries(${MDDFT_TEST} PRIVATE Kokkos::kokkos nvrtc cuda)

FFTX_add_includes_libs_to_target ( ${MDDFT_TEST} )

# -----------------------------------------------------------

set(RCONV_TEST FFTX_Kokkos_GPU_rconv)

add_executable (${RCONV_TEST} FFTX_Kokkos_GPU_rconv.cpp)
target_compile_options(${RCONV_TEST} PRIVATE -DFFTX_CUDA -I$ENV{CUDA_HOME}/include -I${HELPER_CUDA_LOC})
target_link_libraries(${RCONV_TEST} PRIVATE Kokkos::kokkos nvrtc cuda)

FFTX_add_includes_libs_to_target ( ${RCONV_TEST} )

# -----------------------------------------------------------

set(C2C_CONV_TEST FFTX_Kokkos_GPU_C2C_conv)

add_executable (${C2C_CONV_TEST} FFTX_Kokkos_GPU_C2C_conv.cpp)
target_compile_options(${C2C_CONV_TEST} PRIVATE -DFFTX_CUDA -I$ENV{CUDA_HOME}/include -I${HELPER_CUDA_LOC})
target_link_libraries(${C2C_CONV_TEST} PRIVATE Kokkos::kokkos nvrtc cuda)

FFTX_add_includes_libs_to_target ( ${C2C_CONV_TEST} )


