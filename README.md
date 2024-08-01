Kokkos-FFTX project

This is simple example showing how to use FFTX with Kokkos Views

Building:
This project uses out of tree building of both FFTX and Kokkos
See the FFTX and Kokkos documentation to build both projects for CUDA Backend

Please make sure the following environment variables are set:

(These are usually set by FFTX build process)
export FFTX_HOME = <path-to-fftx>
export SPIRAL_HOME = <path-to-spiral-software>

(not set by default with module load, can find root by using which nvcc/nvc++)
export CUDA_HOME = <path-to-cuda-root>

to build this project use the following cmake command:
mkdir build; cd build;
cmake -DCMAKE_INSTALL_PREFIX=<kokkos-fftx-root> -DKokkos_ROOT=<kokkos-root> -DHELPER_CUDA_LOC=<path-to-helper_cuda.h> ..
make -j