#include <iostream>
#include "fftx3.hpp"
#include "interface.hpp"
#include "mddftObj.hpp"
#include "imddftObj.hpp"
#include <string>
#include <fstream>
#include <complex>
#include<Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
  std::cout << "hello world" << std::endl;
  int N = 32;
  Kokkos::View<double*, Kokkos::CudaUVMSpace> input("input", N*N*N*2);
  Kokkos::View<double*, Kokkos::CudaUVMSpace> output("output", N*N*N*2);
  Kokkos::View<double*, Kokkos::CudaUVMSpace> symbol("symbol", N*N*N*2);

  Kokkos::parallel_for("Init", N*N*N*2, KOKKOS_LAMBDA ( int i) {
    input(i) = 1;
    output(i) = 0;
  });
  
  std::vector<void*> args = [&]() {
      static auto output_data = output.data();
      static auto input_data = input.data();
      static auto symbol_data = symbol.data();
      return std::vector<void*>{&output_data, &input_data, &symbol_data};
  }();
  std::vector<int> sizes{N,N,N};
  MDDFTProblem mdp(args, sizes, "mddft");
  mdp.transform();
  for(int i = 0; i < 10; i++)
    std::cout << output(i) << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
