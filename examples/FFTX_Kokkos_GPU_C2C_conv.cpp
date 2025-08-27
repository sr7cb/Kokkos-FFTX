// Test to check FFTX rconv function using Kokkos views
// Test: Take a field with point charge of 1/h^3 and zeros o.w. 
// Then convolve it with the Lattice Green's function data coming from PhiTrimmed file
// Expected result is the Lattice Green's function data on the reduced domain

# include <iostream>
# include <vector>
# include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <Kokkos_Core.hpp>
#include <complex>

#include "fftx3.hpp"
#include "interface.hpp"
#include "mdprdftObj.hpp"
#include "imdprdftObj.hpp"
#include "rconvObj.hpp"
#include "mddftObj.hpp"
#include "imddftObj.hpp"


int main(int argc, char* argv[]){
    Kokkos::initialize(argc, argv);
    {
      //Dims for the physical domain
      const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

      // Dims for computational domain (# of grid points)
      int Nx = 10, Ny = 10, Nz = 10;

      // Grid spacing h
      double hx = Lx/Nx;
      double hy = Ly/Ny;
      double hz = Lz/Nz;
      double h = hx;
      Kokkos::printf("h = %f\n", h);

      // point charge coordinates
      int cx = Nx/2;
      int cy = Ny/2;
      int cz = Nz/2;

      // Using the Kokkos::complex instead of std::complex
      using Complex = Kokkos::complex<double>;

      // 1D Kokkos view for the field with point charge
      Kokkos::View<double*, Kokkos::CudaSpace> Fpc_1D("Fvector", Nx*Ny*Nz);

      // Using kokkos paprallel_for to place the point charge in the center of the domain in Row Major Order
      Kokkos::parallel_for("Point charge field", Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<3>>({0, 0, 0}, {Nx, Ny, Nz}), 
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            int index = k * Ny * Nx + j * Nx + i;
             if(i == cx && j == cy && k == cz){
                Fpc_1D[index] = 1.0/pow(h, 3);
             }
             else{
                Fpc_1D[index] = 0.0;
             }
            //  Kokkos::printf("Fpc_1D[%d] = %f\n", index, Fpc_1D[index]);
        });  

      // Domain doubling the point charge field. It will act as the second input for rconv
      int domaindouble_x = 2*Nx;
      int domaindouble_y = 2*Ny;
      int domaindouble_z = 2*Nz; 

       // 1D Kokkos view for domain double point charge field
      Kokkos::View<double*, Kokkos::CudaSpace> F1D_domaindouble("Fvector", domaindouble_x*domaindouble_y*domaindouble_z);

      Kokkos::parallel_for("Place F1D in doubledomain", Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<3>>({0, 0, 0}, {Nx, Ny, Nz}), 
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            int index_dd = (i + Nx)*4*Ny*Nz + (j + Ny)*2*Nx + k;
            int index_1d = i * Nx * Ny + j * Nx + k;
            F1D_domaindouble[index_dd] = Fpc_1D[index_1d];
            // Kokkos::printf("index_dd = %d\n", index_dd);
            // Kokkos::printf("F1D_dd[%d] = %f\n", index_dd, F1D_domaindouble[index_dd]);
        });

      // // Convert to a complex array for performing complex to complex FFT convolution
      // Kokkos::View<Complex*, Kokkos::CudaSpace> F1D_dd_cmplx("Fvector_complex", domaindouble_x * domaindouble_y * domaindouble_z);
      
      // Kokkos::parallel_for("convert_to_complex", Kokkos::RangePolicy<Kokkos::Cuda>(0, domaindouble_x * domaindouble_y * domaindouble_z), 
      // KOKKOS_LAMBDA(const int i) {
      //   F1D_dd_cmplx(i) = Complex(F1D_domaindouble(i), 0.0);
      //   // printf("real_F1D[%d] = %f, complex_F1D[%d] = (%f,%f)\n", i, F1D_domaindouble[i], i, F1D_dd_cmplx(i).real(), F1D_dd_cmplx(i).imag());
      // });

      // Lattice Green's function
      std::ifstream infileLGF("/home/h82/Documents/Bluestone/P3M/MLC_PPM/LatticeGreensFunction/exec/G_64_Octant");
      std::vector<double> lgf_values(8*Nx*Ny*Nz);
      if(infileLGF.is_open()){
        std::string line;
        int xdir;
        int ydir; 
        int zdir;
        double lgf;
   

        while(getline(infileLGF, line)){
     
        for(char& c : line){
          if (c == '(' || c == ')'){
            c = ' ';
          }
        }

       // Remove extra spaces around commas
        size_t pos = line.find(", ");
        while (pos != std::string::npos) {
          line.erase(pos, 1);  // Erase space after comma
          pos = line.find(", ", pos);
       }

      // Remove leading and trailing spaces
        size_t first = line.find_first_not_of(" \t");
        size_t last = line.find_last_not_of(" \t");

        if (first != std::string::npos && last != std::string::npos) {
          line = line.substr(first, last - first + 1);
        }
        // std::cout << line << std::endl;
        std::stringstream ss(line);
      
        ss >> xdir;
        ss.ignore(1, ',');
        ss >> ydir;
        ss.ignore(1, ',');
        ss >> zdir;
        ss.ignore(1, ',');
        ss >> lgf;

      // if((xdir >= (-Nx) && xdir < (Nx)) && (ydir >= (-Ny) && ydir < (Ny)) && (zdir >= (-Nz) && zdir < (Nz))){
      //     int index_lgf = zdir + Nz + 2*Ny*(ydir + Ny) + 4*Ny*Nx*(xdir+Nx);
      //     lgf_values[index_lgf] = lgf;
      //     // lgf_values.push_back(lgf);
      // }
        if(xdir >= 0 && xdir <= Nx && ydir >= 0 && ydir <= Ny && zdir >= 0 && zdir <= Nz)
          {
            for (int isign = -1; isign < 2; isign+=2)
              {
                for (int jsign = -1; jsign < 2; jsign+=2)
                  {
                    for (int ksign = -1; ksign < 2; ksign+=2)
                      {
                        int zdirsigned = zdir*ksign;
                        int ydirsigned = ydir*jsign;
                        int xdirsigned = xdir*isign;
                        if ((xdirsigned != Nx) &&
                            (ydirsigned != Ny) &&
                            (zdirsigned != Nz))
                          {                        
                            int index = zdirsigned + Nz +
                              2*Nz*(ydirsigned + Ny) +
                              4*Ny*Nz*(xdirsigned + Nx);
                      
                            lgf_values[index] = lgf/h;
                          }
                      }
                  }
              }
          }
      }
      // for(double lgfv : lgf_values){
      //   std::cout << std::setprecision(10) << lgfv << std::endl;
      // }
      infileLGF.close();
    }
    else{
      std::cout << "Unable to open file" << std::endl;
    }
    std::cout << "Size of LGF Vector=" << lgf_values.size()<<std::endl;

    // Creating a host view for the LGF values
    Kokkos::View<double*, Kokkos::HostSpace> host_LGF("h_view", domaindouble_x * domaindouble_y * domaindouble_z);

    // Copying/storing lgf_values into the host view
    for(int i = 0; i < (domaindouble_x * domaindouble_y * domaindouble_z); i++){
      host_LGF[i] = lgf_values.data()[i];
      // std::cout<<"i = "<< i<<",\t host_LGF = " << host_LGF[i] <<",\t" << lgf_values.data()[i] << std::endl;
    }

    // Creating a device view to deep copy host_LGF values 
    Kokkos::View<double*, Kokkos::CudaSpace> dev_LGF("d_view", domaindouble_x * domaindouble_y * domaindouble_z);
    // deepcopy host to device
    Kokkos::deep_copy(dev_LGF, host_LGF);

    // Defining data vectors required for forward DFT in FFTX as Kokkos views
    Kokkos::View<Complex*, Kokkos::CudaSpace> symbol("symbol_view", domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
    Kokkos::View<double*, Kokkos::CudaSpace> dummy1("dummy1_view", domaindouble_x * domaindouble_y * domaindouble_z);
 
    std::vector<void*> args1 = [&]() {
        static auto symbol_data = symbol.data();
        static auto lgf_data = dev_LGF.data();
        static auto dummy1_data = dummy1.data();
        return std::vector<void*>{&symbol_data, &lgf_data, &dummy1_data};
    }();

    // Computing DFT of Lattice Green's Function values and storing it in symbol
    std::vector<int> sizes{domaindouble_x, domaindouble_y, domaindouble_z};
    MDPRDFTProblem r2cdft{args1, sizes, "mdprdft"};
    r2cdft.transform();

    //  Kokkos::parallel_for("Print symbol", Kokkos::RangePolicy<Kokkos::Cuda>(0, domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)), 
    //     KOKKOS_LAMBDA(const int i) {
    //         // printf("i = %d", i);
    //         printf("symbol[%d] = (%f,%f)\n", i, symbol(i).real(), symbol(i).imag());
    //     });


    //--------------------------------------------------------------//
//--------------------------------------------------------------//
// ************* Using RConv function in FFTX *****************//
//  // Using RConv function in FFTX to compute convolution between symbol and F1D_domaindouble
//  Kokkos::View<double*, Kokkos::CudaSpace> out_idft("Rconv output", domaindouble_x * domaindouble_y * domaindouble_z);
//  std::vector<void*> args4 = [&]() {
//       static auto output_data = out_idft.data();
//       static auto F1D_data = F1D_domaindouble.data();
//       static auto symbol1_data = symbol.data();
//       return std::vector<void*>{&output_data, &F1D_data, &symbol1_data};
//   }();
//   std::vector<int> sizes4{domaindouble_x, domaindouble_y, domaindouble_z};
//   //rconv class
//     RCONVProblem conv{args4, sizes4, "rconv"};

//     // // Run the transform
//     conv.transform();

//**************** END of of RCONV ****************************//
    //**************** Using Individual FFTX functions for Convolution *******//

  // Defining data vectors required for forward DFT of the input2 ie F1D in FFTX as Kokkos views
    Kokkos::View<Complex*, Kokkos::CudaSpace> F_dft("fwd_dft_view", domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
    Kokkos::View<double*, Kokkos::CudaSpace> dummy2("dummy2_view", domaindouble_x * domaindouble_y * domaindouble_z);

    std::vector<void*> args2 = [&]() {
        static auto Fdft_data = F_dft.data();
        static auto F1D_data = F1D_domaindouble.data();
        static auto dummy2_data = dummy2.data();
        return std::vector<void*>{&Fdft_data, &F1D_data, &dummy2_data};
    }();

    std::vector<int> sizes2{domaindouble_x, domaindouble_y, domaindouble_z};
    MDDFTProblem c2cdft2{args2, sizes2, "mdprdft"};
    c2cdft2.transform();

    // Kokkos::parallel_for("Print F1DDFT", Kokkos::RangePolicy<Kokkos::Cuda>(0, domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)), 
    //     KOKKOS_LAMBDA(const int i) {
    //         // printf("i = %d", i);
    //         printf("F1D_DFT[%d] = (%f,%f)\n", i, F_dft(i).real(), F_dft(i).imag());
    //     });

    // Pointwise Mulitply
    Kokkos::View<Complex*, Kokkos::CudaSpace> pointwise_mul("fwd_dft_view", domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));

    Kokkos::parallel_for("Pointwise_multiply", Kokkos::RangePolicy<Kokkos::Cuda>(0,domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)), 
        KOKKOS_LAMBDA(const int i) {
            double a = symbol(i).real();
            double b = symbol(i).imag();
            double c = F_dft(i).real();
            double d = F_dft(i).imag();

            //Pointwise multiply symbol and Fpc_dft
            double real_pw = a * c - b * d;
            double img_pw = a * d + b * c;

            pointwise_mul[i] = Complex(real_pw, img_pw);
            // printf("pwise[%d] = (%f,%f)\n", i, pointwise_mul(i).real(), pointwise_mul(i).imag());
        });

    // Calculate the inverse dft to compute the final convolution value
    Kokkos::View<double*, Kokkos::CudaSpace> out_idft("inv_dft_view", domaindouble_x * domaindouble_y * domaindouble_z);
    Kokkos::View<double*, Kokkos::CudaSpace> dummy3("dummy3_view", domaindouble_x * domaindouble_y * domaindouble_z);
    std::vector<void*> args3 = [&]() {
        static auto out_data = out_idft.data();
        static auto pwise_data = pointwise_mul.data();
        static auto dummy3_data = dummy3.data();
        return std::vector<void*>{&out_data, &pwise_data, &dummy3_data};
    }();

    std::vector<int> sizes3{domaindouble_x, domaindouble_y, domaindouble_z};
    IMDDFTProblem c2cdft{args3, sizes3, "imdprdft"};
    c2cdft.transform();

    // Kokkos::parallel_for("Print IF1DDFT", Kokkos::RangePolicy<Kokkos::Cuda>(0, domaindouble_x * domaindouble_y * domaindouble_z), 
    //     KOKKOS_LAMBDA(const int i) {
    //         // printf("i = %d", i);
    //         printf("IF1D_DFT[%d] = %f\n", i, out_idft[i]);
    //     });

    //**************** END of INDIVIDUAL FUNCTIONS ****************************//
    //--------------------------------------------------------------//
    //--------------------------------------------------------------//
    
    // Normalizing the output with h^3/dd^3
    Kokkos::View<double*, Kokkos::CudaSpace> out_normalize("norm_output_view", domaindouble_x * domaindouble_y * domaindouble_z);
    // Normalization Factor
    double norm_factor = pow(h,3)/(domaindouble_x * domaindouble_y * domaindouble_z);
    printf("norm factor = %1.8f\n", norm_factor);
    Kokkos::parallel_for("Normalize output", Kokkos::RangePolicy<Kokkos::Cuda>(0,domaindouble_x * domaindouble_y * domaindouble_z), 
        KOKKOS_LAMBDA(const int i) {
            out_normalize[i] = norm_factor * out_idft[i];
            // Kokkos::printf("norm factor AFTER = %f\n", norm_factor);
            //  Kokkos::printf("out_normalize[%d] = %f\n", i, out_normalize[i]);
        });

    Kokkos::View<double*, Kokkos::CudaSpace> conv_output("Final exatracted output", Nx * Ny * Nz);
    Kokkos::parallel_for("Normalize output", Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<3>>({0, 0, 0}, {Nx, Ny, Nz}), 
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
             //Calculate the index in the domain doubled output vector
            int out_dd_index = i * domaindouble_y * domaindouble_z + j * domaindouble_z + k;
        // Calculate the index in the smaller output of the orginal domain size 
            int out_original_index = i * Nz * Ny + j * Nz + k;
        // Copying the values from larger to smaller output vector
            conv_output[out_original_index] = out_normalize[out_dd_index];
            printf("conv_output[%d] = %f\n", out_original_index, conv_output[out_original_index]);
        }); 
        
         
  }

    
  Kokkos::finalize();
  return 0;
}