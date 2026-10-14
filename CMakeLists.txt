cmake_minimum_required(VERSION 3.24)
# One fatbin for every card we care about: Ampere (A100 / 170HX / 30xx), Ada, Blackwell consumer. Must be set BEFORE project().
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;120")
endif()
project(supr-meow-vela LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(CUDAToolkit REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenSSL)

add_executable(supr-meow-vela
  src/main.cpp src/poisd.cpp src/stratum.cpp src/httpd.cpp src/device.cpp
  src/kernels.cu src/attn.cu src/gemm.cu src/engine.cu)
target_include_directories(supr-meow-vela PRIVATE src vendor ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(supr-meow-vela PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler=-Wall>)
target_link_libraries(supr-meow-vela PRIVATE CUDA::cudart CUDA::cublasLt Threads::Threads)
# Portable binary for mining OS images: static libstdc++/libgcc, CUDA runtime libs shipped in lib/ next to the binary (RPATH $ORIGIN/lib).
option(MEOW_STATIC_LIBSTDCXX "Link libstdc++/libgcc statically" ON)
if(MEOW_STATIC_LIBSTDCXX)
  target_link_options(supr-meow-vela PRIVATE -static-libstdc++ -static-libgcc)
endif()
set_target_properties(supr-meow-vela PROPERTIES BUILD_RPATH "\$ORIGIN/lib;${CUDAToolkit_LIBRARY_DIR}" INSTALL_RPATH "\$ORIGIN/lib" BUILD_WITH_INSTALL_RPATH ON)
if(OPENSSL_FOUND)
  target_compile_definitions(supr-meow-vela PRIVATE MEOW_HAVE_OPENSSL=1)
  target_link_libraries(supr-meow-vela PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
# NVML is loaded with dlopen at runtime (device.cpp) so the binary runs where the driver has no dev package.
target_link_libraries(supr-meow-vela PRIVATE ${CMAKE_DL_LIBS})
