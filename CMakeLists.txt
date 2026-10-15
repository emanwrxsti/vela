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
  src/kernels.cu src/attn.cu src/gemm.cu src/gemm_hl.cu src/engine.cu)
target_include_directories(supr-meow-vela PRIVATE src vendor ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(supr-meow-vela PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-O2 -Wall -Wextra -Wno-unused-parameter>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/O2 /EHsc /bigobj>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Xcompiler=-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/bigobj>)
if(MSVC)
  target_compile_definitions(supr-meow-vela PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_link_libraries(supr-meow-vela PRIVATE CUDA::cudart CUDA::cublasLt Threads::Threads)
# Windows: Winsock. Linux mining-OS images: static libstdc++/libgcc + CUDA runtime libs in lib/ (RPATH $ORIGIN/lib).
if(WIN32)
  target_link_libraries(supr-meow-vela PRIVATE ws2_32)
else()
  option(MEOW_STATIC_LIBSTDCXX "Link libstdc++/libgcc statically" ON)
  if(MEOW_STATIC_LIBSTDCXX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(supr-meow-vela PRIVATE -static-libstdc++ -static-libgcc)
  endif()
  set_target_properties(supr-meow-vela PROPERTIES BUILD_RPATH "\$ORIGIN/lib;${CUDAToolkit_LIBRARY_DIR}" INSTALL_RPATH "\$ORIGIN/lib" BUILD_WITH_INSTALL_RPATH ON)
  # NVML is dlopen'ed at runtime (device.cpp); link libdl where it is a separate library.
  target_link_libraries(supr-meow-vela PRIVATE ${CMAKE_DL_LIBS})
endif()
if(OPENSSL_FOUND)
  target_compile_definitions(supr-meow-vela PRIVATE MEOW_HAVE_OPENSSL=1)
  target_link_libraries(supr-meow-vela PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()

# CPU-only self-test (share-target math + socket loopback); no GPU required, runs in CI.
option(MEOW_BUILD_TESTS "Build CPU self-tests" OFF)
if(MEOW_BUILD_TESTS)
  add_executable(meow-net-test src/test_net.cpp)
  target_include_directories(meow-net-test PRIVATE src)
  target_link_libraries(meow-net-test PRIVATE Threads::Threads)
  if(WIN32)
    target_link_libraries(meow-net-test PRIVATE ws2_32)
  endif()
  if(MSVC)
    target_compile_definitions(meow-net-test PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  enable_testing()
  add_test(NAME compat-selftest COMMAND meow-net-test)
endif()
