cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

# set(Torch_DIR /home/qiu/Downloads/libtorch-cxx11-abi-shared-with-deps-1.5.1/libtorch/share/cmake/Torch)
# set(CMAKE_PREFIX_PATH /home/qiu/Downloads/libtorch-cxx11-abi-shared-with-deps-1.5.1/libtorch/)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++11")
set(Torch_DIR /home/qiu/Downloads/libtorch-cxx11-abi-shared-with-deps-1.6.0/libtorch/share/cmake/Torch)
set(CMAKE_PREFIX_PATH /home/qiu/Downloads/libtorch-cxx11-abi-shared-with-deps-1.6.0/libtorch/)
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(gflags REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS} /home/qiu/intel/compilers_and_libraries_2020/linux/mkl/include /home/qiu/Downloads/protobuf-3.11.4/src)
# include_directories(${OpenCV_INCLUDE_DIRS} /home/qiu/intel/compilers_and_libraries_2020/linux/mkl/include /home/qiu/Downloads/protobuf-3.6.1/src)
include_directories(${OpenCV_INCLUDE_DIRS} /home/qiu/intel/compilers_and_libraries_2020/linux/mkl/include)
link_directories(
    # /home/qiu/Projects/Self-Correction-Human-Parsing/mhp_parsing/deploy/ 
    /home/qiu/Projects/Self-Correction-Human-Parsing/mhp_parsing/deploy/build/    
    )
message(STATUS "Pytorch status:")
message(STATUS "libraries: ${TORCH_LIBRARIES}")
message(STATUS "libraries: ${OpenCV_LIBS}")

# add_library(mhp SHARED mhp.cpp)
# target_link_libraries(mhp "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" gflags glog protobuf -lprotobuf)
add_executable(mhp test.cpp)
target_link_libraries(mhp "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" gflags glog protobuf -lprotobuf -lmhp)


set_property(TARGET mhp PROPERTY CXX_STANDARD 14)
