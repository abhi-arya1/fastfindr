cmake_minimum_required(VERSION 3.16)
project(hnsw_search)

set(CMAKE_CXX_STANDARD 17)

find_path(FAISS_INCLUDE_DIR NAMES faiss/IndexHNSW.h PATHS /opt/homebrew/include)
find_library(FAISS_LIBRARY NAMES faiss PATHS /opt/homebrew/lib)

# Find ONNX Runtime
set(ONNXRUNTIME_ROOT_PATH "/opt/homebrew" CACHE PATH "ONNX Runtime root directory")

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT_PATH}/lib
    REQUIRED
)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
    REQUIRED
)

# Find OpenMP manually for macOS with Homebrew
find_path(OpenMP_CXX_INCLUDE_DIR NAMES omp.h PATHS /opt/homebrew/opt/libomp/include)
find_library(OpenMP_CXX_LIBRARY NAMES omp PATHS /opt/homebrew/opt/libomp/lib)

if(OpenMP_CXX_INCLUDE_DIR AND OpenMP_CXX_LIBRARY)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp")
    set(OpenMP_CXX_FOUND TRUE)
endif()

# Add tokenizers-cpp
add_subdirectory(third_party/tokenizers-cpp EXCLUDE_FROM_ALL)

# Include directories  
include_directories(include)

# Collect source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(hnsw_search main.cpp ${SOURCES})

target_include_directories(hnsw_search PRIVATE 
    ${FAISS_INCLUDE_DIR} 
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_ROOT_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/include
    include)
target_link_libraries(hnsw_search 
    ${FAISS_LIBRARY}
    ${ONNXRUNTIME_LIB}
    tokenizers_cpp)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_include_directories(hnsw_search PRIVATE ${OpenMP_CXX_INCLUDE_DIR})
    target_compile_options(hnsw_search PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(hnsw_search ${OpenMP_CXX_LIBRARY})
endif()