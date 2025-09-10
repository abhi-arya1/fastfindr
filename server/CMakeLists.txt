cmake_minimum_required(VERSION 3.16)
project(search_server)

set(CMAKE_CXX_STANDARD 17)

find_path(FAISS_INCLUDE_DIR NAMES faiss/IndexHNSW.h PATHS /opt/homebrew/include)
find_library(FAISS_LIBRARY NAMES faiss PATHS /opt/homebrew/lib)

# Find httplib
find_path(HTTPLIB_INCLUDE_DIR NAMES httplib.h PATHS /opt/homebrew/include)

# Find nlohmann/json
find_path(NLOHMANN_JSON_INCLUDE_DIR NAMES nlohmann/json.hpp PATHS /opt/homebrew/include)

# Find SQLite3
find_package(SQLite3 REQUIRED)

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

# Collect source files from the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create server executable (server.cpp contains main)
add_executable(server ${SOURCES})

# Configure server target
target_include_directories(server PRIVATE 
    ${FAISS_INCLUDE_DIR} 
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_ROOT_PATH}/include
    ${HTTPLIB_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/include
    include)
    
target_link_libraries(server 
    ${FAISS_LIBRARY}
    ${ONNXRUNTIME_LIB}
    SQLite::SQLite3
    tokenizers_cpp)


# Link OpenMP for server
if(OpenMP_CXX_FOUND)
    target_include_directories(server PRIVATE ${OpenMP_CXX_INCLUDE_DIR})
    target_compile_options(server PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(server ${OpenMP_CXX_LIBRARY})
endif()