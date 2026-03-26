cmake_minimum_required(VERSION 3.18)

project(Onnxify.OperatorSchemaGenerator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime")

set(onnxruntime_MINIMAL_BUILD OFF CACHE BOOL "" FORCE)
set(onnxruntime_DISABLE_CONTRIB_OPS OFF CACHE BOOL "" FORCE)
set(onnxruntime_DISABLE_ML_OPS OFF CACHE BOOL "" FORCE)

set(onnxruntime_ENABLE_TRAINING_OPS ON CACHE BOOL "" FORCE)
set(onnxruntime_ENABLE_TRAINING OFF CACHE BOOL "" FORCE)

set(onnxruntime_USE_FULL_PROTOBUF ON CACHE BOOL "" FORCE)
set(ONNX_DISABLE_DOC_STRINGS OFF CACHE BOOL "" FORCE)
set(__ONNX_NO_DOC_STRINGS OFF CACHE BOOL "" FORCE)

set(onnxruntime_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(onnxruntime_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_CSHARP OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(onnxruntime_RUN_ONNX_TESTS OFF CACHE BOOL "" FORCE)
set(onnxruntime_DEV_MODE OFF CACHE BOOL "" FORCE)
set(onnxruntime_ENABLE_LAZY_TENSOR OFF CACHE BOOL "" FORCE)

add_subdirectory(${ONNXRUNTIME_ROOT}/cmake 
                 ${CMAKE_BINARY_DIR}/onnxruntime-build 
                 EXCLUDE_FROM_ALL)

add_executable(Onnxify.OperatorSchemaGenerator main.cpp)

target_include_directories(Onnxify.OperatorSchemaGenerator PRIVATE
    ${ONNXRUNTIME_ROOT}/include/onnxruntime
    ${CMAKE_BINARY_DIR}/onnxruntime-build/external/onnx
    ${CMAKE_BINARY_DIR}/_deps/nlohmann_json-src/include
)

target_link_libraries(Onnxify.OperatorSchemaGenerator PRIVATE
    onnxruntime
    onnx
    onnx_proto
    protobuf::libprotobuf
    nlohmann_json::nlohmann_json
)

if(MSVC)
    target_link_options(Onnxify.OperatorSchemaGenerator PRIVATE /WHOLEARCHIVE:onnx)
endif()