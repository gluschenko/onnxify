# CMakeList.txt : CMake project for Onnxify.OperatorSchemaGenerator, include source and define
# project specific logic here.
#

set(CMAKE_POLICY_VERSION_MINIMUM 3.10)

set(PROTOBUF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/protobuf)
set(ONNX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnx)
set(JSON_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/json)

# protobuf
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${PROTOBUF_ROOT} ${CMAKE_BINARY_DIR}/protobuf-build)

set(Protobuf_DIR ${CMAKE_BINARY_DIR}/protobuf-build)

# onnx
set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_compile_options(/wd4018 /wd4834)

add_subdirectory(${ONNX_ROOT} ${CMAKE_BINARY_DIR}/onnx-build)

#json
add_subdirectory(${JSON_ROOT} ${CMAKE_BINARY_DIR}/json-build)

# Add source to this project's executable.
add_executable (Onnxify.OperatorSchemaGenerator "main.cpp"   )

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Onnxify.OperatorSchemaGenerator PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(Onnxify.OperatorSchemaGenerator PRIVATE onnx nlohmann_json::nlohmann_json)

target_include_directories(Onnxify.OperatorSchemaGenerator PRIVATE
    ${ONNX_ROOT}
    ${CMAKE_BINARY_DIR}/onnx-build
    ${CMAKE_BINARY_DIR}/onnx-build/onnx
)

# TODO: Add tests and install targets if needed.
