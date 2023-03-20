cmake_minimum_required(VERSION 3.5.1)lm_proto

project(LanguageModel C CXX)

include(../cmake/common.cmake)

# Proto file
get_filename_component(lm_proto "../../protos/llm.proto" ABSOLUTE)
get_filename_component(lm_proto_path "${lm_proto}" PATH)

# Generated sources
set(lm_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/llm.pb.cc")
set(lm_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/llm.pb.h")
set(lm_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/llm.grpc.pb.cc")
set(lm_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/llm.grpc.pb.h")
add_custom_command(
      OUTPUT "${lm_proto_srcs}" "${lm_proto_hdrs}" "${lm_grpc_srcs}" "${lm_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${lm_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${lm_proto}"
      DEPENDS "${lm_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# lm_grpc_proto
add_library(lm_grpc_proto
  ${lm_grpc_srcs}
  ${lm_grpc_hdrs}
  ${lm_proto_srcs}
  ${lm_proto_hdrs})
target_link_libraries(lm_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# llm_helper
add_library(llm_helper
  "helper.h"
  "helper.cc")
target_link_libraries(llm_helper
  lm_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Targets llm_(client|server)
foreach(_target
  llm_client llm_server)
  add_executable(${_target}
    "${_target}.cc")
  target_link_libraries(${_target}
    lm_grpc_proto
    llm_helper
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
endforeach()
