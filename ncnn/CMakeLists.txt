cmake_minimum_required(VERSION 3.10)
project(firered_vad VERSION 1.0.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 可选 OpenMP（某些 ncnn 可能启用 OpenMP）
find_package(OpenMP)
if(TARGET OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP available: will link if needed")
else()
    message(STATUS "OpenMP not found or disabled")
endif()

find_package(Threads)

# NCNN 路径（可通过环境变量覆盖）
if(NOT DEFINED ENV{NCNN_ROOT})
    set(NCNN_ROOT "./3rd/prebuild/MacOS/ncnn" CACHE PATH "NCNN root directory")
else()
    set(NCNN_ROOT $ENV{NCNN_ROOT})
endif()

set(_NCNN_BUILD_LIB "${NCNN_ROOT}/build/src/libncnn.a")
set(_NCNN_INST_LIB  "${NCNN_ROOT}/lib/libncnn.a")
if(EXISTS "${_NCNN_INST_LIB}")
    set(NCNN_INCLUDE_DIR "${NCNN_ROOT}/include")
    set(NCNN_LIB "${_NCNN_INST_LIB}")
elseif(EXISTS "${_NCNN_BUILD_LIB}")
    set(NCNN_INCLUDE_DIR "${NCNN_ROOT}/build/src")
    set(NCNN_LIB "${_NCNN_BUILD_LIB}")
else()
    message(FATAL_ERROR "NCNN not found. Set NCNN_ROOT to the install prefix (with lib/libncnn.a and include/) or the source dir with build/src/libncnn.a.")
endif()

message(STATUS "NCNN Root: ${NCNN_ROOT}")
message(STATUS "NCNN Include: ${NCNN_INCLUDE_DIR}")
message(STATUS "NCNN Lib: ${NCNN_LIB}")

# 源文件
set(FRONTEND_SRCS
    ../common/frontend/fft.cc
)

set(C_API_SRCS
    ./csrc/firered_vad_stream_packed.cpp
)

# 创建共享库
add_library(firered_vad_stream SHARED
    ${C_API_SRCS}
    ${FRONTEND_SRCS}
)

target_include_directories(firered_vad_stream PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>
    $<BUILD_INTERFACE:${NCNN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${NCNN_INCLUDE_DIR}/ncnn>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(firered_vad_stream PUBLIC
    ${NCNN_LIB}
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(firered_vad_stream PUBLIC OpenMP::OpenMP_CXX)
endif()
if(ANDROID)
    target_link_libraries(firered_vad_stream PUBLIC log)
else()
    if(Threads_FOUND)
        target_link_libraries(firered_vad_stream PUBLIC Threads::Threads m)
    endif()
endif()

# 创建测试程序
add_executable(test_vad_stream
    examples/test_vad_stream_packed.cpp
)

target_include_directories(test_vad_stream PRIVATE
    src
    ${NCNN_INCLUDE_DIR}
    ${NCNN_INCLUDE_DIR}/ncnn
)

target_link_libraries(test_vad_stream PRIVATE
    firered_vad_stream
)

# 非流式测试程序（整段一次性推理）
add_executable(test_vad_non_stream
    examples/test_vad_non_stream.cpp
    ../common/frontend/fft.cc
)

target_include_directories(test_vad_non_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${NCNN_INCLUDE_DIR}
    ${NCNN_INCLUDE_DIR}/ncnn
)

target_link_libraries(test_vad_non_stream PRIVATE
    ${NCNN_LIB}
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(test_vad_non_stream PRIVATE OpenMP::OpenMP_CXX)
endif()
if(ANDROID)
    target_link_libraries(test_vad_non_stream PRIVATE log)
else()
    if(Threads_FOUND)
        target_link_libraries(test_vad_non_stream PRIVATE Threads::Threads m)
    endif()
endif()

# AED 非流式测试程序（3类：speech/singing/music）
add_executable(test_aed_non_stream
    examples/test_aed_non_stream.cpp
    ../common/frontend/fft.cc
)

target_include_directories(test_aed_non_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${NCNN_INCLUDE_DIR}
    ${NCNN_INCLUDE_DIR}/ncnn
)

target_link_libraries(test_aed_non_stream PRIVATE
    ${NCNN_LIB}
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(test_aed_non_stream PRIVATE OpenMP::OpenMP_CXX)
endif()
if(ANDROID)
    target_link_libraries(test_aed_non_stream PRIVATE log)
else()
    if(Threads_FOUND)
        target_link_libraries(test_aed_non_stream PRIVATE Threads::Threads m)
    endif()
endif()

# 安装配置
include(GNUInstallDirs)

install(TARGETS firered_vad_stream
    EXPORT firered_vad_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES src/firered_vad_stream_packed.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/firered_vad
)

install(TARGETS test_vad_stream
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 导出配置
install(EXPORT firered_vad_targets
    FILE firered_vadTargets.cmake
    NAMESPACE firered_vad::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/firered_vad
)

message(STATUS "")
message(STATUS "FireRedVAD NCNN Stream build configured successfully!")
message(STATUS "Build with: cmake --build .")
message(STATUS "Install with: cmake --install . --prefix <install_dir>")
message(STATUS "")
