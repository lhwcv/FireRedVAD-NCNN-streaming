cmake_minimum_required(VERSION 3.10)
project(firered_vad VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 查找 OpenMP
find_package(OpenMP REQUIRED)

# NCNN 路径（可通过环境变量覆盖）
if(NOT DEFINED ENV{NCNN_ROOT})
    set(NCNN_ROOT "/opt/ncnn-20260113" CACHE PATH "NCNN root directory")
else()
    set(NCNN_ROOT $ENV{NCNN_ROOT})
endif()

set(NCNN_INCLUDE_DIR ${NCNN_ROOT}/build/src)
set(NCNN_LIB ${NCNN_ROOT}/build/src/libncnn.a)

message(STATUS "NCNN Root: ${NCNN_ROOT}")
message(STATUS "NCNN Include: ${NCNN_INCLUDE_DIR}")
message(STATUS "NCNN Lib: ${NCNN_LIB}")

# 源文件
set(FRONTEND_SRCS
    src/frontend/fft.cc
)

set(C_API_SRCS
    src/c_api/firered_vad_stream_packed.cpp
)

# 创建共享库
add_library(firered_vad_stream SHARED
    ${C_API_SRCS}
    ${FRONTEND_SRCS}
)

target_include_directories(firered_vad_stream PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${NCNN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${NCNN_INCLUDE_DIR}/ncnn>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(firered_vad_stream PUBLIC
    ${NCNN_LIB}
    OpenMP::OpenMP_CXX
    pthread
    m
)

# 创建测试程序
add_executable(test_vad_stream
    src/c_api/test_stream_packed.cpp
)

target_include_directories(test_vad_stream PRIVATE
    src
    ${NCNN_INCLUDE_DIR}
    ${NCNN_INCLUDE_DIR}/ncnn
)

target_link_libraries(test_vad_stream PRIVATE
    firered_vad_stream
)

# 安装配置
include(GNUInstallDirs)

install(TARGETS firered_vad_stream
    EXPORT firered_vad_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES src/c_api/firered_vad_stream_packed.h
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
