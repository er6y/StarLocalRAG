cmake_minimum_required(VERSION 3.4.1)

# 设置项目名称
project(tokenizers_jni)

# 添加日志库
find_library(log-lib log)

# 设置 Rust 库路径
set(RUST_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../libs/tokenizers-ffi/target)

# 对于 x86 和 x86_64，我们使用存根库
if(${ANDROID_ABI} STREQUAL "x86" OR ${ANDROID_ABI} STREQUAL "x86_64")
    message(WARNING "${ANDROID_ABI} ABI is using stub library")
    
    # 为 x86/x86_64 创建一个空的 C++ 库作为替代
    add_library(tokenizers_ffi_stub SHARED
               stub_library.cpp)
               
    # 链接日志库到存根库
    target_link_libraries(tokenizers_ffi_stub
                          ${log-lib})
    
    # 添加 tokenizers_jni 库
    add_library(tokenizers_jni SHARED
                tokenizers_jni.cpp)
    
    # 包含头文件目录
    target_include_directories(tokenizers_jni PRIVATE
                              ${CMAKE_CURRENT_SOURCE_DIR})
    
    # 将 tokenizers_jni 链接到存根库
    target_link_libraries(tokenizers_jni
                          tokenizers_ffi_stub
                          ${log-lib})
else()
    # 对于支持的架构，使用 Rust 库
    
    # 添加 tokenizers_ffi 库
    add_library(tokenizers_ffi SHARED IMPORTED)
    
    # 添加 tokenizers_jni 库
    add_library(tokenizers_jni SHARED
                tokenizers_jni.cpp)
    
    # 包含头文件目录
    target_include_directories(tokenizers_jni PRIVATE
                              ${CMAKE_CURRENT_SOURCE_DIR})
    # 根据ABI选择正确的库路径
    if(${ANDROID_ABI} STREQUAL "arm64-v8a")
        set_target_properties(tokenizers_ffi PROPERTIES IMPORTED_LOCATION
                ${RUST_LIB_DIR}/aarch64-linux-android/release/libtokenizers_ffi.so)
    elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
        set_target_properties(tokenizers_ffi PROPERTIES IMPORTED_LOCATION
                ${RUST_LIB_DIR}/armv7-linux-androideabi/release/libtokenizers_ffi.so)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
    
    # 链接库
    target_link_libraries(tokenizers_jni
                          tokenizers_ffi
                          ${log-lib})
endif()
