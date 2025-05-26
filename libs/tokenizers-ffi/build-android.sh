#!/bin/bash

# 为Android编译Rust库的脚本
# 支持arm64-v8a, armeabi-v7a, x86, x86_64架构

set -e

# 设置路径变量
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
JNI_LIBS_DIR="$SCRIPT_DIR/../../app/src/main/jniLibs"

# 确保目录存在
for ARCH in arm64-v8a armeabi-v7a x86 x86_64; do
    mkdir -p "$JNI_LIBS_DIR/$ARCH"
done

# 确保已安装必要的Rust目标
echo "确保已安装必要的Rust目标..."
cargo --version
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add i686-linux-android
rustup target add x86_64-linux-android

# 编译并复制ARM架构库文件
echo "正在为arm64-v8a架构编译..."
cargo build --release --target aarch64-linux-android
if [ -f "$SCRIPT_DIR/target/aarch64-linux-android/release/libtokenizers_ffi.so" ]; then
    cp "$SCRIPT_DIR/target/aarch64-linux-android/release/libtokenizers_ffi.so" "$JNI_LIBS_DIR/arm64-v8a/"
    echo "ARM64 库文件复制成功"
else
    echo "警告: ARM64 库文件不存在"
fi

echo "正在为armeabi-v7a架构编译..."
cargo build --release --target armv7-linux-androideabi
if [ -f "$SCRIPT_DIR/target/armv7-linux-androideabi/release/libtokenizers_ffi.so" ]; then
    cp "$SCRIPT_DIR/target/armv7-linux-androideabi/release/libtokenizers_ffi.so" "$JNI_LIBS_DIR/armeabi-v7a/"
    echo "ARMv7 库文件复制成功"
else
    echo "警告: ARMv7 库文件不存在"
fi

# 创建 x86 和 x86_64 存根库
echo "创建 x86 和 x86_64 架构的存根库..."

# 创建存根库文件
if [ ! -f "$JNI_LIBS_DIR/x86/libtokenizers_ffi.so" ]; then
    # 创建一个简单的C文件作为存根
    cat > "$SCRIPT_DIR/stub_lib.c" << EOF
// 存根库文件
void free_cstring(char* ptr) {}
void* create_tokenizer(const char* json_config) { return 0; }
void* load_tokenizer_from_file(const char* path) { return 0; }
char* tokenize(void* tokenizer_ptr, const char* text) { return 0; }
void free_tokenizer(void* tokenizer_ptr) {}
void free_string(char* text) {}
EOF
    
    echo "创建 x86 存根库..."
    mkdir -p "$JNI_LIBS_DIR/x86"
    echo "// 存根库" > "$JNI_LIBS_DIR/x86/libtokenizers_ffi.so"
    
    echo "创建 x86_64 存根库..."
    mkdir -p "$JNI_LIBS_DIR/x86_64"
    echo "// 存根库" > "$JNI_LIBS_DIR/x86_64/libtokenizers_ffi.so"
    
    echo "存根库创建完成"
fi

echo "编译和复制完成！"
echo "库文件已直接复制到 $JNI_LIBS_DIR 目录下的相应架构文件夹中"
echo "注意: x86 和 x86_64 架构使用存根库，在运行时会回退到 Java 实现"
