@echo off
REM Android Rust library compilation script
REM Supports arm64-v8a, armeabi-v7a, x86, x86_64 architectures

REM Set path variables
SET SCRIPT_DIR=%~dp0
SET JNI_LIBS_DIR=%SCRIPT_DIR%..\..\app\src\main\jniLibs

REM Ensure directories exist
IF NOT EXIST "%JNI_LIBS_DIR%\arm64-v8a" mkdir "%JNI_LIBS_DIR%\arm64-v8a"
IF NOT EXIST "%JNI_LIBS_DIR%\armeabi-v7a" mkdir "%JNI_LIBS_DIR%\armeabi-v7a"
IF NOT EXIST "%JNI_LIBS_DIR%\x86" mkdir "%JNI_LIBS_DIR%\x86"
IF NOT EXIST "%JNI_LIBS_DIR%\x86_64" mkdir "%JNI_LIBS_DIR%\x86_64"

echo Ensuring required Rust targets are installed...
cargo --version
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add i686-linux-android
rustup target add x86_64-linux-android

REM Compile and copy library files
echo Compiling for arm64-v8a architecture...
cd %SCRIPT_DIR%
cargo build --release --target aarch64-linux-android
IF EXIST "%SCRIPT_DIR%target\aarch64-linux-android\release\libtokenizers_ffi.so" (
    copy "%SCRIPT_DIR%target\aarch64-linux-android\release\libtokenizers_ffi.so" "%JNI_LIBS_DIR%\arm64-v8a\" /Y
    echo ARM64 library file copied successfully
) ELSE (
    echo WARNING: ARM64 library file does not exist
)

echo Compiling for armeabi-v7a architecture...
cd %SCRIPT_DIR%
cargo build --release --target armv7-linux-androideabi
IF EXIST "%SCRIPT_DIR%target\armv7-linux-androideabi\release\libtokenizers_ffi.so" (
    copy "%SCRIPT_DIR%target\armv7-linux-androideabi\release\libtokenizers_ffi.so" "%JNI_LIBS_DIR%\armeabi-v7a\" /Y
    echo ARMv7 library file copied successfully
) ELSE (
    echo WARNING: ARMv7 library file does not exist
)

echo Compiling for x86 architecture...
cd %SCRIPT_DIR%
cargo build --release --target i686-linux-android
IF EXIST "%SCRIPT_DIR%target\i686-linux-android\release\libtokenizers_ffi.so" (
    copy "%SCRIPT_DIR%target\i686-linux-android\release\libtokenizers_ffi.so" "%JNI_LIBS_DIR%\x86\" /Y
    echo x86 library file copied successfully
) ELSE (
    echo WARNING: x86 library file does not exist
)

echo Compiling for x86_64 architecture...
cd %SCRIPT_DIR%
cargo build --release --target x86_64-linux-android
IF EXIST "%SCRIPT_DIR%target\x86_64-linux-android\release\libtokenizers_ffi.so" (
    copy "%SCRIPT_DIR%target\x86_64-linux-android\release\libtokenizers_ffi.so" "%JNI_LIBS_DIR%\x86_64\" /Y
    echo x86_64 library file copied successfully
) ELSE (
    echo WARNING: x86_64 library file does not exist
)

echo Compilation and copying completed!
echo Library files have been copied directly to the corresponding architecture folders in %JNI_LIBS_DIR%
