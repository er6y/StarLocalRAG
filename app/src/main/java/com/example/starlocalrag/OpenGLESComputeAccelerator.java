package com.example.starlocalrag;

import android.opengl.GLES20;
import android.opengl.GLES31;
import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;

/**
 * OpenGL ES计算加速器
 * 利用OpenGL ES 3.2计算着色器进行并行计算加速
 */
public class OpenGLESComputeAccelerator {
    private static final String TAG = "OpenGLESComputeAccelerator";
    
    // 基础计算着色器源码
    private static final String COMPUTE_SHADER_SOURCE = 
        "#version 320 es\n" +
        "layout(local_size_x = 16, local_size_y = 16) in;\n" +
        "layout(binding = 0, rgba32f) uniform image2D inputImage;\n" +
        "layout(binding = 1, rgba32f) uniform image2D outputImage;\n" +
        "void main() {\n" +
        "    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);\n" +
        "    vec4 inputValue = imageLoad(inputImage, coord);\n" +
        "    // 执行基础计算（示例：简单的激活函数）\n" +
        "    vec4 result = max(inputValue, vec4(0.0)); // ReLU激活\n" +
        "    imageStore(outputImage, coord, result);\n" +
        "}";
    
    // 矩阵乘法计算着色器
    private static final String MATRIX_MULTIPLY_SHADER = 
        "#version 320 es\n" +
        "layout(local_size_x = 16, local_size_y = 16) in;\n" +
        "layout(binding = 0, r32f) uniform image2D matrixA;\n" +
        "layout(binding = 1, r32f) uniform image2D matrixB;\n" +
        "layout(binding = 2, r32f) uniform image2D matrixC;\n" +
        "uniform int matrixSize;\n" +
        "void main() {\n" +
        "    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);\n" +
        "    if (coord.x >= matrixSize || coord.y >= matrixSize) return;\n" +
        "    float sum = 0.0;\n" +
        "    for (int k = 0; k < matrixSize; k++) {\n" +
        "        float a = imageLoad(matrixA, ivec2(k, coord.y)).r;\n" +
        "        float b = imageLoad(matrixB, ivec2(coord.x, k)).r;\n" +
        "        sum += a * b;\n" +
        "    }\n" +
        "    imageStore(matrixC, coord, vec4(sum));\n" +
        "}";
    
    private EGLDisplay eglDisplay;
    private EGLContext eglContext;
    private EGLSurface eglSurface;
    
    /**
     * 检查设备是否支持OpenGL ES计算着色器
     */
    public boolean isSupported() {
        try {
            // 检查OpenGL ES 3.2支持
            if (android.os.Build.VERSION.SDK_INT < 24) {
                return false; // OpenGL ES 3.2需要API 24+
            }
            
            // 尝试初始化EGL上下文来检查计算着色器支持
            return checkComputeShaderSupport();
            
        } catch (Exception e) {
            LogManager.logW(TAG, "OpenGL ES支持检查失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 检查计算着色器支持
     */
    private boolean checkComputeShaderSupport() {
        try {
            EGL10 egl = (EGL10) EGLContext.getEGL();
            EGLDisplay display = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);
            
            if (display != EGL10.EGL_NO_DISPLAY) {
                int[] version = new int[2];
                if (egl.eglInitialize(display, version)) {
                    EGLConfig[] configs = new EGLConfig[1];
                    int[] numConfigs = new int[1];
                    int[] attribs = {
                        EGL10.EGL_RENDERABLE_TYPE, 64, // EGL_OPENGL_ES3_BIT
                        EGL10.EGL_NONE
                    };
                    
                    if (egl.eglChooseConfig(display, attribs, configs, 1, numConfigs) && numConfigs[0] > 0) {
                        int[] contextAttribs = {0x3098, 3, 0x30FB, 2, EGL10.EGL_NONE}; // ES 3.2
                        EGLContext context = egl.eglCreateContext(display, configs[0], EGL10.EGL_NO_CONTEXT, contextAttribs);
                        
                        if (context != EGL10.EGL_NO_CONTEXT) {
                            egl.eglDestroyContext(display, context);
                            egl.eglTerminate(display);
                            return true;
                        }
                    }
                    egl.eglTerminate(display);
                }
            }
            return false;
        } catch (Exception e) {
            return false;
        }
    }
    
    private static int computeProgram = -1;
    private static int matrixMultiplyProgram = -1;
    private static boolean initialized = false;
    
    /**
     * 初始化OpenGL ES计算着色器
     */
    public static boolean initializeComputeShader() {
        if (initialized) {
            return true;
        }
        
        try {
            // 检查OpenGL ES版本支持
            if (!checkOpenGLESSupport()) {
                LogManager.logE(TAG, "设备不支持OpenGL ES 3.2计算着色器");
                return false;
            }
            
            // 创建基础计算着色器程序
            computeProgram = createComputeShaderProgram(COMPUTE_SHADER_SOURCE);
            if (computeProgram == -1) {
                LogManager.logE(TAG, "基础计算着色器创建失败");
                return false;
            }
            
            // 创建矩阵乘法计算着色器程序
            matrixMultiplyProgram = createComputeShaderProgram(MATRIX_MULTIPLY_SHADER);
            if (matrixMultiplyProgram == -1) {
                LogManager.logE(TAG, "矩阵乘法计算着色器创建失败");
                return false;
            }
            
            initialized = true;
            LogManager.logI(TAG, "OpenGL ES计算着色器初始化成功");
            return true;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "计算着色器初始化异常: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 检查OpenGL ES支持
     */
    private static boolean checkOpenGLESSupport() {
        try {
            // 检查OpenGL ES版本
            String version = GLES20.glGetString(GLES20.GL_VERSION);
            LogManager.logI(TAG, "OpenGL ES版本: " + version);
            
            // 检查扩展支持
            String extensions = GLES20.glGetString(GLES20.GL_EXTENSIONS);
            
            // 检查关键扩展
            boolean hasComputeShader = extensions.contains("GL_ARB_compute_shader") || 
                                     extensions.contains("GL_ES_VERSION_3_1");
            boolean hasShaderStorage = extensions.contains("GL_ARB_shader_storage_buffer_object");
            boolean hasImageLoad = extensions.contains("GL_ARB_shader_image_load_store");
            
            LogManager.logI(TAG, "OpenGL ES扩展支持:");
            LogManager.logI(TAG, "  计算着色器: " + hasComputeShader);
            LogManager.logI(TAG, "  着色器存储缓冲: " + hasShaderStorage);
            LogManager.logI(TAG, "  图像加载存储: " + hasImageLoad);
            
            return hasComputeShader;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "OpenGL ES支持检查失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 创建计算着色器程序
     */
    private static int createComputeShaderProgram(String shaderSource) {
        try {
            // 创建计算着色器
            int computeShader = GLES31.glCreateShader(GLES31.GL_COMPUTE_SHADER);
            if (computeShader == 0) {
                LogManager.logE(TAG, "创建计算着色器失败");
                return -1;
            }
            
            // 编译着色器
            GLES31.glShaderSource(computeShader, shaderSource);
            GLES31.glCompileShader(computeShader);
            
            // 检查编译状态
            int[] compiled = new int[1];
            GLES31.glGetShaderiv(computeShader, GLES31.GL_COMPILE_STATUS, compiled, 0);
            
            if (compiled[0] == 0) {
                String error = GLES31.glGetShaderInfoLog(computeShader);
                LogManager.logE(TAG, "计算着色器编译失败: " + error);
                GLES31.glDeleteShader(computeShader);
                return -1;
            }
            
            // 创建程序
            int program = GLES31.glCreateProgram();
            if (program == 0) {
                LogManager.logE(TAG, "创建着色器程序失败");
                GLES31.glDeleteShader(computeShader);
                return -1;
            }
            
            // 链接程序
            GLES31.glAttachShader(program, computeShader);
            GLES31.glLinkProgram(program);
            
            // 检查链接状态
            int[] linked = new int[1];
            GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linked, 0);
            
            if (linked[0] == 0) {
                String error = GLES31.glGetProgramInfoLog(program);
                LogManager.logE(TAG, "着色器程序链接失败: " + error);
                GLES31.glDeleteProgram(program);
                GLES31.glDeleteShader(computeShader);
                return -1;
            }
            
            // 清理着色器对象
            GLES31.glDeleteShader(computeShader);
            
            LogManager.logI(TAG, "计算着色器程序创建成功: " + program);
            return program;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "创建计算着色器程序异常: " + e.getMessage());
            return -1;
        }
    }
    
    /**
     * 执行矩阵乘法计算
     */
    public static boolean performMatrixMultiplication(float[] matrixA, float[] matrixB, 
                                                     float[] result, int size) {
        if (!initialized || matrixMultiplyProgram == -1) {
            LogManager.logE(TAG, "计算着色器未初始化");
            return false;
        }
        
        try {
            // 使用矩阵乘法程序
            GLES31.glUseProgram(matrixMultiplyProgram);
            
            // 创建纹理并上传数据
            int[] textures = new int[3];
            GLES31.glGenTextures(3, textures, 0);
            
            // 上传矩阵A
            uploadMatrixToTexture(textures[0], matrixA, size);
            GLES31.glBindImageTexture(0, textures[0], 0, false, 0, 
                                    GLES31.GL_READ_ONLY, GLES31.GL_R32F);
            
            // 上传矩阵B
            uploadMatrixToTexture(textures[1], matrixB, size);
            GLES31.glBindImageTexture(1, textures[1], 0, false, 0, 
                                    GLES31.GL_READ_ONLY, GLES31.GL_R32F);
            
            // 创建结果纹理
            createResultTexture(textures[2], size);
            GLES31.glBindImageTexture(2, textures[2], 0, false, 0, 
                                    GLES31.GL_WRITE_ONLY, GLES31.GL_R32F);
            
            // 设置uniform变量
            int matrixSizeLocation = GLES31.glGetUniformLocation(matrixMultiplyProgram, "matrixSize");
            GLES31.glUniform1i(matrixSizeLocation, size);
            
            // 执行计算
            int workGroupsX = (size + 15) / 16;
            int workGroupsY = (size + 15) / 16;
            GLES31.glDispatchCompute(workGroupsX, workGroupsY, 1);
            
            // 等待计算完成
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            
            // 读取结果
            boolean success = readResultFromTexture(textures[2], result, size);
            
            // 清理资源
            GLES31.glDeleteTextures(3, textures, 0);
            
            LogManager.logI(TAG, "矩阵乘法计算完成");
            return success;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "矩阵乘法计算异常: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 上传矩阵数据到纹理
     */
    private static void uploadMatrixToTexture(int texture, float[] data, int size) {
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, texture);
        GLES31.glTexImage2D(GLES31.GL_TEXTURE_2D, 0, GLES31.GL_R32F, 
                          size, size, 0, GLES31.GL_RED, GLES31.GL_FLOAT, 
                          java.nio.FloatBuffer.wrap(data));
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_NEAREST);
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_NEAREST);
    }
    
    /**
     * 创建结果纹理
     */
    private static void createResultTexture(int texture, int size) {
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, texture);
        GLES31.glTexImage2D(GLES31.GL_TEXTURE_2D, 0, GLES31.GL_R32F, 
                          size, size, 0, GLES31.GL_RED, GLES31.GL_FLOAT, null);
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_NEAREST);
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_NEAREST);
    }
    
    /**
     * 从纹理读取结果
     */
    private static boolean readResultFromTexture(int texture, float[] result, int size) {
        try {
            // 创建帧缓冲
            int[] framebuffer = new int[1];
            GLES31.glGenFramebuffers(1, framebuffer, 0);
            GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, framebuffer[0]);
            
            // 附加纹理到帧缓冲
            GLES31.glFramebufferTexture2D(GLES31.GL_FRAMEBUFFER, GLES31.GL_COLOR_ATTACHMENT0, 
                                        GLES31.GL_TEXTURE_2D, texture, 0);
            
            // 检查帧缓冲状态
            int status = GLES31.glCheckFramebufferStatus(GLES31.GL_FRAMEBUFFER);
            if (status != GLES31.GL_FRAMEBUFFER_COMPLETE) {
                LogManager.logE(TAG, "帧缓冲不完整: " + status);
                return false;
            }
            
            // 读取像素数据
            java.nio.FloatBuffer buffer = java.nio.FloatBuffer.allocate(size * size);
            GLES31.glReadPixels(0, 0, size, size, GLES31.GL_RED, GLES31.GL_FLOAT, buffer);
            
            // 复制到结果数组
            buffer.rewind();
            buffer.get(result);
            
            // 清理帧缓冲
            GLES31.glDeleteFramebuffers(1, framebuffer, 0);
            
            return true;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "读取纹理结果失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 清理资源
     */
    public static void cleanup() {
        try {
            if (computeProgram != -1) {
                GLES31.glDeleteProgram(computeProgram);
                computeProgram = -1;
            }
            
            if (matrixMultiplyProgram != -1) {
                GLES31.glDeleteProgram(matrixMultiplyProgram);
                matrixMultiplyProgram = -1;
            }
            
            initialized = false;
            LogManager.logI(TAG, "OpenGL ES计算加速器资源已清理");
            
        } catch (Exception e) {
            LogManager.logE(TAG, "清理资源异常: " + e.getMessage());
        }
    }
    
    /**
     * 获取OpenGL ES计算能力信息
     */
    public static String getComputeCapabilityInfo() {
        try {
            StringBuilder info = new StringBuilder();
            
            // 获取工作组大小限制
            int[] maxWorkGroupSize = new int[3];
            GLES31.glGetIntegeri_v(GLES31.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, maxWorkGroupSize, 0);
            GLES31.glGetIntegeri_v(GLES31.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, maxWorkGroupSize, 1);
            GLES31.glGetIntegeri_v(GLES31.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, maxWorkGroupSize, 2);
            
            info.append("最大工作组大小: ").append(maxWorkGroupSize[0])
                .append("x").append(maxWorkGroupSize[1])
                .append("x").append(maxWorkGroupSize[2]).append("\n");
            
            // 获取工作组数量限制
            int[] maxWorkGroupCount = new int[3];
            GLES31.glGetIntegeri_v(GLES31.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, maxWorkGroupCount, 0);
            GLES31.glGetIntegeri_v(GLES31.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, maxWorkGroupCount, 1);
            GLES31.glGetIntegeri_v(GLES31.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, maxWorkGroupCount, 2);
            
            info.append("最大工作组数量: ").append(maxWorkGroupCount[0])
                .append("x").append(maxWorkGroupCount[1])
                .append("x").append(maxWorkGroupCount[2]).append("\n");
            
            // 获取共享内存大小
            int[] sharedMemorySize = new int[1];
            GLES31.glGetIntegerv(GLES31.GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, sharedMemorySize, 0);
            info.append("最大共享内存: ").append(sharedMemorySize[0]).append(" bytes\n");
            
            return info.toString();
            
        } catch (Exception e) {
            LogManager.logE(TAG, "获取计算能力信息失败: " + e.getMessage());
            return "获取失败: " + e.getMessage();
        }
    }
}