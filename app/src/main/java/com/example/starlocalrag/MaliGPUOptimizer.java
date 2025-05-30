package com.example.starlocalrag;

import ai.onnxruntime.OrtSession;
import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * Mali GPU优化器
 * 专门针对Mali-G610 GPU的性能优化
 */
public class MaliGPUOptimizer {
    private static final String TAG = "MaliGPUOptimizer";
    
    /**
     * Mali-G610特定的性能优化
     */
    public static void optimizeForMaliG610(OrtSession.SessionOptions sessionOptions) {
        try {
            // Mali GPU架构特定配置
            sessionOptions.addConfigEntry("ep.opencl.enable_winograd_convolution", "1");
            sessionOptions.addConfigEntry("ep.opencl.enable_batched_gemm", "1");
            
            // Mali GPU优化 - 不覆盖已有的线程配置，避免与CPU亲和性配置冲突
            // 注意：线程配置已在LocalLlmHandler中设置，此处不再重复设置以避免冲突
            LogManager.logI(TAG, "Mali GPU优化已启用，使用LocalLlmHandler中的线程配置");
            
            // 内存带宽优化
            sessionOptions.addConfigEntry("ep.opencl.enable_memory_reuse", "1");
            sessionOptions.addConfigEntry("ep.opencl.memory_pool_size", "256MB");
            
            // Mali特定的OpenCL优化
            sessionOptions.addConfigEntry("ep.opencl.enable_fast_math", "1");
            sessionOptions.addConfigEntry("ep.opencl.enable_half_precision", "1");
            
            // 卷积优化
            sessionOptions.addConfigEntry("ep.opencl.conv_algorithm", "WINOGRAD");
            sessionOptions.addConfigEntry("ep.opencl.enable_conv_fusion", "1");
            
            LogManager.logI(TAG, "Mali-G610优化配置已应用");
            
        } catch (Exception e) {
            LogManager.logW(TAG, "Mali GPU优化失败: " + e.getMessage());
        }
    }
    
    /**
     * 检测Mali GPU性能状态
     */
    public static void monitorMaliPerformance() {
        try {
            // 读取Mali GPU使用率
            String gpuUsage = readSysFile("/sys/class/devfreq/gpufreq/load");
            String gpuFreq = readSysFile("/sys/class/devfreq/gpufreq/cur_freq");
            String maxFreq = readSysFile("/sys/class/devfreq/gpufreq/max_freq");
            String minFreq = readSysFile("/sys/class/devfreq/gpufreq/min_freq");
            
            LogManager.logI(TAG, "=== Mali GPU性能监控 ===");
            LogManager.logI(TAG, "GPU使用率: " + gpuUsage + "%");
            LogManager.logI(TAG, "当前频率: " + gpuFreq + "Hz");
            LogManager.logI(TAG, "最大频率: " + maxFreq + "Hz");
            LogManager.logI(TAG, "最小频率: " + minFreq + "Hz");
            
            // 尝试读取GPU温度
            String gpuTemp = readSysFile("/sys/class/thermal/thermal_zone0/temp");
            if (!gpuTemp.equals("未知")) {
                try {
                    int temp = Integer.parseInt(gpuTemp) / 1000; // 转换为摄氏度
                    LogManager.logI(TAG, "GPU温度: " + temp + "°C");
                } catch (NumberFormatException e) {
                    LogManager.logI(TAG, "GPU温度: " + gpuTemp);
                }
            }
            
        } catch (Exception e) {
            LogManager.logW(TAG, "Mali性能监控失败: " + e.getMessage());
        }
    }
    
    /**
     * 优化Mali GPU频率策略
     */
    public static void optimizeFrequencyGovernor() {
        try {
            // 设置GPU频率调节器为性能模式
            Runtime.getRuntime().exec("echo performance > /sys/class/devfreq/gpufreq/governor");
            
            // 设置最小频率为较高值以提升性能
            String maxFreq = readSysFile("/sys/class/devfreq/gpufreq/max_freq");
            if (!maxFreq.equals("未知")) {
                try {
                    long maxFreqValue = Long.parseLong(maxFreq);
                    long targetMinFreq = maxFreqValue * 3 / 4; // 设置为最大频率的75%
                    Runtime.getRuntime().exec("echo " + targetMinFreq + " > /sys/class/devfreq/gpufreq/min_freq");
                    LogManager.logI(TAG, "GPU最小频率已设置为: " + targetMinFreq + "Hz");
                } catch (NumberFormatException e) {
                    LogManager.logW(TAG, "无法解析GPU频率值: " + maxFreq);
                }
            }
            
            LogManager.logI(TAG, "Mali GPU频率优化已应用");
            
        } catch (Exception e) {
            LogManager.logW(TAG, "Mali GPU频率优化失败: " + e.getMessage());
        }
    }
    
    /**
     * 检测Mali GPU架构信息
     */
    public static String detectMaliArchitecture() {
        try {
            // 尝试从多个位置读取GPU信息
            String[] gpuInfoPaths = {
                "/sys/class/misc/mali0/device/gpuinfo",
                "/proc/mali/version",
                "/sys/module/mali/version",
                "/sys/kernel/debug/mali/version"
            };
            
            for (String path : gpuInfoPaths) {
                String info = readSysFile(path);
                if (!info.equals("未知") && !info.isEmpty()) {
                    LogManager.logI(TAG, "Mali架构信息: " + info);
                    return info;
                }
            }
            
            // 如果无法直接读取，通过OpenGL渲染器信息推断
            String renderer = getGPURendererInfo();
            if (renderer != null && renderer.contains("Mali")) {
                LogManager.logI(TAG, "通过OpenGL检测到Mali GPU: " + renderer);
                return renderer;
            }
            
            return "未知Mali架构";
            
        } catch (Exception e) {
            LogManager.logW(TAG, "Mali架构检测失败: " + e.getMessage());
            return "检测失败";
        }
    }
    
    /**
     * 应用Mali GPU内存优化
     */
    public static void optimizeMaliMemory(ai.onnxruntime.OrtSession.SessionOptions sessionOptions) {
        try {
            // Mali GPU特定的内存管理优化
            sessionOptions.addConfigEntry("ep.opencl.memory_type", "device_local");
            sessionOptions.addConfigEntry("ep.opencl.buffer_pool_size", "256");
            sessionOptions.addConfigEntry("ep.opencl.memory_fragment_threshold", "64");
            
            // 设置Mali GPU内存分配策略
            System.setProperty("mali.memory.strategy", "unified");
            
            // 优化内存池大小
            System.setProperty("mali.memory.pool_size", "256MB");
            
            // 启用内存压缩
            System.setProperty("mali.memory.compression", "true");
            
            // 设置内存回收策略
            System.setProperty("mali.memory.gc_threshold", "0.8");
            
            LogManager.logI(TAG, "Mali GPU内存优化已应用");
            
        } catch (Exception e) {
            LogManager.logW(TAG, "Mali GPU内存优化失败: " + e.getMessage());
        }
    }
    
    /**
     * 读取系统文件内容
     */
    private static String readSysFile(String path) {
        try {
            Process process = Runtime.getRuntime().exec("cat " + path);
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line = reader.readLine();
            reader.close();
            return line != null ? line.trim() : "未知";
        } catch (Exception e) {
            return "未知";
        }
    }
    
    /**
     * 检查Mali GPU特性支持
     */
    public static java.util.Map<String, Boolean> checkMaliFeatureSupport(android.content.Context context) {
        java.util.Map<String, Boolean> features = new java.util.HashMap<>();
        
        try {
            // 获取GPU信息
            String gpuInfo = getGPURendererInfo();
            if (gpuInfo == null) {
                gpuInfo = "unknown";
            }
            String lowerGpuInfo = gpuInfo.toLowerCase();
            
            // 基于GPU信息判断特性支持
            features.put("opencl", lowerGpuInfo.contains("mali"));
            features.put("vulkan", lowerGpuInfo.contains("mali-g") && !lowerGpuInfo.contains("mali-g52")); // G52不支持Vulkan
            features.put("compute_shader", lowerGpuInfo.contains("mali"));
            features.put("fp16", lowerGpuInfo.contains("mali-g"));
            
            LogManager.logI(TAG, "Mali GPU特性检查完成: " + features.toString());
            
        } catch (Exception e) {
            LogManager.logW(TAG, "Mali特性检查失败: " + e.getMessage());
        }
        
        return features;
    }
    
    /**
     * 获取GPU渲染器信息（内部方法）
     */
    private static String getGPURendererInfo() {
        try {
            javax.microedition.khronos.egl.EGL10 egl = (javax.microedition.khronos.egl.EGL10) javax.microedition.khronos.egl.EGLContext.getEGL();
            javax.microedition.khronos.egl.EGLDisplay display = egl.eglGetDisplay(javax.microedition.khronos.egl.EGL10.EGL_DEFAULT_DISPLAY);
            
            if (display != javax.microedition.khronos.egl.EGL10.EGL_NO_DISPLAY) {
                int[] version = new int[2];
                if (egl.eglInitialize(display, version)) {
                    javax.microedition.khronos.egl.EGLConfig[] configs = new javax.microedition.khronos.egl.EGLConfig[1];
                    int[] numConfigs = new int[1];
                    int[] attribs = {
                        javax.microedition.khronos.egl.EGL10.EGL_RENDERABLE_TYPE, 4, // EGL_OPENGL_ES2_BIT
                        javax.microedition.khronos.egl.EGL10.EGL_NONE
                    };
                    
                    if (egl.eglChooseConfig(display, attribs, configs, 1, numConfigs) && numConfigs[0] > 0) {
                        javax.microedition.khronos.egl.EGLContext context = egl.eglCreateContext(display, configs[0], javax.microedition.khronos.egl.EGL10.EGL_NO_CONTEXT, new int[]{0x3098, 2, javax.microedition.khronos.egl.EGL10.EGL_NONE});
                        
                        if (context != javax.microedition.khronos.egl.EGL10.EGL_NO_CONTEXT) {
                            javax.microedition.khronos.egl.EGLSurface surface = egl.eglCreatePbufferSurface(display, configs[0], new int[]{javax.microedition.khronos.egl.EGL10.EGL_WIDTH, 1, javax.microedition.khronos.egl.EGL10.EGL_HEIGHT, 1, javax.microedition.khronos.egl.EGL10.EGL_NONE});
                            
                            if (surface != javax.microedition.khronos.egl.EGL10.EGL_NO_SURFACE) {
                                if (egl.eglMakeCurrent(display, surface, surface, context)) {
                                    String renderer = android.opengl.GLES20.glGetString(android.opengl.GLES20.GL_RENDERER);
                                    egl.eglDestroySurface(display, surface);
                                    egl.eglDestroyContext(display, context);
                                    egl.eglTerminate(display);
                                    return renderer;
                                }
                                egl.eglDestroySurface(display, surface);
                            }
                            egl.eglDestroyContext(display, context);
                        }
                    }
                    egl.eglTerminate(display);
                }
            }
        } catch (Exception e) {
            // 忽略异常，返回null
        }
        return null;
    }
}