package com.example.starlocalrag;

import android.content.Context;
import android.os.Build;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;

import java.io.File;
import java.lang.reflect.Method;

/**
 * GPU错误处理工具类
 * 用于处理各种GPU加速相关的错误和警告
 */
public class GPUErrorHandler {
    private static final String TAG = "StarLocalRAG_GPU";
    
    /**
     * 初始化GPU错误处理
     * @param context 应用上下文
     * @param window 窗口对象
     */
    public static void init(Context context, Window window) {
        try {
            // 创建缓存目录
            createCacheDirectories(context);
            
            // 设置窗口格式
            configureWindowFormat(window);
            
            // 处理华为特定问题
            handleHuaweiSpecificIssues();
            
            Log.d(TAG, "GPU错误处理初始化完成");
        } catch (Exception e) {
            Log.e(TAG, "GPU错误处理初始化失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 创建缓存目录
     * @param context 应用上下文
     */
    private static void createCacheDirectories(Context context) {
        try {
            // 创建OpenGL着色器缓存目录
            File openglCacheDir = new File(context.getCacheDir(), "opengl_cache");
            if (!openglCacheDir.exists()) {
                openglCacheDir.mkdirs();
            }
            
            // 创建Skia着色器缓存目录
            File skiaCacheDir = new File(context.getCacheDir(), "skia_cache");
            if (!skiaCacheDir.exists()) {
                skiaCacheDir.mkdirs();
            }
            
            // 设置系统属性（需要root权限，通常无法设置，但尝试一下）
            try {
                System.setProperty("com.android.opengl.shaders_cache", openglCacheDir.getAbsolutePath());
                System.setProperty("com.android.skia.shaders_cache", skiaCacheDir.getAbsolutePath());
            } catch (Exception e) {
                Log.w(TAG, "无法设置着色器缓存系统属性: " + e.getMessage());
            }
            
            Log.d(TAG, "缓存目录创建成功");
        } catch (Exception e) {
            Log.e(TAG, "创建缓存目录失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 配置窗口格式
     * @param window 窗口对象
     */
    private static void configureWindowFormat(Window window) {
        try {
            if (window != null) {
                // 设置窗口格式为RGBA_8888，提高渲染质量
                window.setFormat(android.graphics.PixelFormat.RGBA_8888);
                
                // 根据配置决定是否启用硬件加速
                boolean useGpu = ConfigManager.getBoolean(window.getContext(), ConfigManager.KEY_USE_GPU, false);
                
                if (useGpu) {
                    // 启用硬件加速
                    window.setFlags(
                        WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
                        WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED);
                    Log.d(TAG, "已启用硬件加速");
                } else {
                    // 禁用硬件加速
                    window.clearFlags(WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED);
                    Log.d(TAG, "已禁用硬件加速");
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "配置窗口格式失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 处理华为特定问题
     */
    private static void handleHuaweiSpecificIssues() {
        try {
            // 检测是否是华为设备
            boolean isHuawei = Build.MANUFACTURER.toLowerCase().contains("huawei") || 
                               Build.BRAND.toLowerCase().contains("huawei") ||
                               Build.MODEL.toLowerCase().contains("huawei");
            
            if (isHuawei) {
                Log.d(TAG, "检测到华为设备，应用特定修复");
                
                // 尝试禁用HwEditorHelperImpl
                try {
                    Class<?> hwEditorHelperClass = Class.forName("com.huawei.android.hweditor.HwEditorHelperImpl");
                    Method disableMethod = hwEditorHelperClass.getDeclaredMethod("disable");
                    disableMethod.setAccessible(true);
                    disableMethod.invoke(null);
                    Log.d(TAG, "成功禁用HwEditorHelperImpl");
                } catch (Exception e) {
                    // 大多数情况下会失败，因为这是华为私有API
                    Log.w(TAG, "无法禁用HwEditorHelperImpl: " + e.getMessage());
                }
                
                // 设置系统属性
                try {
                    System.setProperty("hw_editor_disable", "true");
                } catch (Exception e) {
                    Log.w(TAG, "无法设置hw_editor_disable属性: " + e.getMessage());
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "处理华为特定问题失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 检查是否支持GPU加速
     * @param context 应用上下文
     * @return 是否支持GPU加速
     */
    public static boolean isGPUAccelerationSupported(Context context) {
        try {
            // 检查设备是否支持OpenGL ES 2.0或更高版本
            android.app.ActivityManager activityManager = 
                (android.app.ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            
            if (activityManager != null) {
                android.content.pm.ConfigurationInfo configInfo = activityManager.getDeviceConfigurationInfo();
                return configInfo.reqGlEsVersion >= 0x20000;
            }
        } catch (Exception e) {
            Log.e(TAG, "检查GPU加速支持失败: " + e.getMessage(), e);
        }
        
        return false;
    }
}
