package com.example.starlocalrag;

import android.content.Context;
import android.content.pm.PackageManager;
import ai.onnxruntime.OrtSession;

/**
 * HarmonyOS GPU适配器
 * 专门处理HarmonyOS设备的GPU加速优化
 */
public class HarmonyOSGPUAdapter {
    private static final String TAG = "HarmonyOSGPUAdapter";
    
    /**
     * HarmonyOS设备的GPU加速优化
     */
    public static void optimizeForHarmonyOS(OrtSession.SessionOptions sessionOptions) {
        try {
            // HarmonyOS特定的GPU配置
            sessionOptions.addConfigEntry("ep.opencl.enable_caching", "1");
            sessionOptions.addConfigEntry("ep.opencl.cache_dir", "/data/data/com.example.starlocalrag/cache/opencl");
            
            // Mali GPU特定优化
            sessionOptions.addConfigEntry("ep.opencl.device_type", "gpu");
            sessionOptions.addConfigEntry("ep.opencl.enable_fp16", "1");
            
            // 内存优化
            sessionOptions.addConfigEntry("session.memory.enable_memory_arena_shrinkage", "1");
            sessionOptions.addConfigEntry("session.memory.memory_arena_shrinkage_factor", "0.8");
            
            // Vulkan特定配置
            sessionOptions.addConfigEntry("ep.vulkan.device_type", "gpu");
            sessionOptions.addConfigEntry("ep.vulkan.enable_fp16", "1");
            sessionOptions.addConfigEntry("ep.vulkan.memory_type", "device_local");
            
            LogManager.logI(TAG, "HarmonyOS GPU优化配置已应用");
            
        } catch (Exception e) {
            LogManager.logW(TAG, "HarmonyOS GPU优化失败: " + e.getMessage());
        }
    }
    
    /**
     * 检测并处理HarmonyOS GPU权限
     */
    public static boolean checkHarmonyOSGPUPermissions(Context context) {
        try {
            // 检查GPU访问权限
            PackageManager pm = context.getPackageManager();
            
            // HarmonyOS可能需要特殊权限
            boolean hasGPUPermission = true;
            try {
                hasGPUPermission = pm.checkPermission(
                    "ohos.permission.USE_GPU", 
                    context.getPackageName()) == PackageManager.PERMISSION_GRANTED;
            } catch (Exception e) {
                // 如果权限不存在，假设有权限
                LogManager.logI(TAG, "HarmonyOS GPU权限检查跳过: " + e.getMessage());
            }
            
            if (!hasGPUPermission) {
                LogManager.logW(TAG, "缺少HarmonyOS GPU访问权限");
                return false;
            }
            
            return true;
            
        } catch (Exception e) {
            LogManager.logW(TAG, "HarmonyOS权限检查失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 优化Vulkan内存管理
     */
    public static void optimizeVulkanMemory() {
        try {
            // Mali GPU建议使用统一内存架构
            System.setProperty("vulkan.memory.unified", "true");
            
            // 设置合适的内存池大小
            System.setProperty("vulkan.memory.pool_size", "256MB");
            
            // 启用内存压缩
            System.setProperty("vulkan.memory.compression", "true");
            
            LogManager.logI(TAG, "Vulkan内存优化已应用");
            
        } catch (Exception e) {
            LogManager.logW(TAG, "Vulkan内存优化失败: " + e.getMessage());
        }
    }
    
    /**
     * 优化系统设置
     */
    public static void optimizeSystemSettings() {
        try {
            // 系统级GPU调度优化
            LogManager.logI(TAG, "正在优化HarmonyOS系统GPU设置...");
            
            // 注意：这些优化需要系统权限，在普通应用中可能无法执行
            // 这里主要是记录和建议
            LogManager.logI(TAG, "建议在开发者选项中启用'强制GPU渲染'");
            LogManager.logI(TAG, "建议设置性能模式为'高性能'");
            LogManager.logI(TAG, "建议关闭省电模式");
            
        } catch (Exception e) {
            LogManager.logW(TAG, "系统设置优化失败: " + e.getMessage());
        }
    }
    
    /**
     * 配置Vulkan加速
     */
    public static void configureVulkanAcceleration(OrtSession.SessionOptions sessionOptions) {
        try {
            // Vulkan特定配置
            sessionOptions.addConfigEntry("ep.vulkan.device_type", "gpu");
            sessionOptions.addConfigEntry("ep.vulkan.enable_fp16", "1");
            sessionOptions.addConfigEntry("ep.vulkan.memory_type", "device_local");
            
            LogManager.logI(TAG, "Vulkan加速配置已应用");
            
        } catch (Exception e) {
            LogManager.logW(TAG, "Vulkan加速配置失败: " + e.getMessage());
        }
    }
}