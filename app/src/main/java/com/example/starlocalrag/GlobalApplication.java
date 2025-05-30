package com.example.starlocalrag;

import android.app.Application;
import android.content.Context;
import android.app.ActivityManager;
import android.util.Log;

/**
 * 全局应用类，用于提供应用上下文
 */
public class GlobalApplication extends Application {
    private static final String TAG = "GlobalApplication";
    private static Context appContext;
    
    @Override
    public void onCreate() {
        super.onCreate();
        appContext = getApplicationContext();
        
        // 初始化内存监控
        initMemoryMonitoring();
    }
    
    /**
     * 初始化内存监控
     */
    private void initMemoryMonitoring() {
        try {
            // 获取内存信息
            ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
            ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
            activityManager.getMemoryInfo(memoryInfo);
            
            // 获取应用的内存类别
            int memoryClass = activityManager.getMemoryClass();
            int largeMemoryClass = activityManager.getLargeMemoryClass();
            
            // 获取JVM内存信息
            Runtime runtime = Runtime.getRuntime();
            long maxMemory = runtime.maxMemory();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            
            Log.i(TAG, "=== 内存配置信息 ===");
            Log.i(TAG, "系统可用内存: " + (memoryInfo.availMem / (1024 * 1024)) + " MB");
            Log.i(TAG, "系统总内存: " + (memoryInfo.totalMem / (1024 * 1024)) + " MB");
            Log.i(TAG, "应用标准内存类别: " + memoryClass + " MB");
            Log.i(TAG, "应用大内存类别: " + largeMemoryClass + " MB");
            Log.i(TAG, "JVM最大内存: " + (maxMemory / (1024 * 1024)) + " MB");
            Log.i(TAG, "JVM当前分配内存: " + (totalMemory / (1024 * 1024)) + " MB");
            Log.i(TAG, "JVM空闲内存: " + (freeMemory / (1024 * 1024)) + " MB");
            Log.i(TAG, "JVM可用内存: " + ((maxMemory - totalMemory + freeMemory) / (1024 * 1024)) + " MB");
            
            // 检查是否启用了largeHeap
            if (largeMemoryClass > memoryClass) {
                Log.i(TAG, "✓ largeHeap已启用，可用内存增加到: " + largeMemoryClass + " MB");
            } else {
                Log.w(TAG, "⚠ largeHeap未生效，当前内存限制: " + memoryClass + " MB");
            }
            
        } catch (Exception e) {
            Log.e(TAG, "内存监控初始化失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 获取应用上下文
     * @return 应用上下文
     */
    public static Context getAppContext() {
        return appContext;
    }
    
    /**
     * 获取当前可用内存信息
     * @return 可用内存（MB）
     */
    public static long getAvailableMemoryMB() {
        if (appContext == null) return 0;
        
        try {
            ActivityManager activityManager = (ActivityManager) appContext.getSystemService(Context.ACTIVITY_SERVICE);
            ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
            activityManager.getMemoryInfo(memoryInfo);
            return memoryInfo.availMem / (1024 * 1024);
        } catch (Exception e) {
            Log.e(TAG, "获取可用内存失败: " + e.getMessage());
            return 0;
        }
    }
    
    /**
     * 获取JVM最大可用内存
     * @return JVM最大可用内存（MB）
     */
    public static long getJVMMaxMemoryMB() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.maxMemory() / (1024 * 1024);
    }
}
