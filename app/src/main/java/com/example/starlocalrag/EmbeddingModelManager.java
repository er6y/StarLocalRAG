package com.example.starlocalrag;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import java.io.File;
import java.util.Arrays;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 词嵌入模型管理器单例类
 * 负责模型的加载、缓存和卸载，避免重复加载模型
 * 模型只在超时后卸载，其他情况下保持加载状态
 */
public class EmbeddingModelManager {
    private static final String TAG = "EmbeddingModelManager";
    
    // 单例实例
    private static EmbeddingModelManager instance;
    
    // 当前加载的模型
    private EmbeddingModelHandler currentModel;
    
    // 当前模型路径
    private String currentModelPath;
    
    // 最后访问时间
    private long lastAccessTime;
    
    // 卸载定时器
    private Timer unloadTimer;
    
    // 线程池
    private final ExecutorService executor;
    
    // 主线程Handler
    private final Handler mainHandler;
    
    // 模型是否正在加载
    private final AtomicBoolean isLoading = new AtomicBoolean(false);
    
    // 模型是否正在使用中（被某个操作占用）
    private final AtomicBoolean isModelBusy = new AtomicBoolean(false);
    
    // 模型卸载超时（毫秒）- 5分钟
    private static final long UNLOAD_TIMEOUT_MS = 5 * 60 * 1000;
    
    // 上下文
    private Context applicationContext;
    
    // GPU设置
    private boolean useGpu;
    
    /**
     * 模型回调接口
     */
    public interface ModelCallback {
        void onModelReady(EmbeddingModelHandler model);
        void onModelError(Exception e);
    }
    
    /**
     * 模型加载状态监听接口
     */
    public interface ModelLoadListener {
        void onLoadStarted(String modelPath);
        void onLoadProgress(String modelPath, String message);
        void onLoadComplete(String modelPath);
        void onLoadError(String modelPath, Exception e);
    }
    
    // 模型加载监听器
    private ModelLoadListener loadListener;
    
    /**
     * 私有构造函数
     */
    private EmbeddingModelManager() {
        executor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());
    }
    
    /**
     * 获取单例实例
     */
    public static synchronized EmbeddingModelManager getInstance(Context context) {
        if (instance == null) {
            instance = new EmbeddingModelManager();
        }
        
        if (context != null) {
            instance.applicationContext = context.getApplicationContext();
        }
        
        return instance;
    }
    
    /**
     * 设置模型加载监听器
     */
    public void setModelLoadListener(ModelLoadListener listener) {
        this.loadListener = listener;
    }
    
    /**
     * 获取模型实例（同步方法）
     * 如果模型已加载且路径匹配，直接返回
     * 否则加载新模型
     * 如果模型正在被其他操作使用，则抛出异常
     */
    public synchronized EmbeddingModelHandler getModel(String modelPath) throws Exception {
        // 检查模型是否繁忙
        if (isModelBusy.get()) {
            throw new Exception("模型正在被其他操作使用，请稍后再试");
        }
        
        try {
            // 标记模型为繁忙状态
            isModelBusy.set(true);
            
            // 更新最后访问时间
            lastAccessTime = System.currentTimeMillis();
            
            // 取消定时卸载任务
            cancelUnloadTask();
            
            // 检查模型是否需要更换或重新加载
            boolean needReload = false;
            
            // 获取当前GPU设置
            boolean currentGpuSetting = SettingsFragment.getUseGpu(applicationContext);
            
            // 检查当前模型是否有效
            if (currentModel != null) {
                try {
                    // 尝试进行一个简单的操作来验证模型是否有效
                    currentModel.getModelName();
                    
                    // 检查模型路径是否匹配
                    if (!currentModelPath.equals(modelPath)) {
                        Log.d(TAG, "模型路径不匹配，需要重新加载。当前路径: " + currentModelPath + "，请求路径: " + modelPath);
                        LogManager.getInstance(applicationContext).i(TAG, "模型路径不匹配，需要重新加载。当前模型: " + 
                            new File(currentModelPath).getName() + "，请求模型: " + new File(modelPath).getName());
                        needReload = true;
                    } else {
                        // 检查GPU设置是否变更，如果变更则需要重新加载
                        if (this.useGpu != currentGpuSetting) {
                            String gpuStatusMsg = String.format("当前GPU加速设置: %s，已加载模型设置: %s", 
                                currentGpuSetting ? "GPU" : "CPU", 
                                this.useGpu ? "GPU" : "CPU");
                            
                            Log.d(TAG, gpuStatusMsg + "，需要重新加载模型");
                            LogManager.getInstance(applicationContext).i(TAG, gpuStatusMsg + "，需要重新加载模型");
                            needReload = true;
                        } else {
                            String gpuStatusMsg = String.format("当前GPU加速设置: %s，已加载模型设置: %s", 
                                currentGpuSetting ? "GPU" : "CPU", 
                                this.useGpu ? "GPU" : "CPU");
                            
                            Log.d(TAG, gpuStatusMsg + "，直接使用现有模型");
                            LogManager.getInstance(applicationContext).d(TAG, gpuStatusMsg + 
                                "，直接使用现有模型: " + new File(currentModelPath).getName());
                        }
                    }
                } catch (Exception e) {
                    String gpuStatusMsg = String.format("当前GPU加速设置: %s，已加载模型状态: 无效", 
                        currentGpuSetting ? "GPU" : "CPU");
                    
                    Log.e(TAG, gpuStatusMsg + "，需要重新加载: " + e.getMessage(), e);
                    LogManager.getInstance(applicationContext).e(TAG, gpuStatusMsg + "，需要重新加载: " + e.getMessage());
                    needReload = true;
                }
            } else {
                String gpuStatusMsg = String.format("当前GPU加速设置: %s，已加载模型状态: 未加载", 
                    currentGpuSetting ? "GPU" : "CPU");
                
                Log.d(TAG, gpuStatusMsg + "，需要加载模型: " + modelPath);
                LogManager.getInstance(applicationContext).i(TAG, gpuStatusMsg + 
                    "，需要加载模型: " + new File(modelPath).getName());
                needReload = true;
            }
            
            // 如果需要重新加载模型
            if (needReload) {
                Log.d(TAG, "开始加载模型: " + modelPath);
                
                // 关闭现有模型
                if (currentModel != null) {
                    try {
                        String modelName = new File(currentModelPath).getName();
                        Log.d(TAG, "关闭现有模型: " + modelName);
                        LogManager.getInstance(applicationContext).i(TAG, "关闭现有模型: " + modelName);
                        currentModel.close();
                    } catch (Exception e) {
                        Log.e(TAG, "关闭现有模型失败: " + e.getMessage(), e);
                        LogManager.getInstance(applicationContext).e(TAG, "关闭现有模型失败: " + e.getMessage());
                    }
                    currentModel = null;
                } else {
                    LogManager.getInstance(applicationContext).i(TAG, "未加载模型，无需关闭");
                }
                
                // 尝试加载模型
                String loadMsg = String.format("尝试使用%s加速设置加载模型: %s", 
                    currentGpuSetting ? "GPU" : "CPU", 
                    new File(modelPath).getName());
                LogManager.getInstance(applicationContext).i(TAG, loadMsg);
                
                currentModel = loadModelWithGpuFallback(modelPath, currentGpuSetting);
                currentModelPath = modelPath;
                this.useGpu = currentGpuSetting;
                
                // 设置调试模式
                boolean debugMode = SettingsFragment.isDebugModeEnabled(applicationContext);
                if (currentModel != null) {
                    currentModel.setDebugMode(debugMode);
                    Log.d(TAG, "模型加载完成，设置调试模式: " + (debugMode ? "启用" : "禁用"));
                    LogManager.getInstance(applicationContext).d(TAG, "模型 " + new File(modelPath).getName() + 
                        " 加载完成，设置调试模式: " + (debugMode ? "启用" : "禁用"));
                }
            }
            
            return currentModel;
        } finally {
            // 安排定时卸载任务
            scheduleModelUnload();
            
            // 标记模型为非繁忙状态
            isModelBusy.set(false);
        }
    }
    
    /**
     * 异步获取模型
     * 如果模型正在被其他操作使用，则通过回调返回错误
     */
    public void getModelAsync(final String modelPath, final ModelCallback callback) {
        // 检查参数
        if (modelPath == null || modelPath.isEmpty()) {
            if (callback != null) {
                mainHandler.post(() -> callback.onModelError(new IllegalArgumentException("模型路径不能为空")));
            }
            return;
        }
        
        // 检查模型是否繁忙
        if (isModelBusy.get()) {
            if (callback != null) {
                mainHandler.post(() -> callback.onModelError(new Exception("模型正在被其他操作使用，请稍后再试")));
            }
            return;
        }
        
        // 标记模型为繁忙状态
        isModelBusy.set(true);
        
        // 更新最后访问时间
        lastAccessTime = System.currentTimeMillis();
        
        // 取消定时卸载任务
        cancelUnloadTask();
        
        // 获取当前GPU设置
        boolean currentGpuSetting = SettingsFragment.getUseGpu(applicationContext);
        
        // 检查是否需要重新加载模型
        boolean needReload = false;
        
        // 检查当前模型是否有效
        if (currentModel != null) {
            try {
                // 尝试进行一个简单的操作来验证模型是否有效
                currentModel.getModelName();
                
                // 检查模型路径是否匹配
                if (!currentModelPath.equals(modelPath)) {
                    Log.d(TAG, "异步加载 - 模型路径不匹配，需要重新加载。当前路径: " + currentModelPath + "，请求路径: " + modelPath);
                    LogManager.getInstance(applicationContext).i(TAG, "异步加载 - 模型路径不匹配，需要重新加载。当前模型: " + 
                        new File(currentModelPath).getName() + "，请求模型: " + new File(modelPath).getName());
                    needReload = true;
                } else {
                    // 检查GPU设置是否变更，如果变更则需要重新加载
                    if (this.useGpu != currentGpuSetting) {
                        String gpuStatusMsg = String.format("异步加载 - 当前GPU加速设置: %s，已加载模型设置: %s", 
                            currentGpuSetting ? "GPU" : "CPU", 
                            this.useGpu ? "GPU" : "CPU");
                        
                        Log.d(TAG, gpuStatusMsg + "，需要重新加载模型");
                        LogManager.getInstance(applicationContext).i(TAG, gpuStatusMsg + "，需要重新加载模型");
                        needReload = true;
                    } else {
                        String gpuStatusMsg = String.format("异步加载 - 当前GPU加速设置: %s，已加载模型设置: %s", 
                            currentGpuSetting ? "GPU" : "CPU", 
                            this.useGpu ? "GPU" : "CPU");
                        
                        Log.d(TAG, gpuStatusMsg + "，直接使用现有模型");
                        LogManager.getInstance(applicationContext).d(TAG, gpuStatusMsg + 
                            "，直接使用现有模型: " + new File(currentModelPath).getName());
                    }
                }
            } catch (Exception e) {
                String gpuStatusMsg = String.format("异步加载 - 当前GPU加速设置: %s，已加载模型状态: 无效", 
                    currentGpuSetting ? "GPU" : "CPU");
                
                Log.e(TAG, gpuStatusMsg + "，需要重新加载: " + e.getMessage(), e);
                LogManager.getInstance(applicationContext).e(TAG, gpuStatusMsg + "，需要重新加载: " + e.getMessage());
                needReload = true;
            }
        } else {
            String gpuStatusMsg = String.format("异步加载 - 当前GPU加速设置: %s，已加载模型状态: 未加载", 
                currentGpuSetting ? "GPU" : "CPU");
            
            Log.d(TAG, gpuStatusMsg + "，需要加载模型: " + modelPath);
            LogManager.getInstance(applicationContext).i(TAG, gpuStatusMsg + 
                "，需要加载模型: " + new File(modelPath).getName());
            needReload = true;
        }
        
        // 在后台线程中执行，避免阻塞UI线程
        final boolean finalNeedReload = needReload;
        final boolean finalGpuSetting = currentGpuSetting;
        
        executor.execute(() -> {
            try {
                // 更新最后访问时间
                lastAccessTime = System.currentTimeMillis();
                
                // 取消定时卸载任务
                cancelUnloadTask();
                
                // 如果需要重新加载模型
                if (finalNeedReload) {
                    Log.d(TAG, "异步加载 - 开始加载模型: " + modelPath);
                    
                    // 关闭现有模型
                    if (currentModel != null) {
                        try {
                            String modelName = new File(currentModelPath).getName();
                            Log.d(TAG, "关闭现有模型: " + modelName);
                            LogManager.getInstance(applicationContext).i(TAG, "关闭现有模型: " + modelName);
                            currentModel.close();
                        } catch (Exception e) {
                            Log.e(TAG, "关闭现有模型失败: " + e.getMessage(), e);
                            LogManager.getInstance(applicationContext).e(TAG, "关闭现有模型失败: " + e.getMessage());
                        }
                        currentModel = null;
                    } else {
                        LogManager.getInstance(applicationContext).i(TAG, "未加载模型，无需关闭");
                    }
                    
                    // 尝试加载模型
                    String loadMsg = String.format("异步加载 - 尝试使用%s加速设置加载模型: %s", 
                        finalGpuSetting ? "GPU" : "CPU", 
                        new File(modelPath).getName());
                    LogManager.getInstance(applicationContext).i(TAG, loadMsg);
                    
                    currentModel = loadModelWithGpuFallback(modelPath, finalGpuSetting);
                    currentModelPath = modelPath;
                    useGpu = finalGpuSetting;
                    
                    // 设置调试模式
                    boolean debugMode = SettingsFragment.isDebugModeEnabled(applicationContext);
                    if (currentModel != null) {
                        currentModel.setDebugMode(debugMode);
                        Log.d(TAG, "模型加载完成，设置调试模式: " + (debugMode ? "启用" : "禁用"));
                        LogManager.getInstance(applicationContext).d(TAG, "模型 " + new File(modelPath).getName() + 
                            " 加载完成，设置调试模式: " + (debugMode ? "启用" : "禁用"));
                    }
                } else {
                    Log.d(TAG, "异步加载 - 使用现有模型，无需重新加载");
                }
                
                // 通知模型加载完成
                final EmbeddingModelHandler model = currentModel;
                mainHandler.post(() -> callback.onModelReady(model));
            } catch (Exception e) {
                Log.e(TAG, "异步加载 - 加载模型失败: " + e.getMessage(), e);
                LogManager.getInstance(applicationContext).e(TAG, "异步加载 - 加载模型失败: " + e.getMessage());
                mainHandler.post(() -> callback.onModelError(e));
            } finally {
                // 安排定时卸载任务
                scheduleModelUnload();
                
                // 标记模型为非繁忙状态
                isModelBusy.set(false);
            }
        });
    }
    
    /**
     * 使用GPU回退机制加载模型
     * 如果GPU加载失败，会自动降级到CPU模式
     * @param modelPath 模型路径
     * @param useGpu 是否使用GPU
     * @return 加载的模型
     * @throws Exception 如果加载失败
     */
    private EmbeddingModelHandler loadModelWithGpuFallback(String modelPath, boolean useGpu) throws Exception {
        EmbeddingModelHandler model = null;
        
        // 通知加载开始
        notifyLoadStarted(modelPath);
        
        try {
            // 根据设置决定是否使用GPU
            if (useGpu) {
                String loadMsg = String.format("尝试使用GPU加速设置加载模型: %s", new File(modelPath).getName());
                Log.d(TAG, loadMsg);
                notifyLoadProgress(modelPath, "尝试使用GPU加载模型...");
                LogManager.getInstance(applicationContext).i(TAG, loadMsg);
                
                try {
                    // 使用GPU加载模型，传递Context参数
                    model = new EmbeddingModelHandler(applicationContext, modelPath, true);
                    Log.d(TAG, "模型加速: GPU模式加载成功");
                    LogManager.getInstance(applicationContext).i(TAG, "模型加速: GPU模式加载成功 - " + new File(modelPath).getName());
                    notifyLoadComplete(modelPath);
                    return model;
                } catch (Exception e) {
                    Log.w(TAG, "GPU加载失败，降级到CPU模式: " + e.getMessage(), e);
                    LogManager.getInstance(applicationContext).w(TAG, "GPU加载失败，降级到CPU模式: " + e.getMessage());
                    notifyLoadProgress(modelPath, "GPU加载失败，降级到CPU模式...");
                }
            } else {
                // 如果不使用GPU，则直接使用CPU加载
                String loadMsg = String.format("尝试使用CPU加速设置加载模型: %s", new File(modelPath).getName());
                Log.d(TAG, loadMsg);
                notifyLoadProgress(modelPath, "使用CPU加载模型...");
                LogManager.getInstance(applicationContext).i(TAG, loadMsg);
                
                // 传递Context参数
                model = new EmbeddingModelHandler(applicationContext, modelPath, false);
                notifyLoadComplete(modelPath);
                return model;
            }
            
            // 如果GPU加载失败，则使用CPU加载
            String loadMsg = String.format("尝试使用CPU加速设置加载模型: %s", new File(modelPath).getName());
            Log.d(TAG, loadMsg);
            notifyLoadProgress(modelPath, "使用CPU加载模型...");
            LogManager.getInstance(applicationContext).i(TAG, loadMsg);
            
            // 传递Context参数
            model = new EmbeddingModelHandler(applicationContext, modelPath, false);
            notifyLoadComplete(modelPath);
        } catch (Exception e) {
            notifyLoadError(modelPath, e);
            throw e;
        }
        
        Log.d(TAG, "模型加速: CPU模式加载成功");
        LogManager.getInstance(applicationContext).i(TAG, "模型加速: CPU模式加载成功 - " + new File(modelPath).getName());
        return model;
    }
    
    /**
     * 安排模型卸载定时任务
     * 只有在超时未使用时才会卸载模型
     */
    private synchronized void scheduleModelUnload() {
        // 取消现有任务
        cancelUnloadTask();
        
        // 创建新的定时任务
        unloadTimer = new Timer();
        unloadTimer.schedule(new TimerTask() {
            @Override
            public void run() {
                synchronized (EmbeddingModelManager.this) {
                    // 检查是否超时未使用
                    long currentTime = System.currentTimeMillis();
                    if (currentTime - lastAccessTime >= UNLOAD_TIMEOUT_MS && !isModelBusy.get()) {
                        Log.d(TAG, "模型超时未使用，自动卸载");
                        
                        // 卸载模型
                        if (currentModel != null) {
                            try {
                                currentModel.close();
                            } catch (Exception e) {
                                Log.e(TAG, "卸载模型失败: " + e.getMessage(), e);
                            }
                            currentModel = null;
                            currentModelPath = null;
                        }
                        
                        cancelUnloadTask();
                    }
                }
            }
        }, UNLOAD_TIMEOUT_MS, UNLOAD_TIMEOUT_MS); // 首次延迟和周期都是超时时间
    }
    
    /**
     * 取消定时卸载任务
     */
    private synchronized void cancelUnloadTask() {
        if (unloadTimer != null) {
            unloadTimer.cancel();
            unloadTimer = null;
        }
    }
    
    /**
     * 标记模型开始使用
     * 更新最后访问时间并取消定时卸载任务
     */
    public synchronized void markModelInUse() {
        // 更新最后访问时间
        lastAccessTime = System.currentTimeMillis();
        
        // 取消定时卸载任务
        cancelUnloadTask();
        
        Log.d(TAG, "标记模型开始使用");
    }
    
    /**
     * 标记模型使用结束
     * 更新最后访问时间并安排定时卸载任务
     */
    public synchronized void markModelNotInUse() {
        // 更新最后访问时间
        lastAccessTime = System.currentTimeMillis();
        
        // 安排定时卸载任务
        scheduleModelUnload();
        
        Log.d(TAG, "标记模型使用结束，安排定时卸载任务");
    }
    
    /**
     * 显示Toast消息
     */
    private void showToast(final String message) {
        if (applicationContext != null) {
            mainHandler.post(() -> {
                Toast.makeText(applicationContext, message, Toast.LENGTH_SHORT).show();
            });
        }
    }
    
    /**
     * 通知模型加载开始
     */
    private void notifyLoadStarted(String modelPath) {
        if (loadListener != null) {
            mainHandler.post(() -> loadListener.onLoadStarted(modelPath));
        }
    }
    
    /**
     * 通知模型加载进度
     */
    private void notifyLoadProgress(String modelPath, String message) {
        if (loadListener != null) {
            mainHandler.post(() -> loadListener.onLoadProgress(modelPath, message));
        }
    }
    
    /**
     * 通知模型加载完成
     */
    private void notifyLoadComplete(String modelPath) {
        if (loadListener != null) {
            mainHandler.post(() -> loadListener.onLoadComplete(modelPath));
        }
    }
    
    /**
     * 通知模型加载错误
     */
    private void notifyLoadError(String modelPath, Exception e) {
        if (loadListener != null) {
            mainHandler.post(() -> loadListener.onLoadError(modelPath, e));
        }
    }
}
