package com.example.starlocalrag;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 重排模型管理器
 * 采用单例模式，负责重排模型的加载、缓存和卸载
 * 参考EmbeddingModelManager的设计模式
 */
public class RerankerModelManager {
    private static final String TAG = "StarLocalRAG_RerankerModelManager";
    private static final long MODEL_UNLOAD_DELAY_MS = 300000; // 5分钟后卸载模型
    
    private static volatile RerankerModelManager instance;
    private final ExecutorService executor;
    private final Handler mainHandler;
    
    // 模型状态
    private RerankerModelHandler currentModel;
    private String currentModelPath;
    private long lastAccessTime;
    private final AtomicBoolean isModelBusy = new AtomicBoolean(false);
    
    // 卸载任务
    private Runnable unloadTask;
    
    /**
     * 模型回调接口
     */
    public interface RerankerModelCallback {
        void onModelReady(RerankerModelHandler model);
        void onModelError(String error);
    }
    
    /**
     * 重排结果回调接口
     */
    public interface RerankerCallback {
        void onRerankComplete(java.util.List<RerankerModelHandler.RerankResult> results);
        void onRerankError(String error);
        void onRerankProgress(String message);
    }
    
    /**
     * 模型加载监听器
     */
    public interface RerankerModelLoadListener {
        void onLoadStart();
        void onLoadProgress(String message);
        void onLoadComplete();
        void onLoadError(String error);
    }
    
    private RerankerModelLoadListener loadListener;
    
    private RerankerModelManager() {
        executor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());
    }
    
    /**
     * 获取单例实例
     */
    public static RerankerModelManager getInstance(Context context) {
        if (instance == null) {
            synchronized (RerankerModelManager.class) {
                if (instance == null) {
                    instance = new RerankerModelManager();
                }
            }
        }
        return instance;
    }
    
    /**
     * 设置加载监听器
     */
    public void setLoadListener(RerankerModelLoadListener listener) {
        this.loadListener = listener;
    }
    
    /**
     * 异步获取重排模型
     */
    public void getModelAsync(String modelPath, RerankerModelCallback callback) {
        LogManager.logI(TAG, "=== getModelAsync 开始执行 ===");
        LogManager.logI(TAG, "请求的模型路径: " + modelPath);
        LogManager.logI(TAG, "回调对象: " + (callback != null ? "非空" : "空"));
        
        if (modelPath == null || modelPath.trim().isEmpty()) {
            LogManager.logW(TAG, "重排模型路径为空");
            if (callback != null) {
                mainHandler.post(() -> callback.onModelError("重排模型路径为空"));
            }
            return;
        }
        
        LogManager.logI(TAG, "提交异步任务到线程池...");
        executor.execute(() -> {
            try {
                LogManager.logI(TAG, "异步任务开始执行，线程: " + Thread.currentThread().getName());
                RerankerModelHandler model = getModel(modelPath);
                LogManager.logI(TAG, "getModel返回结果: " + (model != null ? "成功" : "失败"));
                
                if (model != null) {
                    LogManager.logI(TAG, "模型获取成功，准备调用onModelReady回调");
                    if (callback != null) {
                        LogManager.logI(TAG, "发送onModelReady回调到主线程");
                        mainHandler.post(() -> {
                            LogManager.logI(TAG, "onModelReady回调开始执行");
                            callback.onModelReady(model);
                            LogManager.logI(TAG, "onModelReady回调执行完成");
                        });
                    } else {
                        LogManager.logW(TAG, "回调对象为空，无法调用onModelReady");
                    }
                } else {
                    LogManager.logE(TAG, "模型获取失败，准备调用onModelError回调");
                    if (callback != null) {
                        mainHandler.post(() -> callback.onModelError("重排模型加载失败"));
                    }
                }
            } catch (Exception e) {
                LogManager.logE(TAG, "异步获取重排模型失败: " + e.getMessage(), e);
                if (callback != null) {
                    mainHandler.post(() -> callback.onModelError("重排模型加载异常: " + e.getMessage()));
                }
            }
        });
    }
    
    /**
     * 异步执行重排任务（推荐使用此方法，避免线程嵌套问题）
     */
    public void rerankAsync(String modelPath, String query, java.util.List<String> documents, int topK, RerankerCallback callback) {
        LogManager.logI(TAG, "=== rerankAsync 开始执行 ===");
        LogManager.logI(TAG, "模型路径: " + modelPath);
        LogManager.logI(TAG, "查询: " + query);
        LogManager.logI(TAG, "文档数: " + (documents != null ? documents.size() : "null"));
        LogManager.logI(TAG, "topK: " + topK);
        
        if (modelPath == null || modelPath.trim().isEmpty()) {
            LogManager.logW(TAG, "重排模型路径为空");
            if (callback != null) {
                mainHandler.post(() -> callback.onRerankError("重排模型路径为空"));
            }
            return;
        }
        
        if (query == null || query.trim().isEmpty()) {
            LogManager.logW(TAG, "查询内容为空");
            if (callback != null) {
                mainHandler.post(() -> callback.onRerankError("查询内容为空"));
            }
            return;
        }
        
        if (documents == null || documents.isEmpty()) {
            LogManager.logW(TAG, "文档列表为空");
            if (callback != null) {
                mainHandler.post(() -> callback.onRerankError("文档列表为空"));
            }
            return;
        }
        
        // 检查模型是否忙碌
        if (isModelBusy.get()) {
            LogManager.logW(TAG, "重排模型正在使用中，请稍候");
            if (callback != null) {
                mainHandler.post(() -> callback.onRerankError("重排模型正在使用中，请稍候"));
            }
            return;
        }
        
        LogManager.logI(TAG, "提交重排任务到线程池...");
        executor.execute(() -> {
            try {
                LogManager.logI(TAG, "重排任务开始执行，线程: " + Thread.currentThread().getName());
                
                // 在线程内部标记模型为忙碌状态
                isModelBusy.set(true);
                
                // 开始加载重排模型（不显示进度信息）
                
                // 获取模型
                RerankerModelHandler model = getModel(modelPath);
                if (model == null) {
                    LogManager.logE(TAG, "重排模型获取失败");
                    if (callback != null) {
                        mainHandler.post(() -> callback.onRerankError("重排模型加载失败"));
                    }
                    return;
                }
                
                LogManager.logI(TAG, "重排模型加载成功，开始执行重排");
                
                // 开始重排文档（不显示进度信息）
                
                // 执行重排
                java.util.List<RerankerModelHandler.RerankResult> results = model.rerank(query, documents, topK, new RerankerModelHandler.RerankProgressCallback() {
                    @Override
                    public void onRerankProgress(int processedCount, int totalCount, double score) {
                        // 在主线程中更新UI
                        mainHandler.post(() -> {
                            if (callback != null) {
                                String progressMessage = String.format("%.3f, ", score);
                                callback.onRerankProgress(progressMessage);
                            }
                        });
                    }
                });
                LogManager.logI(TAG, "重排执行完成，结果数: " + (results != null ? results.size() : "null"));
                
                // 返回结果
                if (callback != null) {
                    if (results != null) {
                        mainHandler.post(() -> callback.onRerankComplete(results));
                    } else {
                        mainHandler.post(() -> callback.onRerankError("重排执行失败，返回结果为空"));
                    }
                }
                
            } catch (Exception e) {
                LogManager.logE(TAG, "重排任务执行失败: " + e.getMessage(), e);
                if (callback != null) {
                    mainHandler.post(() -> callback.onRerankError("重排执行异常: " + e.getMessage()));
                }
            } finally {
                // 标记模型为非忙碌状态
                isModelBusy.set(false);
                LogManager.logI(TAG, "重排任务完成，释放模型忙碌状态");
            }
        });
    }
    
    /**
     * 同步获取重排模型
     */
    public synchronized RerankerModelHandler getModel(String modelPath) {
        LogManager.logI(TAG, "=== getModel 开始执行 ===");
        LogManager.logI(TAG, "请求模型路径: " + modelPath);
        LogManager.logI(TAG, "当前模型路径: " + currentModelPath);
        LogManager.logI(TAG, "当前模型状态: " + (currentModel != null ? "存在" : "不存在"));
        
        if (modelPath == null || modelPath.trim().isEmpty()) {
            LogManager.logW(TAG, "重排模型路径为空");
            return null;
        }
        
        // 更新最后访问时间
        lastAccessTime = System.currentTimeMillis();
        LogManager.logI(TAG, "更新最后访问时间: " + lastAccessTime);
        
        // 取消卸载任务
        cancelUnloadTask();
        LogManager.logI(TAG, "取消卸载任务完成");
        
        // 检查是否需要重新加载模型
        boolean needReloadResult = needReload(modelPath);
        LogManager.logI(TAG, "是否需要重新加载: " + needReloadResult);
        
        if (needReloadResult) {
            LogManager.logI(TAG, "需要加载新的重排模型: " + modelPath);
            
            // 关闭当前模型
            if (currentModel != null) {
                LogManager.logI(TAG, "清理当前模型");
                currentModel.cleanup();
                currentModel = null;
            }
            
            // 加载新模型
            LogManager.logI(TAG, "开始加载新模型");
            currentModel = loadModel(modelPath);
            currentModelPath = modelPath;
            LogManager.logI(TAG, "模型加载完成，结果: " + (currentModel != null ? "成功" : "失败"));
        } else {
            LogManager.logI(TAG, "使用现有模型，无需重新加载");
        }
        
        LogManager.logI(TAG, "getModel 返回结果: " + (currentModel != null ? "成功" : "失败"));
        return currentModel;
    }
    
    /**
     * 检查是否需要重新加载模型
     */
    private boolean needReload(String modelPath) {
        return currentModel == null || 
               !modelPath.equals(currentModelPath) || 
               !currentModel.isInitialized();
    }
    
    /**
     * 加载重排模型
     */
    private RerankerModelHandler loadModel(String modelPath) {
        try {
            LogManager.logI(TAG, "开始加载重排模型: " + modelPath);
            
            // 通知加载开始
            if (loadListener != null) {
                mainHandler.post(() -> loadListener.onLoadStart());
            }
            
            // 检查模型文件是否存在
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                String error = "重排模型文件不存在: " + modelPath;
                LogManager.logE(TAG, error);
                if (loadListener != null) {
                    mainHandler.post(() -> loadListener.onLoadError(error));
                }
                return null;
            }
            
            // 通知加载进度
            if (loadListener != null) {
                mainHandler.post(() -> loadListener.onLoadProgress("正在初始化重排模型..."));
            }
            
            // 创建模型处理器
            RerankerModelHandler handler = new RerankerModelHandler(null, modelPath);
            
            // 初始化模型
            boolean success = handler.initialize();
            if (!success) {
                String error = "重排模型初始化失败";
                LogManager.logE(TAG, error);
                if (loadListener != null) {
                    mainHandler.post(() -> loadListener.onLoadError(error));
                }
                handler.cleanup();
                return null;
            }
            
            LogManager.logI(TAG, "重排模型加载成功: " + modelPath);
            
            // 通知加载完成
            if (loadListener != null) {
                mainHandler.post(() -> loadListener.onLoadComplete());
            }
            
            return handler;
            
        } catch (Exception e) {
            String error = "加载重排模型时发生异常: " + e.getMessage();
            LogManager.logE(TAG, error, e);
            if (loadListener != null) {
                mainHandler.post(() -> loadListener.onLoadError(error));
            }
            return null;
        }
    }
    
    /**
     * 标记模型开始使用
     */
    public void markModelInUse() {
        isModelBusy.set(true);
        lastAccessTime = System.currentTimeMillis();
        cancelUnloadTask();
        LogManager.logD(TAG, "标记重排模型开始使用");
    }
    
    /**
     * 标记模型使用结束
     */
    public void markModelNotInUse() {
        isModelBusy.set(false);
        lastAccessTime = System.currentTimeMillis();
        scheduleUnloadTask();
        LogManager.logD(TAG, "标记重排模型使用结束");
    }
    
    /**
     * 安排卸载任务
     */
    private void scheduleUnloadTask() {
        // 注意：当前禁用自动卸载，与EmbeddingModelManager保持一致
        // 如果需要启用，可以取消下面的注释
        /*
        cancelUnloadTask();
        unloadTask = () -> {
            synchronized (RerankerModelManager.this) {
                if (!isModelBusy.get() && 
                    System.currentTimeMillis() - lastAccessTime >= MODEL_UNLOAD_DELAY_MS) {
                    LogManager.logI(TAG, "自动卸载重排模型");
                    unloadModel();
                }
            }
        };
        mainHandler.postDelayed(unloadTask, MODEL_UNLOAD_DELAY_MS);
        */
    }
    
    /**
     * 取消卸载任务
     */
    private void cancelUnloadTask() {
        if (unloadTask != null) {
            mainHandler.removeCallbacks(unloadTask);
            unloadTask = null;
        }
    }
    
    /**
     * 卸载模型
     */
    public synchronized void unloadModel() {
        if (currentModel != null) {
            LogManager.logI(TAG, "卸载重排模型: " + currentModelPath);
            currentModel.cleanup();
            currentModel = null;
            currentModelPath = null;
        }
        cancelUnloadTask();
    }
    
    /**
     * 获取当前模型路径
     */
    public String getCurrentModelPath() {
        return currentModelPath;
    }
    
    /**
     * 检查模型是否正在使用
     */
    public boolean isModelBusy() {
        return isModelBusy.get();
    }
    
    /**
     * 检查是否有模型加载
     */
    public boolean hasModel() {
        return currentModel != null && currentModel.isInitialized();
    }
    
    /**
     * 关闭管理器
     */
    public void shutdown() {
        LogManager.logI(TAG, "关闭重排模型管理器");
        
        // 卸载模型
        unloadModel();
        
        // 关闭线程池
        if (executor != null && !executor.isShutdown()) {
            executor.shutdown();
        }
        
        // 清理监听器
        loadListener = null;
    }
    
    /**
     * 重置管理器（用于测试或重新初始化）
     */
    public static synchronized void resetManager() {
        if (instance != null) {
            LogManager.logD(TAG, "重置重排模型管理器");
            instance.shutdown();
            instance = null;
        }
    }
}