package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

/**
 * 本地LLM适配器
 * 将本地LLM处理程序与LlmApiAdapter接口对齐，方便上层代码无缝切换
 */
public class LocalLlmAdapter {
    private static final String TAG = "LocalLlmAdapter";
    
    // 上下文
    private final Context context;
    
    // 本地LLM处理程序
    private final LocalLlmHandler localLlmHandler;
    
    // 单例实例
    private static LocalLlmAdapter instance;
    
    /**
     * 获取单例实例
     */
    public static synchronized LocalLlmAdapter getInstance(Context context) {
        if (instance == null) {
            instance = new LocalLlmAdapter(context);
        }
        return instance;
    }
    
    /**
     * 私有构造函数
     */
    private LocalLlmAdapter(Context context) {
        this.context = context;
        this.localLlmHandler = LocalLlmHandler.getInstance(context);
        Log.d(TAG, "LocalLlmAdapter 初始化");
    }
    
    /**
     * 调用本地LLM模型
     * @param modelName 模型名称
     * @param prompt 提示词
     * @param callback 回调接口
     */
    public void callLocalModel(String modelName, String prompt, LlmApiAdapter.ApiCallback callback) {
        Log.d(TAG, "调用本地模型: " + modelName);
        
        // 检查模型是否已加载，如果没有加载或者是不同的模型，则先加载模型
        if (!localLlmHandler.isModelLoaded() || !modelName.equals(localLlmHandler.getCurrentModelName())) {
            Log.d(TAG, "需要加载模型: " + modelName);
            
            // 加载模型
            localLlmHandler.loadModel(modelName, new LocalLlmHandler.LocalLlmCallback() {
                @Override
                public void onToken(String token) {
                    // 加载过程中不会有token回调
                }
                
                @Override
                public void onComplete(String fullResponse) {
                    Log.d(TAG, "模型加载完成，开始推理");
                    // 模型加载完成后，执行推理
                    executeInference(prompt, callback);
                }
                
                @Override
                public void onError(String errorMessage) {
                    Log.e(TAG, "模型加载失败: " + errorMessage);
                    callback.onError("模型加载失败: " + errorMessage);
                }
            });
        } else {
            // 模型已加载，直接执行推理
            Log.d(TAG, "模型已加载，直接执行推理");
            executeInference(prompt, callback);
        }
    }
    
    /**
     * 执行推理
     * @param prompt 提示词
     * @param callback 回调接口
     */
    private void executeInference(String prompt, LlmApiAdapter.ApiCallback callback) {
        localLlmHandler.inference(prompt, new LocalLlmHandler.LocalLlmCallback() {
            @Override
            public void onToken(String token) {
                callback.onStreamingData(token);
            }
            
            @Override
            public void onComplete(String fullResponse) {
                callback.onSuccess(fullResponse);
            }
            
            @Override
            public void onError(String errorMessage) {
                callback.onError(errorMessage);
            }
        });
    }
    
    /**
     * 列出可用的本地模型
     * @return 模型名称数组
     */
    public String[] listAvailableModels() {
        return localLlmHandler.listAvailableModels();
    }
    
    /**
     * 更新GPU设置
     * @param useGpu 是否使用GPU
     */
    public void updateGpuSetting(boolean useGpu) {
        localLlmHandler.setUseGpu(useGpu);
    }
    
    /**
     * 关闭适配器，释放资源
     */
    public void shutdown() {
        if (localLlmHandler != null) {
            localLlmHandler.shutdown();
        }
        instance = null;
    }
}
