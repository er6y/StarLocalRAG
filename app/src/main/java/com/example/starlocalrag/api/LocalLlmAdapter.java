package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;
import com.example.starlocalrag.LogManager;

import com.example.starlocalrag.ConfigManager;

import java.io.File;

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
    
    private volatile boolean isLoading = false;
    private final Object loadingLock = new Object();
    
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
        LogManager.logD(TAG, "LocalLlmAdapter 初始化");
    }
    
    /**
     * 调用本地LLM模型
     * @param modelName 模型名称
     * @param prompt 提示词
     * @param callback 回调接口
     */
    public void callLocalModel(String modelName, String prompt, LlmApiAdapter.ApiCallback callback) {
        LogManager.logD(TAG, "调用本地模型: " + modelName);
        
        synchronized (loadingLock) {
            // 检查模型是否已加载
            if (localLlmHandler.isModelLoaded() && modelName.equals(localLlmHandler.getCurrentModelName())) {
                LogManager.logD(TAG, "模型已加载，直接执行推理: " + modelName);
                executeInference(prompt, callback);
                return;
            }
            
            // 检查是否正在加载
            if (isLoading) {
                LogManager.logW(TAG, "模型正在加载中，请稍后再试: " + modelName);
                callback.onError("模型正在加载中，请稍后再试");
                return;
            }
            
            LogManager.logD(TAG, "加载模型: " + modelName);
            isLoading = true;
        }
        
        // 异步加载模型
        try {
            localLlmHandler.loadModel(modelName, new LocalLlmHandler.StreamingCallback() {
                @Override
                public void onToken(String token) {
                    // 模型加载过程中不需要处理token
                }
                
                @Override
                public void onComplete(String fullResponse) {
                    LogManager.logD(TAG, "模型加载完成: " + modelName);
                    synchronized (loadingLock) {
                        isLoading = false;
                    }
                    // 模型加载完成后执行推理
                    executeInference(prompt, callback);
                }
                
                @Override
                public void onError(String errorMessage) {
                    LogManager.logE(TAG, "模型加载失败: " + modelName + ", 错误: " + errorMessage);
                    synchronized (loadingLock) {
                        isLoading = false;
                    }
                    callback.onError("模型加载失败: " + errorMessage);
                }
            });
        } catch (Exception e) {
            // 确保在任何异常情况下都重置isLoading标志
            LogManager.logE(TAG, "调用模型加载时发生异常: " + modelName, e);
            synchronized (loadingLock) {
                isLoading = false;
            }
            callback.onError("调用模型加载时发生异常: " + e.getMessage());
        }
    }
    
    /**
     * 执行推理
     * @param prompt 提示词
     * @param callback 回调接口
     */
    private void executeInference(String prompt, LlmApiAdapter.ApiCallback callback) {
        LogManager.logD(TAG, "开始执行本地LLM推理，提示词长度: " + prompt.length());
        
        // 注意：标题行应该由调用者添加，这里不再重复添加
        // 避免多次显示“模型回答”标题
        //LogManager.logD(TAG, "不再发送模型回答标题，由调用者负责");
        
        // 调试追踪日志已移除
        localLlmHandler.inference(prompt, new LocalLlmHandler.StreamingCallback() {
            @Override
            public void onToken(String token) {
                // 紧凑打印token内容到控制台并记录到日志文件
                LogManager.print(token);
                
                // 将token发送给UI
                callback.onStreamingData(token);
                
                // 打印调试信息，确认token已发送
                //LogManager.logD(TAG, "流式token已发送到UI");
            }
            
            @Override
            public void onComplete(String fullResponse) {
                LogManager.logD(TAG, "本地LLM推理完成，完整响应长度: " + fullResponse.length());
                // 调试追踪日志已移除
                callback.onSuccess(fullResponse);
            }
            
            @Override
            public void onError(String errorMessage) {
                LogManager.logE(TAG, "本地LLM推理失败: " + errorMessage);
                callback.onError(errorMessage);
            }
        });
    }
    
    /**
     * 列出可用的本地模型
     * @return 模型名称数组
     */
    public String[] listAvailableModels() {
        // 从配置中获取模型路径
        String modelPath = ConfigManager.getModelPath(context);
        LogManager.logD(TAG, "从路径获取模型列表: " + modelPath);
        
        File modelDir = new File(modelPath);
        if (!modelDir.exists() || !modelDir.isDirectory()) {
            LogManager.logE(TAG, "模型目录不存在: " + modelPath);
            return new String[0];
        }
        
        // 获取目录下的所有子目录，每个子目录就是一个模型
        File[] modelDirs = modelDir.listFiles(File::isDirectory);
        if (modelDirs == null || modelDirs.length == 0) {
            LogManager.logW(TAG, "模型目录为空: " + modelPath);
            return new String[0];
        }
        
        // 提取目录名称作为模型名称
        String[] modelNames = new String[modelDirs.length];
        for (int i = 0; i < modelDirs.length; i++) {
            modelNames[i] = modelDirs[i].getName();
            LogManager.logD(TAG, "发现模型: " + modelNames[i]);
        }
        
        return modelNames;
    }
    
    /**
     * 更新GPU设置
     * @param useGpu 是否使用GPU
     */
    public void updateGpuSetting(boolean useGpu) {
        LogManager.logD(TAG, "LocalLLM GPU设置更新: " + (useGpu ? "启用" : "禁用") + "GPU加速");
        
        if (localLlmHandler != null) {
            try {
                localLlmHandler.setUseGpu(useGpu);
                LogManager.logI(TAG, "LocalLLM GPU设置: 成功更新GPU设置为 " + (useGpu ? "启用" : "禁用"));
                
                // 注意：GPU设置变更可能需要重新加载模型才能生效
                LogManager.logW(TAG, "LocalLLM GPU设置: 注意GPU设置变更可能需要重新加载模型才能完全生效");
            } catch (Exception e) {
                LogManager.logE(TAG, "LocalLLM GPU设置: 更新GPU设置失败: " + e.getMessage(), e);
            }
        } else {
            LogManager.logW(TAG, "LocalLLM GPU设置: LocalLlmHandler未初始化，无法更新GPU设置");
        }
    }
    
    /**
     * 停止生成
     */
    public void stopGeneration() {
        // 停止生成逻辑
        // 停止调试日志已移除
        LogManager.logD(TAG, "[停止调试] localLlmHandler状态: " + (localLlmHandler != null ? "已初始化" : "未初始化"));
        
        if (localLlmHandler != null) {
            LogManager.logD(TAG, "[停止调试] 准备调用localLlmHandler.stopInference()");
            localLlmHandler.stopInference();
            // 停止调试日志已移除
        } else {
            LogManager.logW(TAG, "[停止调试] localLlmHandler为null，无法停止推理");
        }
    }
    
    /**
     * 重置停止标志
     */
    public void resetStopFlag() {
        LogManager.logD(TAG, "重置本地LLM停止标志");
        if (localLlmHandler != null) {
            localLlmHandler.resetStopFlag();
        }
    }
    
    /**
     * 重置模型记忆 - 清除KV缓存和对话历史
     */
    public void resetModelMemory() {
        // 新对话调试日志已移除
        if (localLlmHandler != null) {
            // 新对话调试日志已移除
            localLlmHandler.resetModelMemory();
        } else {
            LogManager.logW(TAG, "[新对话调试] localLlmHandler为空，无法重置模型记忆");
        }
        // 新对话调试日志已移除
    }
    
    /**
     * 重置加载状态标志
     * 用于解决并发调用时的状态同步问题
     */
    public void resetLoadingState() {
        synchronized (loadingLock) {
            if (isLoading) {
                LogManager.logW(TAG, "手动重置isLoading标志");
                isLoading = false;
            }
        }
    }
    
    /**
     * 检查当前是否正在加载模型
     * @return true如果正在加载，false否则
     */
    public boolean isLoading() {
        synchronized (loadingLock) {
            return isLoading;
        }
    }
    
    /**
     * 关闭适配器，释放资源
     */
    public void shutdown() {
        if (localLlmHandler != null) {
            localLlmHandler.unloadModel();
        }
        instance = null;
    }
}
