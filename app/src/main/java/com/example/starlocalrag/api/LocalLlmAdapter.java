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
        
        // 简化处理，直接加载模型
        // 在完整实现中，应该检查模型是否已加载以及是否是当前请求的模型
        LogManager.logD(TAG, "加载模型: " + modelName);
        
        // 加载模型
        localLlmHandler.loadModel(modelName, new LocalLlmHandler.LocalLlmCallback() {
            @Override
            public void onToken(String token) {
                // 加载过程中不会有token回调
            }
            
            @Override
            public void onTokenGenerated(String token) {
                // 加载过程中不会有token回调
            }
            
            @Override
            public void onComplete(String fullResponse) {
                LogManager.logD(TAG, "模型加载完成，开始推理");
                // 模型加载完成后，执行推理
                executeInference(prompt, callback);
            }
            
            @Override
            public void onError(String errorMessage) {
                LogManager.logE(TAG, "模型加载失败: " + errorMessage);
                callback.onError("模型加载失败: " + errorMessage);
            }
        });
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
        LogManager.logD(TAG, "不再发送模型回答标题，由调用者负责");
        
        localLlmHandler.inference(prompt, new LocalLlmHandler.LocalLlmCallback() {
            @Override
            public void onToken(String token) {
                // 打印详细日志，包括收到的token内容
                LogManager.logD(TAG, "本地LLM适配器收到流式token[长度" + token.length() + "]: " + 
                      (token.length() > 20 ? token.substring(0, 20) + "..." : token));
                
                // 将token发送给UI
                callback.onStreamingData(token);
                
                // 打印调试信息，确认token已发送
                LogManager.logD(TAG, "流式token已发送到UI");
            }
            
            @Override
            public void onTokenGenerated(String token) {
                // 兼容方法，调用onToken
                onToken(token);
            }
            
            @Override
            public void onComplete(String fullResponse) {
                LogManager.logD(TAG, "本地LLM推理完成，总长度: " + fullResponse.length());
                
                // 打印完整响应的前100个字符，便于调试
                String previewText = fullResponse.length() > 100 ? 
                    fullResponse.substring(0, 100) + "..." : fullResponse;
                LogManager.logD(TAG, "完整响应预览: " + previewText);
                
                // 将完整响应发送给UI
                callback.onSuccess(fullResponse);
                
                // 打印调试信息，确认完整响应已发送
                LogManager.logD(TAG, "完整响应已发送到UI");
            }
            
            @Override
            public void onError(String errorMessage) {
                LogManager.logE(TAG, "本地LLM推理错误: " + errorMessage);
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
        LogManager.logI(TAG, "[停止调试] 收到停止本地LLM生成请求");
        LogManager.logD(TAG, "[停止调试] localLlmHandler状态: " + (localLlmHandler != null ? "已初始化" : "未初始化"));
        
        if (localLlmHandler != null) {
            LogManager.logD(TAG, "[停止调试] 准备调用localLlmHandler.stopInference()");
            localLlmHandler.stopInference();
            LogManager.logI(TAG, "[停止调试] 已调用localLlmHandler.stopInference()");
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
     * 关闭适配器，释放资源
     */
    public void shutdown() {
        if (localLlmHandler != null) {
            localLlmHandler.unloadModel();
        }
        instance = null;
    }
}
