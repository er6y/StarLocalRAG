package com.example.starlocalrag.api;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.example.starlocalrag.ConfigManager;
import com.example.starlocalrag.LogManager;
import com.starlocalrag.llamacpp.LlamaCppInference;
import com.starlocalrag.llamacpp.NativeLibraryLoader;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

// ONNX Runtime相关导入已移除
import java.util.Iterator;
import java.util.Arrays;

/**
 * 本地LLM处理程序
 * 负责加载和管理本地模型，执行本地推理
 * 支持多种模型类型，包括ONNX等
 * 
 * 重构说明：
 * 1. 统一接口规范，明确职责分离
 * 2. 简化回调接口，提高可维护性
 * 3. 为ONNX Runtime GenAI迁移做准备
 */
public class LocalLlmHandler {
    private static final String TAG = "LocalLLMHandler";
    
    // 单例实例
    private static LocalLlmHandler instance;
    
    // 上下文
    private final Context context;
    
    // 线程池
    private final ExecutorService executorService;
    
    // 当前加载的模型名称
    private String currentModelName;
    
    // 模型是否已加载
    private final AtomicBoolean modelLoaded = new AtomicBoolean(false);
    
    // 模型是否正在加载
    private final AtomicBoolean modelLoading = new AtomicBoolean(false);
    
    // 是否使用GPU
    private boolean useGpu = false;
    
    // 推理停止标志
    private final AtomicBoolean shouldStopInference = new AtomicBoolean(false);
    
    // 推理引擎接口（支持多种实现）
    private InferenceEngine inferenceEngine;
    
    // 模型配置
    private ModelConfig modelConfig;
    
    // 词汇表相关字段已不再使用，由 Rust tokenizer 处理
    
    // 特殊token
    private int bosToken = 1;
    private int eosToken = 2;
    private int padToken = 0;
    
    // 最大序列长度
    private int maxSeqLen = 2048;
    
    // 模型类型
    private String modelType = "gguf";
    
    // ONNX相关变量已移除
    
    // 模型配置类
    public static class ModelConfig {
        String modelType; // 模型类型，如"qwen", "deepseek"等
        int vocabSize;    // 词汇表大小
        int hiddenSize;   // 隐藏层大小
        int numLayers;    // 层数
        int numHeads;     // 注意力头数
        private String modelPath; // 模型路径
        private int bosToken;     // 开始标记ID
        private int eosToken;     // 结束标记ID
        
        // 量化相关配置
        private boolean isQuantized = false;     // 是否为量化模型
        private String quantizationType = null; // 量化类型："int8", "int4", "fp16"等
        private float quantizationScale = 1.0f; // 量化缩放因子
        private int quantizationZeroPoint = 0;  // 量化零点
        private boolean enableKVCache = false;  // 是否启用KV缓存，默认禁用以避免兼容性问题
        private int maxBatchSize = 2;           // 最大批处理大小（调查报告建议：从1增加到2-4）
        private int maxSequenceLength = 1024;   // 最大序列长度（动态调整，结合maxSequenceLength配置）
        private Map<String, Object> kvCacheConfig = new HashMap<>(); // KV缓存配置
        
        public ModelConfig(String modelType, int vocabSize, int hiddenSize, int numLayers, int numHeads) {
            this.modelType = modelType;
            this.vocabSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
            this.numHeads = numHeads;
        }
        
        /**
         * 判断模型是否需要注意力掩码
         * @return 是否需要注意力掩码
         */
        public boolean requiresAttentionMask() {
            // 大多数模型都需要注意力掩码
            return true;
        }
        
        /**
         * 判断模型是否需要位置编码
         * @return 是否需要位置编码
         */
        public boolean requiresPositionIds() {
            // 根据模型类型判断是否需要位置编码
            // 例如，某些模型可能使用RoPE等相对位置编码，不需要显式的位置ID
            return "qwen".equalsIgnoreCase(modelType) || "deepseek".equalsIgnoreCase(modelType);
        }
        
        /**
         * 获取模型路径
         * @return 模型路径
         */
        public String getModelPath() {
            return modelPath;
        }
        
        /**
         * 设置模型路径
         * @param modelPath 模型路径
         */
        public void setModelPath(String modelPath) {
            this.modelPath = modelPath;
        }
        
        /**
         * 获取开始标记ID
         * @return 开始标记ID
         */
        public int getBosToken() {
            return bosToken;
        }
        
        /**
         * 设置开始标记ID
         * @param bosToken 开始标记ID
         */
        public void setBosToken(int bosToken) {
            this.bosToken = bosToken;
        }
        
        /**
         * 获取结束标记ID
         * @return 结束标记ID
         */
        public int getEosToken() {
            return eosToken;
        }
        
        /**
         * 设置结束标记ID
         * @param eosToken 结束标记ID
         */
        public void setEosToken(int eosToken) {
            this.eosToken = eosToken;
        }
        
        /**
         * 获取结束标记ID（别名方法）
         * @return 结束标记ID
         */
        public int getEosTokenId() {
            return eosToken;
        }
        
        /**
         * 获取填充标记ID（通常与EOS相同或为0）
         * @return 填充标记ID
         */
        public int getPadTokenId() {
            // 如果没有专门的pad token，通常使用eos token或0
            return eosToken != 0 ? eosToken : 0;
        }
        
        /**
         * 获取注意力头数
         * @return 注意力头数
         */
        public int getNumAttentionHeads() {
            return numHeads;
        }
        
        /**
         * 获取隐藏层大小
         * @return 隐藏层大小
         */
        public int getHiddenSize() {
            return hiddenSize;
        }
        
        /**
         * 获取隐藏层数量
         * @return 隐藏层数量
         */
        public int getNumHiddenLayers() {
            return numLayers;
        }
        
        /**
         * 获取模型类型
         * @return 模型类型
         */
        public String getModelType() {
            return modelType;
        }
        
        /**
         * 获取tokenizer.json文件路径
         * @return tokenizer.json文件路径
         */
        public String getTokenizerJsonPath() {
            if (modelPath == null || modelPath.isEmpty()) {
                return null;
            }
            return modelPath + "/tokenizer.json";
        }
        
        // 量化相关的getter和setter方法
        public boolean isQuantized() {
            return isQuantized;
        }
        
        public void setQuantized(boolean quantized) {
            this.isQuantized = quantized;
        }
        
        public String getQuantizationType() {
            return quantizationType;
        }
        
        public void setQuantizationType(String quantizationType) {
            this.quantizationType = quantizationType;
        }
        
        public float getQuantizationScale() {
            return quantizationScale;
        }
        
        public void setQuantizationScale(float quantizationScale) {
            this.quantizationScale = quantizationScale;
        }
        
        public int getQuantizationZeroPoint() {
            return quantizationZeroPoint;
        }
        
        public void setQuantizationZeroPoint(int quantizationZeroPoint) {
            this.quantizationZeroPoint = quantizationZeroPoint;
        }
        
        public boolean isEnableKVCache() {
            return enableKVCache;
        }
        
        public void setEnableKVCache(boolean enableKVCache) {
            this.enableKVCache = enableKVCache;
        }
        
        public int getMaxBatchSize() {
            return maxBatchSize;
        }
        
        public void setMaxBatchSize(int maxBatchSize) {
            this.maxBatchSize = maxBatchSize;
        }
        
        public int getMaxSequenceLength() {
            return maxSequenceLength;
        }
        
        public void setMaxSequenceLength(int maxSequenceLength) {
            this.maxSequenceLength = maxSequenceLength;
        }
        
        public Map<String, Object> getKvCacheConfig() {
            return kvCacheConfig;
        }
        
        public void setKvCacheConfig(Map<String, Object> kvCacheConfig) {
            this.kvCacheConfig = kvCacheConfig;
        }
        
        /**
         * 动态禁用KV缓存（用于内存不足时的降级策略）
         */
        public void disableKVCache() {
            this.enableKVCache = false;
            LogManager.logW("ModelConfig", "KV缓存已被动态禁用以节省内存");
        }
        
        /**
         * 获取适合当前内存状况的缓存大小
         * @param requestedSize 请求的缓存大小
         * @return 调整后的缓存大小
         */
        public int getAdaptiveCacheSize(int requestedSize) {
            // 获取可用内存
            Runtime runtime = Runtime.getRuntime();
            long maxMemory = runtime.maxMemory();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            long availableMemory = maxMemory - totalMemory + freeMemory;
            
            // 如果可用内存小于50MB，使用最小缓存
            if (availableMemory < 50 * 1024 * 1024) {
                return Math.min(requestedSize, 128);
            }
            // 如果可用内存小于100MB，使用中等缓存
            else if (availableMemory < 100 * 1024 * 1024) {
                return Math.min(requestedSize, 256);
            }
            // 否则使用请求的缓存大小
            else {
                return requestedSize;
            }
        }
    }
    
    /**
     * 本地LLM回调接口
     */
    public interface LocalLlmCallback {
        void onToken(String token);
        void onTokenGenerated(String token); // 添加兼容方法
        void onComplete(String fullResponse);
        void onError(String errorMessage);
    }
    
    /**
     * 获取单例实例
     */
    public static synchronized LocalLlmHandler getInstance(Context context) {
        if (instance == null) {
            instance = new LocalLlmHandler(context);
        }
        return instance;
    }
    
    /**
     * 私有构造函数
     */
    private LocalLlmHandler(Context context) {
        this.context = context.getApplicationContext();
        this.executorService = Executors.newSingleThreadExecutor();
        
        // 初始化GPU设置
        this.useGpu = ConfigManager.getBoolean(context, ConfigManager.KEY_USE_GPU, false);
        LogManager.logD(TAG, "LocalLlmHandler初始化: GPU加速设置为 " + (this.useGpu ? "启用" : "禁用"));
        
        // 推理引擎将在loadModel时根据模型类型自动选择
        // 支持的引擎类型：
        // - LlamaCpp：用于GGUF格式模型
        // - ONNX Runtime GenAI：用于ONNX格式模型（内置分词器）
        // - 传统ONNX Runtime：用于ONNX格式模型（Hugging Face分词器）
        this.inferenceEngine = null;
        LogManager.logI(TAG, "LocalLlmHandler初始化: 推理引擎将根据模型类型自动选择");
    }
    
    /**
     * 设置推理引擎
     * @param engine 推理引擎实例
     */
    public void setInferenceEngine(InferenceEngine engine) {
        if (this.inferenceEngine != null) {
            this.inferenceEngine.release();
        }
        this.inferenceEngine = engine;
        LogManager.logI(TAG, "推理引擎已切换为: " + engine.getEngineType());
    }
    
    // ONNX引擎切换方法已移除
    
    /**
     * 获取当前推理引擎类型
     * @return 引擎类型名称
     */
    public String getCurrentEngineType() {
        return inferenceEngine != null ? inferenceEngine.getEngineType() : "未知";
    }
    
    /**
     * 设置是否使用GPU
     */
    public void setUseGpu(boolean useGpu) {
        this.useGpu = useGpu;
    }
    
    /**
     * 根据配置更新推理引擎
     */
    public void updateEngineFromConfig() {
        // ONNX引擎配置更新逻辑已移除，只支持LlamaCpp引擎
        LogManager.logI(TAG, "只支持LlamaCpp引擎，无需更新引擎配置");
    }
    
    /**
     * 加载本地模型（简化版接口）
     * @param modelName 模型名称（目录名）
     * @param callback 回调接口
     */
    public void loadModel(String modelName, final StreamingCallback callback) {
        loadModel(modelName, new LocalLlmCallback() {
            @Override
            public void onToken(String token) {
                callback.onToken(token);
            }
            
            @Override
            public void onTokenGenerated(String token) {
                callback.onToken(token);
            }
            
            @Override
            public void onComplete(String fullResponse) {
                callback.onComplete(fullResponse);
            }
            
            @Override
            public void onError(String errorMessage) {
                callback.onError(errorMessage);
            }
        });
    }
    
    /**
     * 加载本地模型（兼容旧接口）
     * @param modelName 模型名称（目录名）
     * @param callback 回调接口
     */
    @Deprecated
    public void loadModel(String modelName, final LocalLlmCallback callback) {
        // 如果已经加载了相同的模型，直接返回
        if (modelLoaded.get() && modelName.equals(currentModelName)) {
            if (callback != null) {
                callback.onComplete("模型已加载: " + modelName);
            }
            return;
        }
        
        // 如果正在加载模型，返回
        if (modelLoading.get()) {
            if (callback != null) {
                callback.onError("Model is loading, please try again later");
            }
            return;
        }
        
        // 标记为正在加载
        modelLoading.set(true);
        currentModelName = modelName;
        
        // 在后台线程中加载模型
        executorService.execute(() -> {
            try {
                // 释放之前的资源
                if (inferenceEngine != null) {
                    inferenceEngine.release();
                }
                
                modelLoaded.set(false);
                
                // 1. 确保模型文件存在
                String modelPath = ConfigManager.getModelPath(context);
                File modelDir = new File(modelPath, modelName);
                
                if (!modelDir.exists() || !modelDir.isDirectory()) {
                    throw new IOException("Model file does not exist: " + modelDir.getAbsolutePath());
                }
                
                // 2. 检测模型类型并选择推理引擎
                InferenceEngine selectedEngine = selectInferenceEngine(modelDir);
                if (selectedEngine == null) {
                    throw new IOException("Unable to determine model type or unsupported model format: " + modelDir.getAbsolutePath());
                }
                
                // 如果推理引擎发生变化，释放旧引擎
                if (inferenceEngine != null && !inferenceEngine.getEngineType().equals(selectedEngine.getEngineType())) {
                    inferenceEngine.release();
                }
                inferenceEngine = selectedEngine;
                
                // 3. 加载模型配置（只支持LlamaCpp）
                ModelConfig modelConfig = createBasicModelConfig(modelDir.getAbsolutePath());
                
                // 4. 初始化推理引擎
                inferenceEngine.initialize(modelDir.getAbsolutePath(), modelConfig);
                
                // 4. 标记模型已加载
                modelLoaded.set(true);
                modelLoading.set(false);
                
                LogManager.logI(TAG, "✓ 模型加载成功: " + modelName + " (引擎: " + inferenceEngine.getEngineType() + ")");
                
                if (callback != null) {
                    callback.onComplete("Model loaded successfully: " + modelName);
                }
                
            } catch (Exception e) {
                modelLoaded.set(false);
                modelLoading.set(false);
                
                String errorMsg = "Model loading failed: " + e.getMessage();
                LogManager.logE(TAG, errorMsg, e);
                
                if (callback != null) {
                    callback.onError(errorMsg);
                }
            }
        });
    }
    
    /**
     * 执行推理（简化版接口）
     * @param prompt 输入提示词
     * @param callback 流式回调
     */
    public void inference(String prompt, StreamingCallback callback) {
        InferenceParams params = new InferenceParams();
        // 从配置中获取参数
        params.setThinkingMode(!ConfigManager.getNoThinking(context));
        
        inference(prompt, params, callback);
    }
    
    /**
     * 执行推理（完整参数版本）
     * @param prompt 输入提示词
     * @param params 推理参数
     * @param callback 流式回调
     */
    public void inference(String prompt, InferenceParams params, StreamingCallback callback) {
        if (!modelLoaded.get()) {
            callback.onError("模型未加载，请先加载模型");
            return;
        }
        
        if (inferenceEngine == null) {
            callback.onError("推理引擎未初始化");
            return;
        }
        
        // 重置停止标志
        resetStopFlag();
        
        LogManager.logD(TAG, "开始推理，引擎: " + inferenceEngine.getEngineType() + ", 提示词长度: " + prompt.length());
        // 调试追踪日志已移除
        
        // 执行推理
        inferenceEngine.inference(prompt, params, callback);
    }
    
    /**
     * 执行推理（兼容旧接口）
     * @param prompt 输入提示词
     * @param callback 回调接口
     */
    @Deprecated
    public void inference(String prompt, LocalLlmCallback callback) {
        inference(prompt, (StreamingCallback) callback);
    }

    /**
     * 批处理推理 - 支持多序列并行推理
     * @param inputTexts 输入文本数组
     * @param maxTokens 最大生成token数
     * @param temperature 温度参数
     * @param topK topK参数
     * @param topP topP参数
     * @param callback 流式回调
     * @return 生成的文本数组
     */
    public String[] inferenceStreamBatch(String[] inputTexts, int maxTokens, float temperature, int topK, float topP, LocalLlmCallback callback) {
        LogManager.logE(TAG, "批处理推理不支持，ONNX引擎已移除");
        String[] errorResults = new String[inputTexts.length];
        for (int i = 0; i < inputTexts.length; i++) {
            errorResults[i] = "批处理推理不支持";
        }
        return errorResults;
    }
    
    /**
     * 批处理推理（简化版本）
     * @param inputTexts 输入文本数组
     * @param callback 回调接口
     * @return 生成的文本数组
     */
    public String[] inferenceStreamBatch(String[] inputTexts, LocalLlmCallback callback) {
        // 使用默认参数
        return inferenceStreamBatch(inputTexts, 512, 0.7f, 40, 0.9f, callback);
    }
    
    /**
     * 获取当前模型的批处理能力信息
     * @return 批处理信息字符串
     */
    public String getBatchProcessingInfo() {
        if (modelConfig == null) {
            return "模型未加载";
        }
        
        StringBuilder info = new StringBuilder();
        info.append("批处理支持: ").append(modelConfig.getMaxBatchSize() > 1 ? "是" : "否").append("\n");
        info.append("最大批处理大小: ").append(modelConfig.getMaxBatchSize()).append("\n");
        info.append("最大序列长度: ").append(modelConfig.getMaxSequenceLength()).append("\n");
        info.append("KV缓存: ").append(modelConfig.isEnableKVCache() ? "启用" : "禁用").append("\n");
        info.append("量化类型: ").append(modelConfig.isQuantized() ? modelConfig.getQuantizationType() : "无").append("\n");
        
        return info.toString();
    }
    
    /**
     * 停止当前推理
     */
    public void stopInference() {
        // 停止调试日志已移除
        LogManager.logD(TAG, "[停止调试] 停止标志当前状态: " + shouldStopInference.get());
        
        // 首先调用当前推理引擎的stopInference方法
        if (inferenceEngine != null) {
            // 停止调试日志已移除
            try {
                inferenceEngine.stopInference();
            } catch (Exception e) {
                LogManager.logE(TAG, "推理引擎stopInference调用失败: " + e.getMessage());
            }
        } else {
            LogManager.logW(TAG, "推理引擎为null，无法调用其stopInference方法");
        }
        
        // 然后设置本地停止标志
        shouldStopInference.set(true);
        // 停止调试日志已移除
    }
    
    /**
     * 检查是否应该停止推理
     */
    public boolean shouldStopInference() {
        return shouldStopInference.get();
    }
    
    /**
     * 重置停止标志
     */
    public void resetStopFlag() {
        shouldStopInference.set(false);
    }
    
    /**
     * 重置模型记忆 - 清除KV缓存和对话历史
     */
    public void resetModelMemory() {
        // 新对话调试日志已移除
        
        if (!modelLoaded.get()) {
            LogManager.logW(TAG, "模型未加载，无法重置记忆");
            return;
        }
        
        try {
            // 根据推理引擎类型调用相应的重置方法
            if (inferenceEngine instanceof LocalLLMLlamaCppHandler) {
                ((LocalLLMLlamaCppHandler) inferenceEngine).resetModelMemory();
            } else {
                LogManager.logW(TAG, "未知的推理引擎类型，无法重置记忆");
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "重置模型记忆失败", e);
        }
    }
    
    /**
     * 检查模型是否已加载
     * @return true如果模型已加载，false否则
     */
    public boolean isModelLoaded() {
        return modelLoaded.get();
    }
    
    /**
     * 获取当前加载的模型名称
     * @return 当前模型名称，如果没有加载模型则返回null
     */
    public String getCurrentModelName() {
        return currentModelName;
    }
    
    /**
     * 卸载模型
     */
    public void unloadModel() {
        // 卸载模型逻辑
        LogManager.logD(TAG, "卸载模型");
        // ONNX Handler已移除
        // 设置模型卸载标志
        modelLoaded.set(false);
        
        // 释放内存
        System.gc();
        LogManager.logD(TAG, "已请求垃圾回收");
    }
    
    /**
     * 根据配置的最大序列长度动态计算实际序列长度
     * @param configuredMaxSeqLength 配置的最大序列长度
     * @return 动态计算的序列长度
     */
    private int calculateDynamicSequenceLength(int configuredMaxSeqLength) {
        // 基础输入长度预留（用于提示词、上下文等）
        int baseInputLength = 512;
        
        // 计算总序列长度：基础输入 + 配置的最大长度 + 安全边距
        int calculatedLength = baseInputLength + configuredMaxSeqLength + 128;
        
        // 设置合理的范围限制
        int minLength = 1024;  // 最小序列长度
        int maxLength = 8192;  // 最大序列长度（考虑内存限制）
        
        // 应用范围限制
        calculatedLength = Math.max(minLength, Math.min(maxLength, calculatedLength));
        
        LogManager.logD(TAG, String.format("序列长度计算: 基础输入=%d, 配置值=%d, 计算结果=%d", 
            baseInputLength, configuredMaxSeqLength, calculatedLength));
        
        return calculatedLength;
    }
    
    /**
     * 智能检测模型是否支持KV缓存
     * 通过检查模型输入中是否包含KV缓存相关的输入来判断
     * @param inputInfo 模型输入信息
     * @return true如果模型支持KV缓存，false否则
     */
    private boolean detectKVCacheSupport(Map<String, ai.onnxruntime.NodeInfo> inputInfo) {
        if (inputInfo == null || inputInfo.isEmpty()) {
            LogManager.logW(TAG, "模型输入信息为空，无法检测KV缓存支持");
            return false;
        }
        
        // 检查是否包含KV缓存相关的输入名称
        // 修复：移除attention_mask，它是正常输入不是KV缓存输入
        // 更严格的KV缓存输入名称模式
        String[] kvCachePatterns = {
            "past_key_values",
            "past_key",
            "past_value", 
            "cache_length",
            "key_cache",
            "value_cache",
            "kv_cache"
        };
        
        int kvCacheInputCount = 0;
        int totalInputs = inputInfo.size();
        
        LogManager.logD(TAG, "开始检测KV缓存支持，总输入数量: " + totalInputs);
        
        for (String inputName : inputInfo.keySet()) {
            String lowerInputName = inputName.toLowerCase();
            
            // 检查是否匹配KV缓存模式
            for (String pattern : kvCachePatterns) {
                if (lowerInputName.contains(pattern.toLowerCase())) {
                    kvCacheInputCount++;
                    LogManager.logD(TAG, "发现KV缓存相关输入: " + inputName + " (匹配模式: " + pattern + ")");
                    break;
                }
            }
        }
        
        // 修复判断逻辑：
        // 1. 必须明确发现KV缓存相关的输入名称才认为支持
        // 2. 基础输入通常只有input_ids和attention_mask（1-3个）
        // 3. 移除基于输入数量的模糊判断，避免误判
        
        boolean supportsKVCache = false;
        
        if (kvCacheInputCount > 0) {
            supportsKVCache = true;
            LogManager.logI(TAG, "检测结果: 模型支持KV缓存 (发现 " + kvCacheInputCount + " 个KV缓存相关输入)");
        } else {
            supportsKVCache = false;
            LogManager.logI(TAG, "检测结果: 模型不支持KV缓存 (输入数量: " + totalInputs + ", KV缓存输入: " + kvCacheInputCount + ")");
        }
        
        // 详细记录所有输入名称用于调试
        StringBuilder inputNames = new StringBuilder("所有模型输入: ");
        for (String inputName : inputInfo.keySet()) {
            inputNames.append(inputName).append(", ");
        }
        LogManager.logD(TAG, inputNames.toString());
        
        return supportsKVCache;
    }
    
    // ...
    private void logMemoryInfo() {
        // 记录内存信息逻辑
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        
        LogManager.logD(TAG, "内存信息 - 最大: " + maxMemory / 1024 / 1024 + "MB, 总计: " + totalMemory / 1024 / 1024 + "MB, 已用: " + usedMemory / 1024 / 1024 + "MB, 空闲: " + freeMemory / 1024 / 1024 + "MB");
        
        // 垃圾回收优化：当内存使用率超过75%时，建议进行垃圾回收
        double memoryUsageRatio = (double) usedMemory / maxMemory;
        if (memoryUsageRatio > 0.75) {
            LogManager.logW(TAG, "内存使用率较高 (" + String.format("%.1f", memoryUsageRatio * 100) + "%)，建议进行垃圾回收");
            // 建议垃圾回收，但不强制执行，让系统自行决定
            System.gc();
            LogManager.logI(TAG, "已建议系统进行垃圾回收以优化内存使用");
        }
    }

    // ONNX Runtime GenAI推理引擎已移除

    // ONNX Runtime推理引擎已移除

    /**
     * 加载模型配置
     * @param configFile 配置文件
     * @return 模型配置对象
     * @throws Exception 异常
     */
    private ModelConfig loadModelConfig(File configFile) throws Exception {
        LogManager.logD(TAG, "加载模型配置: " + configFile.getPath());
        
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(configFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
        }
        
        // 检查文件内容是否为空
        String configContent = content.toString().trim();
        if (configContent.isEmpty()) {
            throw new Exception("配置文件为空: " + configFile.getPath());
        }
        
        LogManager.logD(TAG, "配置文件内容长度: " + configContent.length() + " 字符");
        LogManager.logD(TAG, "配置文件前100字符: " + configContent.substring(0, Math.min(100, configContent.length())));
        
        JSONObject config = new JSONObject(configContent);
        
        // 检查是否为ONNX Runtime GenAI格式 (genai_config.json)
        boolean isGenAIConfig = configFile.getName().equals("genai_config.json") || config.has("model");
        
        String modelType;
        int vocabSize;
        int hiddenSize;
        int numLayers;
        int numHeads;
        
        if (isGenAIConfig) {
            // ONNX Runtime GenAI格式 - 使用库内置解析，仅设置基本信息
            LogManager.logI(TAG, "检测到ONNX Runtime GenAI配置格式，使用库内置解析");
            
            // 对于GenAI格式，只需要设置基本信息，具体配置由ONNX Runtime GenAI库自动处理
            modelType = "genai";
            vocabSize = 32000;  // 默认值，实际值由库从配置中读取
            hiddenSize = 4096;  // 默认值，实际值由库从配置中读取
            numLayers = 32;     // 默认值，实际值由库从配置中读取
            numHeads = 32;      // 默认值，实际值由库从配置中读取
        } else {
            // 传统HuggingFace格式解析
            LogManager.logI(TAG, "解析传统HuggingFace配置格式");
            modelType = config.optString("model_type", "unknown");
            vocabSize = config.optInt("vocab_size", 32000);
            hiddenSize = config.optInt("hidden_size", 4096);
            numLayers = config.optInt("num_hidden_layers", 32);
            numHeads = config.optInt("num_attention_heads", 32);
        }
        
        ModelConfig modelConfig = new ModelConfig(modelType, vocabSize, hiddenSize, numLayers, numHeads);
        // 设置模型路径
        modelConfig.setModelPath(configFile.getParentFile().getAbsolutePath());
        
        // 获取特殊token
        if (isGenAIConfig) {
            // ONNX Runtime GenAI格式 - 特殊token由库自动处理，无需手动解析
            LogManager.logI(TAG, "特殊token将由ONNX Runtime GenAI库自动处理");
        } else {
            // 传统格式的token配置
            if (config.has("bos_token_id")) {
                int bosToken = config.getInt("bos_token_id");
                modelConfig.setBosToken(bosToken);
                LogManager.logD(TAG, "设置BOS token: " + bosToken);
            }
            if (config.has("eos_token_id")) {
                int eosToken = config.getInt("eos_token_id");
                modelConfig.setEosToken(eosToken);
                LogManager.logD(TAG, "设置EOS token: " + eosToken);
            }
        }
        
        // 解析量化相关配置
        parseQuantizationConfig(config, modelConfig, configFile);
        
        // 动态调整maxSequenceLength
        int configuredMaxSeqLength = ConfigManager.getMaxSequenceLength(context);
        modelConfig.setMaxSequenceLength(configuredMaxSeqLength);
        
        LogManager.logD(TAG, String.format("模型配置: 类型=%s, 词汇表大小=%d, 隐藏层大小=%d, 层数=%d, 注意力头数=%d",
            modelType, vocabSize, hiddenSize, numLayers, numHeads));
        
        return modelConfig;
    }
    
    /**
     * 检查是否为GGUF模型
     * @param modelDir 模型目录
     * @return 是否为GGUF模型
     */
    private boolean isGgufModel(File modelDir) {
        if (!modelDir.exists() || !modelDir.isDirectory()) {
            return false;
        }
        
        // 检查目录中是否有.gguf文件
        File[] files = modelDir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isFile() && file.getName().toLowerCase().endsWith(".gguf")) {
                    return true;
                }
            }
        }
        return false;
    }
    
    /**
     * 根据模型目录内容选择合适的推理引擎
     * @param modelDir 模型目录
     * @return 推理引擎实例
     */
    private InferenceEngine selectInferenceEngine(File modelDir) {
        LogManager.logI(TAG, "检测模型类型: " + modelDir.getAbsolutePath());
        
        // 检查是否为GGUF模型
        if (isGgufModel(modelDir)) {
            LogManager.logI(TAG, "检测到GGUF模型，选择LlamaCpp推理引擎");
            return new LocalLLMLlamaCppHandler(context);
        }
        
        // ONNX模型支持已移除，只支持GGUF模型
        LogManager.logW(TAG, "只支持GGUF模型，ONNX模型支持已移除");
        return null;
    }
    
    /**
     * 为LlamaCpp模型创建基本配置
     * @param modelPath 模型路径
     * @return 模型配置
     */
    private ModelConfig createBasicModelConfig(String modelPath) {
        LogManager.logI(TAG, "为LlamaCpp模型创建基本配置");
        
        // 创建基本配置
        ModelConfig config = new ModelConfig("llamacpp", 32000, 4096, 32, 32);
        config.setModelPath(modelPath);
        
        // 设置序列长度
        int configuredMaxSeqLength = ConfigManager.getMaxSequenceLength(context);
        config.setMaxSequenceLength(configuredMaxSeqLength);
        
        // 设置基本token
        config.setBosToken(1);
        config.setEosToken(2);
        
        LogManager.logD(TAG, "LlamaCpp基本配置创建完成，最大序列长度: " + configuredMaxSeqLength);
        
        return config;
    }
    
    /**
     * 解析量化配置参数
     * @param config JSON配置对象
     * @param modelConfig 模型配置对象
     * @param configFile 配置文件
     */
    private void parseQuantizationConfig(JSONObject config, ModelConfig modelConfig, File configFile) {
        try {
            // 检查模型文件名是否包含量化标识
            String modelFileName = configFile.getParentFile().getName().toLowerCase();
            boolean isQuantizedByName = modelFileName.contains("int8") || modelFileName.contains("int4") || 
                                      modelFileName.contains("quant") || modelFileName.contains("quantized");
            
            // 从配置文件中读取量化信息
            boolean isQuantized = config.optBoolean("quantized", isQuantizedByName);
            modelConfig.setQuantized(isQuantized);
            
            if (isQuantized) {
                // 确定量化类型
                String quantType = config.optString("quantization_type", "");
                if (quantType.isEmpty()) {
                    // 从文件名推断量化类型
                    if (modelFileName.contains("int8")) {
                        quantType = "int8";
                    } else if (modelFileName.contains("int4")) {
                        quantType = "int4";
                    } else if (modelFileName.contains("fp16")) {
                        quantType = "fp16";
                    } else {
                        quantType = "int8"; // 默认为int8动态量化
                    }
                }
                modelConfig.setQuantizationType(quantType);
                
                // 读取量化参数
                if (config.has("quantization_config")) {
                    JSONObject quantConfig = config.getJSONObject("quantization_config");
                    
                    // 量化缩放因子
                    float scale = (float) quantConfig.optDouble("scale", 1.0);
                    modelConfig.setQuantizationScale(scale);
                    
                    // 量化零点
                    int zeroPoint = quantConfig.optInt("zero_point", 0);
                    modelConfig.setQuantizationZeroPoint(zeroPoint);
                } else {
                    // 使用默认量化参数
                    if ("int8".equals(quantType)) {
                        modelConfig.setQuantizationScale(0.1f);
                        modelConfig.setQuantizationZeroPoint(128);
                    } else if ("int4".equals(quantType)) {
                        modelConfig.setQuantizationScale(0.2f);
                        modelConfig.setQuantizationZeroPoint(8);
                    }
                }
                
                // KV缓存配置
                boolean enableKVCache = config.optBoolean("enable_kv_cache", false);
                modelConfig.setEnableKVCache(enableKVCache);
                
                // 批处理配置
                int maxBatchSize = config.optInt("max_batch_size", 1);
                modelConfig.setMaxBatchSize(maxBatchSize);
                
                LogManager.logI(TAG, "检测到量化模型，类型: " + quantType + ", 启用优化配置");
            }
            
        } catch (Exception e) {
            LogManager.logW(TAG, "解析量化配置失败，使用默认配置: " + e.getMessage());
            // 默认启用int8动态量化
            modelConfig.setQuantized(true);
            modelConfig.setQuantizationType("int8");
            modelConfig.setQuantizationScale(0.1f);
            modelConfig.setQuantizationZeroPoint(128);
        }
    }


    /**
     * 推理引擎接口
     * 统一不同推理后端的调用方式
     */
    public interface InferenceEngine {
        /**
         * 初始化推理引擎
         * @param modelPath 模型路径
         * @param config 模型配置
         * @throws Exception 初始化异常
         */
        void initialize(String modelPath, ModelConfig config) throws Exception;
        
        /**
         * 执行推理
         * @param prompt 输入提示词
         * @param params 推理参数
         * @param callback 流式回调
         */
        void inference(String prompt, InferenceParams params, StreamingCallback callback);
        
        /**
         * 停止推理
         */
        void stopInference();
        
        /**
         * 释放资源
         */
        void release();
        
        /**
         * 获取引擎类型
         * @return 引擎类型名称
         */
        String getEngineType();
    }
    
    /**
     * 推理参数类
     * 集中管理推理相关参数
     */
    public static class InferenceParams {
        private int maxTokens = 512;
        private float temperature = 0.7f;
        private int topK = 40;
        private float topP = 0.9f;
        private boolean thinkingMode = true;
        private float repetitionPenalty = 1.1f;
        private int seed = -1; // -1表示随机种子
        
        // Getter和Setter方法
        public int getMaxTokens() { return maxTokens; }
        public void setMaxTokens(int maxTokens) { this.maxTokens = maxTokens; }
        
        public int getMaxTokenLength() { return maxTokens; } // 兼容方法
        
        public float getTemperature() { return temperature; }
        public void setTemperature(float temperature) { this.temperature = temperature; }
        
        public int getTopK() { return topK; }
        public void setTopK(int topK) { this.topK = topK; }
        
        public float getTopP() { return topP; }
        public void setTopP(float topP) { this.topP = topP; }
        
        public boolean isThinkingMode() { return thinkingMode; }
        public void setThinkingMode(boolean thinkingMode) { this.thinkingMode = thinkingMode; }
        
        public float getRepetitionPenalty() { return repetitionPenalty; }
        public void setRepetitionPenalty(float repetitionPenalty) { this.repetitionPenalty = repetitionPenalty; }
        
        public int getSeed() { return seed; }
        public void setSeed(int seed) { this.seed = seed; }
    }
    
    /**
     * 流式回调接口（简化版）
     * 统一回调接口，简化使用
     */
    public interface StreamingCallback {
        /**
         * 生成新token时调用
         * @param token 生成的token
         */
        void onToken(String token);
        
        /**
         * 推理完成时调用
         * @param fullResponse 完整响应
         */
        void onComplete(String fullResponse);
        
        /**
         * 发生错误时调用
         * @param errorMessage 错误信息
         */
        void onError(String errorMessage);
    }
}
