package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

import com.example.starlocalrag.ConfigManager;
import com.example.starlocalrag.LogManager;
import com.starlocalrag.llamacpp.LlamaCppInference;
import com.starlocalrag.llamacpp.LlamaCppEmbedding;
import com.starlocalrag.llamacpp.NativeLibraryLoader;

import java.io.File;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * LlamaCpp本地LLM推理处理器
 * 直接对标LocalLLMOnnxRuntimeGenAIHandler，实现统一的接口
 * 负责管理llama.cpp模型的加载、推理和资源释放
 * 
 * 主要特性：
 * 1. 支持GGUF格式模型的文本生成和词嵌入
 * 2. 流式文本生成和完整的资源管理
 * 3. 实现token输出总量和速率统计
 * 4. 统一的错误处理和日志记录
 * 
 * @author StarLocalRAG Team
 * @version 2.0
 */
public class LocalLLMLlamaCppHandler implements LocalLlmHandler.InferenceEngine {
    private static final String TAG = "LocalLLMLlamaCppHandler";
    
    // 上下文
    private final Context context;
    
    // LlamaCpp推理引擎句柄
    private long modelHandle = 0;
    private long contextHandle = 0;
    private LlamaCppEmbedding llamaCppEmbedding;
    private ExecutorService executorService;
    
    // 状态管理
    private final AtomicBoolean isInitialized = new AtomicBoolean(false);
    private final AtomicBoolean isGenerating = new AtomicBoolean(false);
    private final AtomicBoolean shouldStop = new AtomicBoolean(false);
    
    // 统计信息
    private final AtomicInteger totalTokensGenerated = new AtomicInteger(0);
    private long generationStartTime = 0;
    
    // 模型配置
    private LocalLlmHandler.ModelConfig modelConfig;
    private String currentModelPath;
    private int maxTokens = 512; // 配置参数 - 从ConfigManager获取
    
    public LocalLLMLlamaCppHandler(Context context) {
        this.context = context;
        this.executorService = Executors.newSingleThreadExecutor();
        
        // 加载本地库
        try {
            NativeLibraryLoader.loadLibrary("llamacpp_jni");
            LogManager.logI(TAG, "LlamaCpp本地库加载成功");
        } catch (Exception e) {
            LogManager.logE(TAG, "加载llama.cpp本地库失败", e);
            throw new RuntimeException("Failed to load llama.cpp native libraries");
        }
        
        LogManager.logI(TAG, "LocalLLMLlamaCppHandler初始化完成");
    }
    
    @Override
    public void initialize(String modelPath, LocalLlmHandler.ModelConfig config) throws Exception {
        if (isInitialized.get()) {
            LogManager.logW(TAG, "推理引擎已经初始化");
            return;
        }
        
        LogManager.logI(TAG, "开始初始化LlamaCpp推理引擎: " + modelPath);
        
        this.modelConfig = config;
        this.currentModelPath = modelPath;
        
        // 验证模型文件
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            throw new Exception("模型文件不存在: " + modelPath);
        }
        
        initializeLlamaCpp(modelPath, config);
        
        isInitialized.set(true);
        LogManager.logI(TAG, "LlamaCpp推理引擎初始化成功");
    }
    
    /**
     * 初始化LlamaCpp推理引擎
     */
    private void initializeLlamaCpp(String modelPath, LocalLlmHandler.ModelConfig config) throws Exception {
        LogManager.logI(TAG, "使用LlamaCpp API初始化...");
        
        // 查找GGUF模型文件
        File modelFile = new File(modelPath);
        File ggufFile = findGgufFile(modelFile);
        if (ggufFile == null) {
            throw new Exception("在目录中未找到GGUF模型文件: " + modelPath);
        }
        
        LogManager.logI(TAG, "找到GGUF模型文件: " + ggufFile.getAbsolutePath());
        
        // 从ConfigManager获取配置参数 - 参考OnnxRuntimeGenAI的做法
        int configMaxSeqLength = ConfigManager.getMaxSequenceLength(context);
        int configThreads = ConfigManager.getThreads(context);
        int configMaxNewTokens = ConfigManager.getMaxNewTokens(context);
        boolean configUseGpu = ConfigManager.getBoolean(context, ConfigManager.KEY_USE_GPU, false);
        
        // 线程数配置 - 与OnnxRuntimeGenAI对齐：MIN(CPU核心数, getThreads)
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int actualThreads = Math.min(configThreads, availableProcessors);
        
        // 设置参数
        maxTokens = configMaxNewTokens; // 使用最大输出token数作为maxTokens
        int contextSize = configMaxSeqLength; // 上下文大小应该是最大序列长度，不是输出token数
        int nGpuLayers = configUseGpu ? -1 : 0; // 根据配置设置GPU层数
        
        LogManager.logI(TAG, String.format("线程配置 - 用户配置: %d线程, CPU核心: %d, 实际使用: %d线程", 
            configThreads, availableProcessors, actualThreads));
        LogManager.logI(TAG, String.format("GPU配置 - 使用GPU: %s, GPU层数: %d", 
            configUseGpu ? "是" : "否", nGpuLayers));
        
        // 使用新的静态方法初始化
        // 1. 初始化后端
        LlamaCppInference.backend_init();
        
        // 2. 加载模型（使用GPU层数参数）
        modelHandle = LlamaCppInference.load_model_with_gpu(ggufFile.getAbsolutePath(), nGpuLayers);
        if (modelHandle == 0) {
            throw new RuntimeException("模型加载失败: " + ggufFile.getAbsolutePath());
        }
        
        // 3. 创建上下文（GPU层数已在模型加载时设置）
        contextHandle = LlamaCppInference.new_context_with_params(modelHandle, contextSize, actualThreads, 0);
        if (contextHandle == 0) {
            LlamaCppInference.free_model(modelHandle);
            throw new RuntimeException("上下文创建失败");
        }
        
        LogManager.logI(TAG, "✓ LlamaCpp引擎初始化完成");
        LogManager.logI(TAG, "模型句柄: " + modelHandle + ", 上下文句柄: " + contextHandle);
        LogManager.logI(TAG, String.format("配置参数 - 最大序列长度: %d, 线程数: %d, 最大输出token数: %d, GPU加速: %s", 
            configMaxSeqLength, actualThreads, configMaxNewTokens, configUseGpu ? "启用" : "禁用"));
    }
    
    /**
     * 查找GGUF模型文件
     */
    private File findGgufFile(File modelDir) {
        if (modelDir.isFile() && modelDir.getName().toLowerCase().endsWith(".gguf")) {
            return modelDir;
        }
        
        if (modelDir.isDirectory()) {
            File[] files = modelDir.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile() && file.getName().toLowerCase().endsWith(".gguf")) {
                        return file;
                    }
                }
            }
        }
        
        return null;
    }
    
    @Override
    public void inference(String prompt, LocalLlmHandler.InferenceParams params, 
                         LocalLlmHandler.StreamingCallback callback) {
        generateText(prompt, params, callback);
    }
    
    public void generateText(String prompt, LocalLlmHandler.InferenceParams params, 
                           LocalLlmHandler.StreamingCallback callback) {
        if (!isInitialized.get()) {
            String error = "推理引擎未初始化";
            LogManager.logE(TAG, error);
            if (callback != null) {
                callback.onError(error);
            }
            return;
        }
        
        if (isGenerating.get()) {
            String error = "正在生成中，请等待当前任务完成";
            LogManager.logW(TAG, error);
            if (callback != null) {
                callback.onError(error);
            }
            return;
        }
        
        // 在后台线程执行推理
        executorService.submit(() -> {
            try {
                isGenerating.set(true);
                shouldStop.set(false);
                generationStartTime = System.currentTimeMillis();
                totalTokensGenerated.set(0);
                
                LogManager.logI(TAG, "开始LlamaCpp流式生成文本");
                
                // 更新推理参数
                if (params != null) {
                    updateInferenceParameters(params);
                }
                
                StringBuilder fullResponse = new StringBuilder();
                
                generateWithLlamaCpp(prompt, params, callback, fullResponse);
                
            } catch (Exception e) {
                LogManager.logE(TAG, "生成文本异常", e);
                if (callback != null) {
                    callback.onError("生成文本时发生异常: " + e.getMessage());
                }
            } finally {
                isGenerating.set(false);
            }
        });
    }
    
    /**
     * 使用LlamaCpp进行推理
     */
    private void generateWithLlamaCpp(String prompt, LocalLlmHandler.InferenceParams params, LocalLlmHandler.StreamingCallback callback, StringBuilder fullResponse) {
        try {
            // 从ConfigManager获取配置参数
            int maxTokens = ConfigManager.getMaxNewTokens(context);
            
            LogManager.logI(TAG, String.format("推理参数 - MaxTokens: %d", maxTokens));
            
            // 动态创建批处理大小，支持超长prompt
            // 首先估算prompt的token数量（粗略估算：字符数/4）
            int estimatedTokens = Math.max(512, prompt.length() / 4 + 100);
            // 限制最大batch大小避免内存问题
            int batchSize = Math.min(estimatedTokens, 2048);
            
            LogManager.logI(TAG, String.format("动态batch大小 - 估算tokens: %d, 使用batch大小: %d", estimatedTokens, batchSize));
            
            long batch = LlamaCppInference.new_batch(batchSize, 0, 1);
            long sampler = params != null ? 
                LlamaCppInference.new_sampler_with_params(
                    params.getTemperature(),
                    params.getTopP(),
                    params.getTopK()
                ) : LlamaCppInference.new_sampler();
            
            try {
                // 清空KV缓存
                LlamaCppInference.kv_cache_clear(contextHandle);
                
                LogManager.logI(TAG, String.format("[调试] 调用completion_init - contextHandle=%d, batch=%d, prompt长度=%d, maxTokens=%d, batchSize=%d", 
                    contextHandle, batch, prompt.length(), maxTokens, 512));
                LogManager.logI(TAG, String.format("[调试] 提示词内容: '%s'", prompt));
                
                // 初始化完成
                int tokenCount = LlamaCppInference.completion_init(contextHandle, batch, prompt, maxTokens, false);
                if (tokenCount < 0) {
                    LogManager.logE(TAG, "初始化完成失败");
                    return;
                }
                
                // 生成循环
                LlamaCppInference.IntVar currentPos = new LlamaCppInference.IntVar(tokenCount);
                
                for (int i = 0; i < maxTokens && !shouldStop.get(); i++) {
                    String token = LlamaCppInference.completion_loop(contextHandle, batch, sampler, maxTokens, currentPos);
                    if (token == null || token.isEmpty()) {
                        break;
                    }
                    
                    fullResponse.append(token);
                    totalTokensGenerated.incrementAndGet();
                    if (callback != null) {
                        callback.onToken(token);
                    }
                }
            } finally {
                // 清理资源
                LlamaCppInference.free_batch(batch);
                LlamaCppInference.free_sampler(sampler);
            }
            
            if (!shouldStop.get() && callback != null) {
                // 计算统计信息
                long duration = System.currentTimeMillis() - generationStartTime;
                int tokens = totalTokensGenerated.get();
                double tokensPerSecond = tokens > 0 && duration > 0 ? 
                    (tokens * 1000.0) / duration : 0.0;
                
                LogManager.logI(TAG, String.format(
                    "生成完成 - 总tokens: %d, 耗时: %dms, 速率: %.2f tokens/s",
                    tokens, duration, tokensPerSecond));
                
                callback.onComplete(fullResponse.toString());
            }
            
        } catch (Exception e) {
            LogManager.logE(TAG, "推理失败: " + e.getMessage(), e);
            if (callback != null) {
                callback.onError("推理失败: " + e.getMessage());
            }
        }
    }
    
    /**
     * 备用方法：使用传统流式生成（如果一站式API有问题）
     */
    private void generateWithTraditionalStreaming(String prompt, LocalLlmHandler.InferenceParams params, LocalLlmHandler.StreamingCallback callback, StringBuilder fullResponse) {
        try {
            // 使用底层JNI方法进行流式生成
            int maxTokens = ConfigManager.getMaxNewTokens(context);
            
            // 动态创建批处理大小，支持超长prompt
            // 首先估算prompt的token数量（粗略估算：字符数/4）
            int estimatedTokens = Math.max(512, prompt.length() / 4 + 100);
            // 限制最大batch大小避免内存问题
            int batchSize = Math.min(estimatedTokens, 2048);
            
            LogManager.logI(TAG, String.format("动态batch大小 - 估算tokens: %d, 使用batch大小: %d", estimatedTokens, batchSize));
            
            long batch = LlamaCppInference.new_batch(batchSize, 0, 1);
            long sampler = params != null ? 
                LlamaCppInference.new_sampler_with_params(
                    params.getTemperature(),
                    params.getTopP(),
                    params.getTopK()
                ) : LlamaCppInference.new_sampler();
            
            try {
                // 清空KV缓存
                LlamaCppInference.kv_cache_clear(contextHandle);
                
                LogManager.logI(TAG, String.format("[调试] 调用completion_init - contextHandle=%d, batch=%d, prompt长度=%d, maxTokens=%d", 
                    contextHandle, batch, prompt.length(), maxTokens));
                LogManager.logI(TAG, String.format("[调试] 提示词内容: '%s'", prompt));
                
                // 初始化完成
                int tokenCount = LlamaCppInference.completion_init(contextHandle, batch, prompt, maxTokens, false);
                if (tokenCount < 0) {
                    LogManager.logE(TAG, "初始化完成失败");
                    if (callback != null) {
                        callback.onError("初始化完成失败");
                    }
                    return;
                }
                
                // 生成循环
                LlamaCppInference.IntVar currentPos = new LlamaCppInference.IntVar(tokenCount);
                
                for (int i = 0; i < maxTokens && !shouldStop.get(); i++) {
                    String token = LlamaCppInference.completion_loop(contextHandle, batch, sampler, maxTokens, currentPos);
                    if (token == null || token.isEmpty()) {
                        break;
                    }
                    
                    fullResponse.append(token);
                    totalTokensGenerated.incrementAndGet();
                    
                    if (callback != null) {
                        callback.onToken(token);
                    }
                }
                
                // 生成完成
                if (callback != null && !shouldStop.get()) {
                    // 计算统计信息
                    long duration = System.currentTimeMillis() - generationStartTime;
                    int tokens = totalTokensGenerated.get();
                    double tokensPerSecond = tokens > 0 && duration > 0 ? 
                        (tokens * 1000.0) / duration : 0.0;
                    
                    LogManager.logI(TAG, String.format(
                        "传统API生成完成 - 总tokens: %d, 耗时: %dms, 速率: %.2f tokens/s",
                        tokens, duration, tokensPerSecond));
                    
                    callback.onComplete(fullResponse.toString());
                }
            } finally {
                // 清理资源
                LlamaCppInference.free_batch(batch);
                LlamaCppInference.free_sampler(sampler);
            }
            
        } catch (Exception e) {
            LogManager.logE(TAG, "传统API推理失败: " + e.getMessage(), e);
            if (callback != null) {
                callback.onError("传统API推理失败: " + e.getMessage());
            }
        }
    }
    
    /**
     * 更新推理参数
     */
    private void updateInferenceParameters(LocalLlmHandler.InferenceParams params) {
        if (modelHandle != 0 && params != null) {
            try {
                // 参数设置已改为通过采样器实现，在创建采样器时设置参数
                // 注意：LlamaCppInference没有setRepeatPenalty方法，暂时跳过
                // LlamaCppInference.setRepeatPenalty(params.getRepetitionPenalty());
                
                LogManager.logD(TAG, String.format(
                    "更新推理参数 - temp: %.2f, top_p: %.2f, top_k: %d, repeat_penalty: %.2f",
                    params.getTemperature(), params.getTopP(), params.getTopK(), 
                    params.getRepetitionPenalty()));
            } catch (Exception e) {
                LogManager.logW(TAG, "更新推理参数失败", e);
            }
        }
    }
    
    /**
     * 生成文本（同步）
     * @param prompt 输入提示词
     * @return 生成的文本
     */
    public CompletableFuture<String> generateTextAsync(String prompt) {
        return CompletableFuture.supplyAsync(() -> {
            if (!isInitialized.get()) {
                Log.e(TAG, "推理引擎未初始化");
                return "";
            }
            
            if (isGenerating.getAndSet(true)) {
                Log.w(TAG, "正在生成中，请等待当前任务完成");
                return "";
            }
            
            try {
                Log.i(TAG, "开始生成文本: " + prompt.substring(0, Math.min(prompt.length(), 100)) + "...");
                
                // 使用底层JNI方法进行同步生成
                int maxTokens = ConfigManager.getMaxNewTokens(context);
                
                // 动态创建批处理大小，支持超长prompt
                // 首先估算prompt的token数量（粗略估算：字符数/4）
                int estimatedTokens = Math.max(512, prompt.length() / 4 + 100);
                // 限制最大batch大小避免内存问题
                int batchSize = Math.min(estimatedTokens, 2048);
                
                LogManager.logI(TAG, String.format("动态batch大小 - 估算tokens: %d, 使用batch大小: %d", estimatedTokens, batchSize));
                
                long batch = LlamaCppInference.new_batch(batchSize, 0, 1);
                // 使用默认参数创建采样器
                long sampler = LlamaCppInference.new_sampler_with_params(
                    1.0f,  // 默认温度
                    0.9f,  // 默认top_p
                    40     // 默认top_k
                );
                
                StringBuilder result = new StringBuilder();
                
                try {
                    // 清空KV缓存
                LlamaCppInference.kv_cache_clear(contextHandle);
                
                LogManager.logI(TAG, String.format("[调试] generateTextAsync调用completion_init - contextHandle=%d, batch=%d, prompt长度=%d, maxTokens=%d", 
                    contextHandle, batch, prompt.length(), maxTokens));
                LogManager.logI(TAG, String.format("[调试] 提示词内容: '%s'", prompt));
                
                // 初始化完成
                int tokenCount = LlamaCppInference.completion_init(contextHandle, batch, prompt, maxTokens, false);
                    if (tokenCount < 0) {
                        LogManager.logE(TAG, "初始化完成失败");
                        return "";
                    }
                    
                    // 生成循环
                    LlamaCppInference.IntVar currentPos = new LlamaCppInference.IntVar(tokenCount);
                    
                    for (int i = 0; i < maxTokens; i++) {
                        String token = LlamaCppInference.completion_loop(contextHandle, batch, sampler, maxTokens, currentPos);
                        if (token == null || token.isEmpty()) {
                            break;
                        }
                        result.append(token);
                    }
                } finally {
                    // 清理资源
                    LlamaCppInference.free_batch(batch);
                    LlamaCppInference.free_sampler(sampler);
                }
                
                String finalResult = result.toString();
                
                if (finalResult == null) {
                    finalResult = "";
                }
                
                Log.i(TAG, "文本生成完成，长度: " + (finalResult != null ? finalResult.length() : 0));
                return finalResult != null ? finalResult : "";
                
            } catch (Exception e) {
                Log.e(TAG, "生成文本异常", e);
                return "";
            } finally {
                isGenerating.set(false);
            }
        }, executorService);
    }
    
    @Override
    public void stopInference() {
        if (isGenerating.get()) {
            LogManager.logI(TAG, "停止LlamaCpp文本生成");
            shouldStop.set(true);
            
            // 使用shouldStop标志来停止生成
            // 不需要调用特定的停止方法，生成循环会检查shouldStop标志
            
            isGenerating.set(false);
        }
    }
    
    /**
     * 初始化词嵌入功能
     * @param embeddingModelPath 嵌入模型路径
     * @return 是否初始化成功
     */
    public boolean initializeEmbedding(String embeddingModelPath) {
        try {
            LogManager.logI(TAG, "初始化LlamaCpp词嵌入功能: " + embeddingModelPath);
            
            // 验证模型文件
            File modelFile = new File(embeddingModelPath);
            if (!modelFile.exists()) {
                LogManager.logE(TAG, "嵌入模型文件不存在: " + embeddingModelPath);
                return false;
            }
            
            // 查找GGUF模型文件
            File ggufFile = findGgufFile(modelFile);
            if (ggufFile == null) {
                LogManager.logE(TAG, "在目录中未找到GGUF嵌入模型文件: " + embeddingModelPath);
                return false;
            }
            
            // 使用专门的LlamaCppEmbedding类
            llamaCppEmbedding = new LlamaCppEmbedding(ggufFile.getAbsolutePath());
            
            LogManager.logI(TAG, "LlamaCpp词嵌入功能初始化成功");
            LogManager.logI(TAG, "嵌入模型信息: " + llamaCppEmbedding.getModelInfo());
            LogManager.logI(TAG, "嵌入维度: " + llamaCppEmbedding.getEmbeddingSize());
            return true;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "初始化词嵌入功能失败", e);
            if (llamaCppEmbedding != null) {
                llamaCppEmbedding.close();
                llamaCppEmbedding = null;
            }
            return false;
        }
    }
    
    /**
     * 计算文本的词嵌入向量
     * @param text 输入文本
     * @return 嵌入向量，失败返回null
     */
    public float[] getEmbedding(String text) {
        if (llamaCppEmbedding == null || !llamaCppEmbedding.isInitialized()) {
            LogManager.logE(TAG, "LlamaCpp嵌入引擎未初始化，无法计算嵌入");
            return null;
        }
        
        if (text == null || text.trim().isEmpty()) {
            LogManager.logW(TAG, "输入文本为空");
            return null;
        }
        
        try {
            LogManager.logD(TAG, "开始计算文本嵌入，文本长度: " + text.length());
            
            // 使用专门的LlamaCppEmbedding计算嵌入
            float[] embedding = llamaCppEmbedding.getEmbedding(text);
            
            if (embedding != null) {
                LogManager.logD(TAG, "嵌入计算完成，维度: " + embedding.length);
            } else {
                LogManager.logW(TAG, "嵌入计算返回null");
            }
            
            return embedding;
        } catch (Exception e) {
            LogManager.logE(TAG, "计算词嵌入失败", e);
            return null;
        }
    }
    
    /**
     * 批量计算文本的词嵌入向量
     * @param texts 输入文本列表
     * @return 嵌入向量列表
     */
    public CompletableFuture<float[][]> getEmbeddingsAsync(java.util.List<String> texts) {
        return CompletableFuture.supplyAsync(() -> {
            if (llamaCppEmbedding == null || !llamaCppEmbedding.isInitialized()) {
                LogManager.logE(TAG, "LlamaCpp嵌入引擎未初始化，无法计算批量嵌入");
                return null;
            }
            
            if (texts == null || texts.isEmpty()) {
                LogManager.logW(TAG, "输入文本列表为空");
                return new float[0][];
            }
            
            try {
                LogManager.logD(TAG, "开始计算批量文本嵌入，文本数量: " + texts.size());
                
                // 使用专门的LlamaCppEmbedding计算批量嵌入
                float[][] embeddings = llamaCppEmbedding.getEmbeddings(texts);
                
                if (embeddings != null) {
                    LogManager.logD(TAG, "批量嵌入计算完成，返回 " + embeddings.length + " 个嵌入向量");
                } else {
                    LogManager.logW(TAG, "批量嵌入计算返回null");
                }
                
                return embeddings;
            } catch (Exception e) {
                LogManager.logE(TAG, "批量计算词嵌入失败: " + e.getMessage(), e);
                return null;
            }
        }, executorService);
    }
    
    /**
     * 批量计算文本的词嵌入向量（同步版本）
     * @param texts 输入文本数组
     * @return 嵌入向量数组
     */
    public float[][] getEmbeddings(String[] texts) {
        if (llamaCppEmbedding == null || !llamaCppEmbedding.isInitialized()) {
            LogManager.logE(TAG, "LlamaCpp嵌入引擎未初始化，无法计算批量嵌入");
            return null;
        }
        
        if (texts == null || texts.length == 0) {
            LogManager.logW(TAG, "输入文本数组为空");
            return new float[0][];
        }
        
        try {
            LogManager.logD(TAG, "开始计算批量文本嵌入，文本数量: " + texts.length);
            
            // 使用专门的LlamaCppEmbedding计算批量嵌入
            float[][] embeddings = llamaCppEmbedding.getEmbeddings(texts);
            
            if (embeddings != null) {
                LogManager.logD(TAG, "批量嵌入计算完成，返回 " + embeddings.length + " 个嵌入向量");
            } else {
                LogManager.logW(TAG, "批量嵌入计算返回null");
            }
            
            return embeddings;
        } catch (Exception e) {
            LogManager.logE(TAG, "批量计算词嵌入失败: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * 获取状态信息
     */
    public boolean isInitialized() {
        return modelHandle != 0 && contextHandle != 0 && isInitialized.get();
    }
    
    public boolean isGenerating() {
        return isGenerating.get();
    }
    
    /**
     * 检查词嵌入功能是否可用
     */
    public boolean isEmbeddingAvailable() {
        return llamaCppEmbedding != null && llamaCppEmbedding.isInitialized() && llamaCppEmbedding.supportsEmbedding();
    }
    
    public String getModelInfo() {
        if (modelHandle != 0 && contextHandle != 0) {
            return "LlamaCpp推理引擎已初始化";
        }
        return "未初始化";
    }
    
    /**
     * 获取统计信息
     */
    public String getStatistics() {
        if (!isInitialized.get()) {
            return "引擎未初始化";
        }
        
        long duration = generationStartTime > 0 ? 
            System.currentTimeMillis() - generationStartTime : 0;
        int tokens = totalTokensGenerated.get();
        double tokensPerSecond = tokens > 0 && duration > 0 ? 
            (tokens * 1000.0) / duration : 0.0;
        
        return String.format(
            "引擎类型: %s\n" +
            "模型路径: %s\n" +
            "是否生成中: %s\n" +
            "总生成tokens: %d\n" +
            "生成速率: %.2f tokens/s\n" +
            "词嵌入可用: %s",
            getEngineType(),
            currentModelPath != null ? currentModelPath : "未设置",
            isGenerating.get() ? "是" : "否",
            tokens,
            tokensPerSecond,
            isEmbeddingAvailable() ? "是" : "否"
        );
    }
    
    @Override
    public void release() {
        LogManager.logI(TAG, "释放LlamaCpp推理引擎资源");
        
        // 停止当前生成
        stopInference();
        
        // 释放LlamaCpp句柄
        if (contextHandle != 0) {
            try {
                LlamaCppInference.free_context(contextHandle);
                LogManager.logI(TAG, "上下文资源释放完成");
            } catch (Exception e) {
                LogManager.logW(TAG, "释放上下文时发生异常", e);
            }
            contextHandle = 0;
        }
        
        if (modelHandle != 0) {
            try {
                LlamaCppInference.free_model(modelHandle);
                LogManager.logI(TAG, "模型资源释放完成");
            } catch (Exception e) {
                LogManager.logW(TAG, "释放模型时发生异常", e);
            }
            modelHandle = 0;
        }
        
        // 释放嵌入资源
        if (llamaCppEmbedding != null) {
            try {
                llamaCppEmbedding.close();
                LogManager.logI(TAG, "LlamaCppEmbedding资源释放完成");
            } catch (Exception e) {
                LogManager.logW(TAG, "释放嵌入引擎时发生异常", e);
            }
            llamaCppEmbedding = null;
        }
        
        // 关闭线程池
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        
        isInitialized.set(false);
        LogManager.logI(TAG, "LlamaCpp推理引擎资源释放完成");
    }
    
    @Override
    public String getEngineType() {
        return "LlamaCpp";
    }
    
    // getEmbeddingSize和supportsEmbedding方法已移至LlamaCppEmbedding.java
    // 请通过getEmbeddingHandler()获取LlamaCppEmbedding实例来使用这些方法
    
    // Embedding相关方法已移至LlamaCppEmbedding.java
    // 如需使用Embedding功能，请直接使用LlamaCppEmbedding实例
    
    public LlamaCppEmbedding getEmbeddingHandler() {
        return llamaCppEmbedding;
    }
    
    /**
     * 流式回调接口
     */
    public interface StreamCallback {
        void onToken(String token);
        void onComplete(String fullText);
        void onError(String error);
    }
}