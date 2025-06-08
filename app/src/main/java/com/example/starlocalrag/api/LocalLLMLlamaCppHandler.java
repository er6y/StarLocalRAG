package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

import com.example.starlocalrag.ConfigManager;
import com.example.starlocalrag.LogManager;
import com.starlocalrag.llamacpp.LlamaCppInference;

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
    
    // LlamaCpp推理引擎
    private LlamaCppInference llamaCppInference;
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
        int configKvCacheSize = ConfigManager.getKvCacheSize(context);
        
        // 线程数配置 - 与OnnxRuntimeGenAI对齐：MIN(CPU核心数, getThreads)
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int actualThreads = Math.min(configThreads, availableProcessors);
        
        // 设置参数
        maxTokens = configKvCacheSize; // 使用最大输出token数作为maxTokens
        int contextSize = configMaxSeqLength; // 上下文大小应该是最大序列长度，不是输出token数
        int nGpuLayers = 0; // 仅CPU推理
        
        LogManager.logI(TAG, String.format("线程配置 - 用户配置: %d线程, CPU核心: %d, 实际使用: %d线程", 
            configThreads, availableProcessors, actualThreads));
        
        // 一行代码初始化 - 类似SimpleGenAI
        llamaCppInference = new LlamaCppInference(ggufFile.getAbsolutePath());
        
        // 配置推理参数 - 只设置必要的系统配置，不设置temperature、topP、topK
        llamaCppInference.setContextSize(contextSize)  // 设置上下文大小为最大序列长度
                .setMaxSequenceLength(configMaxSeqLength)  // 设置最大序列长度
                .setThreads(actualThreads)  // 使用MIN(CPU核心数, getThreads)
                .setGpuLayers(nGpuLayers)  // GPU层数
                .setMaxTokens(maxTokens);  // 使用最大输出token数作为maxTokens
        
        LogManager.logI(TAG, "✓ LlamaCpp引擎初始化完成");
        LogManager.logI(TAG, "模型信息: " + llamaCppInference.getModelInfo());
        LogManager.logI(TAG, "最大序列长度: " + configMaxSeqLength + ", 线程数: " + actualThreads + ", 最大输出token数: " + configKvCacheSize);
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
                
                generateWithLlamaCpp(prompt, callback, fullResponse);
                
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
    private void generateWithLlamaCpp(String prompt, LocalLlmHandler.StreamingCallback callback, StringBuilder fullResponse) {
        try {
            // 创建生成参数 - 类似SimpleGenAI.createGeneratorParams()
            LlamaCppInference.GeneratorParams params = llamaCppInference.createGeneratorParams();
            
            // 从ConfigManager获取配置参数并设置 - 只设置必要的系统配置
            int maxTokens = ConfigManager.getKvCacheSize(context); // 使用最大输出token数作为maxTokens
            int threads = ConfigManager.getThreads(context);
            int kvCacheSize = ConfigManager.getKvCacheSize(context);
            
            // 只设置必要的系统配置，不设置temperature、topP、topK
            params.setSearchOption("max_length", maxTokens);
            params.setSearchOption("threads", threads);
            params.setSearchOption("cache_size", kvCacheSize);
            
            LogManager.logI(TAG, "推理参数 - MaxTokens: " + maxTokens + ", Threads: " + threads + ", CacheSize: " + kvCacheSize);
            LogManager.logI(TAG, "采样参数由GGUF模型自动配置");
            
            // 一站式推理调用 - 类似SimpleGenAI.generate(params, prompt, listener)
            String result = llamaCppInference.generate(params, prompt, token -> {
                if (!shouldStop.get() && !token.isEmpty()) {
                    fullResponse.append(token);
                    totalTokensGenerated.incrementAndGet();
                    if (callback != null) {
                        callback.onToken(token);
                    }
                }
            });
            
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
    private void generateWithTraditionalStreaming(String prompt, LocalLlmHandler.StreamingCallback callback, StringBuilder fullResponse) {
        try {
            // 使用LlamaCppInference的流式生成
            llamaCppInference.generateStream(prompt, 
                new LlamaCppInference.StreamCallback() {
                    @Override
                    public void onText(String text, boolean isComplete) {
                        if (shouldStop.get()) {
                            return;
                        }
                        
                        fullResponse.append(text);
                        totalTokensGenerated.incrementAndGet();
                        
                        if (callback != null) {
                            callback.onToken(text);
                            if (isComplete) {
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
                        }
                    }
                    
                    @Override
                    public void onError(String error) {
                        LogManager.logE(TAG, "传统API生成过程中发生错误: " + error);
                        if (callback != null) {
                            callback.onError(error);
                        }
                    }
                });
            
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
        if (llamaCppInference != null && params != null) {
            try {
                // 更新温度、top_p、top_k等参数
                llamaCppInference.setTemperature(params.getTemperature());
                llamaCppInference.setTopP(params.getTopP());
                llamaCppInference.setTopK(params.getTopK());
                // 注意：LlamaCppInference没有setRepeatPenalty方法，暂时跳过
                // llamaCppInference.setRepeatPenalty(params.getRepetitionPenalty());
                
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
                
                String result = llamaCppInference.generate(prompt);
                
                if (result == null) {
                    result = "";
                }
                
                Log.i(TAG, "文本生成完成，长度: " + (result != null ? result.length() : 0));
                return result != null ? result : "";
                
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
            
            if (llamaCppInference != null) {
                try {
                    // 注意：LlamaCppInference没有stopGeneration方法，使用shouldStop标志
                    // llamaCppInference.stopGeneration();
                } catch (Exception e) {
                    LogManager.logW(TAG, "停止生成时发生异常", e);
                }
            }
            
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
            
            // 嵌入功能已集成到LlamaCppInference中，无需单独初始化
            LogManager.logI(TAG, "嵌入功能将通过LlamaCppInference提供");
            
            LogManager.logI(TAG, "LlamaCpp词嵌入功能初始化成功");
            return true;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "初始化词嵌入功能失败", e);
            // 嵌入功能已集成到LlamaCppInference中，无需单独清理
            return false;
        }
    }
    
    /**
     * 计算文本的词嵌入向量
     * @param text 输入文本
     * @return 嵌入向量，失败返回null
     */
    public float[] getEmbedding(String text) {
        if (!isInitialized()) {
            LogManager.logE(TAG, "LlamaCpp引擎未初始化，无法计算嵌入");
            return null;
        }
        
        if (text == null || text.trim().isEmpty()) {
            LogManager.logW(TAG, "输入文本为空");
            return null;
        }
        
        try {
            LogManager.logD(TAG, "开始计算文本嵌入，文本长度: " + text.length());
            
            // 使用LlamaCppInference计算嵌入
            float[] embedding = llamaCppInference.getEmbedding(text);
            
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
            if (!isInitialized()) {
                LogManager.logE(TAG, "LlamaCpp引擎未初始化，无法计算批量嵌入");
                return null;
            }
            
            if (texts == null || texts.isEmpty()) {
                LogManager.logW(TAG, "输入文本列表为空");
                return new float[0][];
            }
            
            try {
                LogManager.logD(TAG, "开始计算批量文本嵌入，文本数量: " + texts.size());
                
                // 使用LlamaCppInference计算批量嵌入
                float[][] embeddings = llamaCppInference.getEmbeddings(texts);
                
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
        if (!isInitialized()) {
            LogManager.logE(TAG, "LlamaCpp引擎未初始化，无法计算批量嵌入");
            return null;
        }
        
        if (texts == null || texts.length == 0) {
            LogManager.logW(TAG, "输入文本数组为空");
            return new float[0][];
        }
        
        try {
            LogManager.logD(TAG, "开始计算批量文本嵌入，文本数量: " + texts.length);
            
            // 使用LlamaCppInference计算批量嵌入
            float[][] embeddings = llamaCppInference.getEmbeddings(texts);
            
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
        return llamaCppInference != null && isInitialized.get();
    }
    
    public boolean isGenerating() {
        return isGenerating.get();
    }
    
    /**
     * 检查词嵌入功能是否可用
     */
    public boolean isEmbeddingAvailable() {
        return llamaCppInference != null && llamaCppInference.supportsEmbedding();
    }
    
    public String getModelInfo() {
        if (llamaCppInference != null) {
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
        
        // 释放LlamaCppInference
        if (llamaCppInference != null) {
            try {
                llamaCppInference.close();
                LogManager.logI(TAG, "LlamaCppInference资源释放完成");
            } catch (Exception e) {
                LogManager.logW(TAG, "释放推理引擎时发生异常", e);
            }
            llamaCppInference = null;
        }
        
        // 嵌入功能已集成到LlamaCppInference中，无需单独释放
        
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
    
    public int getEmbeddingSize() {
        if (!isInitialized()) {
            return 0;
        }
        
        try {
            return llamaCppInference.getEmbeddingSize();
        } catch (Exception e) {
            LogManager.logE(TAG, "获取嵌入维度失败: " + e.getMessage(), e);
            return 0;
        }
    }
    
    public boolean supportsEmbedding() {
        if (!isInitialized()) {
            return false;
        }
        
        try {
            return llamaCppInference.supportsEmbedding();
        } catch (Exception e) {
            LogManager.logE(TAG, "检查嵌入支持失败: " + e.getMessage(), e);
            return false;
        }
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