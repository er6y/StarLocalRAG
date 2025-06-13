package com.starlocalrag.llamacpp;

import android.util.Log;

/**
 * LlamaCpp JNI接口 - 简洁高效的实现
 * 参考llama.cpp官方示例：embedding.cpp, simple.cpp, llama.android
 * 提供词嵌入和文本生成两大核心功能
 */
public class LlamaCpp {
    private static final String TAG = "LlamaCpp";
    
    // 加载本地库
    static {
        try {
            NativeLibraryLoader.loadLibrary("llamacpp_jni");
            Log.i(TAG, "LlamaCpp native library loaded successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to load LlamaCpp native library", e);
            throw new RuntimeException("Failed to load LlamaCpp native library", e);
        }
    }
    
    // ========== 模型管理 ==========
    
    /**
     * 加载模型
     * @param modelPath GGUF模型文件路径
     * @param nGpuLayers GPU层数（0表示仅CPU）
     * @return 模型句柄，失败返回0
     */
    public static long loadModel(String modelPath, int nGpuLayers) {
        return LlamaCppInference.load_model(modelPath);
    }
    
    /**
     * 释放模型
     * @param model 模型句柄
     */
    public static void freeModel(long model) {
        LlamaCppInference.free_model(model);
    }
    
    /**
     * 创建上下文
     * @param model 模型句柄
     * @param nCtx 上下文大小
     * @param nBatch 批处理大小
     * @param nThreads 线程数
     * @return 上下文句柄
     */
    public static long createContext(long model, int nCtx, int nBatch, int nThreads) {
        return LlamaCppInference.new_context_with_params(model, nCtx, nThreads, 0);
    }
    
    /**
     * 释放上下文
     * @param ctx 上下文句柄
     */
    public static void freeContext(long ctx) {
        LlamaCppInference.free_context(ctx);
    }
    
    // ========== 词嵌入功能 ==========
    
    /**
     * 计算单个文本的词嵌入向量
     * @deprecated 请直接使用 LlamaCppEmbedding 类
     * @param model 模型句柄
     * @param text 输入文本
     * @param normalize 是否归一化向量
     * @return 嵌入向量数组，失败返回null
     */
    @Deprecated
    public static float[] getEmbedding(long model, String text, boolean normalize) {
        // 已废弃：请直接使用 LlamaCppEmbedding 类
        Log.w("LlamaCpp", "getEmbedding方法已废弃，请直接使用LlamaCppEmbedding类");
        return null;
    }
    
    /**
     * 批量计算多个文本的词嵌入向量
     * @deprecated 请直接使用 LlamaCppEmbedding 类
     * @param model 模型句柄
     * @param texts 输入文本数组
     * @param normalize 是否归一化向量
     * @return 嵌入向量矩阵（每行一个向量），失败返回null
     */
    @Deprecated
    public static float[][] getEmbeddings(long model, String[] texts, boolean normalize) {
        // 已废弃：请直接使用 LlamaCppEmbedding 类
        Log.w("LlamaCpp", "getEmbeddings方法已废弃，请直接使用LlamaCppEmbedding类");
        return null;
    }
    
    // ========== 文本生成功能 ==========
    
    /**
     * 生成文本（同步方式）
     * @param ctx 上下文句柄
     * @param prompt 输入提示
     * @param maxTokens 最大生成token数
     * @param temperature 温度参数
     * @param topP top-p采样参数
     * @param topK top-k采样参数
     * @return 生成的文本，失败返回null
     */
    public static String generateText(long ctx, String prompt, int maxTokens, 
                                           float temperature, float topP, int topK) {
        try {
            // 动态创建批处理大小，支持超长prompt
            // 首先估算prompt的token数量（粗略估算：字符数/4）
            int estimatedTokens = Math.max(512, prompt.length() / 4 + 100);
            // 限制最大batch大小避免内存问题
            int batchSize = Math.min(estimatedTokens, 2048);
            
            System.out.println("LlamaCpp动态batch大小 - 估算tokens: " + estimatedTokens + ", 使用batch大小: " + batchSize);
            
            // 创建批处理和采样器
            long batch = LlamaCppInference.new_batch(batchSize, 0, 1);
            long sampler = LlamaCppInference.new_sampler();
            
            // 清空KV缓存
            LlamaCppInference.kv_cache_clear(ctx);
            
            // 初始化完成
            int tokenCount = LlamaCppInference.completion_init(ctx, batch, prompt, maxTokens, false);
            if (tokenCount < 0) {
                LlamaCppInference.free_batch(batch);
                LlamaCppInference.free_sampler(sampler);
                return null;
            }
            
            // 生成循环
            StringBuilder result = new StringBuilder();
            LlamaCppInference.IntVar currentPos = new LlamaCppInference.IntVar(tokenCount);
            
            for (int i = 0; i < maxTokens; i++) {
                String token = LlamaCppInference.completion_loop(ctx, batch, sampler, maxTokens, currentPos);
                if (token == null || token.isEmpty()) {
                    break;
                }
                result.append(token);
            }
            
            // 清理资源
            LlamaCppInference.free_batch(batch);
            LlamaCppInference.free_sampler(sampler);
            
            return result.toString();
        } catch (Exception e) {
            Log.e(TAG, "Text generation failed", e);
            return null;
        }
    }
    
    /**
     * 生成文本（已废弃，需要重新设计接口）
     * @param prompt 输入提示词
     * @param maxTokens 最大生成token数
     * @return 生成的文本
     */
    @Deprecated
    public static String generateText(String prompt, int maxTokens) {
        // 此方法已废弃，需要重新设计接口
        // 建议使用LlamaCppInference的静态方法直接进行推理
        Log.w("LlamaCpp", "generateText method is deprecated. Use LlamaCppInference static methods instead.");
        return null;
    }
    
    /**
     * 开始流式生成文本
     * @param ctx 上下文句柄
     * @param prompt 输入提示
     * @param maxTokens 最大生成token数
     * @param temperature 温度参数
     * @param topP top-p采样参数
     * @param topK top-k采样参数
     * @return 生成会话ID，失败返回0
     */
    public static long startStreamingText(long ctx, String prompt, int maxTokens,
                                            float temperature, float topP, int topK) {
        // 使用LlamaCppInference实例进行流式生成
        return 0; // 暂时返回0，需要重新设计接口
    }
    
    /**
     * 开始文本生成
     * @param ctx 上下文句柄
     * @param prompt 输入提示
     * @return 是否成功开始
     */
    public static boolean startGeneration(long ctx, String prompt) {
        // 使用LlamaCppInference实例
        return false; // 暂时返回false，需要重新设计接口
    }
    
    /**
     * 获取下一个生成的token
     * @param sessionId 生成会话ID
     * @return 生成的文本片段，生成完成返回null
     */
    public static String getNextToken(long sessionId) {
        // 使用LlamaCppInference实例
        return null; // 暂时返回null，需要重新设计接口
    }
    
    /**
     * 停止生成
     * @param sessionId 生成会话ID
     */
    public static void stopGeneration(long sessionId) {
        // 使用LlamaCppInference实例
        // 暂时为空，需要重新设计接口
    }
    
    // ========== 工具函数 ==========
    
    /**
     * 获取模型信息
     * @param model 模型句柄
     * @return 模型信息字符串，包含模型元数据
     */
    public static String getModelInfo(long model) {
        StringBuilder info = new StringBuilder();
        info.append("模型信息:\n");
        
        try {
            // 获取元数据数量
            int metaCount = LlamaCppInference.model_meta_count(model);
            info.append("元数据数量: ").append(metaCount).append("\n");
            
            // 获取所有元数据
            info.append("元数据:\n");
            for (int i = 0; i < metaCount; i++) {
                String key = LlamaCppInference.model_meta_key_by_index(model, i);
                String value = LlamaCppInference.model_meta_val_str_by_index(model, i);
                if (key != null && value != null) {
                    info.append("  ").append(key).append(": ").append(value).append("\n");
                }
            }
            
            // 尝试获取一些常见的元数据
            String[] commonKeys = {"general.architecture", "general.name", "llama.context_length", 
                                  "llama.embedding_length", "llama.block_count", "llama.feed_forward_length", 
                                  "llama.attention.head_count", "llama.attention.head_count_kv", 
                                  "llama.rope.dimension_count", "tokenizer.ggml.model", "tokenizer.ggml.tokens"};
            
            info.append("\n常用参数:\n");
            for (String key : commonKeys) {
                String value = LlamaCppInference.model_meta_val_str(model, key);
                if (value != null) {
                    info.append("  ").append(key).append(": ").append(value).append("\n");
                }
            }
        } catch (Exception e) {
            info.append("获取模型元数据失败: ").append(e.getMessage());
        }
        
        return info.toString();
    }
    
    /**
     * 获取嵌入向量维度
     * @deprecated 请直接使用 LlamaCppEmbedding 类
     * @param model 模型句柄
     * @return 嵌入维度
     */
    @Deprecated
    public static int getEmbeddingSize(long model) {
        // 已废弃：请直接使用 LlamaCppEmbedding 类
        Log.w("LlamaCpp", "getEmbeddingSize方法已废弃，请直接使用LlamaCppEmbedding类");
        return -1;
    }
    
    /**
     * 检查模型是否支持嵌入
     * @deprecated 请直接使用 LlamaCppEmbedding 类
     * @param model 模型句柄
     * @return 是否支持嵌入
     */
    @Deprecated
    public static boolean supportsEmbedding(long model) {
        // 已废弃：请直接使用 LlamaCppEmbedding 类
        Log.w("LlamaCpp", "supportsEmbedding方法已废弃，请直接使用LlamaCppEmbedding类");
        return false;
    }
    
    /**
     * 获取库版本信息
     * @return 版本信息
     */
    public static String getVersion() {
        // 暂时返回固定版本信息
        return "LlamaCpp JNI v1.0";
    }
}