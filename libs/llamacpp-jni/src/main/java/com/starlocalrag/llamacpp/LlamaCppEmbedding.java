package com.starlocalrag.llamacpp;

import android.util.Log;

/**
 * LlamaCpp嵌入向量接口
 * 专门负责文本嵌入功能，对接上层应用的嵌入式应用
 * 
 * 特性：
 * 1. 专注于嵌入向量生成
 * 2. 支持单文本和批量文本嵌入
 * 3. 自动归一化选项
 * 4. 完全自动化资源管理
 */
public class LlamaCppEmbedding implements AutoCloseable {
    private static final String TAG = "LlamaCppEmbedding";
    
    private long modelHandle = 0;
    private boolean isInitialized = false;
    
    // 嵌入相关参数
    private boolean normalizeEmbeddings = true;
    private int nGpuLayers = 0;
    
    /**
     * 默认构造函数
     */
    public LlamaCppEmbedding() {
        Log.d(TAG, "LlamaCppEmbedding created");
    }
    
    /**
     * 一站式构造函数
     * @param modelPath 嵌入模型文件路径
     * @throws RuntimeException 如果初始化失败
     */
    public LlamaCppEmbedding(String modelPath) {
        Log.i(TAG, "LlamaCppEmbedding一站式初始化开始: " + modelPath);
        
        if (!initialize(modelPath)) {
            throw new RuntimeException("Failed to initialize LlamaCppEmbedding with model: " + modelPath);
        }
        
        Log.i(TAG, "✓ LlamaCppEmbedding一站式初始化成功");
    }
    
    /**
     * 初始化嵌入模型
     * @param modelPath 模型文件路径
     * @return 是否初始化成功
     */
    public boolean initialize(String modelPath) {
        if (isInitialized) {
            Log.w(TAG, "Already initialized");
            return true;
        }
        
        try {
            Log.i(TAG, "Initializing embedding model: " + modelPath);
            
            // 加载模型
            modelHandle = LlamaCppInference.load_model(modelPath);
            if (modelHandle == 0) {
                Log.e(TAG, "Failed to load embedding model");
                return false;
            }
            
            // 检查模型是否支持嵌入
            if (!supportsEmbedding()) {
                Log.e(TAG, "Model does not support embedding");
                LlamaCppInference.free_model(modelHandle);
                modelHandle = 0;
                return false;
            }
            
            isInitialized = true;
            Log.i(TAG, "Embedding model initialized successfully");
            return true;
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize embedding model", e);
            cleanup();
            return false;
        }
    }
    
    /**
     * 获取单个文本的嵌入向量
     * @param text 输入文本
     * @return 嵌入向量
     */
    public float[] getEmbedding(String text) {
        if (!isInitialized) {
            Log.e(TAG, "Embedding model not initialized");
            return null;
        }
        
        if (text == null || text.trim().isEmpty()) {
            Log.w(TAG, "Empty text provided for embedding");
            return new float[0];
        }
        
        try {
            Log.d(TAG, "Getting embedding for text: " + text.substring(0, Math.min(50, text.length())) + "...");
            
            float[] embedding = LlamaCpp.getEmbedding(modelHandle, text, normalizeEmbeddings);
            
            if (embedding != null) {
                Log.d(TAG, "Generated embedding with dimension: " + embedding.length);
            } else {
                Log.w(TAG, "Embedding generation returned null");
            }
            
            return embedding;
            
        } catch (Exception e) {
            Log.e(TAG, "Error during embedding generation", e);
            return null;
        }
    }
    
    /**
     * 获取多个文本的嵌入向量
     * @param texts 输入文本数组
     * @return 嵌入向量矩阵
     */
    public float[][] getEmbeddings(String[] texts) {
        if (!isInitialized) {
            Log.e(TAG, "Embedding model not initialized");
            return null;
        }
        
        if (texts == null || texts.length == 0) {
            Log.w(TAG, "Empty text array provided for embeddings");
            return new float[0][];
        }
        
        try {
            Log.d(TAG, "Getting embeddings for " + texts.length + " texts");
            
            float[][] embeddings = LlamaCpp.getEmbeddings(modelHandle, texts, normalizeEmbeddings);
            
            if (embeddings != null) {
                Log.d(TAG, "Generated " + embeddings.length + " embeddings");
            } else {
                Log.w(TAG, "Batch embedding generation returned null");
            }
            
            return embeddings;
            
        } catch (Exception e) {
            Log.e(TAG, "Error during batch embedding generation", e);
            return null;
        }
    }
    
    /**
     * 获取多个文本的嵌入向量（List版本）
     * @param texts 输入文本列表
     * @return 嵌入向量矩阵
     */
    public float[][] getEmbeddings(java.util.List<String> texts) {
        return getEmbeddings(texts.toArray(new String[0]));
    }
    
    /**
     * 获取嵌入向量维度
     * @return 嵌入维度
     */
    public int getEmbeddingSize() {
        if (!isInitialized) {
            return -1;
        }
        return LlamaCpp.getEmbeddingSize(modelHandle);
    }
    
    /**
     * 检查模型是否支持嵌入
     * @return 是否支持嵌入
     */
    public boolean supportsEmbedding() {
        if (!isInitialized) {
            return false;
        }
        return LlamaCpp.supportsEmbedding(modelHandle);
    }
    
    // ========== 参数设置 ==========
    
    /**
     * 设置GPU层数
     * @param nGpuLayers GPU层数
     * @return this
     */
    public LlamaCppEmbedding setGpuLayers(int nGpuLayers) {
        this.nGpuLayers = nGpuLayers;
        return this;
    }
    
    /**
     * 设置是否归一化嵌入向量
     * @param normalize 是否归一化
     * @return this
     */
    public LlamaCppEmbedding setNormalizeEmbeddings(boolean normalize) {
        this.normalizeEmbeddings = normalize;
        return this;
    }
    
    // ========== 状态查询 ==========
    
    public boolean isInitialized() {
        return isInitialized;
    }
    
    public String getModelInfo() {
        if (!isInitialized) {
            return "Embedding model not initialized";
        }
        return LlamaCpp.getModelInfo(modelHandle);
    }
    
    // ========== 资源管理 ==========
    
    private void cleanup() {
        if (modelHandle != 0) {
            LlamaCppInference.free_model(modelHandle);
            modelHandle = 0;
        }
        isInitialized = false;
    }
    
    @Override
    public void close() {
        Log.d(TAG, "Closing LlamaCppEmbedding");
        cleanup();
    }
}