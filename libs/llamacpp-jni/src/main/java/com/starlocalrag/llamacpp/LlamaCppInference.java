package com.starlocalrag.llamacpp;

import android.util.Log;

/**
 * LlamaCppInference - LlamaCpp推理接口
 * 参考官方llama.android示例，提供简洁的JNI接口
 * 直接对接C++的JNI适配器，包含所有与llama_inference.cpp中JNI方法对应的native方法声明
 */
public class LlamaCppInference {
    private static final String TAG = "LlamaCppInference";
    
    // 加载本地库
    static {
        try {
            NativeLibraryLoader.loadLibrary("llamacpp_jni");
            Log.i(TAG, "LlamaCppInference native library loaded successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to load LlamaCppInference native library", e);
            throw new RuntimeException("Failed to load LlamaCppInference native library", e);
        }
    }
    
    // ========== 基础模型和上下文管理 ==========
    
    /**
     * 初始化后端
     */
    public static native void backend_init();
    
    /**
     * 加载模型
     * @param modelPath 模型文件路径
     * @return 模型句柄
     */
    public static native long load_model(String modelPath);
    
    /**
     * 加载模型（带GPU层数参数）
     * @param modelPath 模型文件路径
     * @param gpuLayers GPU层数，-1表示全部使用GPU，0表示仅CPU
     * @return 模型句柄
     */
    public static native long load_model_with_gpu(String modelPath, int gpuLayers);
    
    /**
     * 释放模型
     * @param modelHandle 模型句柄
     */
    public static native void free_model(long modelHandle);
    
    /**
     * 创建上下文
     * @param modelHandle 模型句柄
     * @param contextSize 上下文大小
     * @param threads 线程数
     * @param gpuLayers GPU层数
     * @return 上下文句柄
     */
    public static native long new_context_with_params(long modelHandle, int contextSize, int threads, int gpuLayers);
    
    /**
     * 释放上下文
     * @param contextHandle 上下文句柄
     */
    public static native void free_context(long contextHandle);
    
    // ========== 批处理管理 ==========
    
    /**
     * 创建批处理
     * @param nTokens token数量
     * @param embd 嵌入维度
     * @param nSeqMax 最大序列数
     * @return 批处理句柄
     */
    public static native long new_batch(int nTokens, int embd, int nSeqMax);
    
    /**
     * 释放批处理
     * @param batchHandle 批处理句柄
     */
    public static native void free_batch(long batchHandle);
    
    // ========== 采样器管理 ==========
    
    /**
     * 创建采样器
     * @return 采样器句柄
     */
    public static native long new_sampler();
    
    /**
     * 释放采样器
     * @param samplerHandle 采样器句柄
     */
    public static native void free_sampler(long samplerHandle);
    
    // ========== 推理和生成 ==========
    
    /**
     * 初始化完成
     * @param contextHandle 上下文句柄
     * @param batchHandle 批处理句柄
     * @param prompt 提示词
     * @param maxTokens 最大token数
     * @param addBos 是否添加BOS
     * @return token数量
     */
    public static native int completion_init(long contextHandle, long batchHandle, String prompt, int maxTokens, boolean addBos);
    
    /**
     * 完成循环
     * @param contextHandle 上下文句柄
     * @param batchHandle 批处理句柄
     * @param samplerHandle 采样器句柄
     * @param maxTokens 最大token数
     * @param currentPos 当前位置（输入输出参数）
     * @return 生成的token文本
     */
    public static native String completion_loop(long contextHandle, long batchHandle, long samplerHandle, int maxTokens, IntVar currentPos);
    
    /**
     * 清空KV缓存
     * @param contextHandle 上下文句柄
     */
    public static native void kv_cache_clear(long contextHandle);
    
    // ========== 参数设置（通过采样器实现）==========
    
    /**
     * 创建带参数的采样器
     * @param temperature 温度值
     * @param topP top-p值
     * @param topK top-k值
     * @return 采样器句柄
     */
    public static native long new_sampler_with_params(float temperature, float topP, int topK);
    
    /**
     * 注意：参数设置已改为通过采样器实现
     * 使用 new_sampler_with_params() 代替单独的参数设置方法
     */
    @Deprecated
    public static void setTemperature(float temperature) {
        // 此方法已废弃，参数应通过采样器设置
        android.util.Log.w("LlamaCppInference", "setTemperature已废弃，请使用new_sampler_with_params");
    }
    
    @Deprecated
    public static void setTopP(float topP) {
        // 此方法已废弃，参数应通过采样器设置
        android.util.Log.w("LlamaCppInference", "setTopP已废弃，请使用new_sampler_with_params");
    }
    
    @Deprecated
    public static void setTopK(int topK) {
        // 此方法已废弃，参数应通过采样器设置
        android.util.Log.w("LlamaCppInference", "setTopK已废弃，请使用new_sampler_with_params");
    }
    
    // ========== 辅助类和接口 ==========
    
    /**
     * 整数变量类，用于传递可变整数参数
     */
    public static class IntVar {
        public int value;
        
        public IntVar(int value) {
            this.value = value;
        }
        
        public int getValue() {
            return value;
        }
        
        public void inc() {
            value++;
        }
    }
    
    /**
     * 生成参数类
     */
    public static class GeneratorParams {
        private int maxLength = 512;
        private int threads = 4;
        private int maxNewTokens = 512;
        private boolean useGpu = false;
        
        public GeneratorParams setSearchOption(String key, Object value) {
            switch (key) {
                case "max_length":
                    this.maxLength = (Integer) value;
                    break;
                case "threads":
                    this.threads = (Integer) value;
                    break;
                case "max_new_tokens":
                    this.maxNewTokens = (Integer) value;
                    break;
                case "use_gpu":
                    this.useGpu = (Boolean) value;
                    break;
            }
            return this;
        }
        
        public int getMaxLength() { return maxLength; }
        public int getThreads() { return threads; }
        public int getMaxNewTokens() { return maxNewTokens; }
        public boolean isUseGpu() { return useGpu; }
    }
    
    /**
     * 流式回调接口
     */
    public interface StreamCallback {
        void onText(String text, boolean isComplete);
    }
}