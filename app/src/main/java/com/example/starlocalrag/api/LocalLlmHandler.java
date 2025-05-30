package com.example.starlocalrag.api;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.example.starlocalrag.ConfigManager;
import com.example.starlocalrag.LogManager;

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

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import java.util.Iterator;
import java.util.Arrays;
import com.example.starlocalrag.HarmonyOSGPUAdapter;
import com.example.starlocalrag.MaliGPUOptimizer;
import com.example.starlocalrag.OpenGLESComputeAccelerator;

/**
 * 本地LLM处理程序
 * 负责加载和管理本地模型，执行本地推理
 * 支持多种模型类型，包括ONNX等
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
    
    // ONNX运行时环境
    private OrtEnvironment ortEnvironment;
    
    // ONNX会话
    private OrtSession ortSession;
    
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
    private String modelType = "onnx";
    
    // ONNX处理器
    private LocalLLMOnnxHandler localLlmOnnxHandler;
    
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
    }
    
    /**
     * 设置是否使用GPU
     */
    public void setUseGpu(boolean useGpu) {
        this.useGpu = useGpu;
    }
    
    /**
     * 加载本地模型
     * @param modelName 模型名称（目录名）
     * @param callback 回调接口
     */
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
                callback.onError("模型正在加载中，请稍后再试");
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
                if (ortSession != null) {
                    ortSession.close();
                    ortSession = null;
                }
                
                if (ortEnvironment != null) {
                    ortEnvironment.close();
                    ortEnvironment = null;
                }
                
                modelLoaded.set(false);
                
                // 1. 确保模型文件存在
                // 从ConfigManager获取模型路径
                String modelPath = ConfigManager.getModelPath(context);
                File modelDir = new File(modelPath, modelName);
                
                if (!modelDir.exists() || !modelDir.isDirectory()) {
                    throw new IOException("模型文件不存在: " + modelDir.getAbsolutePath());
                }
                
                // 2. 加载模型配置
                File configFile = new File(modelDir, "config.json");
                if (!configFile.exists()) {
                    throw new IOException("模型配置文件不存在: " + configFile.getPath());
                }
                loadModelConfig(configFile);
                
                // 3. 检查词汇表文件
                File tokenizerFile = new File(modelDir, "tokenizer.json");
                if (!tokenizerFile.exists()) {
                    throw new IOException("词汇表文件不存在: " + tokenizerFile.getPath());
                }
                checkTokenizerFile(tokenizerFile);
                
                // 4. 初始化ONNX运行时环境
                ortEnvironment = OrtEnvironment.getEnvironment();
                
                // 5. 配置会话选项
                SessionOptions sessionOptions = new SessionOptions();
                
                // 设置线程数 - 优化CPU推理性能
                int threads = ConfigManager.getThreads(context);
                int availableProcessors = Runtime.getRuntime().availableProcessors();
                
                // 使用BuildConfig中的线程配置（如果可用）
                int configuredThreads;
                try {
                    configuredThreads = com.example.starlocalrag.BuildConfig.THREAD_COUNT;
                    LogManager.logI(TAG, "使用BuildConfig配置的线程数: " + configuredThreads);
                } catch (Exception e) {
                    configuredThreads = threads;
                    LogManager.logD(TAG, "BuildConfig线程配置不可用，使用默认配置: " + threads);
                }
                
                // 对于CPU推理，使用更多线程可以提高性能
                int optimizedThreads = Math.min(Math.max(configuredThreads, threads) * 2, availableProcessors);
                
                // 根据调查报告建议，优化线程配置以改善缓存利用率
                // 调查报告建议：将IntraOp从8调至4，InterOp从4调至2，减少线程竞争
                Runtime runtime = Runtime.getRuntime();
                long maxMemory = runtime.maxMemory();
                
                // 限制最大线程数，避免过多线程导致缓存未命中
                if (maxMemory > 6 * 1024 * 1024 * 1024L) { // 6GB以上内存
                    optimizedThreads = Math.min(4, availableProcessors); // 最大4线程
                } else if (maxMemory < 3 * 1024 * 1024 * 1024L) { // 3GB以下内存
                    optimizedThreads = Math.min(2, availableProcessors); // 最大2线程
                } else {
                    optimizedThreads = Math.min(3, availableProcessors); // 中等内存设备最大3线程
                }
                
                sessionOptions.setIntraOpNumThreads(optimizedThreads);
                sessionOptions.setInterOpNumThreads(Math.max(1, optimizedThreads / 2));
                
                // 完全禁用CPU亲和性配置以避免移动设备兼容性问题
                // 移动设备的CPU亲和性配置经常导致ORT_FAIL错误
                LogManager.logI(TAG, "已禁用CPU亲和性配置以确保移动设备兼容性");
                
                LogManager.logI(TAG, String.format("ONNX Runtime线程配置 - IntraOp: %d, InterOp: %d (可用处理器: %d)", 
                    optimizedThreads, Math.max(1, optimizedThreads / 2), availableProcessors));
                
                // 启用性能优化
                sessionOptions.setMemoryPatternOptimization(true);
                sessionOptions.setExecutionMode(SessionOptions.ExecutionMode.PARALLEL); // 改为并行模式提高性能
                
                // 启用图优化
                try {
                    sessionOptions.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
                    LogManager.logI(TAG, "已启用ONNX Runtime全部优化");
                } catch (Exception e) {
                    LogManager.logW(TAG, "无法设置优化级别: " + e.getMessage());
                }
                
                // 启用CPU特定优化
                try {
                    // 移除空的affinity配置，避免"Affinity string must not be empty"错误
                    // sessionOptions.addConfigEntry("session.intra_op_thread_affinities", "");
                    // sessionOptions.addConfigEntry("session.inter_op_thread_affinities", "");
                    sessionOptions.addConfigEntry("session.force_spinning_stop", "1");
                    LogManager.logI(TAG, "已启用CPU特定优化配置");
                } catch (Exception e) {
                    LogManager.logW(TAG, "无法设置CPU优化配置: " + e.getMessage());
                }
                
                // 应用量化模型专门配置
                if (modelConfig != null && modelConfig.isQuantized()) {
                    applyQuantizationOptimizations(sessionOptions, modelConfig);
                }
                
                // 检查BuildConfig中的GPU和NNAPI配置
                boolean enableGpuAcceleration = true;
                boolean enableNNAPI = true;
                try {
                    enableGpuAcceleration = com.example.starlocalrag.BuildConfig.ENABLE_GPU_ACCELERATION;
                    enableNNAPI = com.example.starlocalrag.BuildConfig.ENABLE_NNAPI;
                    LogManager.logI(TAG, String.format("BuildConfig配置 - GPU加速: %b, NNAPI: %b", 
                        enableGpuAcceleration, enableNNAPI));
                } catch (Exception e) {
                    LogManager.logD(TAG, "BuildConfig GPU配置不可用，使用默认设置");
                }
                
                // 添加内存管理优化配置
                try {
                    sessionOptions.addConfigEntry("session.enable_cpu_mem_arena", "1");
                    sessionOptions.addConfigEntry("session.enable_mem_pattern", "1");
                    sessionOptions.addConfigEntry("session.use_env_allocators", "1");
                    sessionOptions.addConfigEntry("session.enable_memory_efficient_attention", "1");
                    
                    // 根据调查报告建议，优化内存限制配置
                    // 重用之前定义的runtime变量
                    long maxMemoryMB = maxMemory / (1024 * 1024); // 转换为MB
                    
                    // 动态调整内存限制：根据设备内存容量优化配置
                    long memoryLimit;
                    if (maxMemoryMB > 6144) { // 6GB以上设备
                        memoryLimit = 768; // 使用768MB（调查报告建议）
                    } else if (maxMemoryMB > 4096) { // 4-6GB设备
                        memoryLimit = 512; // 使用512MB
                    } else if (maxMemoryMB > 2048) { // 2-4GB设备
                        memoryLimit = 384; // 使用384MB
                    } else { // 2GB以下设备
                        memoryLimit = 256; // 使用256MB
                    }
                    
                    sessionOptions.addConfigEntry("session.memory_limit_mb", String.valueOf(memoryLimit));
                    
                    LogManager.logI(TAG, String.format("内存管理优化 - 设备内存: %dMB, 会话限制: %dMB (调查报告优化)", 
                        maxMemoryMB, memoryLimit));
                } catch (Exception e) {
                    LogManager.logW(TAG, "内存管理配置失败: " + e.getMessage());
                }
                
                // 如果启用GPU，按优先级尝试不同的GPU加速方式
                if (useGpu && enableGpuAcceleration) {
                    boolean gpuEnabled = false;
                    
                    // 检查系统信息
                    String osVersion = android.os.Build.VERSION.RELEASE;
                    String deviceModel = android.os.Build.MODEL;
                    String manufacturer = android.os.Build.MANUFACTURER;
                    LogManager.logI(TAG, String.format("GPU加速环境检查 - 系统版本: %s, 设备型号: %s, 制造商: %s", 
                        osVersion, deviceModel, manufacturer));
                    
                    // 检查是否为HarmonyOS
                    boolean isHarmonyOS = manufacturer.toLowerCase().contains("huawei") || 
                                         manufacturer.toLowerCase().contains("honor") ||
                                         android.os.Build.DISPLAY.toLowerCase().contains("harmony");
                    if (isHarmonyOS) {
                        LogManager.logI(TAG, "检测到HarmonyOS系统，将尝试兼容性GPU加速方案");
                        
                        // 应用HarmonyOS特定优化
                        try {
                            HarmonyOSGPUAdapter.optimizeForHarmonyOS(sessionOptions);
                            HarmonyOSGPUAdapter.checkHarmonyOSGPUPermissions(context);
                            LogManager.logI(TAG, "HarmonyOS GPU优化配置已应用");
                        } catch (Exception e) {
                            LogManager.logW(TAG, "HarmonyOS GPU优化失败: " + e.getMessage());
                        }
                    }
                    
                    // 检测Mali GPU并应用特定优化
                    try {
                        String maliInfo = MaliGPUOptimizer.detectMaliArchitecture();
                        if (maliInfo != null && maliInfo.toLowerCase().contains("mali")) {
                            LogManager.logI(TAG, "检测到Mali GPU: " + maliInfo);
                            MaliGPUOptimizer.optimizeForMaliG610(sessionOptions);
                            LogManager.logI(TAG, "Mali GPU优化配置已应用");
                        }
                    } catch (Exception e) {
                        LogManager.logW(TAG, "Mali GPU检测和优化失败: " + e.getMessage());
                    }
                    
                    // 初始化OpenGL ES计算加速器（如果支持）
                    try {
                        OpenGLESComputeAccelerator glAccelerator = new OpenGLESComputeAccelerator();
                        if (glAccelerator.isSupported()) {
                            LogManager.logI(TAG, "OpenGL ES计算着色器支持已启用");
                        }
                    } catch (Exception e) {
                        LogManager.logD(TAG, "OpenGL ES计算着色器初始化失败: " + e.getMessage());
                    }
                    
                    // 使用反射机制尝试调用可能存在的GPU加速方法
                    // 针对华为MIS-AL00设备(Mali-G610 GPU)优化加速方案顺序
                    String[] gpuMethods, gpuNames, gpuDescriptions;
                    
                    if (isHarmonyOS) {
                        // HarmonyOS设备优先使用兼容性更好的加速方案
                        if (enableNNAPI) {
                            gpuMethods = new String[]{"addOpenCL", "addVulkan", "addOpenGL", "addNNAPI", "addCUDA"};
                            gpuNames = new String[]{"OpenCL", "Vulkan", "OpenGL", "NNAPI", "CUDA"};
                            gpuDescriptions = new String[]{
                                "开放计算语言 (HarmonyOS兼容性最佳)",
                                "Vulkan API (Mali GPU优化支持)",
                                "OpenGL ES计算着色器 (通用GPU加速)",
                                "Android神经网络API (可能受限)",
                                "NVIDIA CUDA (不适用于Mali GPU)"
                            };
                        } else {
                            gpuMethods = new String[]{"addOpenCL", "addVulkan", "addOpenGL", "addCUDA"};
                            gpuNames = new String[]{"OpenCL", "Vulkan", "OpenGL", "CUDA"};
                            gpuDescriptions = new String[]{
                                "开放计算语言 (HarmonyOS兼容性最佳)",
                                "Vulkan API (Mali GPU优化支持)",
                                "OpenGL ES计算着色器 (通用GPU加速)",
                                "NVIDIA CUDA (不适用于Mali GPU)"
                            };
                        }
                    } else {
                        // 标准Android设备使用原有顺序
                        if (enableNNAPI) {
                            gpuMethods = new String[]{"addNNAPI", "addVulkan", "addOpenCL", "addOpenGL", "addCUDA"};
                            gpuNames = new String[]{"NNAPI", "Vulkan", "OpenCL", "OpenGL", "CUDA"};
                            gpuDescriptions = new String[]{
                                "Android神经网络API (适用于Android 8.1+)",
                                "Vulkan API (现代GPU加速)",
                                "开放计算语言 (跨平台并行计算)",
                                "OpenGL ES计算着色器 (通用GPU加速)",
                                "NVIDIA CUDA (NVIDIA GPU专用)"
                            };
                        } else {
                            gpuMethods = new String[]{"addVulkan", "addOpenCL", "addOpenGL", "addCUDA"};
                            gpuNames = new String[]{"Vulkan", "OpenCL", "OpenGL", "CUDA"};
                            gpuDescriptions = new String[]{
                                "Vulkan API (现代GPU加速)",
                                "开放计算语言 (跨平台并行计算)",
                                "OpenGL ES计算着色器 (通用GPU加速)",
                                "NVIDIA CUDA (NVIDIA GPU专用)"
                            };
                        }
                    }
                    
                    for (int i = 0; i < gpuMethods.length && !gpuEnabled; i++) {
                        try {
                            LogManager.logD(TAG, String.format("尝试启用%s加速 - %s", gpuNames[i], gpuDescriptions[i]));
                            
                            // 尝试通过反射调用方法
                            Method method = SessionOptions.class.getMethod(gpuMethods[i]);
                            method.invoke(sessionOptions);
                            
                            LogManager.logI(TAG, String.format("✓ 成功启用%s加速", gpuNames[i]));
                            gpuEnabled = true;
                            
                        } catch (NoSuchMethodException e) {
                            // 方法不存在，跳过
                            LogManager.logW(TAG, String.format("✗ %s加速方法不可用 - ONNX Runtime版本可能不支持此加速方式", gpuNames[i]));
                            
                        } catch (InvocationTargetException e) {
                            // 方法调用失败，获取具体原因
                            Throwable cause = e.getCause();
                            String errorMsg = cause != null ? cause.getMessage() : e.getMessage();
                            LogManager.logE(TAG, String.format("✗ %s加速启用失败: %s", gpuNames[i], errorMsg));
                            
                            // 针对不同错误提供具体建议
                            if (errorMsg != null) {
                                if (errorMsg.contains("NNAPI") && errorMsg.contains("not supported")) {
                                    LogManager.logW(TAG, "建议: NNAPI可能在此设备上不受支持，这在某些华为/荣耀设备上较常见");
                                } else if (errorMsg.contains("OpenCL") && errorMsg.contains("not found")) {
                                    LogManager.logW(TAG, "建议: OpenCL驱动未找到，可能需要更新GPU驱动或系统版本");
                                    if (isHarmonyOS) {
                                        LogManager.logW(TAG, "HarmonyOS提示: 尝试使用Mali GPU的OpenCL优化配置");
                                        try {
                                            MaliGPUOptimizer.optimizeMaliMemory(sessionOptions);
                                        } catch (Exception ex) {
                                            LogManager.logD(TAG, "Mali OpenCL优化失败: " + ex.getMessage());
                                        }
                                    }
                                } else if (errorMsg.contains("Vulkan")) {
                                    LogManager.logW(TAG, "建议: Vulkan API不可用 - 检查设备是否支持Vulkan 1.0+，或更新GPU驱动");
                                    if (isHarmonyOS) {
                                        LogManager.logW(TAG, "HarmonyOS提示: 确保系统版本支持Vulkan API，某些华为设备需要特定系统版本");
                                        try {
                                            HarmonyOSGPUAdapter.configureVulkanAcceleration(sessionOptions);
                                        } catch (Exception ex) {
                                            LogManager.logD(TAG, "HarmonyOS Vulkan配置失败: " + ex.getMessage());
                                        }
                                    }
                                } else if (errorMsg.contains("OpenGL")) {
                                    LogManager.logW(TAG, "建议: OpenGL ES计算着色器不可用 - 检查设备是否支持OpenGL ES 3.1+");
                                    if (isHarmonyOS) {
                                        LogManager.logW(TAG, "HarmonyOS提示: 在开发者选项中启用'强制进行GPU渲染'可能有助于OpenGL加速");
                                        try {
                                            HarmonyOSGPUAdapter.optimizeSystemSettings();
                                        } catch (Exception ex) {
                                            LogManager.logD(TAG, "HarmonyOS系统设置优化失败: " + ex.getMessage());
                                        }
                                    }
                                } else if (errorMsg.contains("CUDA")) {
                                    LogManager.logW(TAG, "建议: CUDA仅支持NVIDIA GPU，当前设备可能使用其他GPU");
                                }
                            }
                            
                        } catch (Exception e) {
                            // 其他未知错误
                            LogManager.logE(TAG, String.format("✗ %s加速启用失败 (未知错误): %s", gpuNames[i], e.getClass().getSimpleName() + ": " + e.getMessage()));
                        }
                    }
                    
                    if (!gpuEnabled) {
                        LogManager.logW(TAG, "所有GPU加速方式均失败，将使用CPU模式");
                        
                        // 执行GPU诊断
                        try {
                            String diagnosticReport = com.example.starlocalrag.GPUDiagnosticTool.performFullDiagnosis(context);
                            LogManager.logI(TAG, "GPU诊断报告:\n" + diagnosticReport);
                        } catch (Exception e) {
                            LogManager.logE(TAG, "GPU诊断失败: " + e.getMessage(), e);
                        }
                        
                        // 提供针对性建议
                        if (isHarmonyOS) {
                            LogManager.logI(TAG, "HarmonyOS建议: 1) 确保系统版本支持GPU加速 2) 检查开发者选项中的硬件加速设置 3) 尝试重启应用");
                            
                            // 尝试Mali GPU性能监控
                            try {
                                MaliGPUOptimizer.monitorMaliPerformance();
                                MaliGPUOptimizer.optimizeFrequencyGovernor();
                            } catch (Exception e) {
                                LogManager.logD(TAG, "Mali GPU性能监控失败: " + e.getMessage());
                            }
                        } else {
                            LogManager.logI(TAG, "通用建议: 1) 检查设备GPU驱动版本 2) 确认应用权限设置 3) 尝试在开发者选项中启用硬件加速");
                        }
                        
                        LogManager.logI(TAG, "CPU模式性能提示: 虽然无法使用GPU加速，但CPU模式仍可正常运行，只是速度相对较慢");
                        
                        // 检查Mali GPU特性支持
                        try {
                            Map<String, Boolean> maliFeatures = MaliGPUOptimizer.checkMaliFeatureSupport(context);
                            LogManager.logI(TAG, "Mali GPU特性支持: " + maliFeatures.toString());
                        } catch (Exception e) {
                            LogManager.logD(TAG, "Mali GPU特性检查失败: " + e.getMessage());
                        }
                    }
                }
                
                // 6. 加载ONNX模型
                File modelFile = new File(modelDir, "model.onnx");
                if (!modelFile.exists()) {
                    throw new IOException("模型文件不存在: " + modelFile.getPath());
                }
                
                LogManager.logD(TAG, "开始加载ONNX模型: " + modelFile.getPath());
                ortSession = ortEnvironment.createSession(modelFile.getPath(), sessionOptions);
                
                // 7. 打印模型信息
                Map<String, NodeInfo> inputInfo = ortSession.getInputInfo();
                Map<String, NodeInfo> outputInfo = ortSession.getOutputInfo();
                
                LogManager.logD(TAG, "模型输入数量: " + inputInfo.size());
                for (String inputName : inputInfo.keySet()) {
                    LogManager.logD(TAG, "模型输入: " + inputName + ", 类型: " + inputInfo.get(inputName).getInfo());
                }
                
                LogManager.logD(TAG, "模型输出数量: " + outputInfo.size());
                for (String outputName : outputInfo.keySet()) {
                    LogManager.logD(TAG, "模型输出: " + outputName + ", 类型: " + outputInfo.get(outputName).getInfo());
                }
                
                // 7.5. 智能检测KV缓存支持
                boolean modelSupportsKVCache = detectKVCacheSupport(inputInfo);
                if (modelConfig != null) {
                    if (modelSupportsKVCache && !modelConfig.enableKVCache) {
                        LogManager.logI(TAG, "检测到模型支持KV缓存，自动启用以提升性能");
                        modelConfig.enableKVCache = true;
                    } else if (!modelSupportsKVCache && modelConfig.enableKVCache) {
                        LogManager.logW(TAG, "检测到模型不支持KV缓存，自动禁用以避免兼容性问题");
                        modelConfig.enableKVCache = false;
                    }
                    LogManager.logI(TAG, "KV缓存状态: " + (modelConfig.enableKVCache ? "启用" : "禁用"));
                }
                
                // 8. 检测模型类型并初始化相应处理器
                // 默认为ONNX模型
                modelType = "onnx";
                
                // 如果模型配置中指定了模型类型，则使用配置中的类型
                if (modelConfig != null && modelConfig.modelType != null) {
                    modelType = modelConfig.modelType.toLowerCase();
                }
                
                // 根据模型类型初始化处理器
                if ("onnx".equals(modelType)) {
                    LogManager.logI(TAG, "初始化ONNX处理器");
                    localLlmOnnxHandler = new LocalLLMOnnxHandler(context, ortEnvironment, ortSession, modelConfig);
                } else {
                    LogManager.logW(TAG, "未知模型类型: " + modelType + "，默认使用ONNX处理器");
                    modelType = "onnx";
                    localLlmOnnxHandler = new LocalLLMOnnxHandler(context, ortEnvironment, ortSession, modelConfig);
                }
                
                // 9. 标记为已加载
                modelLoaded.set(true);
                LogManager.logI(TAG, "模型加载完成: " + modelName + ", 类型: " + modelType);
                logMemoryInfo();
                
                // 回调成功
                if (callback != null) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onComplete("模型加载成功: " + modelName);
                    });
                }
                
            } catch (Exception e) {
                LogManager.logE(TAG, "加载模型失败: " + e.getMessage(), e);
                currentModelName = null;
                
                // 回调错误
                if (callback != null) {
                    final String errorMessage = e.getMessage();
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onError("加载模型失败: " + errorMessage);
                    });
                }
            } finally {
                // 标记为不再加载中
                modelLoading.set(false);
            }
        });
    }
    
    /**
     * 加载模型配置
     * @param configFile 配置文件
     * @throws Exception 异常
     */
    private void loadModelConfig(File configFile) throws Exception {
        LogManager.logD(TAG, "加载模型配置: " + configFile.getPath());
        
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(configFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
        }
        
        JSONObject config = new JSONObject(content.toString());
        
        String modelType = config.optString("model_type", "unknown");
        int vocabSize = config.optInt("vocab_size", 32000);
        int hiddenSize = config.optInt("hidden_size", 4096);
        int numLayers = config.optInt("num_hidden_layers", 32);
        int numHeads = config.optInt("num_attention_heads", 32);
        
        modelConfig = new ModelConfig(modelType, vocabSize, hiddenSize, numLayers, numHeads);
        // 设置模型路径
        modelConfig.setModelPath(configFile.getParentFile().getAbsolutePath());
        
        // 获取特殊token
        if (config.has("bos_token_id")) {
            bosToken = config.getInt("bos_token_id");
            modelConfig.setBosToken(bosToken);
        }
        if (config.has("eos_token_id")) {
            eosToken = config.getInt("eos_token_id");
            modelConfig.setEosToken(eosToken);
        }
        if (config.has("pad_token_id")) {
            padToken = config.getInt("pad_token_id");
        }
        
        // 解析量化相关配置
        parseQuantizationConfig(config, modelConfig, configFile);
        
        // 动态调整maxSequenceLength，结合配置的最大序列长度
        int configuredMaxSeqLength = ConfigManager.getMaxSequenceLength(context);
        int dynamicMaxSeqLength = calculateDynamicSequenceLength(configuredMaxSeqLength);
        modelConfig.setMaxSequenceLength(dynamicMaxSeqLength);
        
        LogManager.logI(TAG, String.format("序列长度动态调整: 配置值=%d, 计算得出maxSequenceLength=%d", 
            configuredMaxSeqLength, dynamicMaxSeqLength));
        
        LogManager.logD(TAG, String.format("模型配置: 类型=%s, 词汇表大小=%d, 隐藏层大小=%d, 层数=%d, 注意力头数=%d",
            modelType, vocabSize, hiddenSize, numLayers, numHeads));
        LogManager.logD(TAG, String.format("特殊token: BOS=%d, EOS=%d, PAD=%d", bosToken, eosToken, padToken));
        
        if (modelConfig.isQuantized()) {
            LogManager.logI(TAG, String.format("量化模型配置: 类型=%s, 缩放因子=%.4f, 零点=%d", 
                modelConfig.getQuantizationType(), modelConfig.getQuantizationScale(), modelConfig.getQuantizationZeroPoint()));
        }
    }
    
    /**
     * 应用量化模型优化配置
     * @param sessionOptions ONNX Runtime会话选项
     * @param modelConfig 模型配置
     */
    private void applyQuantizationOptimizations(SessionOptions sessionOptions, ModelConfig modelConfig) {
        try {
            String quantType = modelConfig.getQuantizationType();
            LogManager.logI(TAG, "应用量化模型优化配置，类型: " + quantType);
            
            // INT8量化模型专门优化
            if ("int8".equals(quantType)) {
                // 启用INT8优化
                sessionOptions.addConfigEntry("session.disable_prepacking", "0");
                sessionOptions.addConfigEntry("session.enable_cpu_mem_arena", "1");
                sessionOptions.addConfigEntry("session.enable_mem_pattern", "1");
                
                // INT8特定的CPU优化
                sessionOptions.addConfigEntry("session.use_env_allocators", "1");
                sessionOptions.addConfigEntry("session.enable_quant_qdq_cleanup", "1");
                
                // 张量内存管理优化
                sessionOptions.addConfigEntry("session.enable_tensor_memory_reuse", "1");
                sessionOptions.addConfigEntry("session.tensor_memory_pool_size", "64"); // 64MB张量内存池
                sessionOptions.addConfigEntry("session.enable_memory_efficient_attention", "1");
                
                // ARM NEON指令集优化
                sessionOptions.addConfigEntry("session.enable_neon_optimization", "1");
                sessionOptions.addConfigEntry("session.use_arm_neon", "1");
                sessionOptions.addConfigEntry("session.enable_simd_optimization", "1");
                
                // INT8量化优化 - 不重复设置线程配置，使用之前的线程和亲和性配置
                LogManager.logI(TAG, "INT8量化优化已启用，ARM NEON优化: 已启用");
            }
            
            // INT4量化模型优化
            else if ("int4".equals(quantType)) {
                sessionOptions.addConfigEntry("session.enable_cpu_mem_arena", "1");
                sessionOptions.addConfigEntry("session.enable_mem_pattern", "1");
                sessionOptions.addConfigEntry("session.use_env_allocators", "1");
                
                LogManager.logI(TAG, "INT4量化优化已启用");
            }
            
            // KV缓存优化
            if (modelConfig.isEnableKVCache()) {
                sessionOptions.addConfigEntry("session.enable_kv_cache", "1");
                sessionOptions.addConfigEntry("session.kv_cache_max_size", String.valueOf(modelConfig.getMaxSequenceLength()));
                LogManager.logI(TAG, "KV缓存优化已启用，最大序列长度: " + modelConfig.getMaxSequenceLength());
            }
            
            // 批处理和并行优化
            int maxBatchSize = Math.max(modelConfig.getMaxBatchSize(), 2); // 默认最小批处理大小为2
            modelConfig.setMaxBatchSize(maxBatchSize);
            
            sessionOptions.addConfigEntry("session.max_batch_size", String.valueOf(maxBatchSize));
            sessionOptions.addConfigEntry("session.enable_dynamic_batching", "1");
            sessionOptions.addConfigEntry("session.enable_parallel_execution", "1");
            sessionOptions.addConfigEntry("session.batch_timeout_ms", "50"); // 50ms批处理超时
            sessionOptions.addConfigEntry("session.enable_batch_optimization", "1");
            
            LogManager.logI(TAG, "批处理和并行优化已启用，最大批处理大小: " + maxBatchSize);
            
            // 内存访问模式优化
            sessionOptions.addConfigEntry("session.enable_memory_pattern_optimization", "1");
            sessionOptions.addConfigEntry("session.memory_access_pattern", "sequential");
            sessionOptions.addConfigEntry("session.enable_cache_friendly_layout", "1");
            sessionOptions.addConfigEntry("session.prefetch_distance", "2"); // 预取距离
            
            // 内存优化 - 针对量化模型的特殊内存管理
            sessionOptions.addConfigEntry("session.memory_limit_mb", "512"); // 量化模型内存需求较小
            sessionOptions.addConfigEntry("session.enable_memory_efficient_attention", "1");
            
            // 垃圾回收优化
            sessionOptions.addConfigEntry("session.enable_gc_optimization", "1");
            sessionOptions.addConfigEntry("session.gc_threshold_mb", "256"); // 256MB触发GC
            sessionOptions.addConfigEntry("session.enable_memory_pool", "1");
            sessionOptions.addConfigEntry("session.memory_pool_size_mb", "128"); // 128MB内存池
            
            // 系统级优化
            sessionOptions.addConfigEntry("session.enable_low_latency_mode", "1");
            sessionOptions.addConfigEntry("session.cpu_affinity_enabled", "1");
            sessionOptions.addConfigEntry("session.thread_priority", "high");
            
            // 量化特定的数值精度设置
            if (modelConfig.getQuantizationScale() > 0) {
                sessionOptions.addConfigEntry("session.quant_scale", String.valueOf(modelConfig.getQuantizationScale()));
                sessionOptions.addConfigEntry("session.quant_zero_point", String.valueOf(modelConfig.getQuantizationZeroPoint()));
            }
            
            LogManager.logI(TAG, "量化模型优化配置完成");
            
        } catch (Exception e) {
            LogManager.logW(TAG, "应用量化优化失败: " + e.getMessage());
        }
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
                        modelConfig.setQuantizationScale(0.1f); // INT8典型缩放因子
                        modelConfig.setQuantizationZeroPoint(128); // INT8典型零点
                    } else if ("int4".equals(quantType)) {
                        modelConfig.setQuantizationScale(0.2f); // INT4典型缩放因子
                        modelConfig.setQuantizationZeroPoint(8); // INT4典型零点
                    }
                }
                
                // KV缓存配置 - 默认禁用以避免兼容性问题
                boolean enableKVCache = config.optBoolean("enable_kv_cache", false);
                modelConfig.setEnableKVCache(enableKVCache);
                
                // 如果启用了KV缓存，记录警告信息
                if (enableKVCache) {
                    LogManager.logW(TAG, "KV缓存已启用，请确保模型支持KV缓存输入张量");
                }
                
                // 批处理配置
                int maxBatchSize = config.optInt("max_batch_size", 1);
                modelConfig.setMaxBatchSize(maxBatchSize);
                
                // 序列长度配置
                int maxSeqLen = config.optInt("max_sequence_length", 2048);
                modelConfig.setMaxSequenceLength(maxSeqLen);
                
                LogManager.logI(TAG, "检测到量化模型，类型: " + quantType + ", 启用优化配置");
            }
            
        } catch (Exception e) {
            LogManager.logW(TAG, "解析量化配置失败，使用默认配置: " + e.getMessage());
            // 即使解析失败，也尝试从文件名推断，默认启用int8动态量化
            String modelFileName = configFile.getParentFile().getName().toLowerCase();
            if (modelFileName.contains("int8") || modelFileName.contains("quant")) {
                modelConfig.setQuantized(true);
                modelConfig.setQuantizationType("int8");
                LogManager.logI(TAG, "从文件名推断为INT8量化模型");
            } else {
                // 默认启用int8动态量化以提升性能
                modelConfig.setQuantized(true);
                modelConfig.setQuantizationType("int8");
                modelConfig.setQuantizationScale(0.1f);
                modelConfig.setQuantizationZeroPoint(128);
                LogManager.logI(TAG, "默认启用INT8动态量化优化");
            }
        }
    }
    
    /**
     * 检查词汇表文件
     * @param tokenizerFile 词汇表文件
     * @throws Exception 异常
     */
    private void checkTokenizerFile(File tokenizerFile) throws Exception {
        LogManager.logD(TAG, "检查词汇表文件: " + tokenizerFile.getPath());
        
        // 直接检查文件是否存在，不需要读取内容
        // Rust tokenizer 会处理词汇表和特殊 token
        if (!tokenizerFile.exists()) {
            throw new IOException("词汇表文件不存在: " + tokenizerFile.getPath());
        }
        
        LogManager.logD(TAG, "词汇表文件检查完成，将由 Rust tokenizer 处理");
    }
    
    /**
     * 推理接口
     * @param prompt 输入提示词
     * @param callback 回调接口
     */
    public void inference(String prompt, LocalLlmCallback callback) {
        // 重置停止标志
        resetStopFlag();
        
        // 实现推理逻辑
        try {
            // 配置管理
            int maxTokenLength = 512; // 默认最大token长度
            // 从配置中获取思考模式设置
            boolean thinkingMode = !ConfigManager.getNoThinking(context); // 注意：no_thinking=true表示禁用思考模式
            float temperature = 0.7f; // 温度采样参数
            int topK = 5; // top-k 采样参数
            
            LogManager.logD(TAG, "思考模式设置: " + (thinkingMode ? "启用" : "禁用"));

            LogManager.logD(TAG, "开始调用本地LLM流式推理，提示词长度: " + prompt.length());

            // 使用流式推理接口
            localLlmOnnxHandler.inferenceStream(prompt, maxTokenLength, thinkingMode, temperature, topK, 
                new LocalLLMOnnxHandler.StreamingCallback() {
                    @Override
                    public void onToken(String token) {
                        // 检查是否应该停止推理
                        if (shouldStopInference()) {
                            LogManager.logD(TAG, "推理被用户停止");
                            callback.onError("推理被用户停止");
                            return;
                        }
                        LogManager.logD(TAG, "收到流式token: " + (token.length() > 20 ? token.substring(0, 20) + "..." : token));
                        callback.onToken(token);
                    }

                    @Override
                    public void onComplete(String fullResponse) {
                        LogManager.logD(TAG, "流式生成完成，总长度: " + fullResponse.length());
                        callback.onComplete(fullResponse);
                    }

                    @Override
                    public void onError(String errorMessage) {
                        LogManager.logE(TAG, "流式生成错误: " + errorMessage);
                        callback.onError(errorMessage);
                    }
                }, this); // 传递LocalLlmHandler实例
        } catch (Exception e) {
            LogManager.logE(TAG, "调用本地LLM流式推理异常", e);
            callback.onError(e.getMessage());
        }
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
        if (localLlmOnnxHandler == null) {
            LogManager.logE(TAG, "模型未加载，无法进行批处理推理");
            String[] errorResults = new String[inputTexts.length];
            for (int i = 0; i < inputTexts.length; i++) {
                errorResults[i] = "模型未加载";
            }
            return errorResults;
        }
        
        if (modelConfig == null || modelConfig.getMaxBatchSize() <= 1) {
            LogManager.logW(TAG, "当前模型不支持批处理，将使用单序列推理");
            // 回退到单序列推理
            String[] results = new String[inputTexts.length];
            for (int i = 0; i < inputTexts.length; i++) {
                // 使用同步推理方法，需要实现一个简化版本
                try {
                    results[i] = localLlmOnnxHandler.inference(inputTexts[i], maxTokens, false, temperature, topK);
                } catch (Exception e) {
                    LogManager.logE(TAG, "单序列推理失败: " + e.getMessage(), e);
                    results[i] = "推理失败: " + e.getMessage();
                }
            }
            return results;
        }
        
        LogManager.logI(TAG, "开始批处理推理，输入序列数: " + inputTexts.length + ", 最大批处理大小: " + modelConfig.getMaxBatchSize());
        
        // 创建流式回调适配器
         LocalLLMOnnxHandler.StreamingCallback streamCallback = null;
         if (callback != null) {
             streamCallback = new LocalLLMOnnxHandler.StreamingCallback() {
                 @Override
                 public void onToken(String token) {
                     callback.onTokenGenerated(token);
                 }
                 
                 @Override
                 public void onComplete(String fullResponse) {
                     callback.onComplete(fullResponse);
                 }
                 
                 @Override
                 public void onError(String errorMessage) {
                     callback.onError(errorMessage);
                 }
             };
         }
        
        return localLlmOnnxHandler.inferenceStreamBatch(inputTexts, maxTokens, temperature, topK, topP, streamCallback);
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
        LogManager.logI(TAG, "[停止调试] 收到停止推理请求");
        LogManager.logD(TAG, "[停止调试] 停止标志当前状态: " + shouldStopInference.get());
        shouldStopInference.set(true);
        LogManager.logI(TAG, "[停止调试] ✓ 停止标志已设置为true");
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
     * 卸载模型
     */
    public void unloadModel() {
        // 卸载模型逻辑
        LogManager.logD(TAG, "卸载模型");
        // 注意：LocalLLMOnnxHandler没有close方法
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
}
