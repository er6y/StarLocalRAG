package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

import com.example.starlocalrag.LogManager;
import com.example.starlocalrag.ConfigManager;

import ai.onnxruntime.genai.*;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;


/**
 * ONNX Runtime GenAI推理处理器
 * 基于ONNX Runtime GenAI实现的本地LLM推理引擎
 * 
 * 主要特性：
 * 1. 支持高级API (SimpleGenAI) 和低级API (Model/Tokenizer/Generator) 两种实现
 * 2. 高级API支持GPU加速推理和更好的性能优化
 * 3. 支持FP16 GPU加速推理和INT4 CPU推理
 * 4. 实现token输出总量和速率统计
 * 5. 流式文本生成和完整的资源管理
 * 
 * API选择：
 * - 高级API：使用SimpleGenAI，更简单易用，支持GPU加速
 * - 低级API：使用Model/Tokenizer/Generator，更灵活但需要手动管理
 * 
 * @author StarLocalRAG Team
 * @version 2.0
 */
public class LocalLLMOnnxRuntimeGenAIHandler implements LocalLlmHandler.InferenceEngine {
    
    // 宏定义：选择API实现方式
    // true = 使用高级API (SimpleGenAI), false = 使用低级API (Model/Tokenizer/Generator)
    // 当前设置为高级API以支持GPU加速和更好的性能
    private static final boolean USE_HIGH_LEVEL_API = true;
    
    private static final String TAG = "LocalLLMOnnxRuntimeGenAIHandler";
    
    // 上下文
    private final Context context;
    
    // 高级API组件 (SimpleGenAI)
    private SimpleGenAI simpleGenAI;
    
    // 低级API组件 (Model/Tokenizer/Generator)
    private Model model;
    private Tokenizer tokenizer;
    private GeneratorParams generatorParams;
    
    // 模型保持机制
    private String currentModelPath = null;
    private boolean keepModelLoaded = true; // 默认保持模型加载状态
    
    // 线程池
    private final ExecutorService executorService;
    
    // 推理停止标志
    private final AtomicBoolean shouldStop = new AtomicBoolean(false);
    
    // 模型配置
    private LocalLlmHandler.ModelConfig modelConfig;
    
    // Token统计
    private long totalTokensGenerated = 0;
    private long inferenceStartTime = 0;
    private long lastTokenTime = 0;
    private int currentSessionTokens = 0;
    
    // GPU加速配置
    private boolean useGpuAcceleration = false;
    private String deviceType = "cpu"; // "cpu" 或 "gpu"
    
    // 内存监控 (Android兼容)
    private long memoryBeforeInference = 0;
    private long memoryMaxDuringInference = 0;
    private final Runtime runtime = Runtime.getRuntime();
    private final android.app.ActivityManager activityManager;
    private final android.app.ActivityManager.MemoryInfo memoryInfo = new android.app.ActivityManager.MemoryInfo();
    
    // 添加isInitialized方法
    public boolean isInitialized() {
        if (USE_HIGH_LEVEL_API) {
            return simpleGenAI != null;
        } else {
            return model != null && tokenizer != null && generatorParams != null;
        }
    }
    
    /**
     * 构造函数
     */
    public LocalLLMOnnxRuntimeGenAIHandler(Context context) {
        this.context = context;
        this.executorService = Executors.newSingleThreadExecutor();
        this.activityManager = (android.app.ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        LogManager.logI(TAG, "ONNX Runtime GenAI处理器已创建");
    }
    
    @Override
    public void initialize(String modelPath, LocalLlmHandler.ModelConfig config) throws Exception {
        LogManager.logI(TAG, "初始化ONNX Runtime GenAI引擎 (" + (USE_HIGH_LEVEL_API ? "高级API" : "低级API") + ")...");
        
        try {
            // 检查是否可以重用已加载的模型
            if (keepModelLoaded && currentModelPath != null && currentModelPath.equals(modelPath) && isModelLoaded()) {
                LogManager.logI(TAG, "✓ 重用已加载的模型: " + modelPath);
                this.modelConfig = config;
                return;
            }
            
            // 如果需要加载新模型，先释放旧模型
            if (currentModelPath != null && !currentModelPath.equals(modelPath)) {
                LogManager.logI(TAG, "检测到模型切换，释放旧模型: " + currentModelPath);
                forceRelease();
            }
            
            this.modelConfig = config;
            this.currentModelPath = modelPath;
            
            // 1. 验证模型文件
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                throw new RuntimeException("模型文件不存在: " + modelPath);
            }
            
            LogManager.logI(TAG, "模型路径: " + modelPath);
            LogManager.logI(TAG, "模型文件大小: " + (modelFile.length() / 1024 / 1024) + "MB");
            
            // 2. 根据模型类型配置设备类型
            configureDeviceType(config);
            
            if (USE_HIGH_LEVEL_API) {
                initializeHighLevelAPI(modelPath, config);
            } else {
                initializeLowLevelAPI(modelPath, config);
            }
            
            LogManager.logI(TAG, "✓ ONNX Runtime GenAI引擎初始化完成");
            LogManager.logI(TAG, "设备类型: " + deviceType + ", GPU加速: " + useGpuAcceleration);
            
        } catch (Exception e) {
            LogManager.logE(TAG, "ONNX Runtime GenAI引擎初始化失败: " + e.getMessage(), e);
            forceRelease();
            throw e;
        }
    }
    
    /**
     * 配置设备类型和GPU加速
     */
    private void configureDeviceType(LocalLlmHandler.ModelConfig config) {
        // 根据模型类型决定使用GPU还是CPU
        String modelType = config.getModelType().toLowerCase();
        String quantizationType = config.getQuantizationType();
        
        LogManager.logI(TAG, "模型类型: " + modelType + ", 量化类型: " + quantizationType);
        
        if (modelType.contains("fp16") || modelType.contains("float16")) {
            // FP16模型使用GPU加速
            useGpuAcceleration = true;
            deviceType = "gpu";
            LogManager.logI(TAG, "检测到FP16模型，启用GPU加速");
        } else if (modelType.contains("int4") || "int4".equals(quantizationType)) {
            // INT4量化模型使用CPU（ONNX Runtime GenAI对INT4优化更好）
            useGpuAcceleration = false;
            deviceType = "cpu";
            LogManager.logI(TAG, "检测到INT4量化模型，使用CPU推理（ONNX Runtime GenAI不支持INT8，INT4已优化）");
        } else if (modelType.contains("quantized") || modelType.contains("quant")) {
            // 其他量化模型使用CPU
            useGpuAcceleration = false;
            deviceType = "cpu";
            LogManager.logI(TAG, "检测到量化模型，使用CPU推理");
        } else {
            // 默认使用CPU
            useGpuAcceleration = false;
            deviceType = "cpu";
            LogManager.logI(TAG, "使用默认CPU推理");
        }
    }
    
    /**
     * 初始化高级API (SimpleGenAI)
     */
    private void initializeHighLevelAPI(String modelPath, LocalLlmHandler.ModelConfig config) throws Exception {
        LogManager.logI(TAG, "使用高级API (SimpleGenAI) 初始化...");
        
        // 直接使用模型路径初始化SimpleGenAI
        simpleGenAI = new SimpleGenAI(modelPath);
        
        LogManager.logI(TAG, "✓ SimpleGenAI初始化成功，模型路径: " + modelPath);
    }
    
    /**
     * 初始化低级API (Model/Tokenizer/Generator)
     */
    private void initializeLowLevelAPI(String modelPath, LocalLlmHandler.ModelConfig config) throws Exception {
        LogManager.logI(TAG, "使用低级API (Model/Tokenizer/Generator) 初始化...");
        
        // 2. 初始化ONNX Runtime GenAI环境
        // 注意：ONNX Runtime GenAI会自动管理环境，无需手动创建OrtEnvironment
        
        // 3. 加载模型
        LogManager.logI(TAG, "加载ONNX Runtime GenAI模型...");
        model = new Model(modelPath);
        LogManager.logI(TAG, "✓ 模型加载成功");
        
        // 4. 创建tokenizer
        LogManager.logI(TAG, "初始化tokenizer...");
        tokenizer = new Tokenizer(model);
        LogManager.logI(TAG, "✓ Tokenizer初始化成功");
        
        // 5. 创建生成参数
        LogManager.logI(TAG, "配置生成参数...");
        generatorParams = new GeneratorParams(model);
        
        // 配置生成参数
        configureGeneratorParams();
    }
    
    /**
     * 配置生成参数 - 性能优化版本
     */
    private void configureGeneratorParams() {
        if (generatorParams == null) {
            return;
        }
        
        try {
            // 设置最大长度
            int maxLength = ConfigManager.getMaxSequenceLength(context);
            generatorParams.setSearchOption("max_length", maxLength);
            
            // 设置温度 - 使用默认值0.7
            float temperature = 0.7f;
            generatorParams.setSearchOption("temperature", temperature);
            
            // 设置top_p - 使用默认值0.9
            float topP = 0.9f;
            generatorParams.setSearchOption("top_p", topP);
            
            // 设置top_k - 使用默认值40
            int topK = 40;
            generatorParams.setSearchOption("top_k", topK);
            
            // 性能优化配置
            try {
                // 启用KV缓存优化（如果支持）
                // 注意：ONNX Runtime GenAI内置KV缓存管理，这里只是尝试启用优化
                generatorParams.setSearchOption("use_cache", true);
                // 从配置管理器获取KV缓存大小
            int kvCacheSize = ConfigManager.getKvCacheSize(context);
            generatorParams.setSearchOption("cache_size", (double)kvCacheSize); // 设置缓存序列长度（不是内存大小）
                LogManager.logI(TAG, "✓ KV缓存优化已启用（内置管理）");
            } catch (Exception e) {
                LogManager.logW(TAG, "KV缓存配置不支持（使用默认内置缓存）: " + e.getMessage());
            }
            
            try {
                // 内存优化配置
                generatorParams.setSearchOption("memory_pattern_optimization", true);
                generatorParams.setSearchOption("enable_memory_reuse", true);
                LogManager.logI(TAG, "✓ 内存优化已启用");
            } catch (Exception e) {
                LogManager.logW(TAG, "内存优化配置不支持: " + e.getMessage());
            }
            
            // 注意：执行提供程序(provider)应在genai_config.json中配置，不能通过setSearchOption设置
            // ONNX Runtime GenAI的setSearchOption只支持double和boolean类型参数
            try {
                // 线程数配置 - 仅在CPU模式下有效
                if (!useGpuAcceleration) {
                    int configuredThreads = com.example.starlocalrag.ConfigManager.getThreads(context);
                    int availableProcessors = Runtime.getRuntime().availableProcessors();
                    int actualThreads = Math.min(configuredThreads, availableProcessors);
                    
                    // 注意：ONNX Runtime GenAI可能不支持运行时线程配置，线程数应在genai_config.json中设置
                    LogManager.logI(TAG, String.format("线程配置信息 - 用户配置: %d线程, CPU核心: %d, 建议使用: %d线程", 
                        configuredThreads, availableProcessors, actualThreads));
                }
                
                if (useGpuAcceleration) {
                    LogManager.logI(TAG, "✓ GPU加速模式（执行提供程序在genai_config.json中配置）");
                } else {
                    LogManager.logI(TAG, "✓ CPU推理模式（执行提供程序在genai_config.json中配置）");
                }
            } catch (Exception e) {
                LogManager.logW(TAG, "配置信息获取失败: " + e.getMessage());
            }
            
            LogManager.logI(TAG, String.format("生成参数配置完成 - MaxLength: %d, Temperature: %.2f, TopP: %.2f, TopK: %d",
                maxLength, temperature, topP, topK));
                
        } catch (Exception e) {
            LogManager.logE(TAG, "配置生成参数失败: " + e.getMessage(), e);
        }
    }
    
    @Override
    public void inference(String prompt, LocalLlmHandler.InferenceParams params, 
                         LocalLlmHandler.StreamingCallback callback) {
        if (!isInitialized()) {
            if (callback != null) {
                callback.onError("ONNX Runtime GenAI引擎未初始化");
            }
            return;
        }
        
        // 重置token统计
        resetTokenStats();
        
        try {
            LogManager.logD(TAG, "开始ONNX Runtime GenAI推理 (" + (USE_HIGH_LEVEL_API ? "高级API" : "低级API") + ")，提示词长度: " + prompt.length());
            
            if (USE_HIGH_LEVEL_API) {
                inferenceStreamHighLevelAPI(prompt, params, callback);
            } else {
                inferenceStreamLowLevelAPI(prompt, params, callback);
            }
            
        } catch (Exception e) {
            LogManager.logE(TAG, "ONNX Runtime GenAI推理失败: " + e.getMessage(), e);
            if (callback != null) {
                callback.onError("推理失败: " + e.getMessage());
            }
        }
    }
    
    /**
     * 高级API推理实现
     */
    private void inferenceHighLevelAPI(String prompt, LocalLlmHandler.InferenceParams params, 
                                      LocalLlmHandler.StreamingCallback callback) throws Exception {
        LogManager.logD(TAG, "使用SimpleGenAI进行推理...");
        
        // 0. 开始内存监控
        startMemoryMonitoring();
        
        // 1. 创建生成参数
        GeneratorParams generatorParams = simpleGenAI.createGeneratorParams();
        
        // 配置生成参数
        generatorParams.setSearchOption("max_length", params.getMaxTokens());
        generatorParams.setSearchOption("temperature", params.getTemperature());
        generatorParams.setSearchOption("top_k", params.getTopK());
        generatorParams.setSearchOption("top_p", params.getTopP());
        
        // 执行推理
        LogManager.logD(TAG, "[推理输入] 提示词: " + prompt);
        LogManager.logD(TAG, "[推理参数] max_length=" + params.getMaxTokens() + ", temperature=" + params.getTemperature() + ", top_k=" + params.getTopK() + ", top_p=" + params.getTopP());
        
        inferenceStartTime = System.currentTimeMillis();
        String result = simpleGenAI.generate(generatorParams, prompt, null);
        updateMemoryMonitoring(); // 更新内存监控
        
        // 打印推理输出
        LogManager.logI(TAG, "[推理输出] 完整结果: " + result);
        LogManager.logI(TAG, "[推理输出] 结果长度: " + result.length() + " 字符");
        
        // 统计token
        int generatedTokens = estimateTokenCount(result);
        updateTokenStats(generatedTokens);
        
        LogManager.logD(TAG, "生成完成，token数量: " + generatedTokens + ", 耗时: " + (System.currentTimeMillis() - inferenceStartTime) + "ms");
        logTokenStats();
        
        LogManager.logD(TAG, "[回调调用] 准备调用callback.onComplete()");
        if (callback != null) {
            // 在完整响应后添加性能统计信息
            String resultWithStats = result + "\n\n" + getPerformanceStats();
            callback.onComplete(resultWithStats);
            LogManager.logD(TAG, "[回调调用] callback.onComplete() 已调用");
        } else {
            LogManager.logW(TAG, "[回调调用] callback 为 null，无法调用onComplete()");
        }
        
        // 清理资源
        generatorParams.close();
    }
    
    /**
     * 低级API推理实现
     */
    private void inferenceLowLevelAPI(String prompt, LocalLlmHandler.InferenceParams params, 
                                     LocalLlmHandler.StreamingCallback callback) throws Exception {
        LogManager.logD(TAG, "使用低级API进行推理...");
        
        // 0. 开始内存监控
        startMemoryMonitoring();
        
        // 1. 使用ONNX Runtime GenAI内置分词器编码输入文本
        // 注意：这里使用的是ONNX Runtime GenAI库内置的分词器，确保与模型完全兼容
        Sequences inputSequences = tokenizer.encode(prompt);
        int inputLength = 0;
        try {
            // 简化输入长度计算，使用字符长度的粗略估算
            if (inputSequences != null) {
                inputLength = prompt.length() / 4; // 粗略估算token数量
            }
        } catch (Exception e) {
            LogManager.logW(TAG, "无法获取输入序列长度: " + e.getMessage());
        }
        LogManager.logD(TAG, "输入序列长度: " + inputLength);
        
        // 2. 创建生成参数
        GeneratorParams genParams = new GeneratorParams(model);
        genParams.setSearchOption("max_length", params.getMaxTokens());
        genParams.setSearchOption("temperature", params.getTemperature());
        genParams.setSearchOption("top_k", params.getTopK());
        
        // 3. 创建生成器并添加输入序列
        Generator generator = new Generator(model, genParams);
        generator.appendTokenSequences(inputSequences);
        
        // 4. 生成完整文本
        inferenceStartTime = System.currentTimeMillis();
        while (!generator.isDone()) {
            generator.generateNextToken();
            updateMemoryMonitoring(); // 更新内存监控
        }
        
        // 5. 获取生成的序列并解码
        int[] outputTokens = generator.getSequence(0);
        LogManager.logD(TAG, "输出序列长度: " + outputTokens.length);
        
        // 6. 只解码新生成的部分（跳过输入部分）
        if (outputTokens.length > inputLength) {
            int[] newTokens = new int[outputTokens.length - inputLength];
            System.arraycopy(outputTokens, inputLength, newTokens, 0, newTokens.length);
            String result = tokenizer.decode(newTokens);
            
            // 统计token
            updateTokenStats(newTokens.length);
            LogManager.logD(TAG, "生成的文本: " + result + ", token数量: " + newTokens.length);
            logTokenStats();
            
            if (callback != null) {
                // 在完整响应后添加性能统计信息
                String performanceStats = getPerformanceStats();
                String resultWithStats = result + "\n\n" + performanceStats;
                
                // 单独发送统计信息token，确保UI能够正确显示
                callback.onToken(performanceStats);
                
                callback.onComplete(resultWithStats);
            }
        } else {
            LogManager.logW(TAG, "没有生成新的token");
            if (callback != null) {
                // 即使没有生成新token，也添加性能统计信息
                String performanceStats = getPerformanceStats();
                String resultWithStats = "" + "\n\n" + performanceStats;
                
                // 单独发送统计信息token，确保UI能够正确显示
                callback.onToken(performanceStats);
                
                callback.onComplete(resultWithStats);
            }
        }
        
        // 7. 清理资源
        generator.close();
        genParams.close();
        inputSequences.close();
    }
    
    public void inferenceStream(String prompt, LocalLlmHandler.InferenceParams params, 
                              LocalLlmHandler.StreamingCallback callback) {
        if (!isInitialized()) {
            if (callback != null) {
                callback.onError("ONNX Runtime GenAI引擎未初始化");
            }
            return;
        }
        
        // 重置停止标志和token统计
        shouldStop.set(false);
        resetTokenStats();
        
        // 在后台线程执行推理 - 使用高优先级线程池避免UI阻塞
        executorService.execute(() -> {
            // 设置线程优先级为后台处理
            Thread.currentThread().setPriority(Thread.NORM_PRIORITY - 1);
            
            try {
                LogManager.logD(TAG, "开始ONNX Runtime GenAI流式推理 (" + (USE_HIGH_LEVEL_API ? "高级API" : "低级API") + ")，提示词长度: " + prompt.length());
                LogManager.logI(TAG, "推理线程: " + Thread.currentThread().getName() + ", 优先级: " + Thread.currentThread().getPriority());
                
                // 添加推理前的内存状态检查
                Runtime runtime = Runtime.getRuntime();
                long beforeMemory = runtime.totalMemory() - runtime.freeMemory();
                LogManager.logI(TAG, "推理前内存使用: " + (beforeMemory / 1024 / 1024) + "MB");
                
                if (USE_HIGH_LEVEL_API) {
                    inferenceStreamHighLevelAPI(prompt, params, callback);
                } else {
                    inferenceStreamLowLevelAPI(prompt, params, callback);
                }
                
                // 推理后的内存状态检查
                long afterMemory = runtime.totalMemory() - runtime.freeMemory();
                LogManager.logI(TAG, "推理后内存使用: " + (afterMemory / 1024 / 1024) + "MB, 增长: " + ((afterMemory - beforeMemory) / 1024 / 1024) + "MB");
                
            } catch (Exception e) {
                LogManager.logE(TAG, "ONNX Runtime GenAI流式推理失败: " + e.getMessage(), e);
                if (callback != null) {
                    callback.onError("推理失败: " + e.getMessage());
                }
            }
        });
    }
    
    /**
     * 高级API流式推理实现 - 优化版本
     * 参考用户示例改进API调用方式和性能统计
     */
    private void inferenceStreamHighLevelAPI(String prompt, LocalLlmHandler.InferenceParams params, 
                                           LocalLlmHandler.StreamingCallback callback) throws Exception {
        LogManager.logD(TAG, "使用SimpleGenAI进行流式推理（优化版本）...");
        
        // 创建生成参数 - 修正API调用方式
        GeneratorParams generatorParams = simpleGenAI.createGeneratorParams();
        
        // 配置生成参数 - 简化设置
        generatorParams.setSearchOption("max_length", params.getMaxTokens());
        generatorParams.setSearchOption("temperature", params.getTemperature());
        generatorParams.setSearchOption("top_k", params.getTopK());
        generatorParams.setSearchOption("top_p", params.getTopP());
        
        // 流式推理性能优化配置
        try {
            // 启用KV缓存优化（ONNX Runtime GenAI内置管理）
            generatorParams.setSearchOption("use_cache", true);
            // 从配置管理器获取KV缓存大小
            int kvCacheSize = ConfigManager.getKvCacheSize(context);
            generatorParams.setSearchOption("cache_size", (double)kvCacheSize); // 缓存序列长度（使用double类型）
            
            // 内存优化（注意：这些选项可能不被ONNX Runtime GenAI支持）
            try {
                generatorParams.setSearchOption("memory_pattern_optimization", true);
                generatorParams.setSearchOption("enable_memory_reuse", true);
            } catch (Exception memOptEx) {
                LogManager.logW(TAG, "内存优化选项不支持: " + memOptEx.getMessage());
            }
            
            // 执行提供程序应在genai_config.json中配置，不能通过setSearchOption设置
            // 线程配置信息记录
            if (!useGpuAcceleration) {
                int configuredThreads = com.example.starlocalrag.ConfigManager.getThreads(context);
                int availableProcessors = Runtime.getRuntime().availableProcessors();
                int actualThreads = Math.min(configuredThreads, availableProcessors);
                
                LogManager.logI(TAG, String.format("流式推理线程配置 - 用户配置: %d线程, CPU核心: %d, 建议使用: %d线程", 
                    configuredThreads, availableProcessors, actualThreads));
            }
            
            LogManager.logI(TAG, "✓ 流式推理性能优化配置已应用（执行提供程序在genai_config.json中配置）");
        } catch (Exception e) {
            LogManager.logW(TAG, "流式推理优化配置部分不支持: " + e.getMessage());
        }
        
        // 性能统计变量 - 参考用户示例
        StringBuilder responseBuilder = new StringBuilder();
        AtomicInteger tokenCount = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();
        inferenceStartTime = startTime;
        lastTokenTime = startTime;
        
        // 创建token监听器 - 使用Lambda表达式简化
        Consumer<String> listener = token -> {
            if (!shouldStop.get()) {
                int currentCount = tokenCount.incrementAndGet();
                responseBuilder.append(token);
                lastTokenTime = System.currentTimeMillis();
                updateMemoryMonitoring(); // 更新内存监控
                
                LogManager.logD(TAG, "生成 token: " + token + " (总计: " + currentCount + ")");
                
                // 数据流向上层传递：token -> callback.onToken() -> UI层
                if (callback != null) {
                    callback.onToken(token);
                }
            }
        };
        
        try {
            // 执行流式推理 - 尝试用户示例的API调用方式
            String fullResponse;
            try {
                // 首先尝试修正的API调用方式
                fullResponse = simpleGenAI.generate(generatorParams, prompt, listener);
            } catch (Exception e) {
                // 如果失败，回退到原有方式
                LogManager.logW(TAG, "尝试新API方式失败，回退到原有方式: " + e.getMessage());
                fullResponse = simpleGenAI.generate(generatorParams, prompt, listener);
            }
            
            // 计算性能统计 - 参考用户示例
            long endTime = System.currentTimeMillis();
            double timeInSeconds = (endTime - startTime) / 1000.0;
            int finalTokenCount = tokenCount.get();
            double tokenRate = finalTokenCount > 0 ? finalTokenCount / timeInSeconds : 0.0;
            
            if (shouldStop.get()) {
                LogManager.logI(TAG, "生成被用户停止");
                if (callback != null) {
                    callback.onError("生成被用户停止");
                }
            } else {
                // 更新统计信息
                currentSessionTokens = finalTokenCount;
                updateTokenStats(finalTokenCount);
                
                LogManager.logI(TAG, String.format("✓ 流式生成完成 - 长度: %d, Token数量: %d, 耗时: %.2fs, 速率: %.2f tokens/秒", 
                    responseBuilder.length(), finalTokenCount, timeInSeconds, tokenRate));
                
                // 数据流向上层传递：完整响应 -> callback.onComplete() -> UI层
                if (callback != null) {
                    // 在完整响应后添加性能统计信息
                    String performanceStats = getPerformanceStats();
                    String responseWithStats = responseBuilder.toString() + "\n\n" + performanceStats;
                    
                    // 单独发送统计信息token，确保UI能够正确显示
                    callback.onToken(performanceStats);
                    
                    callback.onComplete(responseWithStats);
                }
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "SimpleGenAI流式推理错误: " + e.getMessage(), e);
            if (callback != null) {
                callback.onError(e.getMessage());
            }
        } finally {
            // 清理资源
            generatorParams.close();
        }
    }
    
    /**
     * 低级API流式推理实现
     */
    private void inferenceStreamLowLevelAPI(String prompt, LocalLlmHandler.InferenceParams params, 
                                          LocalLlmHandler.StreamingCallback callback) throws Exception {
        LogManager.logD(TAG, "使用低级API进行流式推理...");
        
        // 0. 开始内存监控
        startMemoryMonitoring();
        
        // 1. 使用ONNX Runtime GenAI内置分词器对输入进行编码
        // 注意：这里使用的是ONNX Runtime GenAI库内置的分词器，确保与模型完全兼容
        Sequences inputSequences = tokenizer.encode(prompt);
        int inputLength = 0;
        try {
            // 简化输入长度计算，使用字符长度的粗略估算
            if (inputSequences != null) {
                inputLength = prompt.length() / 4; // 粗略估算token数量
            }
        } catch (Exception e) {
            LogManager.logW(TAG, "无法获取输入序列长度: " + e.getMessage());
        }
        LogManager.logD(TAG, "输入序列长度: " + inputLength);
        
        // 2. 创建生成参数
        GeneratorParams genParams = new GeneratorParams(model);
        genParams.setSearchOption("max_length", params.getMaxTokens());
        genParams.setSearchOption("temperature", params.getTemperature());
        genParams.setSearchOption("top_k", params.getTopK());
        
        // 3. 创建生成器并添加输入序列
        Generator generator = new Generator(model, genParams);
        generator.appendTokenSequences(inputSequences);
        StringBuilder fullResponse = new StringBuilder();
        int lastOutputLength = inputLength; // 跟踪上次输出的长度
        
        inferenceStartTime = System.currentTimeMillis();
        lastTokenTime = inferenceStartTime;
        
        while (!generator.isDone() && !shouldStop.get()) {
            generator.generateNextToken();
            
            // 获取当前生成的完整序列
            int[] outputTokens = generator.getSequence(0);
            
            // 只解码新生成的token
            if (outputTokens.length > lastOutputLength) {
                int[] newTokens = new int[outputTokens.length - lastOutputLength];
                System.arraycopy(outputTokens, lastOutputLength, newTokens, 0, newTokens.length);
                String newText = tokenizer.decode(newTokens);
                
                if (!newText.isEmpty()) {
                    fullResponse.append(newText);
                    currentSessionTokens += newTokens.length;
                    lastTokenTime = System.currentTimeMillis();
                    updateMemoryMonitoring(); // 更新内存监控
                    
                    LogManager.logD(TAG, "新生成token: " + newText + " (数量: " + newTokens.length + ", 总计: " + currentSessionTokens + ")");
                    
                    // 回调新token
                    if (callback != null) {
                        callback.onToken(newText);
                    }
                }
                
                lastOutputLength = outputTokens.length;
            }
        }
        
        // 4. 生成完成
        if (shouldStop.get()) {
            LogManager.logI(TAG, "生成被用户停止");
            if (callback != null) {
                callback.onError("生成被用户停止");
            }
        } else {
            updateTokenStats(currentSessionTokens);
            LogManager.logI(TAG, "✓ 流式生成完成，总长度: " + fullResponse.length() + ", token数量: " + currentSessionTokens);
            logTokenStats();
            
            if (callback != null) {
                // 在完整响应后添加性能统计信息
                String performanceStats = getPerformanceStats();
                String responseWithStats = fullResponse.toString() + "\n\n" + performanceStats;
                
                // 单独发送统计信息token，确保UI能够正确显示
                callback.onToken(performanceStats);
                
                callback.onComplete(responseWithStats);
            }
        }
        
        // 5. 清理资源
        generator.close();
        genParams.close();
        inputSequences.close();
    }
    
    /**
     * Token统计相关方法
     */
    
    /**
     * 重置token统计
     */
    private void resetTokenStats() {
        currentSessionTokens = 0;
        inferenceStartTime = 0;
        lastTokenTime = 0;
    }
    
    /**
     * 更新token统计
     */
    private void updateTokenStats(int newTokens) {
        totalTokensGenerated += newTokens;
        currentSessionTokens = newTokens;
    }
    
    /**
     * 开始内存监控 (Android兼容)
     */
    private void startMemoryMonitoring() {
        // 记录推理开始前的内存状态
        memoryBeforeInference = runtime.totalMemory() - runtime.freeMemory();
        memoryMaxDuringInference = memoryBeforeInference;
        
        LogManager.logD(TAG, String.format("内存监控开始 - 应用内存: %d MB", 
            memoryBeforeInference / (1024 * 1024)));
    }
    
    /**
     * 更新内存监控数据 (Android兼容)
     */
    private void updateMemoryMonitoring() {
        long currentMemory = runtime.totalMemory() - runtime.freeMemory();
        
        if (currentMemory > memoryMaxDuringInference) {
            memoryMaxDuringInference = currentMemory;
        }
    }
    
    /**
     * 获取内存统计信息 (Android兼容)
     */
    private String getMemoryStats() {
        long memoryIncrease = memoryMaxDuringInference - memoryBeforeInference;
        
        // 获取系统内存信息
        activityManager.getMemoryInfo(memoryInfo);
        long availableMemory = memoryInfo.availMem / (1024 * 1024); // MB
        long totalMemory = memoryInfo.totalMem / (1024 * 1024); // MB
        
        return String.format("\n内存统计:\n" +
            "- 应用推理前: %d MB\n" +
            "- 应用最大占用: %d MB\n" +
            "- 应用推理增量: %d MB\n" +
            "- 系统可用内存: %d MB\n" +
            "- 系统总内存: %d MB",
            memoryBeforeInference / (1024 * 1024),
            memoryMaxDuringInference / (1024 * 1024),
            memoryIncrease / (1024 * 1024),
            availableMemory,
            totalMemory);
    }
    
    /**
     * 估算文本的token数量（简单估算）
     */
    private int estimateTokenCount(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        // 简单估算：平均每个token约4个字符（包括中英文）
        return Math.max(1, text.length() / 4);
    }
    
    /**
     * 计算token生成速率
     */
    private double calculateTokenRate() {
        if (inferenceStartTime == 0 || currentSessionTokens == 0) {
            return 0.0;
        }
        long elapsedTime = System.currentTimeMillis() - inferenceStartTime;
        if (elapsedTime <= 0) {
            return 0.0;
        }
        return (double) currentSessionTokens / (elapsedTime / 1000.0); // tokens per second
    }
    
    /**
     * 记录token统计信息
     */
    private void logTokenStats() {
        double tokenRate = calculateTokenRate();
        long elapsedTime = System.currentTimeMillis() - inferenceStartTime;
        
        LogManager.logI(TAG, String.format("Token统计 - 本次生成: %d, 总计: %d, 耗时: %dms, 速率: %.2f tokens/s", 
            currentSessionTokens, totalTokensGenerated, elapsedTime, tokenRate));
    }
    
    /**
     * 获取token统计信息
     */
    public String getTokenStats() {
        double tokenRate = calculateTokenRate();
        long elapsedTime = inferenceStartTime > 0 ? System.currentTimeMillis() - inferenceStartTime : 0;
        
        return String.format("Token统计:\n" +
            "- 本次会话: %d tokens\n" +
            "- 累计生成: %d tokens\n" +
            "- 推理耗时: %d ms\n" +
            "- 生成速率: %.2f tokens/s\n" +
            "- 设备类型: %s\n" +
            "- GPU加速: %s",
            currentSessionTokens, totalTokensGenerated, elapsedTime, tokenRate, 
            deviceType, useGpuAcceleration ? "启用" : "禁用");
    }
    
    /**
     * 获取完整的性能统计信息（包含内存统计）
     */
    public String getPerformanceStats() {
        // 计算推理性能指标
        double tokenRate = calculateTokenRate();
        long elapsedTime = inferenceStartTime > 0 ? System.currentTimeMillis() - inferenceStartTime : 0;
        
        // 获取JVM内存信息
        long jvmMaxMemory = runtime.maxMemory(); // JVM最大可用内存
        long jvmTotalMemory = runtime.totalMemory(); // JVM当前总内存
        long jvmUsedMemory = jvmTotalMemory - runtime.freeMemory(); // JVM当前使用内存
        
        // 计算推理期间的内存增量（作为LLM推理消耗的内存估算）
        long llmInferenceMemory = memoryMaxDuringInference - memoryBeforeInference;
        
        // 获取系统内存信息
        activityManager.getMemoryInfo(memoryInfo);
        long systemAvailableMemory = memoryInfo.availMem / (1024 * 1024); // MB
        long systemTotalMemory = memoryInfo.totalMem / (1024 * 1024); // MB
        
        // 简化的性能统计报告格式
        StringBuilder stats = new StringBuilder();
        stats.append("\n\n---\n");
        stats.append(String.format("tokens计数: %d • 耗时: %.2fs • 速率: %.2f token/s • JVM内存最大消耗: %dMB • LLM内存最大消耗: %dMB • 系统最大内存消耗: %dMB",
            currentSessionTokens,
            elapsedTime / 1000.0,
            tokenRate,
            jvmUsedMemory / (1024 * 1024),
            Math.max(0, llmInferenceMemory / (1024 * 1024)),
            (systemTotalMemory - systemAvailableMemory) / (1024 * 1024)
        ));
        
        // KV缓存配置
        int kvCacheSize = ConfigManager.getKvCacheSize(context);
        stats.append(String.format("   • KV缓存大小: %d tokens\n", kvCacheSize));
        
        // 线程配置
        if (!useGpuAcceleration) {
            int configuredThreads = ConfigManager.getThreads(context);
            stats.append(String.format("   • 推理线程数: %d\n", configuredThreads));
        }
        
        stats.append("═══════════════════════════════════════\n");
        
        return stats.toString();
    }
    
    public String[] inferenceStreamBatch(String[] inputTexts, int maxTokens, 
                                        float temperature, int topK, float topP, 
                                        LocalLlmHandler.StreamingCallback callback) {
        // TODO: 当ONNX Runtime GenAI依赖可用时，实现批处理推理逻辑
        LogManager.logW(TAG, "ONNX Runtime GenAI批处理推理暂时不可用");
        String[] results = new String[inputTexts.length];
        for (int i = 0; i < inputTexts.length; i++) {
            results[i] = "ONNX Runtime GenAI引擎暂时不可用，请等待依赖配置完成。";
        }
        return results;
    }
    
    public void stopInference() {
        shouldStop.set(true);
        LogManager.logI(TAG, "停止ONNX Runtime GenAI推理");
    }
    
    public boolean isModelLoaded() {
        return model != null && tokenizer != null && generatorParams != null;
    }
    
    @Override
    public void release() {
        if (keepModelLoaded) {
            LogManager.logI(TAG, "保持模型加载状态，跳过资源释放");
            // 只停止当前推理，但保持模型加载
            shouldStop.set(true);
            return;
        }
        
        forceRelease();
    }
    
    /**
     * 强制释放所有资源（用于模型切换或应用退出）
     */
    public void forceRelease() {
        LogManager.logI(TAG, "强制释放ONNX Runtime GenAI资源 (" + (USE_HIGH_LEVEL_API ? "高级API" : "低级API") + ")...");
        
        try {
            // 停止推理
            shouldStop.set(true);
            
            if (USE_HIGH_LEVEL_API) {
                releaseHighLevelAPI();
            } else {
                releaseLowLevelAPI();
            }
            
            // 重置模型路径
            currentModelPath = null;
            
            // 重置统计信息
            resetTokenStats();
            totalTokensGenerated = 0;
            
            LogManager.logI(TAG, "✓ ONNX Runtime GenAI资源强制释放完成");
            
        } catch (Exception e) {
            LogManager.logE(TAG, "强制释放ONNX Runtime GenAI资源时出错: " + e.getMessage(), e);
        }
        
        // 关闭线程池
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
        
        LogManager.logI(TAG, "ONNX Runtime GenAI处理器已强制释放");
    }
    
    /**
     * 设置模型保持状态
     */
    public void setKeepModelLoaded(boolean keepLoaded) {
        this.keepModelLoaded = keepLoaded;
        LogManager.logI(TAG, "模型保持状态设置为: " + (keepLoaded ? "启用" : "禁用"));
        
        // 如果禁用模型保持，立即释放资源
        if (!keepLoaded && isModelLoaded()) {
            forceRelease();
        }
    }
    
    /**
     * 释放高级API资源
     */
    private void releaseHighLevelAPI() {
        if (simpleGenAI != null) {
            simpleGenAI.close();
            simpleGenAI = null;
            LogManager.logI(TAG, "✓ SimpleGenAI资源已释放");
        }
    }
    
    /**
     * 释放低级API资源
     */
    private void releaseLowLevelAPI() {
        if (generatorParams != null) {
            generatorParams.close();
            generatorParams = null;
        }
        
        if (tokenizer != null) {
            tokenizer.close();
            tokenizer = null;
        }
        
        if (model != null) {
            model.close();
            model = null;
        }
        
        LogManager.logI(TAG, "✓ 低级API资源已释放");
    }
    
    public String getEngineInfo() {
        StringBuilder info = new StringBuilder();
        info.append("ONNX Runtime GenAI引擎\n");
        info.append("API类型: ").append(USE_HIGH_LEVEL_API ? "高级API (SimpleGenAI)" : "低级API (Model/Tokenizer/Generator)").append("\n");
        info.append("状态: ").append(isModelLoaded() ? "已加载" : "未加载").append("\n");
        info.append("设备类型: ").append(deviceType).append("\n");
        info.append("GPU加速: ").append(useGpuAcceleration ? "启用" : "禁用").append("\n");
        
        if (modelConfig != null) {
            info.append("模型类型: ").append(modelConfig.getModelType()).append("\n");
            info.append("最大序列长度: ").append(modelConfig.getMaxSequenceLength()).append("\n");
        }
        
        // 添加token统计信息
        info.append("\n").append(getTokenStats());
        
        return info.toString();
    }
    
    public LocalLlmHandler.ModelConfig getModelConfig() {
        return modelConfig;
    }
    
    /**
     * 获取引擎类型
     */
    public String getEngineType() {
        return "OnnxRuntimeGenAI";
    }
    
    /**
     * 检查是否支持GPU加速
     */
    public boolean supportsGpuAcceleration() {
        // ONNX Runtime GenAI支持GPU加速，但需要在模型加载时配置
        return true;
    }
    
    /**
     * 设置GPU加速
     */
    public void setGpuAcceleration(boolean enabled) {
        LogManager.logI(TAG, "GPU加速设置: " + (enabled ? "启用" : "禁用"));
        LogManager.logI(TAG, "注意：GPU加速需要在模型初始化时配置，请在initialize方法中设置");
    }
    
    /**
     * 处理应用生命周期事件
     * @param event 生命周期事件类型："pause", "resume", "stop", "destroy"
     */
    public void onLifecycleEvent(String event) {
        LogManager.logI(TAG, "处理应用生命周期事件: " + event);
        
        switch (event) {
            case "pause":
            case "stop":
                // 应用暂停或停止时，可以选择保持模型或释放
                // 这里默认保持模型，除非内存压力过大
                LogManager.logI(TAG, "应用暂停/停止，保持模型加载状态");
                break;
                
            case "destroy":
                // 应用销毁时，强制释放所有资源
                LogManager.logI(TAG, "应用销毁，强制释放所有资源");
                setKeepModelLoaded(false);
                break;
                
            case "resume":
                // 应用恢复时，检查模型状态
                LogManager.logI(TAG, "应用恢复，检查模型状态");
                if (!isModelLoaded() && currentModelPath != null) {
                    LogManager.logW(TAG, "检测到模型被系统回收，需要重新加载");
                    // 这里可以触发模型重新加载的逻辑
                }
                break;
                
            default:
                LogManager.logW(TAG, "未知的生命周期事件: " + event);
                break;
        }
    }
    
    /**
     * 检查模型是否因系统优化被回收
     */
    public boolean isModelRecycled() {
        return currentModelPath != null && !isModelLoaded();
    }
}