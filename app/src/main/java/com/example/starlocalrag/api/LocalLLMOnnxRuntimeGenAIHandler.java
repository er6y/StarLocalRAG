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
import java.util.function.Consumer;

/**
 * ONNX Runtime GenAI推理处理器
 * 基于ONNX Runtime GenAI实现的本地LLM推理引擎
 * 
 * 主要特性：
 * 1. 使用ONNX Runtime GenAI的高级API
 * 2. 支持流式文本生成
 * 3. 内置tokenizer和生成逻辑
 * 4. 优化的内存管理
 * 5. 支持多种生成参数配置
 * 
 * TODO: 当ONNX Runtime GenAI依赖可用时，取消注释相关代码
 * 
 * @author StarLocalRAG Team
 * @version 1.0
 */
public class LocalLLMOnnxRuntimeGenAIHandler implements LocalLlmHandler.InferenceEngine {
    
    // 添加isInitialized方法
    public boolean isInitialized() {
        return model != null && tokenizer != null && generatorParams != null;
    }
    private static final String TAG = "LocalLLMOnnxRuntimeGenAIHandler";
    
    // 上下文
    private final Context context;
    
    // ONNX Runtime GenAI组件
    private Model model;
    private Tokenizer tokenizer;
    private GeneratorParams generatorParams;
    
    // 线程池
    private final ExecutorService executorService;
    
    // 推理停止标志
    private final AtomicBoolean shouldStop = new AtomicBoolean(false);
    
    // 模型配置
    private LocalLlmHandler.ModelConfig modelConfig;
    
    /**
     * 构造函数
     */
    public LocalLLMOnnxRuntimeGenAIHandler(Context context) {
        this.context = context;
        this.executorService = Executors.newSingleThreadExecutor();
        LogManager.logI(TAG, "ONNX Runtime GenAI处理器已创建");
    }
    
    @Override
    public void initialize(String modelPath, LocalLlmHandler.ModelConfig config) throws Exception {
        LogManager.logI(TAG, "初始化ONNX Runtime GenAI引擎...");
        
        try {
            this.modelConfig = config;
            
            // 1. 验证模型文件
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                throw new RuntimeException("模型文件不存在: " + modelPath);
            }
            
            LogManager.logI(TAG, "模型路径: " + modelPath);
            LogManager.logI(TAG, "模型文件大小: " + (modelFile.length() / 1024 / 1024) + "MB");
            
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
            
            LogManager.logI(TAG, "✓ ONNX Runtime GenAI引擎初始化完成");
            
        } catch (Exception e) {
            LogManager.logE(TAG, "ONNX Runtime GenAI引擎初始化失败: " + e.getMessage(), e);
            release();
            throw e;
        }
    }
    
    /**
     * 配置生成参数
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
            
            LogManager.logI(TAG, String.format("生成参数配置完成 - MaxLength: %d, Temperature: %.2f, TopP: %.2f, TopK: %d",
                maxLength, temperature, topP, topK));
                
        } catch (Exception e) {
            LogManager.logE(TAG, "配置生成参数失败: " + e.getMessage(), e);
        }
    }
    
    @Override
    public void inference(String prompt, LocalLlmHandler.InferenceParams params, 
                         LocalLlmHandler.StreamingCallback callback) {
        if (model == null || tokenizer == null) {
            if (callback != null) {
                callback.onError("ONNX Runtime GenAI引擎未初始化");
            }
            return;
        }
        
        try {
            LogManager.logD(TAG, "开始ONNX Runtime GenAI推理，提示词长度: " + prompt.length());
            
            // 1. 使用ONNX Runtime GenAI内置分词器编码输入文本
            // 注意：这里使用的是ONNX Runtime GenAI库内置的分词器，确保与模型完全兼容
            Sequences inputSequences = tokenizer.encode(prompt);
            int inputLength = inputSequences.getSequenceLength(0);
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
            while (!generator.isDone()) {
                generator.generateNextToken();
            }
            
            // 5. 获取生成的序列并解码
            int[] outputTokens = generator.getSequence(0);
            LogManager.logD(TAG, "输出序列长度: " + outputTokens.length);
            
            // 6. 只解码新生成的部分（跳过输入部分）
            if (outputTokens.length > inputLength) {
                int[] newTokens = new int[outputTokens.length - inputLength];
                System.arraycopy(outputTokens, inputLength, newTokens, 0, newTokens.length);
                String result = tokenizer.decode(newTokens);
                LogManager.logD(TAG, "生成的文本: " + result);
                
                if (callback != null) {
                    callback.onComplete(result);
                }
            } else {
                LogManager.logW(TAG, "没有生成新的token");
                if (callback != null) {
                    callback.onComplete("");
                }
            }
            
            // 7. 清理资源
            generator.close();
            genParams.close();
            inputSequences.close();
            
        } catch (Exception e) {
            LogManager.logE(TAG, "ONNX Runtime GenAI推理失败: " + e.getMessage(), e);
            if (callback != null) {
                callback.onError("推理失败: " + e.getMessage());
            }
        }
    }
    
    public void inferenceStream(String prompt, LocalLlmHandler.InferenceParams params, 
                              LocalLlmHandler.StreamingCallback callback) {
        if (model == null || tokenizer == null) {
            if (callback != null) {
                callback.onError("ONNX Runtime GenAI引擎未初始化");
            }
            return;
        }
        
        // 重置停止标志
        shouldStop.set(false);
        
        // 在后台线程执行推理
        executorService.execute(() -> {
            try {
                LogManager.logD(TAG, "开始ONNX Runtime GenAI流式推理，提示词长度: " + prompt.length());
                
                // 1. 使用ONNX Runtime GenAI内置分词器对输入进行编码
                // 注意：这里使用的是ONNX Runtime GenAI库内置的分词器，确保与模型完全兼容
                Sequences inputSequences = tokenizer.encode(prompt);
                int inputLength = inputSequences.getSequenceLength(0);
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
                            LogManager.logD(TAG, "新生成token: " + newText);
                            
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
                    LogManager.logI(TAG, "✓ 流式生成完成，总长度: " + fullResponse.length());
                    if (callback != null) {
                        callback.onComplete(fullResponse.toString());
                    }
                }
                
                // 5. 清理资源
                generator.close();
                genParams.close();
                inputSequences.close();
                
            } catch (Exception e) {
                LogManager.logE(TAG, "ONNX Runtime GenAI流式推理失败: " + e.getMessage(), e);
                if (callback != null) {
                    callback.onError("推理失败: " + e.getMessage());
                }
            }
        });
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
        LogManager.logI(TAG, "释放ONNX Runtime GenAI资源...");
        
        try {
            // 停止推理
            shouldStop.set(true);
            
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
            
            LogManager.logI(TAG, "✓ ONNX Runtime GenAI资源释放完成");
            
        } catch (Exception e) {
            LogManager.logE(TAG, "释放ONNX Runtime GenAI资源时出错: " + e.getMessage(), e);
        }
        
        // 关闭线程池
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
        
        LogManager.logI(TAG, "ONNX Runtime GenAI处理器已释放");
    }
    
    public String getEngineInfo() {
        StringBuilder info = new StringBuilder();
        info.append("ONNX Runtime GenAI引擎\n");
        info.append("状态: ").append(isModelLoaded() ? "已加载" : "未加载").append("\n");
        if (modelConfig != null) {
            info.append("模型类型: ").append(modelConfig.getModelType()).append("\n");
            info.append("最大序列长度: ").append(modelConfig.getMaxSequenceLength()).append("\n");
        }
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
}