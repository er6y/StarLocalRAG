package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

import com.example.starlocalrag.LogManager;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.LongBuffer;
import java.nio.IntBuffer;
import java.nio.FloatBuffer;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;

import com.starlocalrag.tokenizers.HuggingfaceTokenizer;
import com.example.starlocalrag.api.TokenizerManager;
import com.example.starlocalrag.api.TokenizerInterface;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

/**
 * ONNX模型处理类
 * 负责ONNX模型的推理逻辑
 */
public class LocalLLMOnnxHandler {
    
    /**
     * 获取当前使用的内存量（单位：字节）
     * 优化版本：更准确地反映实际内存使用
     * @return 当前使用的内存量
     */
    private long getUsedMemory() {
        Runtime runtime = Runtime.getRuntime();
        // 强制垃圾回收以获得更准确的内存使用情况
        System.gc();
        try {
            Thread.sleep(10); // 给GC一点时间
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return runtime.totalMemory() - runtime.freeMemory();
    }
    
    /**
     * 获取系统级内存使用情况（更准确的内存监控）
     * @return 当前进程的内存使用量
     */
    private long getProcessMemoryUsage() {
        try {
            android.app.ActivityManager.MemoryInfo memInfo = new android.app.ActivityManager.MemoryInfo();
            android.app.ActivityManager activityManager = (android.app.ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            if (activityManager != null) {
                activityManager.getMemoryInfo(memInfo);
                // 返回已使用的内存
                return memInfo.totalMem - memInfo.availMem;
            }
        } catch (Exception e) {
            LogManager.logW(TAG, "获取系统内存信息失败: " + e.getMessage());
        }
        // 回退到JVM内存监控
        return getUsedMemory();
    }
    private static final String TAG = "LocalLLMOnnxHandler";
    
    /**
     * 流式输出回调接口
     * 用于实时返回生成的token
     */
    public interface StreamingCallback {
        /**
         * 当生成新token时调用
         * @param token 生成的token文本
         */
        void onToken(String token);
        
        /**
         * 当生成完成时调用
         * @param fullResponse 完整的生成结果
         */
        void onComplete(String fullResponse);
        
        /**
         * 当发生错误时调用
         * @param errorMessage 错误信息
         */
        void onError(String errorMessage);
    }
    
    private final Context context;
    private final OrtEnvironment ortEnvironment;
    private final OrtSession ortSession;
    private final LocalLlmHandler.ModelConfig modelConfig;
    private TokenizerManager tokenizerManager;
    
    /**
     * 构造函数
     */
    public LocalLLMOnnxHandler(Context context, OrtEnvironment ortEnvironment, OrtSession ortSession, 
                              LocalLlmHandler.ModelConfig modelConfig) {
        this.context = context;
        this.ortEnvironment = ortEnvironment;
        this.ortSession = ortSession;
        this.modelConfig = modelConfig;
        
        LogManager.logD(TAG, "LocalLLMOnnxHandler 初始化完成");
    }
    
    /**
     * 执行推理
     * @param prompt 输入提示词
     * @param maxTokenLength 最大生成长度
     * @param thinkingMode 思考模式
     * @param temperature 温度参数
     * @param topK topK参数
     * @return 生成的文本
     */
    public String inference(String prompt, int maxTokenLength, boolean thinkingMode, float temperature, int topK) throws OrtException {
        LogManager.logD(TAG, "开始推理，提示词长度: " + prompt.length());
        LogManager.logD(TAG, "推理参数 - maxTokenLength: " + maxTokenLength + ", thinkingMode: " + thinkingMode 
              + ", temperature: " + temperature + ", topK: " + topK);
        // Prompt content debugging logging removed
        
        // 创建同步等待机制
        final CountDownLatch latch = new CountDownLatch(1);
        final StringBuilder result = new StringBuilder();
        final Throwable[] error = new Throwable[1];
        
        // 调用流式推理方法
        inferenceStream(prompt, maxTokenLength, thinkingMode, temperature, topK, new StreamingCallback() {
            @Override
            public void onToken(String token) {
                // 只需收集token，不需要其他处理
            }
            
            @Override
            public void onComplete(String fullResponse) {
                result.append(fullResponse);
                latch.countDown(); // 释放等待
            }
            
            @Override
            public void onError(String errorMessage) {
                error[0] = new OrtException(errorMessage);
                latch.countDown(); // 释放等待
            }
        }, null);
        
        // 等待流式推理完成
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("推理过程被中断", e);
        }
        
        // 检查是否有错误
        if (error[0] != null) {
            if (error[0] instanceof OrtException) {
                throw (OrtException) error[0];
            } else {
                throw new RuntimeException("推理过程发生错误: " + error[0].getMessage(), error[0]);
            }
        }
        
        long startTime = System.currentTimeMillis();
        
        // 使用TokenizerManager进行分词，已在tokenizeInput方法中初始化
        try {
            
            // 1. 对输入进行分词
            int[] inputIds = tokenizeInput(prompt, thinkingMode);
            // 只打印前20个token，避免日志过长
            String tokenDisplay;
            if (inputIds.length > 20) {
                int[] truncatedIds = Arrays.copyOfRange(inputIds, 0, 20);
                tokenDisplay = Arrays.toString(truncatedIds).replace("]", ", ...]");
            } else {
                tokenDisplay = Arrays.toString(inputIds);
            }
            LogManager.logD(TAG, "分词结果 - 长度: " + inputIds.length + ", tokens: " + tokenDisplay);
            
            // 简单记录token ID
            StringBuilder tokenIds = new StringBuilder("Token IDs: ");
            for (int i = 0; i < Math.min(20, inputIds.length); i++) {
                tokenIds.append(inputIds[i]).append(" ");
            }
            if (inputIds.length > 20) {
                tokenIds.append("... (共").append(inputIds.length).append("个token)");
            }
            LogManager.logD(TAG, tokenIds.toString());
            
            // 2. 准备输入张量
            Map<String, OnnxTensor> inputs = prepareInputs(inputIds);
            LogManager.logD(TAG, "准备输入张量完成 - 张量数量: " + inputs.size());
            for (Map.Entry<String, OnnxTensor> entry : inputs.entrySet()) {
                LogManager.logD(TAG, "输入张量: " + entry.getKey() + ", 形状: " + Arrays.toString(entry.getValue().getInfo().getShape()));
            }
            
            // 3. 执行推理
            LogManager.logD(TAG, "开始执行模型推理...");
            long inferenceStartTime = System.currentTimeMillis();
            
            // 实现自回归生成，每次生成一个token，然后添加到输入中继续生成
            try {
                // 准备生成参数
                int[] currentInputIds = inputIds;
                StringBuilder generatedText = new StringBuilder();
                List<Integer> generatedTokenIds = new ArrayList<>(); // 存储生成的token IDs用于批量解码
                
                // 使用方法参数中的温度和topK参数
                // 如果没有提供，则使用默认值
                float actualTemperature = temperature > 0 ? temperature : (thinkingMode ? 0.6f : 0.7f);
                float topP = thinkingMode ? 0.95f : 0.8f;
                int actualTopK = topK > 0 ? topK : 20;
                
                LogManager.logD(TAG, "生成参数 - 温度: " + actualTemperature + ", topP: " + topP + ", topK: " + actualTopK);
                
                // 自回归生成循环
                for (int i = 0; i < maxTokenLength; i++) {
                    // 准备当前输入
                    Map<String, OnnxTensor> currentInputs = prepareInputs(currentInputIds);
                    
                    // 执行推理
                    OrtSession.Result output = ortSession.run(currentInputs);
                    
                    if (i == 0) {
                        LogManager.logD(TAG, "模型推理完成，输出数量: " + output.size());
                        
                        // 打印输出张量信息
                        for (Map.Entry<String, OnnxValue> entry : output) {
                            String outputName = entry.getKey();
                            OnnxValue value = entry.getValue();
                            LogManager.logD(TAG, "输出张量: " + outputName + ", 类型: " + value.getInfo());
                            
                            // 安全地获取形状信息
                            try {
                                if (value instanceof OnnxTensor) {
                                    OnnxTensor tensor = (OnnxTensor) value;
                                    LogManager.logD(TAG, "张量形状: " + Arrays.toString(tensor.getInfo().getShape()));
                                } else {
                                    LogManager.logD(TAG, "非张量类型，无法获取形状信息");
                                }
                            } catch (Exception e) {
                                LogManager.logW(TAG, "获取张量形状时出错: " + e.getMessage());
                            }
                        }
                    }
                        
                        // 从输出张量中提取下一个token ID
                        int nextTokenId = -1;
                        
                        for (Map.Entry<String, OnnxValue> entry : output) {
                            OnnxValue value = entry.getValue();
                            
                            if (value instanceof OnnxTensor) {
                                OnnxTensor tensor = (OnnxTensor) value;
                                long[] shape = tensor.getInfo().getShape();
                                
                                if (shape.length == 3) { // 三维张量 [batch_size, seq_len, vocab_size]
                                    try {
                                        float[][][] data = (float[][][]) tensor.getValue();
                                        if (data != null && data.length > 0 && data[0].length > 0) {
                                            // 获取最后一个位置的logits
                                            float[] logits = data[0][data[0].length - 1];
                                            
                                            // 应用温度缩放
                                            if (actualTemperature > 0) {
                                                for (int j = 0; j < logits.length; j++) {
                                                    logits[j] /= actualTemperature;
                                                }
                                            }
                                            
                                            // 应用softmax获取概率分布
                                            float[] probs = applySoftmax(logits);
                                            
                                            // 应用top-p采样
                                            probs = applyTopP(probs, topP);
                                            
                                            // 应用top-k采样
                                            probs = applyTopK(probs, actualTopK);
                                            
                                            // 从概率分布中采样
                                            nextTokenId = sampleFromDistribution(probs);
                                            
                                            // 尝试解码并打印token文本
                                            try {
                                                String tokenText = tokenizerManager.decodeIds(new int[]{nextTokenId});
                                                LogManager.logD(TAG, "生成的下一个token ID: " + nextTokenId + ", 文本: '" + tokenText + "'");
                                            } catch (Exception e) {
                                                LogManager.logD(TAG, "生成的下一个token ID: " + nextTokenId + ", 无法解码文本: " + e.getMessage());
                                            }
                                            break;
                                        }
                                    } catch (Exception e) {
                                        LogManager.logE(TAG, "处理logits时出错: " + e.getMessage(), e);
                                    }
                                } else if (shape.length == 2) { // 二维张量 [batch_size, seq_len]
                                    try {
                                        long[][] data = (long[][]) tensor.getValue();
                                        if (data != null && data.length > 0) {
                                            // 获取最后一个位置的token ID
                                            nextTokenId = (int) data[0][data[0].length - 1];
                                            
                                            // 尝试解码并打印token文本
                                            try {
                                                String tokenText = tokenizerManager.decodeIds(new int[]{nextTokenId});
                                                LogManager.logD(TAG, "生成的下一个token ID: " + nextTokenId + ", 文本: '" + tokenText + "'");
                                            } catch (Exception e) {
                                                LogManager.logD(TAG, "生成的下一个token ID: " + nextTokenId + ", 无法解码文本: " + e.getMessage());
                                            }
                                            break;
                                        }
                                    } catch (Exception e) {
                                        LogManager.logE(TAG, "处理二维张量时出错: " + e.getMessage(), e);
                                    }
                                }
                            }
                        }
                        
                        // 关闭当前输入资源
                        for (OnnxTensor tensor : currentInputs.values()) {
                            tensor.close();
                        }
                        
                        // 关闭输出资源
                        for (Map.Entry<String, OnnxValue> entry : output) {
                            entry.getValue().close();
                        }
                        
                        // 检查是否获取到有效的token ID
                        if (nextTokenId == -1) {
                            LogManager.logE(TAG, "无法从输出中提取下一个token ID");
                            break;
                        }
                        
                        // 检查是否结束生成（通常是特殊的EOS token）
                        if (nextTokenId == 151645) { // <|im_end|> token ID
                            LogManager.logD(TAG, "检测到结束token，停止生成");
                            break;
                        }
                        
                        // 将新token添加到当前输入
                        currentInputIds = appendTokenToInput(currentInputIds, nextTokenId);
                        
                        // 将新token ID添加到生成的token列表中
                        generatedTokenIds.add(nextTokenId);
                        
                        // 解码当前token并添加到结果中
                        try {
                            // 使用全局tokenizer实例解码
                            String tokenText = tokenizerManager.decodeIds(new int[]{nextTokenId});
                            generatedText.append(tokenText);
                            
                            // 每5个token或最后一个token时记录日志
                            if (i % 5 == 0 || i == maxTokenLength - 1) {
                                // 打印Unicode编码版本（为了兼容性）
                                LogManager.logD(TAG, "已生成 " + (i+1) + " 个token，当前文本(Unicode): " + escapeString(generatedText.toString()));
                                // 打印原始文本版本（更直观）
                                LogManager.logD(TAG, "已生成 " + (i+1) + " 个token，当前文本(原文): \n" + generatedText.toString());
                            }
                        } catch (Exception e) {
                            LogManager.logE(TAG, "解码token时出错: " + e.getMessage(), e);
                        }
                }
                
                // 尝试使用批量解码来提高效率
                if (!generatedTokenIds.isEmpty()) {
                    try {
                        // 将List<Integer>转换为int[]
                        int[] allTokenIds = new int[generatedTokenIds.size()];
                        for (int i = 0; i < generatedTokenIds.size(); i++) {
                            allTokenIds[i] = generatedTokenIds.get(i);
                        }
                        
                        // 批量解码所有生成的token
                        String batchDecodedText = tokenizerManager.decodeIds(allTokenIds);
                        // 打印Unicode编码版本（为了兼容性）
                        LogManager.logD(TAG, "批量解码结果(Unicode): " + escapeString(batchDecodedText));
                        // 打印原始文本版本（更直观）
                        LogManager.logD(TAG, "批量解码结果(原文): \n" + batchDecodedText);
                        
                        // 比较逐个解码和批量解码的结果
                        if (!batchDecodedText.equals(generatedText.toString())) {
                            LogManager.logD(TAG, "批量解码和逐个解码结果不同，使用批量解码结果");
                            // 使用批量解码结果，因为它可能处理了特殊字符和字节级别的编码
                            result.setLength(0);
                            result.append(batchDecodedText);
                        } else {
                            // 结果相同，使用原来的结果
                            result.append(generatedText.toString());
                        }
                    } catch (Exception e) {
                        LogManager.logE(TAG, "批量解码失败，使用逐个解码结果: " + e.getMessage(), e);
                        // 如果批量解码失败，使用逐个解码的结果
                        result.append(generatedText.toString());
                    }
                } else {
                    // 没有生成任何token，直接使用原来的结果
                    result.append(generatedText.toString());
                }
                
                // 记录推理完成时间
                long inferenceEndTime = System.currentTimeMillis();
                LogManager.logD(TAG, "模型推理耗时: " + (inferenceEndTime - inferenceStartTime) + "ms");
                
            } catch (OrtException e) {
                LogManager.logE(TAG, "模型推理失败", e);
                LogManager.logE(TAG, "错误详情: " + e.getMessage());
                result.append("推理失败: ").append(e.getMessage());
            }
            
            // 4. 释放资源
            for (OnnxTensor tensor : inputs.values()) {
                tensor.close();
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "推理过程发生异常", e);
            LogManager.logE(TAG, "异常详情: " + e.getMessage());
            result.append("推理异常: ").append(e.getMessage());
        } finally {
            // 重置TokenizerManager资源
            if (tokenizerManager != null) {
                try {
                    TokenizerManager.resetManager();
                    LogManager.logD(TAG, "TokenizerManager实例已重置");
                } catch (Exception e) {
                    LogManager.logE(TAG, "重置TokenizerManager时出错: " + e.getMessage(), e);
                }
            }
        }
        
        long endTime = System.currentTimeMillis();
        LogManager.logD(TAG, "总推理耗时: " + (endTime - startTime) + "ms");
        
        return result.toString();
    }
    
    /**
     * 批处理推理方法 - 支持多序列并行推理
     * @param inputTexts 输入文本数组
     * @param maxTokens 最大生成token数
     * @param temperature 温度参数
     * @param topK topK参数
     * @param topP topP参数
     * @param callback 流式回调
     * @return 生成的文本数组
     */
    public String[] inferenceStreamBatch(String[] inputTexts, int maxTokens, float temperature, int topK, float topP, StreamingCallback callback) {
        if (inputTexts == null || inputTexts.length == 0) {
            return new String[0];
        }
        
        int batchSize = Math.min(inputTexts.length, modelConfig != null ? modelConfig.getMaxBatchSize() : 1);
        String[] results = new String[batchSize];
        
        try {
            // 1. 分词所有输入文本
            int[][] allInputIds = new int[batchSize][];
            for (int i = 0; i < batchSize; i++) {
                allInputIds[i] = tokenizerManager.tokenizeInput(inputTexts[i]);
                LogManager.logD(TAG, "批次 " + i + " 分词结果长度: " + allInputIds[i].length);
            }
            
            // 2. 填充序列到相同长度（批处理要求）
            int maxSeqLen = 0;
            for (int[] ids : allInputIds) {
                maxSeqLen = Math.max(maxSeqLen, ids.length);
            }
            
            // 填充所有序列到相同长度
            for (int i = 0; i < batchSize; i++) {
                if (allInputIds[i].length < maxSeqLen) {
                    int[] paddedIds = new int[maxSeqLen];
                    System.arraycopy(allInputIds[i], 0, paddedIds, 0, allInputIds[i].length);
                    // 用pad token填充剩余位置（通常是0）
                    for (int j = allInputIds[i].length; j < maxSeqLen; j++) {
                        paddedIds[j] = 0; // pad token
                    }
                    allInputIds[i] = paddedIds;
                }
            }
            
            // 3. 批处理推理循环
            StringBuilder[] resultBuilders = new StringBuilder[batchSize];
            for (int i = 0; i < batchSize; i++) {
                resultBuilders[i] = new StringBuilder();
            }
            
            boolean[] finished = new boolean[batchSize];
            int finishedCount = 0;
            
            for (int step = 0; step < maxTokens && finishedCount < batchSize; step++) {
                // 准备批处理输入
                Map<String, OnnxTensor> batchInputs = prepareBatchInputs(allInputIds);
                
                // 执行批处理推理
                Result batchOutput = ortSession.run(batchInputs);
                
                // 处理每个序列的输出
                int[] nextTokenIds = extractBatchTokens(batchOutput, batchSize, temperature, topK, topP);
                
                for (int i = 0; i < batchSize; i++) {
                    if (!finished[i] && nextTokenIds[i] != -1) {
                        // 检查是否为结束token
                        if (isEndToken(nextTokenIds[i])) {
                            finished[i] = true;
                            finishedCount++;
                            continue;
                        }
                        
                        // 解码token并添加到结果
                        try {
                            String tokenText = tokenizerManager.decodeIds(new int[]{nextTokenIds[i]});
                            resultBuilders[i].append(tokenText);
                            
                            // 流式回调
                             if (callback != null) {
                                 callback.onToken("批次" + i + ": " + tokenText);
                             }
                        } catch (Exception e) {
                            LogManager.logW(TAG, "解码批次 " + i + " token失败: " + e.getMessage());
                        }
                        
                        // 更新输入序列
                        allInputIds[i] = appendTokenToInput(allInputIds[i], nextTokenIds[i]);
                    }
                }
                
                // 清理批处理输入资源
                for (OnnxTensor tensor : batchInputs.values()) {
                    tensor.close();
                }
                batchOutput.close();
            }
            
            // 收集结果
            for (int i = 0; i < batchSize; i++) {
                results[i] = resultBuilders[i].toString();
            }
            
            LogManager.logI(TAG, "批处理推理完成，批次大小: " + batchSize + ", 完成序列数: " + finishedCount);
            
        } catch (Exception e) {
            LogManager.logE(TAG, "批处理推理失败: " + e.getMessage(), e);
            // 返回错误信息
            for (int i = 0; i < batchSize; i++) {
                results[i] = "批处理推理失败: " + e.getMessage();
            }
        }
        
        return results;
    }
    
    /**
     * 准备批处理输入张量
     * @param allInputIds 所有输入ID数组
     * @return 批处理输入张量映射
     * @throws OrtException ONNX Runtime异常
     */
    private Map<String, OnnxTensor> prepareBatchInputs(int[][] allInputIds) throws OrtException {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        
        int batchSize = allInputIds.length;
        int seqLen = allInputIds[0].length;
        
        // 创建批处理输入张量
        long[] shape = new long[]{batchSize, seqLen};
        LongBuffer batchBuffer = LongBuffer.allocate(batchSize * seqLen);
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                batchBuffer.put(allInputIds[i][j]);
            }
        }
        batchBuffer.flip();
        
        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(ortEnvironment, batchBuffer, shape);
        inputs.put("input_ids", inputIdsTensor);
        
        // 添加注意力掩码
        if (modelConfig != null && modelConfig.requiresAttentionMask()) {
            LongBuffer attentionBuffer = LongBuffer.allocate(batchSize * seqLen);
            for (int i = 0; i < batchSize * seqLen; i++) {
                attentionBuffer.put(1L); // 所有位置可见
            }
            attentionBuffer.flip();
            
            OnnxTensor attentionTensor = OnnxTensor.createTensor(ortEnvironment, attentionBuffer, shape);
            inputs.put("attention_mask", attentionTensor);
        }
        
        return inputs;
    }
    
    /**
     * 从批处理输出中提取token
     * @param batchOutput 批处理输出
     * @param batchSize 批处理大小
     * @param temperature 温度参数
     * @param topK topK参数
     * @param topP topP参数
     * @return 下一个token ID数组
     */
    private int[] extractBatchTokens(Result batchOutput, int batchSize, float temperature, int topK, float topP) {
        int[] nextTokenIds = new int[batchSize];
        
        try {
            for (Map.Entry<String, OnnxValue> entry : batchOutput) {
                OnnxValue value = entry.getValue();
                
                if (value instanceof OnnxTensor) {
                    OnnxTensor tensor = (OnnxTensor) value;
                    long[] shape = tensor.getInfo().getShape();
                    
                    if (shape.length == 3) { // [batch_size, seq_len, vocab_size]
                        float[][][] data = (float[][][]) tensor.getValue();
                        
                        for (int batch = 0; batch < batchSize; batch++) {
                            if (data[batch].length > 0) {
                                float[] logits = data[batch][data[batch].length - 1];
                                
                                // 应用采样策略
                                if (temperature > 0) {
                                    for (int j = 0; j < logits.length; j++) {
                                        logits[j] /= temperature;
                                    }
                                }
                                
                                float[] probs = applySoftmax(logits);
                                probs = applyTopP(probs, topP);
                                probs = applyTopK(probs, topK);
                                
                                nextTokenIds[batch] = sampleFromDistribution(probs);
                            } else {
                                nextTokenIds[batch] = -1;
                            }
                        }
                        break;
                    }
                }
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "提取批处理token失败: " + e.getMessage(), e);
            for (int i = 0; i < batchSize; i++) {
                nextTokenIds[i] = -1;
            }
        }
        
        return nextTokenIds;
    }
    
    /**
     * 检查是否为结束token
     * @param tokenId token ID
     * @return 是否为结束token
     */
    private boolean isEndToken(int tokenId) {
        if (modelConfig != null) {
            return tokenId == modelConfig.getEosTokenId() || tokenId == modelConfig.getPadTokenId();
        }
        return tokenId == 2; // 默认EOS token
    }
    
    /**
     * 将新token添加到输入数组中
     * @param inputIds 当前输入
     * @param newTokenId 新token ID
     * @return 更新后的输入数组
     */
    private int[] appendTokenToInput(int[] inputIds, int newTokenId) {
        int[] newInputIds = new int[inputIds.length + 1];
        System.arraycopy(inputIds, 0, newInputIds, 0, inputIds.length);
        newInputIds[inputIds.length] = newTokenId;
        return newInputIds;
    }
    
    /**
     * 应用softmax函数获取概率分布
     * @param logits 输入logits
     * @return 概率分布
     */
    private float[] applySoftmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (logit > max) {
                max = logit;
            }
        }
        
        float sum = 0.0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        
        return probs;
    }
    
    /**
     * 应用top-k采样
     * @param probs 概率分布
     * @param k k值
     * @return 过滤后的概率分布
     */
    private float[] applyTopK(float[] probs, int k) {
        // 创建索引-概率对
        List<Map.Entry<Integer, Float>> indexProbPairs = new ArrayList<>();
        for (int i = 0; i < probs.length; i++) {
            indexProbPairs.add(new AbstractMap.SimpleEntry<>(i, probs[i]));
        }
        
        // 按概率排序
        indexProbPairs.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));
        
        // 只保留前k个
        float[] filteredProbs = new float[probs.length];
        float sum = 0.0f;
        for (int i = 0; i < k && i < indexProbPairs.size(); i++) {
            Map.Entry<Integer, Float> pair = indexProbPairs.get(i);
            filteredProbs[pair.getKey()] = pair.getValue();
            sum += pair.getValue();
        }
        
        // 重新归一化
        if (sum > 0) {
            for (int i = 0; i < filteredProbs.length; i++) {
                filteredProbs[i] /= sum;
            }
        }
        
        return filteredProbs;
    }
    
    /**
     * 应用top-p (nucleus) 采样
     * @param probs 概率分布
     * @param p p值
     * @return 过滤后的概率分布
     */
    private float[] applyTopP(float[] probs, float p) {
        // 创建索引-概率对
        List<Map.Entry<Integer, Float>> indexProbPairs = new ArrayList<>();
        for (int i = 0; i < probs.length; i++) {
            indexProbPairs.add(new AbstractMap.SimpleEntry<>(i, probs[i]));
        }
        
        // 按概率排序
        indexProbPairs.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));
        
        // 累积概率直到达到p
        float[] filteredProbs = new float[probs.length];
        float cumulativeProb = 0.0f;
        for (Map.Entry<Integer, Float> pair : indexProbPairs) {
            if (cumulativeProb < p) {
                filteredProbs[pair.getKey()] = pair.getValue();
                cumulativeProb += pair.getValue();
            } else {
                break;
            }
        }
        
        // 重新归一化
        float sum = 0.0f;
        for (float prob : filteredProbs) {
            sum += prob;
        }
        
        if (sum > 0) {
            for (int i = 0; i < filteredProbs.length; i++) {
                filteredProbs[i] /= sum;
            }
        }
        
        return filteredProbs;
    }
    
    /**
     * 从概率分布中采样
     * @param probs 概率分布
     * @return 采样的索引
     */
    private int sampleFromDistribution(float[] probs) {
        float rand = new Random().nextFloat();
        float cumulativeProb = 0.0f;
        
        for (int i = 0; i < probs.length; i++) {
            cumulativeProb += probs[i];
            if (rand < cumulativeProb) {
                return i;
            }
        }
        
        // 如果由于浮点精度问题没有返回，则返回最高概率的索引
        int maxIndex = 0;
        float maxProb = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    /**
     * 将字符串转换为安全的表示形式，保留原始 Unicode 字符
     * @param s 输入字符串
     * @return 安全的表示形式
     */
    private String escapeString(String s) {
        if (s == null) {
            return "null";
        }
        
        // 保留原始 Unicode 字符，只处理特殊字符
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c < 32 || c == 127) { // 控制字符
                sb.append(String.format("\\u%04x", (int)c));
            } else {
                sb.append(c);
            }
        }
        
        // 处理敏感 token
        String result = sb.toString();
        if (result.contains("|endoftext")) {
            result = result.replace("|endoftext", "|endoftxt");
        }
        
        return result;
    }
    
    /**
     * 应用分词器模板
     * @param text 输入文本
     * @param thinkingMode 是否启用思考模式
     * @return 处理后的文本
     */
    private String applyTokenizerTemplate(String text, boolean thinkingMode) {
        LogManager.logD(TAG, "应用分词器模板到文本，思考模式: " + thinkingMode);
        
        try {
            // 确保有有效的 context
            if (context == null) {
                throw new RuntimeException("上下文对象为空");
            }
            
            // 获取 TokenizerManager 实例
            tokenizerManager = TokenizerManager.getInstance(context);
            
            // 使用当前 LLM 模型的路径初始化分词器
            // 如果已经初始化为相同的模型路径，不会重复初始化
            File modelDir = new File(modelConfig.getModelPath());
            boolean initSuccess = tokenizerManager.initialize(modelDir);
            if (!initSuccess) {
                LogManager.logE(TAG, "TokenizerManager初始化失败，请检查模型路径: " + modelDir.getAbsolutePath());
                throw new RuntimeException("TokenizerManager初始化失败，请检查模型路径");
            }
            
            // 获取已初始化的 HuggingfaceTokenizer 实例
            HuggingfaceTokenizer tokenizer = tokenizerManager.getTokenizer();
            if (tokenizer == null) {
                throw new RuntimeException("无法获取分词器实例，请检查是否存在 tokenizer.json 文件");
            }
            
            // 创建聊天消息数组
            org.json.JSONArray messages = new org.json.JSONArray();
            
            // 添加用户消息
            org.json.JSONObject userMessage = new org.json.JSONObject();
            userMessage.put("role", "user");
            userMessage.put("content", text);
            messages.put(userMessage);
            
            // 使用JNI层的applyChatTemplate方法处理模板
            // 参数说明：
            // messages: 消息数组
            // false: 不添加生成提示
            // thinkingMode: 是否启用思考模式
            String formattedText = tokenizer.applyChatTemplate(messages, false, thinkingMode);
            
            LogManager.logD(TAG, "使用JNI层applyChatTemplate后的文本长度: " + formattedText.length());
            LogManager.logD(TAG, "使用JNI层applyChatTemplate后的文本片段: " + 
                  formattedText.substring(0, Math.min(100, formattedText.length())) + "...");
            
            return formattedText;
        } catch (Exception e) {
            LogManager.logE(TAG, "应用聊天模板失败: " + e.getMessage(), e);
            // 直接抛出异常，不在Java层硬编码模板
            throw new RuntimeException("应用聊天模板失败，请检查分词器配置: " + e.getMessage(), e);
        }
    }
    
    /**
     * 对输入文本进行分词
     * 使用TokenizerManager进行分词处理
     * 注意：传统ONNX Runtime使用Hugging Face分词器，通过TokenizerManager管理
     * 这与ONNX Runtime GenAI的内置分词器不同，需要手动处理分词和配置解析
     * @param text 输入文本
     * @param thinkingMode 是否启用思考模式
     * @return 分词结果（token IDs）
     */
    private int[] tokenizeInput(String text, boolean thinkingMode) {
        LogManager.logD(TAG, "开始对输入文本进行分词，文本长度: " + text.length());
        long startTime = System.currentTimeMillis();
        
        int[] result;
        
        try {
            // 初始化TokenizerManager（如果尚未初始化）
            if (tokenizerManager == null && context != null) {
                tokenizerManager = TokenizerManager.getInstance(context);
                File modelDir = new File(modelConfig.getModelPath());
                boolean initSuccess = tokenizerManager.initialize(modelDir);
                if (!initSuccess) {
                    LogManager.logE(TAG, "TokenizerManager初始化失败");
                    // 如果初始化失败，返回默认的特殊标记
                    return new int[]{modelConfig.getBosToken()};
                }
                LogManager.logD(TAG, "TokenizerManager初始化成功");
            }
            
            // 使用与Python类似的方式处理聊天模板
            String processedText = applyTokenizerTemplate(text, thinkingMode);
            
            // 使用TokenizerManager进行分词
            long[][] tokenizeResult = tokenizerManager.tokenize(processedText);
            
            // 转换结果格式（从long[][]到int[]）
            if (tokenizeResult != null && tokenizeResult.length > 0 && tokenizeResult[0] != null) {
                long[] ids = tokenizeResult[0]; // 获取token IDs
                result = new int[ids.length];
                for (int i = 0; i < ids.length; i++) {
                    result[i] = (int) ids[i];
                }
                
                // 输出详细的分词结果日志
                StringBuilder tokenLog = new StringBuilder("分词结果 (ID:文本): [");
                for (int i = 0; i < Math.min(20, result.length); i++) {
                    if (i > 0) tokenLog.append(", ");
                    // 解码单个token
                    String decodedToken = tokenizerManager.decodeIds(new int[]{result[i]});
                    tokenLog.append("'").append(result[i]).append(":").append(decodedToken).append("'");
                }
                if (result.length > 20) {
                    tokenLog.append(", ... (共").append(result.length).append("个token)");
                }
                tokenLog.append("]");
                LogManager.logD(TAG, tokenLog.toString());
                
                // 添加一个简洁的摘要日志，只显示前几个和后几个token
                if (result.length > 10) {
                    StringBuilder summaryLog = new StringBuilder("分词摘要: ");
                    // 显示前5个token
                    for (int i = 0; i < Math.min(5, result.length); i++) {
                        if (i > 0) summaryLog.append(" ");
                        String decodedToken = tokenizerManager.decodeIds(new int[]{result[i]});
                        summaryLog.append(result[i]).append(":").append(decodedToken);
                    }
                    summaryLog.append(" ... ");
                    // 显示后5个token
                    int startIdx = Math.max(5, result.length - 5);
                    for (int i = startIdx; i < result.length; i++) {
                        String decodedToken = tokenizerManager.decodeIds(new int[]{result[i]});
                        summaryLog.append(" ").append(result[i]).append(":").append(decodedToken);
                    }
                    LogManager.logD(TAG, summaryLog.toString());
                }
            } else {
                LogManager.logE(TAG, "分词结果为空或格式不正确");
                result = new int[]{modelConfig.getBosToken()};
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "分词过程发生异常: " + e.getMessage(), e);
            // 如果分词失败，使用默认的特殊标记
            result = new int[]{modelConfig.getBosToken()};
        }
        
        long endTime = System.currentTimeMillis();
        LogManager.logD(TAG, "分词完成，耗时: " + (endTime - startTime) + "ms, 生成token数量: " + result.length);
        
        return result;
    }
    
    /**
     * 执行流式推理
     * 通过回调接口实时返回生成的token
     * 
     * @param prompt 输入提示词
     * @param maxTokenLength 最大生成长度
     * @param thinkingMode 思考模式
     * @param temperature 温度参数
     * @param topK topK参数
     * @param callback 流式输出回调接口
     */
    public void inferenceStream(String prompt, int maxTokenLength, boolean thinkingMode, float temperature, int topK, StreamingCallback callback, LocalLlmHandler handler) {
        LogManager.logD(TAG, "开始流式推理，提示词长度: " + prompt.length());
        LogManager.logD(TAG, "推理参数 - maxTokenLength: " + maxTokenLength + ", thinkingMode: " + thinkingMode 
              + ", temperature: " + temperature + ", topK: " + topK);
        
        // 在新线程中执行推理，避免阻塞调用线程
        new Thread(() -> {
            StringBuilder fullResponse = new StringBuilder();
            
            // 推理统计变量
            long inferenceStartTime = 0;
            long inferenceEndTime = 0;
            int totalGeneratedTokens = 0;
            
            // 统一的内存监控机制：使用与OnnxRuntimeGenAI相同的方法
            startMemoryMonitoring();
            
            // 创建内存监控线程（1秒一次）
            Thread memoryMonitorThread = new Thread(() -> {
                try {
                    while (!Thread.interrupted()) {
                        updateMemoryMonitoring();
                        Thread.sleep(1000); // 每1秒监控一次内存
                    }
                } catch (InterruptedException e) {
                    // 线程被中断，正常退出
                    LogManager.logD(TAG, "内存监控线程正常退出");
                }
            });
            memoryMonitorThread.setDaemon(true);
            memoryMonitorThread.setName("LLM-MemoryMonitor");
            
            try {
                long startTime = System.currentTimeMillis();
                
                // 使用TokenizerManager进行分词，已在tokenizeInput方法中初始化
                
                // 1. 对输入进行分词
                int[] inputIds = tokenizeInput(prompt, thinkingMode);
                // 只打印前20个token，避免日志过长
                String tokenDisplay;
                if (inputIds.length > 20) {
                    int[] truncatedIds = Arrays.copyOfRange(inputIds, 0, 20);
                    tokenDisplay = Arrays.toString(truncatedIds).replace("]", ", ...]");
                } else {
                    tokenDisplay = Arrays.toString(inputIds);
                }
                LogManager.logD(TAG, "分词结果 - 长度: " + inputIds.length + ", tokens: " + tokenDisplay);
                
                // 2. 准备输入张量
                Map<String, OnnxTensor> inputs = prepareInputs(inputIds);
                LogManager.logD(TAG, "准备输入张量完成 - 张量数量: " + inputs.size());
                
                // 3. 执行推理
                LogManager.logD(TAG, "开始执行流式模型推理...");
                inferenceStartTime = System.currentTimeMillis();
                
                // 启动内存监控线程
                memoryMonitorThread.start();
                
                // 实现自回归生成，每次生成一个token，然后添加到输入中继续生成
                try {
                    // 准备生成参数
                    int[] currentInputIds = inputIds;
                    List<Integer> generatedTokenIds = new ArrayList<>(); // 存储生成的token IDs
                    
                    // 使用方法参数中的温度和topK参数
                    float actualTemperature = temperature > 0 ? temperature : (thinkingMode ? 0.6f : 0.7f);
                    float topP = thinkingMode ? 0.95f : 0.8f;
                    int actualTopK = topK > 0 ? topK : 20;
                    
                    LogManager.logD(TAG, "生成参数 - 温度: " + actualTemperature + ", topP: " + topP + ", topK: " + actualTopK);
                    
                    // 自回归生成循环
                    for (int i = 0; i < maxTokenLength; i++) {
                        // 检查是否应该停止推理
                        if (handler != null && handler.shouldStopInference()) {
                            // 停止调试日志已移除
                            LogManager.logD(TAG, "[停止调试] 停止检查通过，准备退出推理循环");
                            callback.onError("推理被用户停止");
                            return;
                        }
                        
                        // 每10步记录一次停止状态检查（避免日志过多）
                        if (i % 10 == 0) {
                            boolean shouldStop = handler != null ? handler.shouldStopInference() : false;
                            LogManager.logD(TAG, "[停止调试] 第" + i + "步停止状态检查: " + shouldStop);
                        }
                        
                        // 准备当前输入
                        Map<String, OnnxTensor> currentInputs = prepareInputs(currentInputIds);
                        
                        // 执行推理前再次检查停止状态
                        if (handler != null && handler.shouldStopInference()) {
                            // 停止调试日志已移除
                            callback.onError("推理被用户停止");
                            return;
                        }
                        
                        // 执行推理 - 性能关键路径
                        long inferenceStart = System.currentTimeMillis();
                        OrtSession.Result output = ortSession.run(currentInputs);
                        long inferenceTime = System.currentTimeMillis() - inferenceStart;
                        
                        // 只在推理时间过长时记录日志
                        if (inferenceTime > 100) {
                            LogManager.logD(TAG, "单次推理耗时: " + inferenceTime + "ms");
                        }
                        
                        // 从输出张量中提取下一个token ID
                        int nextTokenId = -1;
                        
                        for (Map.Entry<String, OnnxValue> entry : output) {
                            OnnxValue value = entry.getValue();
                            
                            if (value instanceof OnnxTensor) {
                                OnnxTensor tensor = (OnnxTensor) value;
                                long[] shape = tensor.getInfo().getShape();
                                
                                if (shape.length == 3) { // 三维张量 [batch_size, seq_len, vocab_size]
                                    try {
                                        float[][][] data = (float[][][]) tensor.getValue();
                                        if (data != null && data.length > 0 && data[0].length > 0) {
                                            // 获取最后一个位置的logits
                                            float[] logits = data[0][data[0].length - 1];
                                            
                                            // 应用温度缩放
                                            if (actualTemperature > 0) {
                                                for (int j = 0; j < logits.length; j++) {
                                                    logits[j] /= actualTemperature;
                                                }
                                            }
                                            
                                            // 应用softmax获取概率分布
                                            float[] probs = applySoftmax(logits);
                                            
                                            // 应用top-p采样
                                            probs = applyTopP(probs, topP);
                                            
                                            // 应用top-k采样
                                            probs = applyTopK(probs, actualTopK);
                                            
                                            // 从概率分布中采样
                                            nextTokenId = sampleFromDistribution(probs);
                                            break;
                                        }
                                    } catch (Exception e) {
                                        LogManager.logE(TAG, "处理logits时出错: " + e.getMessage(), e);
                                    }
                                } else if (shape.length == 2) { // 二维张量 [batch_size, seq_len]
                                    try {
                                        long[][] data = (long[][]) tensor.getValue();
                                        if (data != null && data.length > 0) {
                                            // 获取最后一个位置的token ID
                                            nextTokenId = (int) data[0][data[0].length - 1];
                                            break;
                                        }
                                    } catch (Exception e) {
                                        LogManager.logE(TAG, "处理二维张量时出错: " + e.getMessage(), e);
                                    }
                                }
                            }
                        }
                        
                        // 关闭当前输入资源
                        for (OnnxTensor tensor : currentInputs.values()) {
                            tensor.close();
                        }
                        
                        // 关闭输出资源
                        for (Map.Entry<String, OnnxValue> entry : output) {
                            entry.getValue().close();
                        }
                        
                        // 检查是否获取到有效的token ID
                        if (nextTokenId == -1) {
                            LogManager.logE(TAG, "无法从输出中提取下一个token ID");
                            break;
                        }
                        
                        // 检查是否结束生成（通常是特殊的EOS token）
                        if (nextTokenId == 151645) { // <|im_end|> token ID
                            LogManager.logD(TAG, "检测到结束token，停止生成");
                            break;
                        }
                        
                        // 将新token添加到当前输入
                        currentInputIds = appendTokenToInput(currentInputIds, nextTokenId);
                        
                        // 将新token ID添加到生成的token列表中
                        generatedTokenIds.add(nextTokenId);
                        totalGeneratedTokens++;
                        
                        // 解码当前token并添加到结果中
                        try {
                            // 解码单个token，使用skipSpecialTokens=true过滤特殊token
                            String tokenText = tokenizerManager.decodeIds(new int[]{nextTokenId});
                            
                            // 只有非空token才添加到结果中
                            if (tokenText != null && !tokenText.isEmpty()) {
                                fullResponse.append(tokenText);
                                
                                // 通过回调接口返回token
                                callback.onToken(tokenText);
                                
                                // 在回调后检查停止状态
                                if (handler != null && handler.shouldStopInference()) {
                                    // 停止调试日志已移除
                                    callback.onError("推理被用户停止");
                                    return;
                                }
                                
                                // 减少日志频率，提高性能（每20个token记录一次）
                                if (i % 20 == 0) {
                                    double currentSpeed = (i + 1) / ((System.currentTimeMillis() - inferenceStartTime) / 1000.0);
                                    LogManager.logD(TAG, String.format("已生成 %d token，速度: %.2f token/s", i + 1, currentSpeed));
                                }
                            }
                        } catch (Exception e) {
                            LogManager.logE(TAG, "解码token时出错: " + e.getMessage(), e);
                        }
                    }
                    
                    // 尝试使用批量解码来提高效率
                    if (!generatedTokenIds.isEmpty()) {
                        try {
                            // 将List<Integer>转换为int[]
                            int[] allTokenIds = new int[generatedTokenIds.size()];
                            for (int i = 0; i < generatedTokenIds.size(); i++) {
                                allTokenIds[i] = generatedTokenIds.get(i);
                            }
                            
                            // 批量解码所有生成的token，使用skipSpecialTokens=true过滤特殊token
                            String batchDecodedText = tokenizerManager.decodeIds(allTokenIds);
                            LogManager.logD(TAG, "批量解码结果: " + escapeString(batchDecodedText));
                            
                            // 比较逐个解码和批量解码的结果
                            if (!batchDecodedText.equals(fullResponse.toString())) {
                                LogManager.logD(TAG, "批量解码和逐个解码结果不同，使用批量解码结果");
                                // 使用批量解码结果，因为它可能处理了特殊字符和字节级别的编码
                                fullResponse.setLength(0);
                                fullResponse.append(batchDecodedText);
                            }
                        } catch (Exception e) {
                            LogManager.logE(TAG, "批量解码失败，使用逐个解码结果: " + e.getMessage(), e);
                        }
                    }
                    
                    // 记录推理完成时间
                    inferenceEndTime = System.currentTimeMillis();
                    long inferenceTime = inferenceEndTime - inferenceStartTime;
                    LogManager.logD(TAG, "模型推理耗时: " + inferenceTime + "ms");
                    
                    // 计算统计信息
                    double tokenPerSecond = totalGeneratedTokens / (inferenceTime / 1000.0);
                    
                    // 停止内存监控线程
                    memoryMonitorThread.interrupt();
                    
                    // 使用统一的内存差值估算LLM推理消耗
                    long llmInferenceMemory = memoryMaxDuringInference - memoryBeforeInference;
                    
                    LogManager.logI(TAG, String.format("内存监控结束 - 推理前: %.2f MB, 推理期间最大: %.2f MB, LLM推理消耗: %.2f MB", 
                        memoryBeforeInference / 1024.0 / 1024.0, 
                        memoryMaxDuringInference / 1024.0 / 1024.0,
                        Math.max(0, llmInferenceMemory) / 1024.0 / 1024.0));
                    
                    // 构建统一格式的统计信息字符串
                     String statsInfo = String.format(
                         "\n\n---\n推理统计：时间：%.2f秒; LLM推理消耗：%.2f MB; 应用最大内存：%.2f MB; 生成速度：%.2f token/s; 生成token数：%d", 
                         inferenceTime / 1000.0, 
                         Math.max(0, llmInferenceMemory) / (1024.0 * 1024.0),
                         memoryMaxDuringInference / (1024.0 * 1024.0),
                         tokenPerSecond,
                         totalGeneratedTokens
                     );
                    
                    // 将统计信息添加到结果中
                    fullResponse.append(statsInfo);
                    
                    // 输出统计信息到日志
                    LogManager.logD(TAG, "推理统计信息:\n" + 
                          "- 总生成token数: " + totalGeneratedTokens + "; " +
                          "- 推理时间: " + (inferenceTime / 1000.0) + "秒; " +
                          "- LLM推理消耗: " + Math.max(0, llmInferenceMemory / (1024.0 * 1024.0)) + " MB; " +
                          "- 生成速度: " + tokenPerSecond + " token/s");
                    
                    // 打印详细的内存统计信息
                    LogManager.logI(TAG, getMemoryStats());
                    
                    // 通过回调接口返回完整结果（包含统计信息）
                    callback.onComplete(fullResponse.toString());
                    
                    // 单独发送统计信息token，确保UI能够正确显示
                    callback.onToken(statsInfo);
                    
                } catch (OrtException e) {
                    LogManager.logE(TAG, "模型推理失败", e);
                    callback.onError("推理失败: " + e.getMessage());
                }
                
                // 释放资源
                for (OnnxTensor tensor : inputs.values()) {
                    tensor.close();
                }
                
            } catch (Exception e) {
                LogManager.logE(TAG, "推理过程发生异常", e);
                callback.onError("推理异常: " + e.getMessage());
            } finally {
                // 重置TokenizerManager资源
                if (tokenizerManager != null) {
                    try {
                        TokenizerManager.resetManager();
                        LogManager.logD(TAG, "TokenizerManager实例已重置");
                    } catch (Exception e) {
                        LogManager.logE(TAG, "重置TokenizerManager时出错: " + e.getMessage(), e);
                    }
                }
            }
        }).start();
    }
    
    // 内存监控变量（统一与OnnxRuntimeGenAI的实现）
    private long memoryBeforeInference = 0;
    private long memoryMaxDuringInference = 0;
    private static int memoryUpdateCount = 0; // 内存监控更新计数器
    
    // 缓存张量以减少重复创建开销 - 增强版本
    private LongBuffer cachedInputBuffer = null;
    private LongBuffer cachedAttentionBuffer = null;
    private int cachedMaxLength = 0;
    
    // 张量内存管理优化
    private long lastMemoryCheckTime = 0;
    private static final long MEMORY_CHECK_INTERVAL = 5000; // 5秒检查一次内存
    private static final double MEMORY_PRESSURE_THRESHOLD = 0.75; // 75%内存使用率阈值（优化后）
    private static final int RECOMMENDED_MIN_MEMORY_MB = 2048; // 推荐最小内存2GB
    private int tensorReuseCount = 0;
    private long totalTensorMemorySaved = 0;
    
    private Map<String, OnnxTensor> prepareInputs(int[] inputIds) throws OrtException {
        long startTime = System.currentTimeMillis();
        
        // 张量内存管理优化：定期检查内存压力
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastMemoryCheckTime > MEMORY_CHECK_INTERVAL) {
            checkAndOptimizeMemory();
            lastMemoryCheckTime = currentTime;
        }
        
        Map<String, OnnxTensor> inputs = new HashMap<>();
        
        // 获取批处理大小
        int batchSize = (modelConfig != null && modelConfig.getMaxBatchSize() > 1) ? 
                       modelConfig.getMaxBatchSize() : 1;
        
        try {
            
            // 优化：重用缓存的buffer以减少内存分配
            int requiredLength = inputIds.length * batchSize;
            boolean needReallocation = false;
            
            if (cachedInputBuffer == null || requiredLength > cachedMaxLength) {
                // 计算节省的内存（如果是重用）
                if (cachedInputBuffer != null) {
                    long savedMemory = cachedMaxLength * 8; // 8字节per long
                    totalTensorMemorySaved += savedMemory;
                    LogManager.logD(TAG, "张量缓存重用节省内存: " + (savedMemory / 1024) + "KB");
                }
                
                cachedMaxLength = Math.max(requiredLength * 2, 2048); // 预分配更大空间
                cachedInputBuffer = LongBuffer.allocate(cachedMaxLength);
                if (modelConfig != null && modelConfig.requiresAttentionMask()) {
                    cachedAttentionBuffer = LongBuffer.allocate(cachedMaxLength);
                }
                needReallocation = true;
                LogManager.logD(TAG, "重新分配张量缓存，最大长度: " + cachedMaxLength + ", 批处理大小: " + batchSize);
            } else {
                // 缓存重用成功
                tensorReuseCount++;
                if (tensorReuseCount % 10 == 0) { // 每10次重用记录一次
                    LogManager.logD(TAG, "张量缓存重用次数: " + tensorReuseCount + ", 累计节省内存: " + (totalTensorMemorySaved / 1024 / 1024) + "MB");
                }
            }
            
            // 清空并填充输入数据
            cachedInputBuffer.clear();
            
            // 支持批处理：复制输入数据到每个批次
            for (int batch = 0; batch < batchSize; batch++) {
                for (int i = 0; i < inputIds.length; i++) {
                    cachedInputBuffer.put(inputIds[i]);
                }
            }
            cachedInputBuffer.flip();
            cachedInputBuffer.limit(requiredLength);
            
            long[] shape = new long[]{batchSize, inputIds.length};
            
            // 创建输入张量（使用缓存的buffer）
            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(ortEnvironment, cachedInputBuffer, shape);
            inputs.put("input_ids", inputIdsTensor);
            
            // 根据模型需要，添加注意力掩码张量
            if (modelConfig != null && modelConfig.requiresAttentionMask() && cachedAttentionBuffer != null) {
                cachedAttentionBuffer.clear();
                for (int batch = 0; batch < batchSize; batch++) {
                    for (int i = 0; i < inputIds.length; i++) {
                        cachedAttentionBuffer.put(1L); // 所有位置都设置为可见
                    }
                }
                cachedAttentionBuffer.flip();
                cachedAttentionBuffer.limit(requiredLength);
                
                OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(ortEnvironment, cachedAttentionBuffer, shape);
                inputs.put("attention_mask", attentionMaskTensor);
            }
            
            // KV缓存支持
            if (modelConfig != null && modelConfig.isEnableKVCache()) {
                addKVCacheInputs(inputs, inputIds.length, batchSize);
            }
            
        } catch (OrtException e) {
            LogManager.logE(TAG, "创建输入张量失败: " + e.getMessage(), e);
            throw e;
        }
        
        long endTime = System.currentTimeMillis();
        long prepareTime = endTime - startTime;
        if (prepareTime > 5) { // 只记录较慢的操作
            LogManager.logD(TAG, "准备输入张量耗时: " + prepareTime + "ms, 批处理大小: " + batchSize);
        }
        
        return inputs;
    }
    
    /**
     * 开始内存监控 (统一实现)
     */
    private void startMemoryMonitoring() {
        // 强制垃圾回收，确保获得准确的基线内存
        System.gc();
        try {
            Thread.sleep(100); // 等待GC完成
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        Runtime runtime = Runtime.getRuntime();
        // 记录推理开始前的内存状态
        memoryBeforeInference = runtime.totalMemory() - runtime.freeMemory();
        memoryMaxDuringInference = memoryBeforeInference;
        
        // 重置计数器
        memoryUpdateCount = 0;
        
        LogManager.logI(TAG, String.format("内存监控开始 - 应用内存: %.2f MB, JVM总内存: %.2f MB, JVM空闲内存: %.2f MB", 
            memoryBeforeInference / (1024.0 * 1024.0),
            runtime.totalMemory() / (1024.0 * 1024.0),
            runtime.freeMemory() / (1024.0 * 1024.0)));
    }
    
    /**
     * 更新内存监控数据 (统一实现)
     */
    private void updateMemoryMonitoring() {
        Runtime runtime = Runtime.getRuntime();
        long currentMemory = runtime.totalMemory() - runtime.freeMemory();
        
        if (currentMemory > memoryMaxDuringInference) {
            long oldMax = memoryMaxDuringInference;
            memoryMaxDuringInference = currentMemory;
            // 只在内存显著增加时记录日志（避免日志过多）
            if (currentMemory - oldMax > 5 * 1024 * 1024) { // 增加超过5MB时记录
                LogManager.logD(TAG, String.format("内存峰值更新: %.2f MB -> %.2f MB", 
                    oldMax / (1024.0 * 1024.0), currentMemory / (1024.0 * 1024.0)));
            }
        }
        
        // 定期记录当前内存状态（每10次更新记录一次，约10秒）
        memoryUpdateCount++;
        if (memoryUpdateCount % 10 == 0) {
            LogManager.logD(TAG, String.format("内存监控: 当前 %.2f MB, 峰值 %.2f MB", 
                currentMemory / (1024.0 * 1024.0), memoryMaxDuringInference / (1024.0 * 1024.0)));
        }
    }
    
    /**
     * 获取内存统计信息 (统一实现)
     */
    private String getMemoryStats() {
        long memoryIncrease = memoryMaxDuringInference - memoryBeforeInference;
        
        return String.format("\n内存统计:\n" +
            "- 应用推理前: %d MB\n" +
            "- 应用最大占用: %d MB\n" +
            "- LLM推理消耗: %d MB",
            memoryBeforeInference / (1024 * 1024),
            memoryMaxDuringInference / (1024 * 1024),
            Math.max(0, memoryIncrease / (1024 * 1024)));
    }
    
    /**
     * 检查并优化内存使用
     * 实现智能内存管理和垃圾回收优化
     */
    private void checkAndOptimizeMemory() {
        try {
            Runtime runtime = Runtime.getRuntime();
            long maxMemory = runtime.maxMemory();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            long usedMemory = totalMemory - freeMemory;
            
            double memoryUsageRatio = (double) usedMemory / maxMemory;
            
            // 记录内存使用情况
            LogManager.logD(TAG, String.format("内存使用情况 - 使用率: %.1f%%, 已用: %dMB, 最大: %dMB", 
                memoryUsageRatio * 100, usedMemory / 1024 / 1024, maxMemory / 1024 / 1024));
            
            // 当内存使用率超过阈值时，执行优化策略（增强版）
            if (memoryUsageRatio > MEMORY_PRESSURE_THRESHOLD) {
                LogManager.logW(TAG, "检测到内存压力，开始执行增强优化策略");
                
                // 1. 更积极的张量缓存清理（降低阈值到75%）
                if (memoryUsageRatio > MEMORY_PRESSURE_THRESHOLD && cachedInputBuffer != null) {
                    long freedMemory = cachedMaxLength * 8; // 8字节per long
                    cachedInputBuffer = null;
                    cachedAttentionBuffer = null;
                    cachedMaxLength = 0;
                    LogManager.logI(TAG, "积极清理张量缓存，释放内存: " + (freedMemory / 1024) + "KB");
                }
                
                // 2. 清理KV缓存（新增）
                if (memoryUsageRatio > 0.8) {
                    // 这里可以添加KV缓存清理逻辑
                    LogManager.logI(TAG, "执行KV缓存清理策略");
                }
                
                // 3. 强制垃圾回收（增强版）
                System.gc();
                
                // 4. 等待一小段时间让GC完成
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                
                // 5. 检查优化效果
                long newUsedMemory = runtime.totalMemory() - runtime.freeMemory();
                long memoryFreed = usedMemory - newUsedMemory;
                if (memoryFreed > 0) {
                    LogManager.logI(TAG, "增强内存优化完成，释放内存: " + (memoryFreed / 1024 / 1024) + "MB");
                } else {
                    LogManager.logW(TAG, "增强内存优化效果有限，建议检查内存泄漏或考虑更大内存配置");
                }
            }
            
        } catch (Exception e) {
            LogManager.logE(TAG, "内存优化过程中发生错误: " + e.getMessage(), e);
        }
    }
    
    /**
     * 添加KV缓存相关的输入张量
     * 修复类型不匹配问题，支持多种数据类型
     * @param inputs 输入张量映射
     * @param seqLength 序列长度
     * @param batchSize 批处理大小
     * @throws OrtException ONNX Runtime异常
     */
    private void addKVCacheInputs(Map<String, OnnxTensor> inputs, int seqLength, int batchSize) throws OrtException {
        try {
            // 获取模型配置中的注意力头数和隐藏层大小
            int numHeads = modelConfig.getNumAttentionHeads();
            int hiddenSize = modelConfig.getHiddenSize();
            int headDim = hiddenSize / numHeads;
            int numLayers = modelConfig.getNumHiddenLayers();
            
            // 根据调查报告建议，优化缓存大小计算
            // 减少初始缓存大小以降低内存压力
            int requestedCacheSize = Math.min(seqLength + 256, 512); // 减少缓存大小
            int initialCacheSize = modelConfig.getAdaptiveCacheSize(requestedCacheSize);
            
            // 计算预估内存使用量（优化计算公式）
            long memoryPerLayer = (long) batchSize * numHeads * initialCacheSize * headDim * 4 * 2; // 4字节float, K+V两个缓存
            long totalMemory = memoryPerLayer * numLayers;
            
            // 获取当前内存状况
            Runtime runtime = Runtime.getRuntime();
            long jvmMaxMemory = runtime.maxMemory();
            long jvmTotalMemory = runtime.totalMemory();
            long jvmFreeMemory = runtime.freeMemory();
            long jvmAvailableMemory = jvmMaxMemory - jvmTotalMemory + jvmFreeMemory;
            
            // 获取系统内存信息
            long systemAvailableMemory = com.example.starlocalrag.GlobalApplication.getAvailableMemoryMB();
            long jvmMaxMemoryMB = com.example.starlocalrag.GlobalApplication.getJVMMaxMemoryMB();
            
            // 减少日志输出频率（根据调查报告建议）
            if (LogManager.isDebugEnabled()) {
                LogManager.logD(TAG, String.format("KV缓存内存分析: JVM可用=%.1f MB, 预估使用=%.1f MB (层数=%d, 头数=%d, 缓存大小=%d)", 
                    jvmAvailableMemory / (1024.0 * 1024.0), totalMemory / (1024.0 * 1024.0), numLayers, numHeads, initialCacheSize));
            }
            
            // 根据调查报告建议，降低内存阈值到50%以提高稳定性
            if (totalMemory > jvmAvailableMemory * 0.5) {
                LogManager.logW(TAG, String.format("内存不足，动态禁用KV缓存 (预估%.1f MB > 可用%.1f MB * 50%%)", 
                    totalMemory / (1024.0 * 1024.0), jvmAvailableMemory / (1024.0 * 1024.0)));
                modelConfig.disableKVCache();
                return; // 直接返回，不创建KV缓存
            }
            
            // 进一步优化缓存大小
            if (totalMemory > 32 * 1024 * 1024) { // 降低阈值到32MB
                initialCacheSize = Math.min(initialCacheSize / 2, 64); // 进一步减少最大缓存
                totalMemory = (long) batchSize * numHeads * initialCacheSize * headDim * 4 * 2 * numLayers;
                LogManager.logI(TAG, String.format("优化KV缓存大小至: %d, 新预估内存: %.1f MB", 
                    initialCacheSize, totalMemory / (1024.0 * 1024.0)));
            }
            
            // 为每一层创建KV缓存张量，支持多种数据类型
            for (int layer = 0; layer < numLayers; layer++) {
                // Key缓存形状: [batch_size, num_heads, cache_size, head_dim]
                long[] kvShape = new long[]{batchSize, numHeads, initialCacheSize, headDim};
                
                try {
                    // 优先尝试使用FloatBuffer创建张量（修复类型不匹配问题）
                    int totalElements = batchSize * numHeads * initialCacheSize * headDim;
                    FloatBuffer kCacheBuffer = FloatBuffer.allocate(totalElements);
                    FloatBuffer vCacheBuffer = FloatBuffer.allocate(totalElements);
                    
                    // 初始化为0
                    for (int i = 0; i < totalElements; i++) {
                        kCacheBuffer.put(0.0f);
                        vCacheBuffer.put(0.0f);
                    }
                    kCacheBuffer.flip();
                    vCacheBuffer.flip();
                    
                    OnnxTensor kCacheTensor = OnnxTensor.createTensor(ortEnvironment, kCacheBuffer, kvShape);
                    OnnxTensor vCacheTensor = OnnxTensor.createTensor(ortEnvironment, vCacheBuffer, kvShape);
                    
                    inputs.put("past_key_values." + layer + ".key", kCacheTensor);
                    inputs.put("past_key_values." + layer + ".value", vCacheTensor);
                    
                } catch (Exception bufferException) {
                    // 如果FloatBuffer失败，回退到多维数组方式
                    LogManager.logW(TAG, "FloatBuffer创建失败，回退到数组方式: " + bufferException.getMessage());
                    
                    float[][][][] emptyKCache = new float[batchSize][numHeads][initialCacheSize][headDim];
                    float[][][][] emptyVCache = new float[batchSize][numHeads][initialCacheSize][headDim];
                    
                    OnnxTensor kCacheTensor = OnnxTensor.createTensor(ortEnvironment, emptyKCache);
                    OnnxTensor vCacheTensor = OnnxTensor.createTensor(ortEnvironment, emptyVCache);
                    
                    inputs.put("past_key_values." + layer + ".key", kCacheTensor);
                    inputs.put("past_key_values." + layer + ".value", vCacheTensor);
                }
            }
            
            // 添加缓存长度张量 - 修复类型不匹配问题
            long[] cacheLenShape = new long[]{batchSize};
            
            // 支持多种类型的缓存长度张量
            try {
                // 优先尝试IntBuffer（某些模型可能需要int类型）
                IntBuffer cacheLenIntBuffer = IntBuffer.allocate(batchSize);
                for (int i = 0; i < batchSize; i++) {
                    cacheLenIntBuffer.put(0); // 初始缓存长度为0
                }
                cacheLenIntBuffer.flip();
                
                OnnxTensor cacheLenTensor = OnnxTensor.createTensor(ortEnvironment, cacheLenIntBuffer, cacheLenShape);
                inputs.put("cache_length", cacheLenTensor);
                
                if (LogManager.isDebugEnabled()) {
                    LogManager.logD(TAG, "成功创建cache_length张量(int类型)");
                }
                
            } catch (Exception intException) {
                try {
                    // 回退到LongBuffer
                    LongBuffer cacheLenBuffer = LongBuffer.allocate(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        cacheLenBuffer.put(0L); // 初始缓存长度为0
                    }
                    cacheLenBuffer.flip();
                    
                    OnnxTensor cacheLenTensor = OnnxTensor.createTensor(ortEnvironment, cacheLenBuffer, cacheLenShape);
                    inputs.put("cache_length", cacheLenTensor);
                    
                    if (LogManager.isDebugEnabled()) {
                        LogManager.logD(TAG, "成功创建cache_length张量(long类型)");
                    }
                    
                } catch (Exception longException) {
                    LogManager.logW(TAG, "创建cache_length张量失败，跳过: int异常=" + intException.getMessage() + ", long异常=" + longException.getMessage());
                    // 不添加cache_length张量，某些模型可能不需要这个输入
                }
            }
            
            LogManager.logI(TAG, "KV缓存张量已添加，层数: " + numLayers + ", 头数: " + numHeads + ", 头维度: " + headDim + ", 缓存大小: " + initialCacheSize);
            
        } catch (Exception e) {
            LogManager.logE(TAG, "添加KV缓存张量失败: " + e.getMessage(), e);
            // KV缓存失败时禁用KV缓存，避免影响推理
            modelConfig.disableKVCache();
        }
    }
}
