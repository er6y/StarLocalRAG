package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.LongBuffer;
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

/**
 * ONNX模型处理类
 * 负责ONNX模型的推理逻辑
 */
public class LocalLLMOnnxHandler {
    
    /**
     * 获取当前使用的内存量（单位：字节）
     * @return 当前使用的内存量
     */
    private long getUsedMemory() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
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
        
        Log.d(TAG, "LocalLLMOnnxHandler 初始化完成");
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
        Log.d(TAG, "开始推理，提示词长度: " + prompt.length());
        Log.d(TAG, "推理参数 - maxTokenLength: " + maxTokenLength + ", thinkingMode: " + thinkingMode 
              + ", temperature: " + temperature + ", topK: " + topK);
        Log.d(TAG, "提示词内容: " + prompt);
        
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
        });
        
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
            Log.d(TAG, "分词结果 - 长度: " + inputIds.length + ", tokens: " + tokenDisplay);
            
            // 简单记录token ID
            StringBuilder tokenIds = new StringBuilder("Token IDs: ");
            for (int i = 0; i < Math.min(20, inputIds.length); i++) {
                tokenIds.append(inputIds[i]).append(" ");
            }
            if (inputIds.length > 20) {
                tokenIds.append("... (共").append(inputIds.length).append("个token)");
            }
            Log.d(TAG, tokenIds.toString());
            
            // 2. 准备输入张量
            Map<String, OnnxTensor> inputs = prepareInputs(inputIds);
            Log.d(TAG, "准备输入张量完成 - 张量数量: " + inputs.size());
            for (Map.Entry<String, OnnxTensor> entry : inputs.entrySet()) {
                Log.d(TAG, "输入张量: " + entry.getKey() + ", 形状: " + Arrays.toString(entry.getValue().getInfo().getShape()));
            }
            
            // 3. 执行推理
            Log.d(TAG, "开始执行模型推理...");
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
                
                Log.d(TAG, "生成参数 - 温度: " + actualTemperature + ", topP: " + topP + ", topK: " + actualTopK);
                
                // 自回归生成循环
                for (int i = 0; i < maxTokenLength; i++) {
                    // 准备当前输入
                    Map<String, OnnxTensor> currentInputs = prepareInputs(currentInputIds);
                    
                    // 执行推理
                    OrtSession.Result output = ortSession.run(currentInputs);
                    
                    if (i == 0) {
                        Log.d(TAG, "模型推理完成，输出数量: " + output.size());
                        
                        // 打印输出张量信息
                        for (Map.Entry<String, OnnxValue> entry : output) {
                            String outputName = entry.getKey();
                            OnnxValue value = entry.getValue();
                            Log.d(TAG, "输出张量: " + outputName + ", 类型: " + value.getInfo());
                            
                            // 安全地获取形状信息
                            try {
                                if (value instanceof OnnxTensor) {
                                    OnnxTensor tensor = (OnnxTensor) value;
                                    Log.d(TAG, "张量形状: " + Arrays.toString(tensor.getInfo().getShape()));
                                } else {
                                    Log.d(TAG, "非张量类型，无法获取形状信息");
                                }
                            } catch (Exception e) {
                                Log.w(TAG, "获取张量形状时出错: " + e.getMessage());
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
                                                Log.d(TAG, "生成的下一个token ID: " + nextTokenId + ", 文本: '" + tokenText + "'");
                                            } catch (Exception e) {
                                                Log.d(TAG, "生成的下一个token ID: " + nextTokenId + ", 无法解码文本: " + e.getMessage());
                                            }
                                            break;
                                        }
                                    } catch (Exception e) {
                                        Log.e(TAG, "处理logits时出错: " + e.getMessage(), e);
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
                                                Log.d(TAG, "生成的下一个token ID: " + nextTokenId + ", 文本: '" + tokenText + "'");
                                            } catch (Exception e) {
                                                Log.d(TAG, "生成的下一个token ID: " + nextTokenId + ", 无法解码文本: " + e.getMessage());
                                            }
                                            break;
                                        }
                                    } catch (Exception e) {
                                        Log.e(TAG, "处理二维张量时出错: " + e.getMessage(), e);
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
                            Log.e(TAG, "无法从输出中提取下一个token ID");
                            break;
                        }
                        
                        // 检查是否结束生成（通常是特殊的EOS token）
                        if (nextTokenId == 151645) { // <|im_end|> token ID
                            Log.d(TAG, "检测到结束token，停止生成");
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
                                Log.d(TAG, "已生成 " + (i+1) + " 个token，当前文本(Unicode): " + escapeString(generatedText.toString()));
                                // 打印原始文本版本（更直观）
                                Log.d(TAG, "已生成 " + (i+1) + " 个token，当前文本(原文): \n" + generatedText.toString());
                            }
                        } catch (Exception e) {
                            Log.e(TAG, "解码token时出错: " + e.getMessage(), e);
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
                        Log.d(TAG, "批量解码结果(Unicode): " + escapeString(batchDecodedText));
                        // 打印原始文本版本（更直观）
                        Log.d(TAG, "批量解码结果(原文): \n" + batchDecodedText);
                        
                        // 比较逐个解码和批量解码的结果
                        if (!batchDecodedText.equals(generatedText.toString())) {
                            Log.d(TAG, "批量解码和逐个解码结果不同，使用批量解码结果");
                            // 使用批量解码结果，因为它可能处理了特殊字符和字节级别的编码
                            result.setLength(0);
                            result.append(batchDecodedText);
                        } else {
                            // 结果相同，使用原来的结果
                            result.append(generatedText.toString());
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "批量解码失败，使用逐个解码结果: " + e.getMessage(), e);
                        // 如果批量解码失败，使用逐个解码的结果
                        result.append(generatedText.toString());
                    }
                } else {
                    // 没有生成任何token，直接使用原来的结果
                    result.append(generatedText.toString());
                }
                
                // 记录推理完成时间
                long inferenceEndTime = System.currentTimeMillis();
                Log.d(TAG, "模型推理耗时: " + (inferenceEndTime - inferenceStartTime) + "ms");
                
            } catch (OrtException e) {
                Log.e(TAG, "模型推理失败", e);
                Log.e(TAG, "错误详情: " + e.getMessage());
                result.append("推理失败: ").append(e.getMessage());
            }
            
            // 4. 释放资源
            for (OnnxTensor tensor : inputs.values()) {
                tensor.close();
            }
        } catch (Exception e) {
            Log.e(TAG, "推理过程发生异常", e);
            Log.e(TAG, "异常详情: " + e.getMessage());
            result.append("推理异常: ").append(e.getMessage());
        } finally {
            // 重置TokenizerManager资源
            if (tokenizerManager != null) {
                try {
                    tokenizerManager.resetManager();
                    Log.d(TAG, "TokenizerManager实例已重置");
                } catch (Exception e) {
                    Log.e(TAG, "重置TokenizerManager时出错: " + e.getMessage(), e);
                }
            }
        }
        
        long endTime = System.currentTimeMillis();
        Log.d(TAG, "总推理耗时: " + (endTime - startTime) + "ms");
        
        return result.toString();
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
        Log.d(TAG, "应用分词器模板到文本，思考模式: " + thinkingMode);
        
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
                Log.e(TAG, "TokenizerManager初始化失败，请检查模型路径: " + modelDir.getAbsolutePath());
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
            
            Log.d(TAG, "使用JNI层applyChatTemplate后的文本长度: " + formattedText.length());
            Log.d(TAG, "使用JNI层applyChatTemplate后的文本片段: " + 
                  formattedText.substring(0, Math.min(100, formattedText.length())) + "...");
            
            return formattedText;
        } catch (Exception e) {
            Log.e(TAG, "应用聊天模板失败: " + e.getMessage(), e);
            // 直接抛出异常，不在Java层硬编码模板
            throw new RuntimeException("应用聊天模板失败，请检查分词器配置: " + e.getMessage(), e);
        }
    }
    
    /**
     * 对输入文本进行分词
     * 使用TokenizerManager进行分词处理
     * @param text 输入文本
     * @param thinkingMode 是否启用思考模式
     * @return 分词结果（token IDs）
     */
    private int[] tokenizeInput(String text, boolean thinkingMode) {
        Log.d(TAG, "开始对输入文本进行分词，文本长度: " + text.length());
        long startTime = System.currentTimeMillis();
        
        int[] result;
        
        try {
            // 初始化TokenizerManager（如果尚未初始化）
            if (tokenizerManager == null && context != null) {
                tokenizerManager = TokenizerManager.getInstance(context);
                File modelDir = new File(modelConfig.getModelPath());
                boolean initSuccess = tokenizerManager.initialize(modelDir);
                if (!initSuccess) {
                    Log.e(TAG, "TokenizerManager初始化失败");
                    // 如果初始化失败，返回默认的特殊标记
                    return new int[]{modelConfig.getBosToken()};
                }
                Log.d(TAG, "TokenizerManager初始化成功");
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
                Log.d(TAG, tokenLog.toString());
                
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
                    Log.d(TAG, summaryLog.toString());
                }
            } else {
                Log.e(TAG, "分词结果为空或格式不正确");
                result = new int[]{modelConfig.getBosToken()};
            }
        } catch (Exception e) {
            Log.e(TAG, "分词过程发生异常: " + e.getMessage(), e);
            // 如果分词失败，使用默认的特殊标记
            result = new int[]{modelConfig.getBosToken()};
        }
        
        long endTime = System.currentTimeMillis();
        Log.d(TAG, "分词完成，耗时: " + (endTime - startTime) + "ms, 生成token数量: " + result.length);
        
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
    public void inferenceStream(String prompt, int maxTokenLength, boolean thinkingMode, float temperature, int topK, StreamingCallback callback) {
        Log.d(TAG, "开始流式推理，提示词长度: " + prompt.length());
        Log.d(TAG, "推理参数 - maxTokenLength: " + maxTokenLength + ", thinkingMode: " + thinkingMode 
              + ", temperature: " + temperature + ", topK: " + topK);
        
        // 在新线程中执行推理，避免阻塞调用线程
        new Thread(() -> {
            StringBuilder fullResponse = new StringBuilder();
            
            // 推理统计变量
            long inferenceStartTime = 0;
            long inferenceEndTime = 0;
            int totalGeneratedTokens = 0;
            long initialMemory = getUsedMemory();
            long peakMemory = initialMemory;
            final long[] currentMemory = {initialMemory};
            
            // 创建内存监控线程
            Thread memoryMonitorThread = new Thread(() -> {
                try {
                    while (!Thread.interrupted()) {
                        long memory = getUsedMemory();
                        currentMemory[0] = memory;
                        if (memory > peakMemory) {
                            peakMemory = memory;
                        }
                        Thread.sleep(1000); // 每秒监控一次
                    }
                } catch (InterruptedException e) {
                    // 线程被中断，正常退出
                }
            });
            memoryMonitorThread.setDaemon(true);
            
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
                Log.d(TAG, "分词结果 - 长度: " + inputIds.length + ", tokens: " + tokenDisplay);
                
                // 2. 准备输入张量
                Map<String, OnnxTensor> inputs = prepareInputs(inputIds);
                Log.d(TAG, "准备输入张量完成 - 张量数量: " + inputs.size());
                
                // 3. 执行推理
                Log.d(TAG, "开始执行流式模型推理...");
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
                    
                    Log.d(TAG, "生成参数 - 温度: " + actualTemperature + ", topP: " + topP + ", topK: " + actualTopK);
                    
                    // 自回归生成循环
                    for (int i = 0; i < maxTokenLength; i++) {
                        // 准备当前输入
                        Map<String, OnnxTensor> currentInputs = prepareInputs(currentInputIds);
                        
                        // 执行推理
                        OrtSession.Result output = ortSession.run(currentInputs);
                        
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
                                        Log.e(TAG, "处理logits时出错: " + e.getMessage(), e);
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
                                        Log.e(TAG, "处理二维张量时出错: " + e.getMessage(), e);
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
                            Log.e(TAG, "无法从输出中提取下一个token ID");
                            break;
                        }
                        
                        // 检查是否结束生成（通常是特殊的EOS token）
                        if (nextTokenId == 151645) { // <|im_end|> token ID
                            Log.d(TAG, "检测到结束token，停止生成");
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
                                
                                // 每5个token记录日志
                                if (i % 5 == 0) {
                                    Log.d(TAG, "已生成 " + (i+1) + " 个token，当前文本: " + escapeString(fullResponse.toString()));
                                }
                            }
                        } catch (Exception e) {
                            Log.e(TAG, "解码token时出错: " + e.getMessage(), e);
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
                            Log.d(TAG, "批量解码结果: " + escapeString(batchDecodedText));
                            
                            // 比较逐个解码和批量解码的结果
                            if (!batchDecodedText.equals(fullResponse.toString())) {
                                Log.d(TAG, "批量解码和逐个解码结果不同，使用批量解码结果");
                                // 使用批量解码结果，因为它可能处理了特殊字符和字节级别的编码
                                fullResponse.setLength(0);
                                fullResponse.append(batchDecodedText);
                            }
                        } catch (Exception e) {
                            Log.e(TAG, "批量解码失败，使用逐个解码结果: " + e.getMessage(), e);
                        }
                    }
                    
                    // 记录推理完成时间
                    inferenceEndTime = System.currentTimeMillis();
                    long inferenceTime = inferenceEndTime - inferenceStartTime;
                    Log.d(TAG, "模型推理耗时: " + inferenceTime + "ms");
                    
                    // 计算统计信息
                    double tokenPerSecond = totalGeneratedTokens / (inferenceTime / 1000.0);
                    long memoryUsage = peakMemory - initialMemory;
                    
                    // 停止内存监控线程
                    memoryMonitorThread.interrupt();
                    
                    // 构建统计信息字符串
                    String statsInfo = String.format(
                        "\n\n---\n推理统计：\n- 推理时间：%.2f秒\n- 内存占用：%.2f MB\n- 生成速度：%.2f token/s", 
                        inferenceTime / 1000.0, 
                        memoryUsage / (1024.0 * 1024.0), 
                        tokenPerSecond
                    );
                    
                    // 将统计信息添加到结果中
                    fullResponse.append(statsInfo);
                    
                    // 输出统计信息到日志
                    Log.d(TAG, "推理统计信息:\n" + 
                          "- 总生成token数: " + totalGeneratedTokens + "\n" +
                          "- 推理时间: " + (inferenceTime / 1000.0) + "秒\n" +
                          "- 内存占用: " + (memoryUsage / (1024.0 * 1024.0)) + " MB\n" +
                          "- 生成速度: " + tokenPerSecond + " token/s");
                    
                    // 通过回调接口返回完整结果（包含统计信息）
                    callback.onComplete(fullResponse.toString());
                    
                    // 单独发送统计信息token，确保UI能够正确显示
                    callback.onToken(statsInfo);
                    
                } catch (OrtException e) {
                    Log.e(TAG, "模型推理失败", e);
                    callback.onError("推理失败: " + e.getMessage());
                }
                
                // 释放资源
                for (OnnxTensor tensor : inputs.values()) {
                    tensor.close();
                }
                
            } catch (Exception e) {
                Log.e(TAG, "推理过程发生异常", e);
                callback.onError("推理异常: " + e.getMessage());
            } finally {
                // 重置TokenizerManager资源
                if (tokenizerManager != null) {
                    try {
                        tokenizerManager.resetManager();
                        Log.d(TAG, "TokenizerManager实例已重置");
                    } catch (Exception e) {
                        Log.e(TAG, "重置TokenizerManager时出错: " + e.getMessage(), e);
                    }
                }
            }
        }).start();
    }
    
    private Map<String, OnnxTensor> prepareInputs(int[] inputIds) throws OrtException {
        Log.d(TAG, "开始准备ONNX模型输入张量");
        long startTime = System.currentTimeMillis();
        
        Map<String, OnnxTensor> inputs = new HashMap<>();
        
        try {
            // 创建输入IDs张量
            long[] longInputIds = new long[inputIds.length];
            for (int i = 0; i < inputIds.length; i++) {
                longInputIds[i] = inputIds[i];
            }
            
            long[] shape = new long[]{1, longInputIds.length};
            Log.d(TAG, "创建 input_ids 张量，形状: " + Arrays.toString(shape));
            
            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(ortEnvironment, LongBuffer.wrap(longInputIds), shape);
            inputs.put("input_ids", inputIdsTensor);
            Log.d(TAG, "input_ids 张量创建成功");
            
            // 根据模型需要，添加注意力掩码张量
            if (modelConfig != null && modelConfig.requiresAttentionMask()) {
                // 创建全为1的注意力掩码
                long[] attentionMask = new long[inputIds.length];
                for (int i = 0; i < inputIds.length; i++) {
                    attentionMask[i] = 1L; // 所有位置都设置为可见
                }
                
                OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(ortEnvironment, LongBuffer.wrap(attentionMask), shape);
                inputs.put("attention_mask", attentionMaskTensor);
                Log.d(TAG, "attention_mask 张量创建成功，形状: " + Arrays.toString(shape));
            }
            
            // 根据模型需要，可能还需要添加其他输入张量
            if (modelConfig != null) {
                // 根据模型配置添加其他必要的输入张量
                if (modelConfig.requiresPositionIds()) {
                    Log.d(TAG, "模型需要 position_ids 张量，正在创建...");
                    // 创建位置编码张量
                    // 实现位置编码逻辑...
                }
                
                // 其他可能的输入张量，如token_type_ids等
            }
            
            Log.d(TAG, "所有输入张量准备完成，共 " + inputs.size() + " 个张量");
            
        } catch (OrtException e) {
            Log.e(TAG, "创建输入张量失败", e);
            Log.e(TAG, "错误详情: " + e.getMessage());
            throw e; // 重新抛出异常，以便上层方法处理
        }
        
        long endTime = System.currentTimeMillis();
        Log.d(TAG, "准备输入张量耗时: " + (endTime - startTime) + "ms");
        
        return inputs;
    }
}
