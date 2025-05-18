package com.example.starlocalrag.api;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.example.starlocalrag.ConfigManager;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
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

/**
 * 本地LLM处理程序
 * 负责加载和管理本地ONNX模型，执行本地推理
 */
public class LocalLLMHandler {
    private static final String TAG = "LocalLLMHandler";
    
    // 单例实例
    private static LocalLLMHandler instance;
    
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
    
    // ONNX运行时环境
    private OrtEnvironment ortEnvironment;
    
    // ONNX会话
    private OrtSession ortSession;
    
    // 模型配置
    private ModelConfig modelConfig;
    
    // 词汇表
    private Map<String, Integer> tokenizer;
    private Map<Integer, String> reverseTokenizer;
    
    // 特殊token
    private int bosToken = 1;
    private int eosToken = 2;
    private int padToken = 0;
    
    // 最大序列长度
    private int maxSeqLen = 2048;
    
    // 模型配置类
    private static class ModelConfig {
        String modelType; // 模型类型，如"qwen", "deepseek"等
        int vocabSize;    // 词汇表大小
        int hiddenSize;   // 隐藏层大小
        int numLayers;    // 层数
        int numHeads;     // 注意力头数
        
        public ModelConfig(String modelType, int vocabSize, int hiddenSize, int numLayers, int numHeads) {
            this.modelType = modelType;
            this.vocabSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
            this.numHeads = numHeads;
        }
    }
    
    /**
     * 本地LLM回调接口
     */
    public interface LocalLlmCallback {
        void onToken(String token);
        void onComplete(String fullResponse);
        void onError(String errorMessage);
    }
    
    /**
     * 获取单例实例
     */
    public static synchronized LocalLLMHandler getInstance(Context context) {
        if (instance == null) {
            instance = new LocalLLMHandler(context);
        }
        return instance;
    }
    
    /**
     * 私有构造函数
     */
    private LocalLLMHandler(Context context) {
        this.context = context;
        this.executorService = Executors.newSingleThreadExecutor();
        
        // 从配置中获取是否使用GPU
        this.useGpu = ConfigManager.getBoolean(context, ConfigManager.KEY_USE_GPU, false);
        
        Log.d(TAG, "LocalLLMHandler 初始化, 使用GPU: " + useGpu);
    }
    
    /**
     * 加载本地模型
     * @param modelName 模型名称（目录名）
     * @param callback 回调接口
     */
    public void loadModel(String modelName, final LocalLlmCallback callback) {
        // 防止重复加载
        if (modelLoading.get()) {
            if (callback != null) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    callback.onError("模型正在加载中，请稍后再试");
                });
            }
            return;
        }
        
        // 如果当前有模型已加载，先卸载
        if (modelLoaded.get()) {
            unloadModel();
        }
        
        // 标记为正在加载
        modelLoading.set(true);
        currentModelName = modelName;
        
        // 在后台线程中执行加载
        executorService.execute(() -> {
            try {
                Log.i(TAG, "开始加载模型: " + modelName);
                logMemoryInfo();
                
                // 1. 确保模型文件存在
                // 模型一般放在SD卡中，这里直接检查模型目录
                File modelDir = new File(android.os.Environment.getExternalStorageDirectory(), "models/" + modelName);
                
                if (!modelDir.exists() || !modelDir.isDirectory()) {
                    throw new IOException("模型文件不存在: " + modelDir.getAbsolutePath());
                }
                
                // 2. 加载模型配置
                File configFile = new File(modelDir, "config.json");
                if (!configFile.exists()) {
                    throw new IOException("模型配置文件不存在: " + configFile.getPath());
                }
                loadModelConfig(configFile);
                
                // 3. 加载词汇表
                File tokenizerFile = new File(modelDir, "tokenizer.json");
                if (!tokenizerFile.exists()) {
                    throw new IOException("词汇表文件不存在: " + tokenizerFile.getPath());
                }
                loadTokenizer(tokenizerFile);
                
                // 4. 初始化ONNX运行时环境
                ortEnvironment = OrtEnvironment.getEnvironment();
                
                // 5. 配置会话选项
                SessionOptions sessionOptions = new SessionOptions();
                
                // 设置线程数
                int threads = ConfigManager.getThreads(context);
                sessionOptions.setIntraOpNumThreads(threads);
                
                // 启用内存优化
                sessionOptions.setMemoryPatternOptimization(true);
                sessionOptions.setExecutionMode(SessionOptions.ExecutionMode.SEQUENTIAL);
                
                // 如果启用GPU，设置GPU加速
                if (useGpu) {
                    try {
                        // 尝试启用GPU加速
                        sessionOptions.addCUDA();
                        Log.d(TAG, "已启用CUDA GPU加速");
                    } catch (Exception e) {
                        Log.w(TAG, "启用GPU加速失败，将使用CPU: " + e.getMessage());
                    }
                }
                
                // 6. 加载ONNX模型
                File modelFile = new File(modelDir, "model.onnx");
                if (!modelFile.exists()) {
                    throw new IOException("模型文件不存在: " + modelFile.getPath());
                }
                
                Log.d(TAG, "开始加载ONNX模型: " + modelFile.getPath());
                ortSession = ortEnvironment.createSession(modelFile.getPath(), sessionOptions);
                
                // 7. 打印模型信息
                Map<String, NodeInfo> inputInfo = ortSession.getInputInfo();
                Map<String, NodeInfo> outputInfo = ortSession.getOutputInfo();
                
                Log.d(TAG, "模型输入数量: " + inputInfo.size());
                for (String inputName : inputInfo.keySet()) {
                    Log.d(TAG, "模型输入: " + inputName + ", 类型: " + inputInfo.get(inputName).getInfo());
                }
                
                Log.d(TAG, "模型输出数量: " + outputInfo.size());
                for (String outputName : outputInfo.keySet()) {
                    Log.d(TAG, "模型输出: " + outputName + ", 类型: " + outputInfo.get(outputName).getInfo());
                }
                
                // 8. 标记为已加载
                modelLoaded.set(true);
                Log.i(TAG, "模型加载完成: " + modelName);
                logMemoryInfo();
                
                // 回调成功
                if (callback != null) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onComplete("模型加载成功: " + modelName);
                    });
                }
                
            } catch (Exception e) {
                Log.e(TAG, "加载模型失败: " + e.getMessage(), e);
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
        Log.d(TAG, "加载模型配置: " + configFile.getPath());
        
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
        
        // 获取特殊token
        if (config.has("bos_token_id")) {
            bosToken = config.getInt("bos_token_id");
        }
        if (config.has("eos_token_id")) {
            eosToken = config.getInt("eos_token_id");
        }
        if (config.has("pad_token_id")) {
            padToken = config.getInt("pad_token_id");
        }
        
        Log.d(TAG, String.format("模型配置: 类型=%s, 词汇表大小=%d, 隐藏层大小=%d, 层数=%d, 注意力头数=%d",
            modelType, vocabSize, hiddenSize, numLayers, numHeads));
        Log.d(TAG, String.format("特殊token: BOS=%d, EOS=%d, PAD=%d", bosToken, eosToken, padToken));
    }
    
    /**
     * 加载词汇表
     * @param tokenizerFile 词汇表文件
     * @throws Exception 异常
     */
    private void loadTokenizer(File tokenizerFile) throws Exception {
        Log.d(TAG, "加载词汇表: " + tokenizerFile.getPath());
        
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(tokenizerFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
        }
        
        JSONObject tokenizerJson = new JSONObject(content.toString());
        
        // 初始化词汇表
        tokenizer = new HashMap<>();
        reverseTokenizer = new HashMap<>();
        
        // 解析词汇表
        if (tokenizerJson.has("model") && tokenizerJson.getJSONObject("model").has("vocab")) {
            JSONObject vocab = tokenizerJson.getJSONObject("model").getJSONObject("vocab");
            Iterator<String> keys = vocab.keys();
            
            while (keys.hasNext()) {
                String token = keys.next();
                int id = vocab.getInt(token);
                tokenizer.put(token, id);
                reverseTokenizer.put(id, token);
            }
        }
        
        Log.d(TAG, "词汇表加载完成，大小: " + tokenizer.size());
    }
    
    /**
     * 卸载当前模型
     */
    public void unloadModel() {
        executorService.execute(this::unloadModelInternal);
    }
    
    /**
     * 内部卸载模型方法
     */
    private void unloadModelInternal() {
        Log.d(TAG, "卸载模型: " + currentModelName);
        
        // 关闭ONNX会话
        if (ortSession != null) {
            try {
                ortSession.close();
                ortSession = null;
                Log.d(TAG, "ONNX会话关闭成功");
            } catch (Exception e) {
                Log.w(TAG, "关闭ONNX会话失败: " + e.getMessage());
            }
        }
        
        // 重置状态
        modelLoaded.set(false);
        currentModelName = null;
        tokenizer = null;
        reverseTokenizer = null;
        modelConfig = null;
        
        // 建议进行垃圾回收
        System.gc();
        
        Log.d(TAG, "模型卸载完成");
        logMemoryInfo();
    }
    
    /**
     * 记录当前内存使用情况
     */
    private void logMemoryInfo() {
        try {
            ActivityManager activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
            activityManager.getMemoryInfo(memoryInfo);
            
            Runtime runtime = Runtime.getRuntime();
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            long freeMemory = runtime.freeMemory();
            
            Log.i(TAG, String.format("内存使用情况: 已用=%.2fMB, 空闲=%.2fMB",
                usedMemory / (1024.0 * 1024.0),
                freeMemory / (1024.0 * 1024.0)
            ));
        } catch (Exception e) {
            Log.e(TAG, "获取内存信息失败: " + e.getMessage());
        }
    }
    
    /**
     * 安全关闭可关闭资源
     * @param closeable 可关闭的资源
     * @param resourceName 资源名称（用于日志）
     */
    private void safeClose(AutoCloseable closeable, String resourceName) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                Log.w(TAG, "关闭" + resourceName + "时发生异常: " + e.getMessage());
            }
        }
    }
    
    /**
     * 执行本地模型推理
     * @param prompt 提示词
     * @param callback 回调接口
     */
    public void inference(String prompt, final LocalLlmCallback callback) {
        // 检查模型是否已加载
        if (!modelLoaded.get()) {
            if (callback != null) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    callback.onError("模型未加载");
                });
            }
            return;
        }
        
        // 使用自回归生成方法
        generateText(prompt, callback);
    }
    
    /**
     * 执行自回归生成，支持连续生成多个 token
     * @param prompt 提示词
     * @param callback 回调接口
     */
    public void generateText(String prompt, final LocalLlmCallback callback) {
        // 检查模型是否已加载
        if (!modelLoaded.get()) {
            if (callback != null) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    callback.onError("模型未加载");
                });
            }
            return;
        }
        
        // 从配置中获取生成参数
        int maxNewTokens = ConfigManager.getMaxNewTokens(context);
        boolean noThinking = ConfigManager.getNoThinking(context);
        
        Log.d(TAG, "自回归生成参数: maxNewTokens=" + maxNewTokens + ", noThinking=" + noThinking);
        
        // 在后台线程中执行推理
        final String finalPrompt = prompt;
        executorService.execute(() -> {
            try {
                Log.i(TAG, "开始生成文本，最大token数: " + maxNewTokens);
                
                // 简化版实现，直接返回预设响应
                String response = "这是一个简化的ONNX模型推理实现。在实际应用中，这里会执行真正的模型推理过程，" +
                        "包括分词、创建输入张量、执行推理、采样生成等步骤。\n\n" +
                        "您的提示词是: " + finalPrompt + "\n\n" +
                        "在完整实现中，模型会根据提示词生成连续的文本输出。";
                
                // 模拟生成过程
                simulateTokenGeneration(response, callback);
                
            } catch (Exception e) {
                Log.e(TAG, "推理过程中发生异常: " + e.getMessage(), e);
                if (callback != null) {
                    final String errorMessage = e.getMessage();
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onError("推理失败: " + errorMessage);
                    });
                }
            }
        });
    }
    
    /**
     * 模拟token生成过程
     * @param text 要生成的文本
     * @param callback 回调接口
     */
    private void simulateTokenGeneration(String text, LocalLlmCallback callback) {
        // 将文本分成字符，模拟token生成
        char[] chars = text.toCharArray();
        StringBuilder fullResponse = new StringBuilder();
        
        // 创建一个新线程来模拟生成过程
        new Thread(() -> {
            try {
                for (char c : chars) {
                    // 添加到完整响应
                    fullResponse.append(c);
                    
                    // 回调单个token
                    if (callback != null) {
                        final String token = String.valueOf(c);
                        new Handler(Looper.getMainLooper()).post(() -> {
                            callback.onToken(token);
                        });
                    }
                    
                    // 模拟生成延迟
                    Thread.sleep(50);
                }
                
                // 回调完整响应
                if (callback != null) {
                    final String response = fullResponse.toString();
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onComplete(response);
                    });
                }
                
            } catch (InterruptedException e) {
                Log.e(TAG, "模拟生成过程被中断: " + e.getMessage());
            }
        }).start();
    }
}
