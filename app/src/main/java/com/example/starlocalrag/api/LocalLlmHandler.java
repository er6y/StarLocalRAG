package com.example.starlocalrag.api;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.example.starlocalrag.ConfigManager;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Method;
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
                
                // 设置线程数
                int threads = ConfigManager.getThreads(context);
                sessionOptions.setIntraOpNumThreads(threads);
                
                // 启用内存优化
                sessionOptions.setMemoryPatternOptimization(true);
                sessionOptions.setExecutionMode(SessionOptions.ExecutionMode.SEQUENTIAL);
                
                // 如果启用GPU，按优先级尝试不同的GPU加速方式
                if (useGpu) {
                    boolean gpuEnabled = false;
                    
                    // 使用反射机制尝试调用可能存在的GPU加速方法
                    String[] gpuMethods = {"addNNAPI", "addOpenCL", "addCUDA"};
                    String[] gpuNames = {"NNAPI", "OpenCL", "CUDA"};
                    
                    for (int i = 0; i < gpuMethods.length && !gpuEnabled; i++) {
                        try {
                            // 尝试通过反射调用方法
                            Method method = SessionOptions.class.getMethod(gpuMethods[i]);
                            method.invoke(sessionOptions);
                            Log.i(TAG, "成功启用" + gpuNames[i] + "加速");
                            gpuEnabled = true;
                        } catch (NoSuchMethodException e) {
                            // 方法不存在，跳过
                            Log.d(TAG, gpuNames[i] + "加速方法不可用");
                        } catch (Exception e) {
                            // 其他错误，如调用失败
                            Log.d(TAG, "启用" + gpuNames[i] + "加速失败: " + e.getMessage());
                        }
                    }
                    
                    if (!gpuEnabled) {
                        Log.w(TAG, "所有GPU加速方式均失败，将使用CPU模式");
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
                
                // 8. 检测模型类型并初始化相应处理器
                // 默认为ONNX模型
                modelType = "onnx";
                
                // 如果模型配置中指定了模型类型，则使用配置中的类型
                if (modelConfig != null && modelConfig.modelType != null) {
                    modelType = modelConfig.modelType.toLowerCase();
                }
                
                // 根据模型类型初始化处理器
                if ("onnx".equals(modelType)) {
                    Log.i(TAG, "初始化ONNX处理器");
                    localLlmOnnxHandler = new LocalLLMOnnxHandler(context, ortEnvironment, ortSession, modelConfig);
                } else {
                    Log.w(TAG, "未知模型类型: " + modelType + "，默认使用ONNX处理器");
                    modelType = "onnx";
                    localLlmOnnxHandler = new LocalLLMOnnxHandler(context, ortEnvironment, ortSession, modelConfig);
                }
                
                // 9. 标记为已加载
                modelLoaded.set(true);
                Log.i(TAG, "模型加载完成: " + modelName + ", 类型: " + modelType);
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
        
        Log.d(TAG, String.format("模型配置: 类型=%s, 词汇表大小=%d, 隐藏层大小=%d, 层数=%d, 注意力头数=%d",
            modelType, vocabSize, hiddenSize, numLayers, numHeads));
        Log.d(TAG, String.format("特殊token: BOS=%d, EOS=%d, PAD=%d", bosToken, eosToken, padToken));
    }
    
    /**
     * 检查词汇表文件
     * @param tokenizerFile 词汇表文件
     * @throws Exception 异常
     */
    private void checkTokenizerFile(File tokenizerFile) throws Exception {
        Log.d(TAG, "检查词汇表文件: " + tokenizerFile.getPath());
        
        // 直接检查文件是否存在，不需要读取内容
        // Rust tokenizer 会处理词汇表和特殊 token
        if (!tokenizerFile.exists()) {
            throw new IOException("词汇表文件不存在: " + tokenizerFile.getPath());
        }
        
        Log.d(TAG, "词汇表文件检查完成，将由 Rust tokenizer 处理");
    }
    

    
    public void inference(String prompt, LocalLlmCallback callback) {
        // 实现推理逻辑
        new Thread(() -> {
            try {
                // 配置管理
                int maxTokenLength = 512; // 默认最大token长度
                boolean thinkingMode = false; // 默认关闭思考模式
                float temperature = 0.7f; // 温度采样参数
                int topK = 5; // top-k 采样参数
                
                // 在这里调用实际的推理逻辑
                String result = localLlmOnnxHandler.inference(prompt, maxTokenLength, thinkingMode, temperature, topK);
                callback.onToken(result);
                callback.onComplete(result);
            } catch (Exception e) {
                callback.onError(e.getMessage());
            }
        }).start();
    }
    
    public void unloadModel() {
        // 卸载模型逻辑
        Log.d(TAG, "卸载模型");
    }
    
    private void logMemoryInfo() {
        // 记录内存信息逻辑
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        Log.d(TAG, "内存信息 - 最大: " + maxMemory / 1024 / 1024 + "MB, 总计: " + totalMemory / 1024 / 1024 + "MB, 空闲: " + freeMemory / 1024 / 1024 + "MB");
    }
}
