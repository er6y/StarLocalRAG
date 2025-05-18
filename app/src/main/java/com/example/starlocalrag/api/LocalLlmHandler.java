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
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.nio.LongBuffer;
import android.content.res.AssetManager;
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
public class LocalLlmHandler {
    private static final String TAG = "LocalLlmHandler";
    
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
        this.context = context;
        this.executorService = Executors.newSingleThreadExecutor();
        
        // 从配置中获取是否使用GPU
        this.useGpu = ConfigManager.getBoolean(context, ConfigManager.KEY_USE_GPU, false);
        
        Log.d(TAG, "LocalLlmHandler 初始化, 使用GPU: " + useGpu);
    }
    
    /**
     * 加载本地模型
     * @param modelName 模型名称（目录名）
     * @param callback 回调接口
     */
    public void loadModel(String modelName, final LocalLlmCallback callback) {
        // 防止重复加载
        if (modelLoading.get()) {
            Log.w(TAG, "模型正在加载中，请勿重复加载");
            if (callback != null) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    callback.onError("模型正在加载中，请勿重复加载");
                });
            }
            return;
        }
        
        // 设置加载状态
        modelLoading.set(true);
        
        // 在后台线程中加载模型
        executorService.execute(() -> {
            try {
                Log.d(TAG, "开始加载模型: " + modelName);
                
                // 如果当前有加载的模型，先卸载
                if (modelLoaded.get()) {
                    unloadModelInternal();
                }
                
                // 从ConfigManager获取模型基础路径
                String baseModelPath = ConfigManager.getModelPath(context);
                File modelDir = new File(baseModelPath, modelName);
                
                Log.d(TAG, "检查模型目录: " + modelDir.getAbsolutePath());
                
                // 检查目录是否存在
                if (!modelDir.exists() || !modelDir.isDirectory()) {
                    String errorMsg = "模型目录不存在或无法访问: " + modelDir.getAbsolutePath();
                    Log.e(TAG, errorMsg);
                    
                    // 检查基础目录
                    File baseDir = modelDir.getParentFile();
                    if (baseDir != null) {
                        Log.d(TAG, "基础模型目录: " + baseDir.getAbsolutePath() + 
                              " 存在: " + baseDir.exists() + 
                              " 可读: " + baseDir.canRead() + 
                              " 可写: " + baseDir.canWrite() + 
                              " 可执行: " + baseDir.canExecute());
                    }
                    
                    throw new IOException(errorMsg);
                }
                
                Log.d(TAG, "模型目录存在，检查权限...");
                Log.d(TAG, "可读: " + modelDir.canRead() + 
                      " 可写: " + modelDir.canWrite() + 
                      " 可执行: " + modelDir.canExecute());
                
                // 输出模型目录内容
                Log.d(TAG, "模型目录存在，内容:");
                File[] files = modelDir.listFiles();
                if (files != null) {
                    for (File file : files) {
                        Log.d(TAG, "  - " + file.getName() + (file.isDirectory() ? " [目录]" : " [文件, " + file.length() + " 字节]"));
                    }
                } else {
                    Log.e(TAG, "无法列出模型目录内容");
                }
                
                // 模型文件路径
                File modelFile = new File(modelDir, "model.onnx");
                File configFile = new File(modelDir, "config.json");
                File tokenizerFile = new File(modelDir, "tokenizer.json");
                
                Log.d(TAG, "检查模型文件:");
                Log.d(TAG, "  - " + modelFile.getAbsolutePath() + " 存在: " + modelFile.exists());
                Log.d(TAG, "  - " + configFile.getAbsolutePath() + " 存在: " + configFile.exists());
                Log.d(TAG, "  - " + tokenizerFile.getAbsolutePath() + " 存在: " + tokenizerFile.exists());
                
                if (!modelFile.exists()) {
                    String errorMsg = "模型文件不存在: " + modelFile.getAbsolutePath();
                    Log.e(TAG, errorMsg);
                    throw new IOException(errorMsg);
                }
                
                // 创建ONNX环境
                if (ortEnvironment == null) {
                    try {
                        Log.d(TAG, "正在创建ONNX环境...");
                        ortEnvironment = OrtEnvironment.getEnvironment();
                        Log.d(TAG, "创建ONNX环境成功");
                    } catch (Exception e) {
                        String errorMsg = "创建ONNX环境失败: " + e.getMessage();
                        Log.e(TAG, errorMsg, e);
                        throw new Exception(errorMsg, e);
                    }
                } else {
                    Log.d(TAG, "ONNX环境已存在，跳过创建");
                }
                
                // 加载配置文件
                loadModelConfig(configFile);
                Log.d(TAG, "加载模型配置成功");
                
                // 加载词汇表
                loadTokenizer(tokenizerFile);
                Log.d(TAG, "加载词汇表成功");
                
                // 创建会话选项
                SessionOptions sessionOptions = new SessionOptions();
                
                // 设置执行提供程序
                if (useGpu) {
                    // 尝试使用GPU
                    try {
                        sessionOptions.addCUDA();
                        Log.d(TAG, "启用CUDA GPU执行");
                    } catch (Exception e) {
                        Log.w(TAG, "启用GPU失败，回退到CPU: " + e.getMessage());
                    }
                } else {
                    Log.d(TAG, "使用CPU执行");
                }
                
                // 启用内存优化
                sessionOptions.setMemoryPatternOptimization(true);
                // SEQUENTIAL模式在一些版本中不可用，改为简单配置
                // sessionOptions.setExecutionMode(ExecutionMode.SEQUENTIAL);
                
                // 设置线程数
                sessionOptions.setIntraOpNumThreads(4);
                sessionOptions.setInterOpNumThreads(1);
                
                // 创建ONNX会话
                try {
                    String modelPath = modelFile.getAbsolutePath();
                    Log.d(TAG, "正在创建ONNX会话，模型路径: " + modelPath);
                    long startTime = System.currentTimeMillis();
                    ortSession = ortEnvironment.createSession(modelPath, sessionOptions);
                    long endTime = System.currentTimeMillis();
                    Log.d(TAG, String.format("创建ONNX会话成功，耗时: %d 毫秒", (endTime - startTime)));
                    
                    // 输出会话信息
                    if (ortSession != null) {
                        Log.d(TAG, "ONNX会话信息:");
                        Log.d(TAG, "  输入数量: " + ortSession.getInputNames().size());
                        for (String inputName : ortSession.getInputNames()) {
                            Log.d(TAG, "  - 输入: " + inputName);
                        }
                        Log.d(TAG, "  输出数量: " + ortSession.getOutputNames().size());
                        for (String outputName : ortSession.getOutputNames()) {
                            Log.d(TAG, "  - 输出: " + outputName);
                        }
                    }
                } catch (Exception e) {
                    String errorMsg = "创建ONNX会话失败: " + e.getMessage();
                    Log.e(TAG, errorMsg, e);
                    throw new Exception(errorMsg, e);
                }
                
                // 更新状态
                currentModelName = modelName;
                modelLoaded.set(true);
                
                Log.d(TAG, "模型加载成功: " + modelName);
                
                // 回调成功
                if (callback != null) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onComplete("模型加载成功: " + modelName);
                    });
                }
                
            } catch (Exception e) {
                Log.e(TAG, "加载模型失败: " + e.getMessage(), e);
                
                // 回调错误
                if (callback != null) {
                    final String errorMessage = "加载模型失败: " + e.getMessage();
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onError(errorMessage);
                    });
                }
                
                // 重置状态
                modelLoaded.set(false);
                currentModelName = null;
                
            } finally {
                // 重置加载中状态
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
        if (!configFile.exists()) {
            Log.w(TAG, "模型配置文件不存在，使用默认配置");
            // 使用默认配置（Qwen3-0.6B）
            modelConfig = new ModelConfig("qwen", 151936, 1024, 24, 16);
            return;
        }
        
        // 读取配置文件
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new java.io.FileReader(configFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
            
            // 解析JSON
            JSONObject config = new JSONObject(content.toString());
            
            // 获取模型类型
            String modelType = "qwen"; // 默认为Qwen
            if (config.has("model_type")) {
                modelType = config.getString("model_type");
            }
            
            // 获取模型参数
            int vocabSize = config.optInt("vocab_size", 151936);
            int hiddenSize = config.optInt("hidden_size", 1024);
            int numLayers = config.optInt("num_hidden_layers", 24);
            int numHeads = config.optInt("num_attention_heads", 16);
            
            // 创建模型配置
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
            
            // 获取最大序列长度
            if (config.has("max_position_embeddings")) {
                maxSeqLen = config.getInt("max_position_embeddings");
            }
            
            Log.d(TAG, "加载模型配置成功: " + modelType + ", 词汇表大小: " + vocabSize);
            
        } catch (Exception e) {
            Log.e(TAG, "解析模型配置文件失败: " + e.getMessage(), e);
            throw new Exception("解析模型配置文件失败: " + e.getMessage());
        }
    }
    
    /**
     * 加载词汇表
     * @param tokenizerFile 词汇表文件
     * @throws Exception 异常
     */
    private void loadTokenizer(File tokenizerFile) throws Exception {
        tokenizer = new HashMap<>();
        reverseTokenizer = new HashMap<>();
        
        if (!tokenizerFile.exists()) {
            Log.w(TAG, "词汇表文件不存在，使用简化的词汇表");
            // 创建一个简化的词汇表，只包含基本的ASCII字符
            for (int i = 0; i < 128; i++) {
                String token = String.valueOf((char)i);
                tokenizer.put(token, i);
                reverseTokenizer.put(i, token);
            }
            return;
        }
        
        BufferedReader reader = null;
        try {
            // 读取词汇表文件
            StringBuilder content = new StringBuilder();
            reader = new BufferedReader(new java.io.FileReader(tokenizerFile));
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
            
            // 解析JSON
            JSONObject tokenizerJson = new JSONObject(content.toString());
            JSONObject vocab = tokenizerJson.getJSONObject("model").getJSONObject("vocab");
            
            // 填充词汇表
            Iterator<String> keys = vocab.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                int value = vocab.getInt(key);
                tokenizer.put(key, value);
                reverseTokenizer.put(value, key);
            }
            
            Log.d(TAG, "加载词汇表成功，词汇表大小: " + tokenizer.size());
            
        } catch (Exception e) {
            Log.e(TAG, "加载词汇表失败: " + e.getMessage(), e);
            throw new Exception("加载词汇表失败: " + e.getMessage());
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭词汇表文件失败: " + e.getMessage());
                }
            }
        }
    }
    
    /**
     * 卸载当前模型
     */
    public void unloadModel() {
        executorService.execute(this::unloadModelInternal);
    }
    
    /**
     * 从 assets 复制模型文件到应用私有目录
     * @param modelName 模型名称
     * @return 是否复制成功
     */
    private boolean copyModelFilesFromAssets(String modelName) {
        try {
            AssetManager assetManager = context.getAssets();
            String[] files = assetManager.list("models/" + modelName);
            
            if (files == null || files.length == 0) {
                Log.e(TAG, "在 assets 中找不到模型文件: models/" + modelName);
                return false;
            }
            
            // 确保目标目录存在
            File modelsDir = new File(context.getFilesDir(), "models");
            File modelDir = new File(modelsDir, modelName);
            if (!modelDir.exists() && !modelDir.mkdirs()) {
                Log.e(TAG, "无法创建模型目录: " + modelDir.getAbsolutePath());
                return false;
            }
            
            // 复制文件
            for (String filename : files) {
                try (InputStream in = assetManager.open("models/" + modelName + "/" + filename);
                     FileOutputStream out = new FileOutputStream(new File(modelDir, filename))) {
                    
                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = in.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                    Log.d(TAG, "已复制文件: " + filename);
                }
            }
            
            return true;
            
        } catch (IOException e) {
            Log.e(TAG, "复制模型文件失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 内部卸载模型方法
     */
    private void unloadModelInternal() {
        if (modelLoaded.get()) {
            try {
                Log.d(TAG, "卸载模型: " + currentModelName);
                
                // 根据用户要求，除非用户更换模型或系统生命周期控制，否则不卸载模型
                // 这里我们只在必要时才卸载，以便重复调用
                if (ortSession != null) {
                    try {
                        ortSession.close();
                        Log.d(TAG, "ONNX会话关闭成功");
                    } catch (Exception e) {
                        Log.e(TAG, "关闭ONNX会话失败: " + e.getMessage(), e);
                    } finally {
                        ortSession = null;
                    }
                }
                
                // 重置状态
                modelLoaded.set(false);
                currentModelName = null;
                
                Log.d(TAG, "模型卸载成功");
                
            } catch (Exception e) {
                Log.e(TAG, "卸载模型失败: " + e.getMessage(), e);
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
            Log.e(TAG, "模型未加载，无法执行推理");
            if (callback != null) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    callback.onError("模型未加载，无法执行推理");
                });
            }
            return;
        }
        
        // 在后台线程中执行推理
        executorService.execute(() -> {
            // 推理前记录内存使用情况
            Log.i(TAG, "推理前内存状态:");
            logMemoryInfo();
            
            OnnxTensor inputTensor = null;
            OnnxTensor attentionMask = null;
            OnnxTensor positionIds = null;
            
            try {
                // 1. 对输入进行分词
                List<Integer> inputIds = tokenizePrompt(prompt);
                
                // 2. 添加特殊token（如BOS）
                inputIds.add(0, bosToken);
                
                // 3. 创建输入张量
                long[] inputShape = {1, inputIds.size()};
                inputTensor = createInputTensor(inputIds, inputShape);
                
                // 4. 创建attention mask（全1）
                long[] attentionMaskData = new long[inputIds.size()];
                Arrays.fill(attentionMaskData, 1);
                LongBuffer attentionMaskBuffer = LongBuffer.wrap(attentionMaskData);
                attentionMask = OnnxTensor.createTensor(ortEnvironment, attentionMaskBuffer, inputShape);
                
                // 5. 创建position ids（0到seq_len-1）
                long[] positionIdsData = new long[inputIds.size()];
                for (int i = 0; i < positionIdsData.length; i++) {
                    positionIdsData[i] = i;
                }
                LongBuffer positionIdsBuffer = LongBuffer.wrap(positionIdsData);
                positionIds = OnnxTensor.createTensor(ortEnvironment, positionIdsBuffer, inputShape);
                
                // 6. 打印模型输入信息
                Map<String, NodeInfo> inputInfo = ortSession.getInputInfo();
                Log.d(TAG, "模型期望输入数量: " + inputInfo.size());
                for (String inputName : inputInfo.keySet()) {
                    Log.d(TAG, "模型输入名称: " + inputName);
                }
                
                // 7. 准备输入Map
                Map<String, OnnxTensor> inputs = new HashMap<>();
                
                // 只添加模型期望的输入
                if (inputInfo.containsKey("input_ids")) {
                    inputs.put("input_ids", inputTensor);
                }
                if (inputInfo.containsKey("attention_mask")) {
                    inputs.put("attention_mask", attentionMask);
                }
                if (inputInfo.containsKey("position_ids")) {
                    inputs.put("position_ids", positionIds);
                }
                
                Log.d(TAG, "实际提供的输入数量: " + inputs.size());
                
                // 7. 执行推理
                Log.d(TAG, "开始执行推理，输入长度: " + inputIds.size());
                OrtSession.Result results = ortSession.run(inputs);
                
                // 8. 处理输出
                Object outputObj = results.get(0).getValue();
                String response;
                int predictedToken = -1;
                
                if (outputObj instanceof float[][]) {
                    // 二维输出 [batch_size, vocab_size]
                    float[][] output2d = (float[][]) outputObj;
                    predictedToken = argmax(output2d[output2d.length - 1]);
                    response = detokenize(predictedToken);
                } else if (outputObj instanceof float[][][]) {
                    // 三维输出 [batch_size, seq_len, vocab_size]
                    float[][][] output3d = (float[][][]) outputObj;
                    // 取最后一个时间步的输出
                    float[] lastTimestep = output3d[0][output3d[0].length - 1];
                    predictedToken = argmax(lastTimestep);
                    response = detokenize(predictedToken);
                } else {
                    throw new RuntimeException("不支持的输出类型: " + 
                        (outputObj != null ? outputObj.getClass().getSimpleName() : "null"));
                }
                
                // 9. 回调结果
                if (callback != null) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onToken(response);
                        callback.onComplete(response);
                    });
                }
                
                Log.d(TAG, "推理完成，输出token: " + predictedToken + ", 文本: " + response);
                
            } catch (Exception e) {
                Log.e(TAG, "推理过程中发生异常: " + e.getMessage(), e);
                if (callback != null) {
                    final String errorMessage = e.getMessage();
                    new Handler(Looper.getMainLooper()).post(() -> {
                        callback.onError("推理失败: " + errorMessage);
                    });
                }
            } finally {
                // 释放资源
                safeClose(inputTensor, "inputTensor");
                safeClose(attentionMask, "attentionMask");
                safeClose(positionIds, "positionIds");
                
                // 推理后记录内存使用情况
                Log.i(TAG, "推理后内存状态:");
                logMemoryInfo();
            }
        });
    }
    
    /**
     * 获取已安装的模型列表
     * @return 模型名称数组
     */
    public String[] listAvailableModels() {
        File modelsDir = new File(context.getFilesDir(), "models");
        if (!modelsDir.exists() || !modelsDir.isDirectory()) {
            return new String[]{"无可用模型"};
        }
        
        File[] modelDirs = modelsDir.listFiles(File::isDirectory);
        if (modelDirs == null || modelDirs.length == 0) {
            return new String[]{"无可用模型"};
        }
        
        String[] modelNames = new String[modelDirs.length];
        for (int i = 0; i < modelDirs.length; i++) {
            modelNames[i] = modelDirs[i].getName();
            Log.d(TAG, "找到模型: " + modelNames[i]);
        }
        
        return modelNames;
    }
    
    /**
     * 设置是否使用GPU
     * @param useGpu 是否使用GPU
     */
    public void setUseGpu(boolean useGpu) {
        if (this.useGpu != useGpu) {
            this.useGpu = useGpu;
            Log.d(TAG, "更新GPU设置: " + useGpu);
            
            // 如果模型已加载，需要重新加载以应用新设置
            if (modelLoaded.get()) {
                final String modelToReload = currentModelName;
                unloadModel();
                loadModel(modelToReload, null);
            }
        }
    }
    
    /**
     * 检查模型是否已加载
     * @return 是否已加载
     */
    public boolean isModelLoaded() {
        return modelLoaded.get();
    }
    
    /**
     * 获取当前加载的模型名称
     * @return 模型名称
     */
    public String getCurrentModelName() {
        return currentModelName;
    }
    
    /**
     * 关闭处理程序，释放资源
     */
    public void shutdown() {
        try {
            // 卸载模型
            unloadModel();
            
            // 关闭ONNX环境
            if (ortEnvironment != null) {
                try {
                    ortEnvironment.close();
                    Log.d(TAG, "ONNX环境关闭成功");
                } catch (Exception e) {
                    Log.e(TAG, "关闭ONNX环境失败: " + e.getMessage(), e);
                } finally {
                    ortEnvironment = null;
                }
            }
            
            // 关闭线程池
            if (executorService != null && !executorService.isShutdown()) {
                executorService.shutdown();
                Log.d(TAG, "线程池关闭成功");
            }
            
            // 清空词汇表
            if (tokenizer != null) {
                tokenizer.clear();
            }
            if (reverseTokenizer != null) {
                reverseTokenizer.clear();
            }
            
            // 重置单例
            instance = null;
            
            Log.d(TAG, "LocalLlmHandler关闭成功");
            
        } catch (Exception e) {
            Log.e(TAG, "LocalLlmHandler关闭失败: " + e.getMessage(), e);
        }
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
            long maxMemory = runtime.maxMemory();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            long usedMemory = totalMemory - freeMemory;
            
            Log.i(TAG, String.format(
                "内存状态: \n" +
                "- 系统可用内存: %.2f MB\n" +
                "- 系统内存低: %b\n" +
                "- JVM最大内存: %.2f MB\n" +
                "- JVM已分配内存: %.2f MB\n" +
                "- JVM已使用内存: %.2f MB\n" +
                "- JVM可用内存: %.2f MB",
                memoryInfo.availMem / (1024.0 * 1024.0),
                memoryInfo.lowMemory,
                maxMemory / (1024.0 * 1024.0),
                totalMemory / (1024.0 * 1024.0),
                usedMemory / (1024.0 * 1024.0),
                freeMemory / (1024.0 * 1024.0)
            ));
        } catch (Exception e) {
            Log.e(TAG, "获取内存信息失败: " + e.getMessage());
        }
    }
    
    /**
     * 将提示词分词为token ID列表
     * @param prompt 提示词
     * @return token ID列表
     */
    private List<Integer> tokenizePrompt(String prompt) {
        // 简化的分词实现，实际应用中需要使用专业的分词器
        List<Integer> ids = new ArrayList<>();
        
        // 如果词汇表为空，使用字符级别的分词
        if (tokenizer == null || tokenizer.isEmpty()) {
            for (char c : prompt.toCharArray()) {
                ids.add((int) c % 128); // 简化处理，只使用ASCII范围
            }
            return ids;
        }
        
        // 如果有词汇表，尝试使用词汇表分词
        // 这里的实现非常简化，实际应用中需要更复杂的分词算法
        StringBuilder current = new StringBuilder();
        for (char c : prompt.toCharArray()) {
            current.append(c);
            String token = current.toString();
            
            if (tokenizer.containsKey(token)) {
                ids.add(tokenizer.get(token));
                current = new StringBuilder();
            } else if (current.length() > 10) { // 防止无限积累
                // 如果当前字符串过长但找不到匹配，回退到字符级别
                for (char backupChar : current.toString().toCharArray()) {
                    ids.add((int) backupChar % 128);
                }
                current = new StringBuilder();
            }
        }
        
        // 处理最后可能剩下的字符
        if (current.length() > 0) {
            for (char backupChar : current.toString().toCharArray()) {
                ids.add((int) backupChar % 128);
            }
        }
        
        return ids;
    }
    
    /**
     * 将token ID转换为文本
     * @param tokenId token ID
     * @return 文本
     */
    private String detokenize(int tokenId) {
        // 如果在反向词汇表中找到，直接返回
        if (reverseTokenizer.containsKey(tokenId)) {
            return reverseTokenizer.get(tokenId);
        }
        
        // 如果找不到，尝试使用ASCII字符
        if (tokenId >= 0 && tokenId < 128) {
            return String.valueOf((char) tokenId);
        }
        
        // 如果都不满足，返回空字符串
        return "";
    }
    
    /**
     * 创建ONNX输入张量
     * @param inputIds 输入token IDs
     * @param shape 形状
     * @return ONNX张量
     * @throws OrtException 异常
     */
    private OnnxTensor createInputTensor(List<Integer> inputIds, long[] shape) throws OrtException {
        // 创建LongBuffer并填充数据
        LongBuffer buffer = LongBuffer.allocate(inputIds.size());
        for (int id : inputIds) {
            buffer.put(id);
        }
        buffer.rewind();
        
        // 创建ONNX张量
        return OnnxTensor.createTensor(ortEnvironment, buffer, shape);
    }
    
    /**
     * 找到数组中最大值的索引
     * @param array 数组
     * @return 最大值的索引
     */
    private int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
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
}
