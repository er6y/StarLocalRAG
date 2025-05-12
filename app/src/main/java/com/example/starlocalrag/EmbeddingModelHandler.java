package com.example.starlocalrag;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.TimeUnit;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * 处理词嵌入模型的工具类，支持TorchScript和ONNX格式
 */
public class EmbeddingModelHandler {
    private static final String TAG = "StarLocalRAG_EmbeddingModel";
    private static final int MODEL_LOAD_TIMEOUT_SECONDS = 60;
    
    // ONNX会话状态常量
    private static final int SESSION_STATE_NONE = 0;      // 未初始化
    private static final int SESSION_STATE_LOADING = 1;   // 正在加载
    private static final int SESSION_STATE_READY = 2;     // 已就绪
    private static final int SESSION_STATE_ERROR = 3;     // 错误状态
    
    // 会话重试相关常量
    private static final int MAX_SESSION_RETRY = 3;       // 最大重试次数
    private static final long SESSION_RETRY_DELAY_MS = 500; // 重试间隔
    
    // 模型类型枚举
    public enum ModelType {
        TORCH_SCRIPT,
        ONNX,
        UNKNOWN
    }
    
    private Module torchModel;
    private OrtSession onnxSession;
    private OrtEnvironment ortEnvironment;
    private ModelType modelType;
    private String modelPath;
    private BertTokenizer tokenizer; // 添加tokenizer字段
    private JSONObject configJson; // 添加模型配置字段
    private String modelName; // 添加模型名称字段
    
    // 是否使用一致性处理
    private boolean useConsistentProcessing = false;
    
    // 是否启用调试模式
    private boolean debugMode = false;
    
    // 是否使用 GPU 加速
    private boolean useGpu = false;
    
    // 添加Context引用，用于获取TokenizerManager
    private Context context;
    
    // 嵌入向量维度
    private int embeddingSize = 0;
    
    // 最大序列长度
    private int maxSequenceLength = 512;
    
    // 是否使用平均池化
    private boolean useMeanPooling = true;
    
    // 会话状态变量
    private final Object sessionLock = new Object();      // 会话锁
    private int sessionState = SESSION_STATE_NONE;        // 当前会话状态
    private int sessionRetryCount = 0;                    // 当前重试次数
    private long lastSessionCheckTime = 0;                // 上次会话检查时间
    
    /**
     * 默认构造函数
     */
    public EmbeddingModelHandler() {
        // 默认构造函数
    }
    
    /**
     * 带参数的构造函数
     * @param modelPath 模型文件路径
     * @throws Exception 如果加载失败
     */
    public EmbeddingModelHandler(String modelPath) throws Exception {
        this(modelPath, false);
    }
    
    /**
     * 带参数的构造函数，支持 GPU 加速
     * @param modelPath 模型文件路径
     * @param useGpu 是否使用 GPU 加速
     * @throws Exception 如果加载失败
     */
    public EmbeddingModelHandler(String modelPath, boolean useGpu) throws Exception {
        this.modelPath = modelPath;
        this.modelType = determineModelType(modelPath);
        this.useGpu = useGpu;
        
        try {
            if (this.modelType == ModelType.TORCH_SCRIPT) {
                // 加载TorchScript模型
                this.torchModel = Module.load(modelPath);
                Log.d(TAG, "TorchScript模型加载成功");
            } else if (this.modelType == ModelType.ONNX) {
                // 加载ONNX模型
                loadOnnxModel(modelPath);
            } else {
                throw new RuntimeException("不支持的模型类型");
            }
        } catch (Exception e) {
            // 如果使用GPU加速失败，尝试降级到CPU模式
            if (useGpu && (e.getMessage().contains("OpenGLRenderer") || 
                          e.getMessage().contains("HwEditorHelperImpl") || 
                          e.getMessage().contains("GPU") || 
                          e.getMessage().contains("gpu"))) {
                Log.w(TAG, "GPU加速失败，降级到CPU模式: " + e.getMessage(), e);
                this.useGpu = false;
                
                // 重新尝试加载模型，但不使用GPU
                if (this.modelType == ModelType.TORCH_SCRIPT) {
                    this.torchModel = Module.load(modelPath);
                    Log.d(TAG, "TorchScript模型使用CPU模式加载成功");
                } else if (this.modelType == ModelType.ONNX) {
                    loadOnnxModel(modelPath);
                }
            } else {
                // 其他错误，直接抛出
                throw e;
            }
        }
    }
    
    /**
     * 带参数的构造函数，支持 GPU 加速和上下文
     * @param context 应用上下文
     * @param modelPath 模型文件路径
     * @param useGpu 是否使用 GPU 加速
     * @throws Exception 如果加载失败
     */
    public EmbeddingModelHandler(Context context, String modelPath, boolean useGpu) throws Exception {
        this(modelPath, useGpu);
        this.context = context;
    }
    
    /**
     * 根据模型文件路径创建嵌入模型处理器
     * @param modelPath 模型文件的完整路径
     * @return 创建的模型处理器，如果创建失败则返回null
     */
    public static EmbeddingModelHandler create(String modelPath) {
        Log.d(TAG, "开始创建嵌入模型处理器，模型路径: " + modelPath);
        
        if (modelPath == null || modelPath.isEmpty()) {
            Log.e(TAG, "模型路径为空");
            return null;
        }
        
        try {
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                Log.e(TAG, "模型文件不存在: " + modelPath);
                return null;
            }
            
            Log.d(TAG, "模型文件存在，大小: " + modelFile.length() + " 字节");
            
            // 检查是否是目录
            if (modelFile.isDirectory()) {
                Log.d(TAG, "指定的路径是目录，正在查找模型文件...");
                
                // 查找模型文件，支持递归搜索
                File modelFileFound = findModelFileInDirectory(modelFile);
                if (modelFileFound != null) {
                    Log.d(TAG, "找到模型文件: " + modelFileFound.getAbsolutePath() + "，大小: " + modelFileFound.length() + " 字节");
                    
                    // 检查是否存在必要的配置文件
                    File modelDir = modelFileFound.getParentFile();
                    checkRequiredConfigFiles(modelDir);
                    
                    return loadModelWithTimeout(modelFileFound.getAbsolutePath());
                } else {
                    Log.e(TAG, "在目录及其子目录中没有找到模型文件: " + modelPath);
                    return null;
                }
            }
            
            // 如果是文件，直接加载
            return loadModelWithTimeout(modelPath);
        } catch (Exception e) {
            Log.e(TAG, "创建嵌入模型处理器失败: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * 检查模型目录中是否存在必要的配置文件
     * @param modelDir 模型目录
     */
    private static void checkRequiredConfigFiles(File modelDir) {
        if (modelDir == null || !modelDir.exists() || !modelDir.isDirectory()) {
            Log.w(TAG, "模型目录无效，无法检查配置文件");
            return;
        }
        
        // 检查config.json
        File configFile = new File(modelDir, "config.json");
        if (configFile.exists()) {
            Log.d(TAG, "找到config.json: " + configFile.getAbsolutePath() + ", 大小: " + configFile.length() + " 字节");
        } else {
            Log.w(TAG, "未找到config.json文件");
        }
        
        // 检查tokenizer.json
        File tokenizerFile = new File(modelDir, "tokenizer.json");
        if (tokenizerFile.exists()) {
            Log.d(TAG, "找到tokenizer.json: " + tokenizerFile.getAbsolutePath() + ", 大小: " + tokenizerFile.length() + " 字节");
        } else {
            Log.w(TAG, "未找到tokenizer.json文件");
        }
        
        // 检查special_tokens_map.json
        File specialTokensMapFile = new File(modelDir, "special_tokens_map.json");
        if (specialTokensMapFile.exists()) {
            Log.d(TAG, "找到special_tokens_map.json: " + specialTokensMapFile.getAbsolutePath() + ", 大小: " + specialTokensMapFile.length() + " 字节");
        } else {
            Log.w(TAG, "未找到special_tokens_map.json文件");
        }
        
        // 检查tokenizer_config.json
        File tokenizerConfigFile = new File(modelDir, "tokenizer_config.json");
        if (tokenizerConfigFile.exists()) {
            Log.d(TAG, "找到tokenizer_config.json: " + tokenizerConfigFile.getAbsolutePath() + ", 大小: " + tokenizerConfigFile.length() + " 字节");
        } else {
            Log.w(TAG, "未找到tokenizer_config.json文件");
        }
    }
    
    /**
     * 在目录及其子目录中查找模型文件
     * @param directory 要搜索的目录
     * @return 找到的模型文件，如果没找到则返回null
     */
    private static File findModelFileInDirectory(File directory) {
        if (!directory.isDirectory()) {
            return null;
        }
        
        // 只支持 .pt 和 .onnx 词嵌入模型
        String[] supportedExtensions = {".pt", ".onnx"};
        
        // 打印目录内容，用于调试
        Log.d(TAG, "查找目录内容: " + directory.getAbsolutePath());
        File[] allFiles = directory.listFiles();
        if (allFiles != null) {
            // 检查必要的配置文件是否存在
            boolean hasConfigJson = false;
            boolean hasTokenizerJson = false;
            boolean hasSpecialTokensMap = false;
            boolean hasTokenizerConfig = false;
            
            for (File file : allFiles) {
                Log.d(TAG, "  - " + file.getName() + (file.isDirectory() ? " [目录]" : " [文件, " + file.length() + " 字节]"));
                
                // 检查必要的配置文件
                if (file.getName().equals("config.json")) {
                    hasConfigJson = true;
                    Log.d(TAG, "找到config.json文件");
                } else if (file.getName().equals("tokenizer.json")) {
                    hasTokenizerJson = true;
                    Log.d(TAG, "找到tokenizer.json文件");
                } else if (file.getName().equals("special_tokens_map.json")) {
                    hasSpecialTokensMap = true;
                    Log.d(TAG, "找到special_tokens_map.json文件");
                } else if (file.getName().equals("tokenizer_config.json")) {
                    hasTokenizerConfig = true;
                    Log.d(TAG, "找到tokenizer_config.json文件");
                }
            }
            
            // 记录配置文件状态
            Log.d(TAG, "配置文件检查结果: config.json=" + hasConfigJson + 
                      ", tokenizer.json=" + hasTokenizerJson + 
                      ", special_tokens_map.json=" + hasSpecialTokensMap + 
                      ", tokenizer_config.json=" + hasTokenizerConfig);
            
            // 首先查找model.onnx文件
            for (File file : allFiles) {
                if (file.isFile() && file.getName().equals("model.onnx")) {
                    Log.d(TAG, "找到model.onnx文件: " + file.getAbsolutePath());
                    return file;
                }
            }
            
            // 然后查找其他.onnx或.pt文件
            for (File file : allFiles) {
                if (file.isFile()) {
                    String fileName = file.getName().toLowerCase();
                    for (String ext : supportedExtensions) {
                        if (fileName.endsWith(ext)) {
                            Log.d(TAG, "找到模型文件: " + file.getAbsolutePath());
                            return file;
                        }
                    }
                }
            }
            
            // 递归搜索子目录
            for (File file : allFiles) {
                if (file.isDirectory()) {
                    File modelFile = findModelFileInDirectory(file);
                    if (modelFile != null) {
                        return modelFile;
                    }
                }
            }
        } else {
            Log.d(TAG, "  无法列出目录内容");
        }
        
        return null;
    }
    
    /**
     * 使用超时机制加载模型
     * @param modelPath 模型文件路径
     * @return 加载的模型处理器，如果加载失败则返回null
     */
    private static EmbeddingModelHandler loadModelWithTimeout(String modelPath) {
        Log.d(TAG, "使用超时机制加载模型: " + modelPath);
        
        // 创建线程池
        ExecutorService executor = Executors.newSingleThreadExecutor();
        
        try {
            // 提交加载任务
            Future<EmbeddingModelHandler> future = executor.submit(() -> {
                try {
                    EmbeddingModelHandler handler = new EmbeddingModelHandler(modelPath);
                    return handler;
                } catch (Exception e) {
                    Log.e(TAG, "模型加载失败: " + e.getMessage(), e);
                    throw new RuntimeException("模型加载失败: " + e.getMessage(), e);
                }
            });
            
            // 等待加载完成，设置超时
            return future.get(MODEL_LOAD_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            Log.e(TAG, "模型加载超时: " + e.getMessage(), e);
        } catch (Exception e) {
            Log.e(TAG, "模型加载失败: " + e.getMessage(), e);
        } finally {
            // 关闭线程池
            executor.shutdownNow();
        }
        
        return null;
    }
    
    /**
     * 加载ONNX模型
     * @param modelPath 模型路径
     * @return 是否加载成功
     */
    private boolean loadOnnxModel(String modelPath) {
        try {
            Log.d(TAG, "开始加载ONNX模型: " + modelPath);
            
            // 检查模型路径是否为目录
            File modelFile = new File(modelPath);
            if (modelFile.isDirectory()) {
                Log.d(TAG, "指定的路径是一个目录，尝试在目录中查找模型文件");
                File foundModelFile = findModelFileInDirectory(modelFile);
                if (foundModelFile != null) {
                    modelPath = foundModelFile.getAbsolutePath();
                    Log.d(TAG, "在目录中找到模型文件: " + modelPath);
                } else {
                    Log.e(TAG, "在目录中未找到有效的模型文件: " + modelPath);
                    return false;
                }
            } else if (!modelFile.exists()) {
                Log.e(TAG, "模型文件不存在: " + modelPath);
                return false;
            }
            
            // 创建ONNX运行时环境
            ortEnvironment = OrtEnvironment.getEnvironment();
            Log.d(TAG, "成功创建ONNX运行时环境");
            
            // 配置会话选项
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            
            // 动态设置线程数为系统CPU核心数的一半
            int availableProcessors = Runtime.getRuntime().availableProcessors();
            int numThreads = Math.max(1, availableProcessors / 2); // 至少使用1个线程
            
            sessionOptions.setIntraOpNumThreads(numThreads);
            sessionOptions.setInterOpNumThreads(numThreads);
            Log.d(TAG, "动态设置线程数 - 可用CPU核心: " + availableProcessors + 
                  ", 使用内部线程: " + numThreads + ", 外部线程: " + numThreads);
            
            // 设置优化级别
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            Log.d(TAG, "设置优化级别: ALL_OPT");
            
            // 加载模型
            Log.d(TAG, "开始加载ONNX模型: " + modelPath);
            try {
                onnxSession = ortEnvironment.createSession(modelPath, sessionOptions);
                Log.d(TAG, "ONNX模型加载成功: " + modelPath);
            } catch (OrtException e) {
                Log.e(TAG, "加载ONNX模型失败: " + e.getMessage(), e);
                return false;
            }
            
            // 加载tokenizer和配置
            loadTokenizerAndConfig();
            
            // 设置会话状态为就绪
            synchronized (sessionLock) {
                sessionState = SESSION_STATE_READY;
                sessionRetryCount = 0;
            }
            
            return true;
        } catch (Exception e) {
            Log.e(TAG, "加载ONNX模型失败: " + e.getMessage(), e);
            
            // 设置会话状态为错误
            synchronized (sessionLock) {
                sessionState = SESSION_STATE_ERROR;
            }
            
            return false;
        }
    }
    
    /**
     * 确定模型类型
     * @param modelPath 模型路径
     * @return 模型类型
     */
    private ModelType determineModelType(String modelPath) {
        File modelFile = new File(modelPath);
        
        if (!modelFile.exists()) {
            Log.e(TAG, "模型文件不存在: " + modelPath);
            return ModelType.UNKNOWN;
        }
        
        // 检查文件扩展名
        String fileName = modelFile.getName().toLowerCase();
        if (fileName.endsWith(".pt") || fileName.endsWith(".pth") || fileName.endsWith(".ptl")) {
            Log.d(TAG, "检测到TorchScript模型: " + modelPath);
            return ModelType.TORCH_SCRIPT;
        } else if (fileName.endsWith(".onnx")) {
            Log.d(TAG, "检测到ONNX模型: " + modelPath);
            return ModelType.ONNX;
        }
        
        // 检查文件是否是目录
        if (modelFile.isDirectory()) {
            // 检查目录中是否包含ONNX模型文件
            File[] files = modelFile.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.getName().toLowerCase().endsWith(".onnx")) {
                        String onnxPath = file.getAbsolutePath();
                        Log.d(TAG, "在目录中找到ONNX模型: " + onnxPath);
                        this.modelPath = onnxPath; // 更新模型路径
                        return ModelType.ONNX;
                    }
                }
                
                // 检查是否包含TorchScript模型文件
                for (File file : files) {
                    if (file.getName().toLowerCase().endsWith(".pt") || 
                        file.getName().toLowerCase().endsWith(".pth") || 
                        file.getName().toLowerCase().endsWith(".ptl")) {
                        String torchPath = file.getAbsolutePath();
                        Log.d(TAG, "在目录中找到TorchScript模型: " + torchPath);
                        this.modelPath = torchPath; // 更新模型路径
                        return ModelType.TORCH_SCRIPT;
                    }
                }
            }
        }
        
        // 尝试读取文件头部字节来判断
        try {
            byte[] header = new byte[4];
            try (FileInputStream fis = new FileInputStream(modelFile)) {
                if (fis.read(header) == 4) {
                    // ONNX文件通常以"ONNX"字符串开头
                    if (header[0] == 0x08 && header[1] == 0x00) {
                        Log.d(TAG, "通过文件头检测到ONNX模型");
                        return ModelType.ONNX;
                    }
                    
                    // PyTorch文件通常以特定的魔数开头
                    if (header[0] == 0x80 && header[1] == 0x02) {
                        Log.d(TAG, "通过文件头检测到TorchScript模型");
                        return ModelType.TORCH_SCRIPT;
                    }
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "读取文件头失败: " + e.getMessage());
        }
        
        Log.d(TAG, "无法确定模型类型，默认尝试作为ONNX模型");
        return ModelType.ONNX; // 默认尝试作为ONNX模型
    }
    
    /**
     * 加载tokenizer和配置文件
     */
    private void loadTokenizerAndConfig() {
        try {
            File modelDir = new File(modelPath).getParentFile();
            if (modelDir == null || !modelDir.exists()) {
                Log.w(TAG, "模型目录不存在");
                return;
            }
            
            // 首先尝试从TokenizerManager获取tokenizer
            if (context != null) {
                TokenizerManager tokenizerManager = TokenizerManager.getInstance(context);
                
                // 如果TokenizerManager已初始化，直接使用其tokenizer
                if (tokenizerManager.isInitialized()) {
                    this.tokenizer = tokenizerManager.getTokenizer();
                    Log.d(TAG, "从TokenizerManager获取已初始化的tokenizer");
                } 
                // 否则尝试初始化TokenizerManager
                else {
                    boolean success = tokenizerManager.initialize(modelDir);
                    if (success) {
                        this.tokenizer = tokenizerManager.getTokenizer();
                        Log.d(TAG, "通过TokenizerManager初始化tokenizer成功");
                    } else {
                        Log.w(TAG, "TokenizerManager初始化失败，将使用本地tokenizer");
                        loadLocalTokenizer(modelDir);
                    }
                }
            } else {
                // 如果没有Context，使用本地tokenizer
                Log.d(TAG, "Context为null，使用本地tokenizer");
                loadLocalTokenizer(modelDir);
            }
            
            // 加载模型配置
            try {
                File configFile = new File(modelDir, "config.json");
                if (configFile.exists()) {
                    String configContent = readFileContent(configFile);
                    configJson = new JSONObject(configContent);
                    Log.d(TAG, "模型配置加载成功");
                    
                    // 从配置中提取模型名称
                    if (configJson.has("model_name")) {
                        modelName = configJson.getString("model_name");
                        Log.d(TAG, "从配置中提取模型名称: " + modelName);
                    } else if (configJson.has("model_type")) {
                        modelName = configJson.getString("model_type");
                        Log.d(TAG, "从配置中提取模型类型作为名称: " + modelName);
                    } else if (configJson.has("architectures") && configJson.getJSONArray("architectures").length() > 0) {
                        modelName = configJson.getJSONArray("architectures").getString(0);
                        Log.d(TAG, "从配置中提取架构作为名称: " + modelName);
                    }
                } else {
                    Log.d(TAG, "未找到config.json文件");
                }
            } catch (Exception e) {
                Log.e(TAG, "加载模型配置失败: " + e.getMessage(), e);
            }
            
            // 如果仍然没有模型名称，尝试从路径中提取
            if (modelName == null || modelName.isEmpty()) {
                modelName = extractModelNameFromPath(modelPath);
                Log.d(TAG, "从路径中提取模型名称: " + modelName);
            }
        } catch (Exception e) {
            Log.e(TAG, "加载tokenizer和配置失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 加载本地tokenizer（当TokenizerManager不可用时）
     * @param modelDir 模型目录
     */
    private void loadLocalTokenizer(File modelDir) {
        try {
            // 加载tokenizer
            tokenizer = new BertTokenizer();
            boolean tokenizerLoaded = tokenizer.loadFromDirectory(modelDir);
            if (tokenizerLoaded) {
                Log.d(TAG, "本地Tokenizer加载成功，词汇表大小: " + tokenizer.getVocabSize());
            } else {
                Log.w(TAG, "本地Tokenizer加载失败，将使用简化的分词方法");
            }
        } catch (Exception e) {
            Log.e(TAG, "加载本地tokenizer失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 使用ONNX模型生成嵌入向量
     * @param text 输入文本
     * @return 嵌入向量
     * @throws Exception 如果生成失败
     */
    private synchronized float[] generateEmbeddingWithOnnx(String text) {
        if (onnxSession == null) {
            Log.e(TAG, "ONNX会话为空，无法生成嵌入");
            throw new RuntimeException("ONNX会话为空，无法生成嵌入");
        }

        // 记录模型推理开始时间
        long inferenceStartTime = System.currentTimeMillis();
        
        try {
            // 检查文本输入是否有效
            if (text == null || text.isEmpty()) {
                Log.e(TAG, "输入文本为null或空");
                throw new RuntimeException("输入文本为null或空");
            }
            
            // 使用tokenizer处理文本
            long[][] tokenResults = tokenizeText(text);
            if (tokenResults == null || tokenResults.length == 0) {
                Log.e(TAG, "分词结果为空");
                throw new RuntimeException("分词结果为空");
            }
            
            // 提取第一个批次的结果
            long[] inputIds = tokenResults[0];
            
            // 创建注意力掩码数组，全部填充为1
            long[] attentionMask = new long[inputIds.length];
            Arrays.fill(attentionMask, 1L);
            
            // 记录输入张量形状和示例
            Log.d(TAG, "输入张量形状: [1, " + inputIds.length + "]");
            Log.d(TAG, "输入ID示例: " + Arrays.toString(Arrays.copyOfRange(inputIds, 0, Math.min(10, inputIds.length))));
            Log.d(TAG, "输入数据类型: INT64 (与PC端保持一致)");
            
            // 将数组转换为LongBuffer (对应INT64类型)
            LongBuffer inputIdsBuffer = LongBuffer.allocate(inputIds.length);
            inputIdsBuffer.put(inputIds);
            inputIdsBuffer.rewind();
            
            LongBuffer attentionMaskBuffer = LongBuffer.allocate(attentionMask.length);
            attentionMaskBuffer.put(attentionMask);
            attentionMaskBuffer.rewind();
            
            // 设置输入形状
            long[] inputShape = new long[]{1, inputIds.length};
            
            // 创建输入张量 - 确保使用INT64类型
            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(ortEnvironment, inputIdsBuffer, inputShape);
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(ortEnvironment, attentionMaskBuffer, inputShape);
            
            // 记录输入张量信息
            Log.d(TAG, "输入张量类型 - input_ids: " + inputIdsTensor.getInfo().type + ", attention_mask: " + attentionMaskTensor.getInfo().type);
            
            // 准备输入数据
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            
            // 执行模型推理
            OrtSession.Result result = onnxSession.run(inputs);
            
            // 获取输出张量，通常是embedding
            OnnxTensor outputTensor = (OnnxTensor) result.get(0);
            
            // 提取嵌入向量数据
            float[][] embeddingData = (float[][]) outputTensor.getValue();
            float[] embedding = embeddingData[0]; // 获取第一个（也是唯一的）样本的向量
            
            // 记录向量信息
            Log.d(TAG, "原始嵌入向量维度: " + embedding.length);
            Log.d(TAG, "嵌入向量样例 (前5个值): " + 
                Arrays.toString(Arrays.copyOfRange(embedding, 0, Math.min(5, embedding.length))));
            
            // 对embedding进行L2归一化
            embedding = normalizeVector(embedding);
            
            // 创建一个副本，避免在关闭张量后访问其内存
            float[] embeddingCopy = Arrays.copyOf(embedding, embedding.length);
            
            // 清空输入映射
            inputs.clear();
            
            // 记录总耗时
            long inferenceEndTime = System.currentTimeMillis();
            Log.d(TAG, "ONNX推理总耗时: " + (inferenceEndTime - inferenceStartTime) + "ms");
            
            // 注意：我们不显式关闭任何资源，以避免崩溃
            // 让垃圾回收器处理这些资源
            
            return embeddingCopy;
        } catch (Exception e) {
            Log.e(TAG, "执行ONNX推理失败: " + e.getMessage(), e);
            throw new RuntimeException("执行ONNX推理失败", e);
        }
    }
    
    /**
     * 检查ONNX会话状态并尝试恢复
     * @return 会话是否可用
     */
    private boolean checkAndRecoverOnnxSession() {
        // 记录当前线程信息
        Log.d(TAG, "检查会话状态 [线程ID: " + Thread.currentThread().getId() + "]");
        
        synchronized (sessionLock) {
            // 记录上次检查时间
            lastSessionCheckTime = System.currentTimeMillis();
            
            // 检查会话是否可用
            if (onnxSession != null && sessionState == SESSION_STATE_READY) {
                Log.d(TAG, "会话状态正常，可以使用");
                return true;
            }
            
            // 如果会话正在加载中，等待一段时间
            if (sessionState == SESSION_STATE_LOADING) {
                Log.d(TAG, "会话正在加载中，等待...");
                try {
                    // 等待最多3秒
                    for (int i = 0; i < 30; i++) {
                        // 每100毫秒检查一次
                        Thread.sleep(100);
                        if (onnxSession != null && sessionState == SESSION_STATE_READY) {
                            Log.d(TAG, "会话已加载完成，可以使用");
                            return true;
                        }
                    }
                } catch (InterruptedException e) {
                    Log.e(TAG, "等待会话加载被中断: " + e.getMessage());
                    Thread.currentThread().interrupt();
                }
            }
            
            // 如果会话不可用或处于错误状态，尝试恢复
            if (onnxSession == null || sessionState == SESSION_STATE_ERROR) {
                // 检查重试次数
                if (sessionRetryCount >= MAX_SESSION_RETRY) {
                    Log.e(TAG, "会话恢复失败，已达到最大重试次数: " + sessionRetryCount);
                    return false;
                }
                
                Log.d(TAG, "尝试恢复会话，当前重试次数: " + sessionRetryCount);
                
                // 更新会话状态
                int oldState = sessionState;
                sessionState = SESSION_STATE_LOADING;
                Log.d(TAG, "会话状态变更: " + oldState + " -> " + sessionState + " (开始恢复)");
                
                // 增加重试计数
                sessionRetryCount++;
                
                try {
                    // 先关闭之前的会话
                    if (onnxSession != null) {
                        try {
                            onnxSession.close();
                            Log.d(TAG, "已关闭旧的ONNX会话");
                        } catch (Exception e) {
                            Log.e(TAG, "关闭旧的ONNX会话失败: " + e.getMessage(), e);
                        } finally {
                            onnxSession = null;
                        }
                    }
                    
                    // 重新加载ONNX模型
                    try {
                        Log.d(TAG, "重新加载ONNX模型: " + modelPath);
                        loadOnnxModel(modelPath);
                        
                        // 检查会话是否加载成功
                        if (onnxSession != null) {
                            Log.d(TAG, "ONNX会话恢复成功");
                            sessionState = SESSION_STATE_READY;
                            return true;
                        } else {
                            Log.e(TAG, "ONNX会话恢复失败，会话为null");
                            sessionState = SESSION_STATE_ERROR;
                            return false;
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "重新加载ONNX模型失败: " + e.getMessage(), e);
                        sessionState = SESSION_STATE_ERROR;
                        return false;
                    }
                } finally {
                    // 如果会话仍然为null，确保状态为ERROR
                    if (onnxSession == null && sessionState != SESSION_STATE_ERROR) {
                        Log.e(TAG, "会话为null但状态不是ERROR，更正状态");
                        sessionState = SESSION_STATE_ERROR;
                    }
                    
                    // 记录会话恢复结果
                    boolean success = onnxSession != null && sessionState == SESSION_STATE_READY;
                    Log.d(TAG, "会话恢复" + (success ? "成功" : "失败") + 
                           "，最终状态: " + sessionState + 
                           "，重试次数: " + sessionRetryCount + "/" + MAX_SESSION_RETRY);
                }
            }
            
            // 会话仍然不可用
            Log.e(TAG, "会话检查结束，会话仍然不可用，状态: " + sessionState);
            return false;
        }
    }
    
    /**
     * 从路径中提取模型名称
     * @param path 模型路径
     * @return 提取的模型名称
     */
    private String extractModelNameFromPath(String path) {
        try {
            File file = new File(path);
            File parentDir = file.getParentFile();
            
            // 首先尝试从目录名称检测是否是bge-m3
            if (parentDir != null) {
                String dirName = parentDir.getName();
                Log.d(TAG, "检查目录名称是否包含bge-m3: " + dirName);
                
                if (dirName.toLowerCase().contains("bge-m3")) {
                    Log.d(TAG, "检测到bge-m3模型: " + dirName);
                    return "BGE-M3";
                }
                
                // 检查tokenizer.json文件是否存在，并尝试从中提取模型信息
                File tokenizerFile = new File(parentDir, "tokenizer.json");
                if (tokenizerFile.exists()) {
                    String modelNameFromTokenizer = extractModelNameFromTokenizer(tokenizerFile);
                    if (modelNameFromTokenizer != null && !modelNameFromTokenizer.isEmpty()) {
                        Log.d(TAG, "从tokenizer.json成功提取模型名称: " + modelNameFromTokenizer);
                        return modelNameFromTokenizer;
                    }
                }
            }
            
            // 尝试从config.json提取
            if (parentDir != null) {
                String configName = extractModelNameFromConfig(parentDir);
                if (configName != null && !configName.isEmpty()) {
                    Log.d(TAG, "从config.json成功提取模型名称: " + configName);
                    return configName;
                }
            }
            
            // 尝试从目录名称提取
            if (parentDir != null) {
                String dirName = parentDir.getName();
                Log.d(TAG, "尝试从目录名称提取模型名称: " + dirName);
                
                // 检查是否包含常见的模型名称关键词
                List<String> modelKeywords = Arrays.asList(
                    "bge", "bert", "roberta", "clip", "e5", "gpt", "llama", "sbert", 
                    "sentence", "transformer", "embedding", "encoder", "minilm", "mpnet"
                );
                
                for (String keyword : modelKeywords) {
                    if (dirName.toLowerCase().contains(keyword.toLowerCase())) {
                        Log.d(TAG, "从目录名称提取到模型关键词: " + keyword);
                        return dirName;
                    }
                }
                
                // 如果目录名称不包含关键词，但看起来像是模型名称（包含字母和数字的组合），也返回它
                if (dirName.matches(".*[a-zA-Z].*") && dirName.matches(".*[0-9].*")) {
                    Log.d(TAG, "目录名称看起来像模型名称: " + dirName);
                    return dirName;
                }
            }
            
            // 如果无法从目录名称中提取，返回文件名
            String fileName = file.getName()
                .replace(".onnx", "")
                .replace(".pt", "")
                .replace("model", "")
                .replace("_", " ")
                .trim();
            
            if (!fileName.isEmpty()) {
                Log.d(TAG, "从文件名提取模型名称: " + fileName);
                return fileName;
            }
            
            // 如果所有方法都失败，返回一个默认名称
            Log.d(TAG, "无法提取模型名称，使用默认名称");
            return "Embedding Model";
        } catch (Exception e) {
            Log.e(TAG, "提取模型名称失败: " + e.getMessage(), e);
            return "Embedding Model";
        }
    }

    /**
     * 从tokenizer.json文件中提取模型名称
     * @param tokenizerFile tokenizer.json文件
     * @return 提取的模型名称，如果无法提取则返回null
     */
    private String extractModelNameFromTokenizer(File tokenizerFile) {
        try {
            Log.d(TAG, "尝试从tokenizer.json提取模型名称: " + tokenizerFile.getAbsolutePath());
            
            // 读取tokenizer.json文件的前1000个字符，足够检测模型类型
            StringBuilder content = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new FileReader(tokenizerFile))) {
                char[] buffer = new char[1000];
                int read = reader.read(buffer);
                if (read > 0) {
                    content.append(buffer, 0, read);
                }
            }
            
            String contentStr = content.toString();
            
            // 检查是否是bge-m3模型的tokenizer
            if (contentStr.contains("\"name\":") && contentStr.contains("\"model\":")) {
                // 尝试提取name字段
                int nameIndex = contentStr.indexOf("\"name\":");
                if (nameIndex != -1) {
                    int nameStartIndex = contentStr.indexOf("\"", nameIndex + 7);
                    if (nameStartIndex != -1) {
                        int nameEndIndex = contentStr.indexOf("\"", nameStartIndex + 1);
                        if (nameEndIndex != -1) {
                            String name = contentStr.substring(nameStartIndex + 1, nameEndIndex);
                            Log.d(TAG, "从tokenizer.json的name字段提取模型名称: " + name);
                            
                            // 如果名称包含bge-m3，直接返回标准化的名称
                            if (name.toLowerCase().contains("bge-m3")) {
                                return "BGE-M3";
                            }
                            
                            return name;
                        }
                    }
                }
                
                // 如果找不到name字段或name字段不包含bge-m3，但内容特征符合bge-m3
                if (contentStr.contains("[\"<s>\",0]") && contentStr.contains("[\"<pad>\",0]") && 
                    contentStr.contains("[\"</s>\",0]") && contentStr.contains("[\"<unk>\",0]")) {
                    Log.d(TAG, "从tokenizer.json内容特征识别为bge-m3模型");
                    return "BGE-M3";
                }
            }
            
            return null;
        } catch (Exception e) {
            Log.e(TAG, "从tokenizer.json提取模型名称失败: " + e.getMessage(), e);
            return null;
        }
    }

    /**
     * 从config.json文件中提取模型名称
     * @param modelDir 模型目录
     * @return 提取的模型名称，如果无法提取则返回null
     */
    private String extractModelNameFromConfig(File modelDir) {
        try {
            File configFile = new File(modelDir, "config.json");
            if (!configFile.exists()) {
                Log.d(TAG, "config.json文件不存在: " + configFile.getAbsolutePath());
                return null;
            }
            
            Log.d(TAG, "尝试从config.json提取模型名称: " + configFile.getAbsolutePath());
            
            // 读取config.json文件内容
            StringBuilder content = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new FileReader(configFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    content.append(line);
                }
            }
            
            // 解析JSON
            JSONObject config = new JSONObject(content.toString());
            Log.d(TAG, "成功解析config.json");
            
            // 尝试从不同字段提取模型名称
            String name = null;
            
            // 尝试从model_type字段提取
            if (config.has("model_type") && !config.isNull("model_type")) {
                name = config.getString("model_type");
                Log.d(TAG, "从model_type字段提取模型名称: " + name);
            }
            
            // 尝试从architectures字段提取
            if ((name == null || name.isEmpty()) && config.has("architectures") && !config.isNull("architectures")) {
                JSONArray architectures = config.getJSONArray("architectures");
                if (architectures.length() > 0) {
                    name = architectures.getString(0);
                    Log.d(TAG, "从architectures字段提取模型名称: " + name);
                }
            }
            
            // 尝试从_name_or_path字段提取
            if ((name == null || name.isEmpty()) && config.has("_name_or_path") && !config.isNull("_name_or_path")) {
                name = config.getString("_name_or_path");
                Log.d(TAG, "从_name_or_path字段提取模型名称: " + name);
            }
            
            // 尝试从name字段提取
            if ((name == null || name.isEmpty()) && config.has("name") && !config.isNull("name")) {
                name = config.getString("name");
                Log.d(TAG, "从name字段提取模型名称: " + name);
            }
            
            // 尝试从hidden_size和embedding_size推断模型大小
            String modelSize = "";
            if (config.has("hidden_size") && !config.isNull("hidden_size")) {
                int hiddenSize = config.getInt("hidden_size");
                
                // 根据hidden_size推断模型大小
                if (hiddenSize <= 384) {
                    modelSize = "Tiny";
                } else if (hiddenSize <= 512) {
                    modelSize = "Mini";
                } else if (hiddenSize <= 768) {
                    modelSize = "Small";
                } else if (hiddenSize <= 1024) {
                    modelSize = "Medium";
                } else {
                    modelSize = "Large";
                }
                
                Log.d(TAG, "从hidden_size推断模型大小: " + modelSize + " (hidden_size=" + hiddenSize + ")");
            }
            
            // 组合模型名称和大小
            if (name != null && !name.isEmpty()) {
                if (!modelSize.isEmpty()) {
                    return name + "-" + modelSize;
                }
                return name;
            }
            
            return null;
        } catch (Exception e) {
            Log.e(TAG, "从config.json提取模型名称失败: " + e.getMessage(), e);
            return null;
        }
    }

    /**
     * 读取文件内容
     * @param file 文件
     * @return 文件内容
     * @throws IOException 如果读取失败
     */
    private String readFileContent(File file) throws IOException {
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
        }
        return content.toString();
    }

    /**
     * 获取向量生成的调试信息
     * @param text 输入文本
     * @param embedding 生成的向量
     * @param startTime 开始时间
     * @return 向量生成的调试信息
     */
    public String getVectorDebugInfo(String text, float[] embedding, long startTime) {
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        StringBuilder vectorInfo = new StringBuilder();
        vectorInfo.append("向量生成信息:\n");
        vectorInfo.append("模型: ").append(modelName).append("\n");
        vectorInfo.append("输入文本长度: ").append(text.length()).append(" 字符\n");
        vectorInfo.append("向量维度: ").append(embedding.length).append("\n");
        vectorInfo.append("处理时间: ").append(duration).append(" 毫秒\n");
        
        // 添加向量前10个和后10个值的样本
        vectorInfo.append("向量样本(前10个值): ");
        for (int i = 0; i < Math.min(10, embedding.length); i++) {
            vectorInfo.append(String.format("%.4f", embedding[i]));
            if (i < Math.min(10, embedding.length) - 1) {
                vectorInfo.append(", ");
            }
        }
        vectorInfo.append("\n");
        
        if (embedding.length > 20) {
            vectorInfo.append("向量样本(后10个值): ");
            for (int i = Math.max(0, embedding.length - 10); i < embedding.length; i++) {
                vectorInfo.append(String.format("%.4f", embedding[i]));
                if (i < embedding.length - 1) {
                    vectorInfo.append(", ");
                }
            }
            vectorInfo.append("\n");
        }
        
        return vectorInfo.toString();
    }

    /**
     * 生成文本的嵌入向量
     * @param text 输入文本
     * @return 嵌入向量（浮点数数组）
     * @throws Exception 如果生成嵌入失败
     */
    public float[] generateEmbedding(String text) throws Exception {
        Log.d(TAG, "开始生成嵌入向量，文本长度: " + text.length());
        
        if (text == null || text.isEmpty()) {
            Log.w(TAG, "输入文本为空，返回空向量");
            return new float[0];
        }
        
        try {
            // 根据模型类型生成嵌入向量
            float[] embedding;
            long startTime = System.currentTimeMillis();
            
            if (modelType == ModelType.ONNX) {
                // 检查ONNX会话状态并尝试恢复
                if (!checkAndRecoverOnnxSession()) {
                    Log.e(TAG, "ONNX会话不可用，无法生成嵌入向量");
                    throw new RuntimeException("ONNX会话不可用");
                }
                
                embedding = generateEmbeddingWithOnnx(text);
            } else if (modelType == ModelType.TORCH_SCRIPT) {
                // Torch模型的实现（暂未实现）
                Log.e(TAG, "TorchScript模型尚未实现");
                return null;
            } else {
                Log.e(TAG, "未知的模型类型");
                return null;
            }
            
            // 记录生成时间
            long endTime = System.currentTimeMillis();
            Log.d(TAG, "生成嵌入向量耗时: " + (endTime - startTime) + "ms");
            
            // 对向量进行L2归一化
            embedding = normalizeVector(embedding);
            
            // 打印前5个和后5个向量值，帮助调试
            if (embedding != null && embedding.length > 10) {
                StringBuilder sb = new StringBuilder("嵌入向量样例 (前5个值): ");
                for (int i = 0; i < 5; i++) {
                    sb.append(embedding[i]).append(", ");
                }
                sb.append(" ... (后5个值): ");
                for (int i = embedding.length - 5; i < embedding.length; i++) {
                    sb.append(embedding[i]).append(", ");
                }
                Log.d(TAG, sb.toString());
                
                // 计算向量范数，帮助确认归一化是否有效
                double norm = 0.0;
                for (float v : embedding) {
                    norm += v * v;
                }
                norm = Math.sqrt(norm);
                Log.d(TAG, "嵌入向量L2范数: " + norm + " (应接近1.0)");
            }
            
            return embedding;
        } catch (Exception e) {
            Log.e(TAG, "生成嵌入向量失败: " + e.getMessage(), e);
            throw e;
        }
    }
    
    /**
     * 获取嵌入向量维度
     * @return 嵌入向量维度
     */
    public int getEmbeddingDimension() {
        try {
            // 尝试从模型配置中获取维度
            if (configJson != null && configJson.has("hidden_size")) {
                int hiddenSize = configJson.getInt("hidden_size");
                Log.d(TAG, "从配置文件获取向量维度: " + hiddenSize + ", 模型路径: " + modelPath);
                return hiddenSize;
            }
            
            // 如果配置中没有，则使用默认值
            if (modelName != null) {
                if (modelName.contains("bge-small") || modelName.contains("bge-base")) {
                    Log.d(TAG, "根据模型名称判断向量维度: 768, 模型名称: " + modelName + ", 模型路径: " + modelPath);
                    return 768;
                } else if (modelName.contains("bge-large")) {
                    Log.d(TAG, "根据模型名称判断向量维度: 1024, 模型名称: " + modelName + ", 模型路径: " + modelPath);
                    return 1024;
                } else if (modelName.contains("bge-m3")) {
                    Log.d(TAG, "根据模型名称判断向量维度: 1024, 模型名称: " + modelName + ", 模型路径: " + modelPath);
                    return 1024;
                }
            }
            
            // 默认维度
            Log.w(TAG, "无法确定向量维度，使用默认值1024, 模型名称: " + modelName + ", 模型路径: " + modelPath);
            return 1024; // 修改默认值为1024
        } catch (Exception e) {
            Log.e(TAG, "获取嵌入向量维度失败: " + e.getMessage() + ", 模型名称: " + modelName + ", 模型路径: " + modelPath, e);
            return 1024; // 修改默认值为1024
        }
    }
    
    /**
     * 设置是否使用一致性处理
     * 确保在知识库构建和查询过程中使用相同的分词逻辑
     * @param consistent 是否使用一致性处理
     */
    public void setUseConsistentProcessing(boolean consistent) {
        this.useConsistentProcessing = consistent;
        Log.d(TAG, "设置一致性处理: " + consistent);
        
        // 如果tokenizer已初始化，同步设置其一致性分词策略
        if (tokenizer != null) {
            tokenizer.setUseConsistentTokenization(consistent);
            Log.d(TAG, "同步设置tokenizer的一致性分词策略: " + consistent);
        }
    }
    
    /**
     * 获取当前是否使用一致性处理
     * @return 是否使用一致性处理
     */
    public boolean isUsingConsistentProcessing() {
        return this.useConsistentProcessing;
    }
    
    /**
     * 设置调试模式
     * @param debug 是否启用调试
     */
    public void setDebugMode(boolean debug) {
        this.debugMode = debug;
        Log.d(TAG, "设置调试模式: " + debug);
        
        // 如果tokenizer已初始化，同步设置其调试模式
        if (tokenizer != null) {
            tokenizer.setDebugMode(debug);
            Log.d(TAG, "同步设置tokenizer的调试模式: " + debug);
        }
    }
    
    /**
     * 获取当前是否启用调试模式
     * @return 是否启用调试模式
     */
    public boolean isDebugModeEnabled() {
        return this.debugMode;
    }
    
    /**
     * 输出调试日志
     * 只有在调试模式启用时才会输出
     * @param message 日志信息
     */
    private void debugLog(String message) {
        if (debugMode) {
            Log.d(TAG, message);
        }
    }

    /**
     * 获取嵌入模型名称
     * @return 嵌入模型名称
     */
    public String getEmbeddingModel() {
        return modelName != null ? modelName : extractModelNameFromPath(modelPath);
    }
    
    /**
     * 获取模型名称
     * @return 模型名称
     */
    public String getModelName() {
        if (modelName != null && !modelName.isEmpty()) {
            return modelName;
        }
        
        // 检查路径中是否包含bge-m3
        if (modelPath != null && modelPath.toLowerCase().contains("bge-m3")) {
            Log.d(TAG, "从路径中检测到bge-m3模型");
            return "BGE-M3";
        }
        
        return extractModelNameFromPath(modelPath);
    }
    
    /**
     * 获取模型类型
     * @return 模型类型
     */
    public ModelType getModelType() {
        return modelType;
    }
    
    /**
     * 获取模型路径
     * @return 模型路径
     */
    public String getModelPath() {
        return modelPath;
    }
    
    /**
     * 关闭模型资源
     */
    public void close() {
        try {
            Log.d(TAG, "开始关闭模型资源");
            
            synchronized (sessionLock) {
                if (torchModel != null) {
                    // PyTorch模型不需要显式关闭
                    Log.d(TAG, "释放TorchScript模型资源");
                    torchModel = null;
                }
                
                if (onnxSession != null) {
                    try {
                        Log.d(TAG, "关闭ONNX会话");
                        onnxSession.close();
                    } catch (Exception e) {
                        Log.e(TAG, "关闭ONNX会话失败: " + e.getMessage(), e);
                    } finally {
                        onnxSession = null;
                        sessionState = SESSION_STATE_NONE;
                    }
                }
                
                // 重置会话状态
                sessionRetryCount = 0;
                lastSessionCheckTime = 0;
            }
            
            Log.d(TAG, "模型资源已关闭");
        } catch (Exception e) {
            Log.e(TAG, "关闭模型资源失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 对向量进行L2归一化
     * @param vector 输入向量
     * @return 归一化后的向量
     */
    private float[] normalizeVector(float[] vector) {
        Log.d(TAG, "对嵌入向量进行L2归一化，向量长度: " + vector.length);
        
        float[] normalized = new float[vector.length];
        float norm = 0.0f;
        
        // 计算L2范数（欧几里德范数）
        for (float v : vector) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);
        
        // 记录归一化前的范数
        Log.d(TAG, "归一化前的向量L2范数: " + norm);
        
        // 如果范数太小，返回零向量以避免数值问题
        if (norm < 1e-6) {
            Log.w(TAG, "向量范数接近零 (" + norm + ")，返回零向量");
            return new float[vector.length];
        }
        
        // 归一化
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        
        // 验证归一化结果
        float newNorm = 0.0f;
        for (float v : normalized) {
            newNorm += v * v;
        }
        newNorm = (float) Math.sqrt(newNorm);
        
        Log.d(TAG, "归一化后的向量L2范数: " + newNorm + " (应接近1.0)");
        
        return normalized;
    }
    
    /**
     * 对文本进行分词处理
     * 这是一个改进的实现，尝试模拟专业tokenizer的行为
     * @param text 输入文本
     * @return 包含input_ids和attention_mask的数组
     */
    private long[][] tokenizeText(String text) {
        try {
            // 首先尝试从TokenizerManager获取tokenizer
            if (context != null) {
                TokenizerManager tokenizerManager = TokenizerManager.getInstance(context);
                
                // 如果TokenizerManager已初始化，直接使用其tokenizer
                if (tokenizerManager.isInitialized()) {
                    BertTokenizer globalTokenizer = tokenizerManager.getTokenizer();
                    if (globalTokenizer != null) {
                        debugLog("使用TokenizerManager中的全局tokenizer进行分词");
                        
                        // 设置一致性分词策略
                        if (useConsistentProcessing) {
                            tokenizerManager.setUseConsistentTokenization(true);
                            debugLog("设置一致性分词策略: " + useConsistentProcessing);
                            debugLog("为当前分词操作启用一致性分词策略");
                        }
                        
                        return globalTokenizer.tokenize(text);
                    }
                }
            }
            
            // 如果TokenizerManager不可用，使用本地tokenizer
            if (tokenizer != null) {
                debugLog("使用本地tokenizer进行分词");
                
                // 使用一致性分词策略
                if (useConsistentProcessing && !tokenizer.isUseConsistentTokenization()) {
                    tokenizer.setUseConsistentTokenization(true);
                    debugLog("为当前分词操作启用一致性分词策略");
                }
                
                return tokenizer.tokenize(text);
            }
            
            // 如果tokenizer未初始化，使用简单的分词方法
            Log.d(TAG, "tokenizer未初始化，使用简单分词方法");
            
            // 简单的空格分词
            String[] words = text.split("\\s+");
            
            // 创建input_ids和attention_mask
            int seqLength = words.length + 2; // 加上[CLS]和[SEP]
            long[] inputIds = new long[seqLength];
            long[] attentionMask = new long[seqLength];
            
            // 添加[CLS]标记
            inputIds[0] = 0; // [CLS] token ID
            attentionMask[0] = 1;
            
            // 填充单词ID (这里只是简单地使用位置作为ID)
            for (int i = 0; i < words.length; i++) {
                inputIds[i + 1] = i + 4; // 从4开始，避开特殊token ID
                attentionMask[i + 1] = 1;
            }
            
            // 添加[SEP]标记
            inputIds[seqLength - 1] = 2; // [SEP] token ID
            attentionMask[seqLength - 1] = 1;
            
            return new long[][] { inputIds, attentionMask };
        } catch (Exception e) {
            Log.e(TAG, "分词处理失败: " + e.getMessage(), e);
            
            // 返回最基本的序列：只包含[CLS]和[SEP]
            long[] inputIds = new long[] { 0, 2 }; // [CLS], [SEP]
            long[] attentionMask = new long[] { 1, 1 };
            return new long[][] { inputIds, attentionMask };
        }
    }
}
