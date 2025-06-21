package com.example.starlocalrag;

import android.content.Context;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import com.example.starlocalrag.api.TokenizerManager;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 重排模型处理器
 * 支持ONNX格式的Cross-Encoder重排模型，如bge-reranker-m3
 */
public class RerankerModelHandler {
    private static final String TAG = "StarLocalRAG_RerankerModel";
    private static final int MAX_SEQUENCE_LENGTH = 512; // 最大序列长度
    private static final int MAX_BATCH_SIZE = 8; // 批处理大小，避免内存过大
    
    // ONNX会话状态常量
    private static final int SESSION_STATE_NONE = 0;      // 未初始化
    private static final int SESSION_STATE_LOADING = 1;   // 正在加载
    private static final int SESSION_STATE_READY = 2;     // 已就绪
    private static final int SESSION_STATE_ERROR = 3;     // 错误状态
    
    // 会话重试相关常量
    private static final int MAX_SESSION_RETRY = 3;       // 最大重试次数
    private static final long SESSION_RETRY_DELAY_MS = 500; // 重试间隔
    
    private final Context context;
    private final String modelPath;
    private OrtEnvironment environment;
    private OrtSession session;
    private final AtomicBoolean isInitialized = new AtomicBoolean(false);
    private final AtomicBoolean isLoading = new AtomicBoolean(false);
    
    // 会话状态管理变量
    private final Object sessionLock = new Object();      // 会话锁
    private int sessionState = SESSION_STATE_NONE;        // 当前会话状态
    private int sessionRetryCount = 0;                    // 当前重试次数
    private long lastSessionCheckTime = 0;                // 上次会话检查时间
    
    // TokenizerManager实例
    private TokenizerManager tokenizerManager;
    
    // 模型输入输出名称（根据具体模型调整）
    private static final String INPUT_IDS = "input_ids";
    private static final String ATTENTION_MASK = "attention_mask";
    private static final String TOKEN_TYPE_IDS = "token_type_ids";
    private static final String OUTPUT_LOGITS = "logits";
    
    /**
     * 重排结果类
     */
    public static class RerankResult implements Comparable<RerankResult> {
        public final String text;
        public final float score;
        public final int originalIndex;
        
        public RerankResult(String text, float score, int originalIndex) {
            this.text = text;
            this.score = score;
            this.originalIndex = originalIndex;
        }
        
        @Override
        public int compareTo(RerankResult other) {
            // 按分数降序排列
            return Float.compare(other.score, this.score);
        }
    }
    
    public RerankerModelHandler(Context context, String modelPath) {
        this.context = context;
        this.modelPath = modelPath;
    }
    
    /**
     * 初始化模型
     */
    public boolean initialize() {
        if (isInitialized.get()) {
            return true;
        }
        
        synchronized (sessionLock) {
            // 双重检查锁定模式
            if (isInitialized.get()) {
                return true;
            }
            
            if (isLoading.get()) {
                LogManager.logW(TAG, "模型正在加载中，请稍候");
                return false;
            }
            
            isLoading.set(true);
            sessionState = SESSION_STATE_LOADING;
            LogManager.logD(TAG, "会话状态变更: " + sessionState + " (开始加载)");
        
        try {
            LogManager.logI(TAG, "开始初始化重排模型: " + modelPath);
            
            // 检查模型文件是否存在
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                LogManager.logE(TAG, "重排模型文件不存在: " + modelPath);
                return false;
            }
            
            // 打印模型文件详细信息
            long fileSize = modelFile.length();
            LogManager.logI(TAG, "模型文件大小: " + (fileSize / 1024 / 1024) + " MB");
            LogManager.logI(TAG, "模型文件可读: " + modelFile.canRead());
            LogManager.logI(TAG, "模型文件绝对路径: " + modelFile.getAbsolutePath());
            
            // 检查可用内存
            Runtime runtime = Runtime.getRuntime();
            long maxMemory = runtime.maxMemory();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            long usedMemory = totalMemory - freeMemory;
            LogManager.logI(TAG, "内存状态 - 最大: " + (maxMemory / 1024 / 1024) + "MB, " +
                           "已用: " + (usedMemory / 1024 / 1024) + "MB, " +
                           "可用: " + (freeMemory / 1024 / 1024) + "MB");
            
            // 创建ONNX环境
            LogManager.logI(TAG, "创建ONNX环境...");
            environment = OrtEnvironment.getEnvironment();
            LogManager.logI(TAG, "ONNX环境创建成功");
            
            // 创建会话选项
            LogManager.logI(TAG, "创建会话选项...");
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            
            // 设置线程数（从ConfigManager获取配置）
            int threadCount = ConfigManager.getThreads(context);
            LogManager.logI(TAG, "设置线程数: " + threadCount);
            sessionOptions.setIntraOpNumThreads(threadCount);
            sessionOptions.setInterOpNumThreads(threadCount);
            
            // 设置内存优化选项
            LogManager.logI(TAG, "设置内存优化选项...");
            sessionOptions.setMemoryPatternOptimization(true);
            
            // 设置执行模式为顺序执行，避免并发问题
            LogManager.logI(TAG, "设置执行模式为顺序执行...");
            sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            
            // 设置其他选项
            LogManager.logI(TAG, "设置会话选项完成");
            
            // 加载模型
            LogManager.logI(TAG, "开始加载ONNX模型...");
            LogManager.logI(TAG, "会话创建前 - 线程: " + Thread.currentThread().getName());
            LogManager.logI(TAG, "会话创建前 - 时间戳: " + System.currentTimeMillis());
            
            session = environment.createSession(modelPath, sessionOptions);
            
            LogManager.logI(TAG, "ONNX模型加载成功");
            LogManager.logI(TAG, "会话创建后 - 时间戳: " + System.currentTimeMillis());
            
            // 打印模型信息
            LogManager.logI(TAG, "模型输入数量: " + session.getInputNames().size());
            LogManager.logI(TAG, "模型输出数量: " + session.getOutputNames().size());
            LogManager.logI(TAG, "模型输入名称: " + session.getInputNames());
            LogManager.logI(TAG, "模型输出名称: " + session.getOutputNames());
            
            // 初始化TokenizerManager
            //LogManager.logI(TAG, "开始初始化TokenizerManager...");
            try {
                tokenizerManager = TokenizerManager.getInstance(context);
                
                // 获取tokenizer.json文件所在的目录
                File modelDir = modelFile.getParentFile();
                
                if (modelDir != null && modelDir.exists()) {
                    LogManager.logI(TAG, "尝试从模型目录初始化tokenizer: " + modelDir.getAbsolutePath());
                    boolean tokenizerSuccess = tokenizerManager.initialize(modelDir);
                    
                    if (tokenizerSuccess) {
                        //LogManager.logI(TAG, "TokenizerManager初始化成功");
                        //LogManager.logI(TAG, "Tokenizer特殊token数量: " + tokenizerManager.getSpecialTokensSize());
                    } else {
                        LogManager.logW(TAG, "TokenizerManager初始化失败，将使用简化tokenizer");
                        tokenizerManager = null;
                    }
                } else {
                    LogManager.logW(TAG, "无法获取模型目录，将使用简化tokenizer");
                    tokenizerManager = null;
                }
            } catch (Exception e) {
                LogManager.logE(TAG, "TokenizerManager初始化异常: " + e.getMessage(), e);
                tokenizerManager = null;
            }
            
            isInitialized.set(true);
            sessionState = SESSION_STATE_READY;
            sessionRetryCount = 0; // 重置重试计数
            //LogManager.logI(TAG, "重排模型初始化成功");
            //LogManager.logD(TAG, "会话状态变更: " + sessionState + " (初始化成功)");
            
            return true;
            
        } catch (OutOfMemoryError e) {
            LogManager.logE(TAG, "重排模型初始化失败 - 内存不足");
            LogManager.logE(TAG, "异常类型: " + e.getClass().getSimpleName());
            LogManager.logE(TAG, "异常消息: " + e.getMessage());
            LogManager.logE(TAG, "异常详情: ", e);
            
            sessionState = SESSION_STATE_ERROR;
            LogManager.logD(TAG, "会话状态变更: " + sessionState + " (内存不足错误)");
            
            // 强制垃圾回收
            System.gc();
            
            cleanup();
            return false;
        } catch (Exception e) {
            LogManager.logE(TAG, "重排模型初始化失败");
            LogManager.logE(TAG, "异常类型: " + e.getClass().getSimpleName());
            LogManager.logE(TAG, "异常消息: " + e.getMessage());
            LogManager.logE(TAG, "异常详情: ", e);
            
            sessionState = SESSION_STATE_ERROR;
            LogManager.logD(TAG, "会话状态变更: " + sessionState + " (初始化异常)");
            
            // 打印更多调试信息
            if (e.getCause() != null) {
                LogManager.logE(TAG, "根本原因: " + e.getCause().getClass().getSimpleName() + ": " + e.getCause().getMessage());
            }
            
            // 检查是否是内存相关问题
            if (e.getMessage() != null && e.getMessage().toLowerCase().contains("memory")) {
                LogManager.logE(TAG, "可能是内存相关问题导致的失败");
                // 强制垃圾回收
                System.gc();
            }
            
            // 检查是否是文件访问问题
            if (e.getMessage() != null && 
                (e.getMessage().toLowerCase().contains("file") || 
                 e.getMessage().toLowerCase().contains("path") ||
                 e.getMessage().toLowerCase().contains("access"))) {
                LogManager.logE(TAG, "可能是文件访问权限问题");
            }
            
            cleanup();
            return false;
        } finally {
            isLoading.set(false);
        }
        } // 结束synchronized块
    }
    
    /**
     * 重排进度回调接口
     */
    public interface RerankProgressCallback {
        void onRerankProgress(int processedCount, int totalCount, double score);
    }
    
    /**
     * 对文档进行重排
     * @param query 查询文本
     * @param documents 待重排的文档列表
     * @param topK 返回前K个结果
     * @return 重排后的结果列表
     */
    public List<RerankResult> rerank(String query, List<String> documents, int topK) {
        return rerank(query, documents, topK, null);
    }
    
    /**
     * 对文档进行重排（带进度回调）
     * @param query 查询文本
     * @param documents 待重排的文档列表
     * @param topK 返回前K个结果
     * @param progressCallback 进度回调
     * @return 重排后的结果列表
     */
    public List<RerankResult> rerank(String query, List<String> documents, int topK, RerankProgressCallback progressCallback) {
        // 立即输出日志，确认方法被调用
        LogManager.logI(TAG, "=== RERANK方法开始执行 ===");
        //LogManager.logI(TAG, "查询文本: " + (query != null ? query.substring(0, Math.min(50, query.length())) + "..." : "null"));
        //LogManager.logI(TAG, "文档数量: " + (documents != null ? documents.size() : 0));
        //LogManager.logI(TAG, "topK: " + topK);
        //LogManager.logI(TAG, "模型初始化状态: " + isInitialized.get());
        
        if (!isInitialized.get()) {
            LogManager.logE(TAG, "重排模型未初始化");
            return convertToRerankResults(documents); // 返回原始顺序
        }
        
        // 检查并恢复会话状态
        if (!checkAndRecoverSession()) {
            LogManager.logE(TAG, "重排模型会话不可用，返回原始顺序");
            return convertToRerankResults(documents);
        }
        
        if (documents == null || documents.isEmpty()) {
            LogManager.logW(TAG, "文档列表为空");
            return new ArrayList<>();
        }
        
        if (query == null || query.trim().isEmpty()) {
            LogManager.logW(TAG, "查询文本为空");
            return convertToRerankResults(documents);
        }
        
        try {
            //LogManager.logD(TAG, "开始重排，查询: " + query.substring(0, Math.min(50, query.length())) + 
            //               "..., 文档数: " + documents.size());
            
            List<RerankResult> results = new ArrayList<>();
            
            // 分批处理以避免内存过大
            for (int i = 0; i < documents.size(); i += MAX_BATCH_SIZE) {
                int endIndex = Math.min(i + MAX_BATCH_SIZE, documents.size());
                List<String> batch = documents.subList(i, endIndex);
                
                List<RerankResult> batchResults = processBatch(query, batch, i, progressCallback);
                results.addAll(batchResults);
            }
            
            // 按分数排序
            Collections.sort(results);
            
            // 返回前topK个结果
            int resultSize = Math.min(topK, results.size());
            List<RerankResult> topResults = results.subList(0, resultSize);
            
            LogManager.logD(TAG, "重排完成，返回前" + resultSize + "个结果");
            return topResults;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "重排过程中发生错误");
            LogManager.logE(TAG, "异常类型: " + e.getClass().getSimpleName());
            LogManager.logE(TAG, "异常消息: " + e.getMessage());
            LogManager.logE(TAG, "查询文本: " + (query != null ? query.substring(0, Math.min(100, query.length())) : "null"));
            LogManager.logE(TAG, "文档数量: " + (documents != null ? documents.size() : 0));
            LogManager.logE(TAG, "异常详情: ", e);
            
            if (e.getCause() != null) {
                LogManager.logE(TAG, "根本原因: " + e.getCause().getClass().getSimpleName() + ": " + e.getCause().getMessage());
            }
            
            return convertToRerankResults(documents); // 返回原始顺序
        }
    }
    
    /**
     * 处理一个批次的文档
     */
    private List<RerankResult> processBatch(String query, List<String> documents, int startIndex) throws OrtException {
        return processBatch(query, documents, startIndex, null);
    }
    
    /**
     * 处理一个批次的文档（带进度回调）
     */
    private List<RerankResult> processBatch(String query, List<String> documents, int startIndex, RerankProgressCallback progressCallback) throws OrtException {
        List<RerankResult> results = new ArrayList<>();
        
        //LogManager.logI(TAG, "开始处理批次，文档数量: " + documents.size() + ", 起始索引: " + startIndex);
        
        for (int i = 0; i < documents.size(); i++) {
            long docStartTime = System.currentTimeMillis();
            int globalIndex = startIndex + i;
            
            LogManager.logI(TAG, "=== 开始处理文档 " + (i + 1) + "/" + documents.size() + " (全局索引: " + globalIndex + ") ===");
            
            String document = documents.get(i);
            if (document == null || document.trim().isEmpty()) {
                LogManager.logW(TAG, "跳过空文档 " + globalIndex);
                continue;
            }
            
            // 构建输入文本（标记输入格式：[Q] query [SEP] [D] document [SEP]）
            String inputText = "[Q] " + query + " [SEP] [D] " + document + " [SEP]";
            //LogManager.logI(TAG, "构建标记输入格式文本:");
            //LogManager.logI(TAG, "  - Query长度: " + query.length() + " 字符");
            //LogManager.logI(TAG, "  - Document长度: " + document.length() + " 字符");
            //LogManager.logI(TAG, "  - 合并后总长度: " + inputText.length() + " 字符");
            //LogManager.logI(TAG, "  - 格式: [Q] query [SEP] [D] document [SEP]");
            LogManager.logI(TAG, "输入文本预览: " + inputText.substring(0, Math.min(150, inputText.length())) + "...");
            
            //LogManager.logW(TAG, "🔍 性能分析: 重排vs嵌入的关键差异:");
            //LogManager.logW(TAG, "  1. 嵌入模型: 只需处理单个文档，输出固定维度向量");
            //LogManager.logW(TAG, "  2. 重排模型: 需要处理query+document组合，每个文档都要单独推理");
            //LogManager.logW(TAG, "  3. 当前文档数量: " + documents.size() + "，意味着需要" + documents.size() + "次独立推理");
            
            // 开始文本分词处理
            long tokenizeStartTime = System.currentTimeMillis();
            //LogManager.logI(TAG, "=== 开始第" + (i+1) + "/" + documents.size() + "个文档的TOKENIZATION ===");
            //LogManager.logI(TAG, "🔤 输入文本统计:");
            //LogManager.logI(TAG, "  - 原始query长度: " + query.length() + " 字符");
            //LogManager.logI(TAG, "  - 原始document长度: " + document.length() + " 字符");
            //LogManager.logI(TAG, "  - 合并后文本长度: " + inputText.length() + " 字符");
            //LogManager.logI(TAG, "  - 预计token数量: ~" + (inputText.length() / 4) + " (估算)");
            //LogManager.logI(TAG, "🎯 开始调用tokenizeInput方法...");
            
            Map<String, OnnxTensor> inputs;
            try {
                inputs = tokenizeInput(inputText);
                long tokenizeTime = System.currentTimeMillis() - tokenizeStartTime;
                //LogManager.logI(TAG, "✅ TOKENIZATION完成，总耗时: " + tokenizeTime + "ms");
                //LogManager.logI(TAG, "📊 输出张量统计:");
                //LogManager.logI(TAG, "  - 张量数量: " + inputs.size());
                
                // 打印张量维度信息
                for (Map.Entry<String, OnnxTensor> entry : inputs.entrySet()) {
                    long[] shape = entry.getValue().getInfo().getShape();
                    //LogManager.logI(TAG, "  - 张量 [" + entry.getKey() + "] 维度: " + java.util.Arrays.toString(shape));
                    //LogManager.logI(TAG, "  - 张量 [" + entry.getKey() + "] 元素总数: " + java.util.Arrays.stream(shape).reduce(1, (a, b) -> a * b));
                }
            } catch (Exception e) {
                LogManager.logE(TAG, "❌ TOKENIZATION失败: " + e.getMessage(), e);
                LogManager.logE(TAG, "失败的输入文本长度: " + inputText.length());
                LogManager.logE(TAG, "失败的输入文本预览: " + inputText.substring(0, Math.min(100, inputText.length())));
                continue;
            }
            
            try {
                // 运行推理
                long inferenceStartTime = System.currentTimeMillis();
                //LogManager.logI(TAG, "=== 开始模型推理阶段 ===");
                //LogManager.logI(TAG, "🧠 推理环境检查:");
                //LogManager.logI(TAG, "  - 当前线程: " + Thread.currentThread().getName());
                //LogManager.logI(TAG, "  - 会话状态: " + (session != null ? "有效" : "无效"));
                //LogManager.logI(TAG, "  - 输入张量数量: " + inputs.size());
                //LogManager.logI(TAG, "  - 系统内存: " + (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024 + "MB / " + Runtime.getRuntime().totalMemory() / 1024 / 1024 + "MB");
                
                // 关键的推理调用
                //LogManager.logI(TAG, "🚀 调用 session.run()...");
                LogManager.logI(TAG, "推理超时设置: 5分钟");
                
                // 使用Future来实现超时机制
                java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newSingleThreadExecutor();
                java.util.concurrent.Future<OrtSession.Result> future = executor.submit(() -> {
                    LogManager.logI(TAG, "⚡ 推理线程开始执行...");
                    //LogManager.logI(TAG, "推理线程ID: " + Thread.currentThread().getId());
                    OrtSession.Result result = session.run(inputs);
                    LogManager.logI(TAG, "✅ 推理线程执行完成");
                    return result;
                });
                
                OrtSession.Result output;
                try {
                    // 设置5ss分钟超时
                    output = future.get(300, java.util.concurrent.TimeUnit.SECONDS);
                    // LogManager.logI(TAG, "✅ session.run() 返回成功");
                } catch (java.util.concurrent.TimeoutException e) {
                    LogManager.logE(TAG, "⏰ 推理超时（5分钟），取消任务");
                    future.cancel(true);
                    executor.shutdownNow();
                    continue; // 跳过这个文档
                } catch (java.util.concurrent.ExecutionException e) {
                    LogManager.logE(TAG, "❌ 推理执行异常: " + e.getCause().getMessage(), e.getCause());
                    executor.shutdownNow();
                    continue;
                } catch (InterruptedException e) {
                    LogManager.logE(TAG, "🛑 推理被中断: " + e.getMessage());
                    Thread.currentThread().interrupt();
                    executor.shutdownNow();
                    continue;
                } finally {
                    executor.shutdown();
                }
                
                long inferenceTime = System.currentTimeMillis() - inferenceStartTime;
                // LogManager.logI(TAG, "✅ 模型推理完成，耗时: " + inferenceTime + "ms");
                
                // 获取logits
                long outputProcessStartTime = System.currentTimeMillis();
                //LogManager.logI(TAG, "=== 开始处理模型输出 ===");
                //LogManager.logI(TAG, "📤 模型输出分析:");
                //LogManager.logI(TAG, "  - 输出对象类型: " + output.getClass().getName());
                //LogManager.logI(TAG, "  - 输出键数量: " + output.size());
                
                // 列出所有输出
                //LogManager.logI(TAG, "📋 输出张量信息:");
                //LogManager.logI(TAG, "  - 输出张量数量: " + output.size());
                for (int j = 0; j < output.size(); j++) {
                    try {
                        OnnxTensor tensor = (OnnxTensor) output.get(j);
                        //LogManager.logI(TAG, "  - 张量[" + j + "] 形状: " + java.util.Arrays.toString(tensor.getInfo().getShape()));
                    } catch (Exception e) {
                        LogManager.logI(TAG, "  - 张量[" + j + "] 获取失败: " + e.getMessage());
                    }
                }
                
                //LogManager.logI(TAG, "🎯 尝试获取logits张量 (索引: 0)...");
                OnnxTensor logitsTensor = null;
                try {
                    logitsTensor = (OnnxTensor) output.get(0);
                } catch (Exception e) {
                    LogManager.logE(TAG, "❌ 获取logits张量失败: " + e.getMessage());
                }
                
                if (logitsTensor != null) {
                    //LogManager.logI(TAG, "✅ 成功获取logits张量");
                    Object logitsValue = logitsTensor.getValue();
                    //LogManager.logI(TAG, "📊 Logits详细信息:");
                    //LogManager.logI(TAG, "  - 数据类型: " + logitsValue.getClass().getName());
                    //LogManager.logI(TAG, "  - 张量形状: " + java.util.Arrays.toString(logitsTensor.getInfo().getShape()));
                    
                    float score;
                    if (logitsValue instanceof float[][][]) {
                        // 三维输出：[batch_size, sequence_length, num_classes]
                        float[][][] logits3D = (float[][][]) logitsValue;
                        //LogManager.logI(TAG, "🔢 检测到三维logits输出");
                        //LogManager.logI(TAG, "  - 维度: [" + logits3D.length + ", " + 
                        //               (logits3D.length > 0 ? logits3D[0].length : 0) + ", " + 
                        //               (logits3D.length > 0 && logits3D[0].length > 0 ? logits3D[0][0].length : 0) + "]");
                        //LogManager.logI(TAG, "  - 解释: [batch_size, sequence_length, num_classes]");
                        
                        // 通常取第一个batch的第一个token的logits
                        if (logits3D.length > 0 && logits3D[0].length > 0) {
                            //LogManager.logI(TAG, "📈 使用第一个batch的第一个token计算得分");
                            //LogManager.logI(TAG, "  - 原始logits长度: " + logits3D[0][0].length);
                            if (logits3D[0][0].length > 0) {
                                //LogManager.logI(TAG, "  - 原始logits前5个值: " + java.util.Arrays.toString(java.util.Arrays.copyOf(logits3D[0][0], Math.min(5, logits3D[0][0].length))));
                            }
                            score = calculateRelevanceScore(logits3D[0][0]);
                        } else {
                            LogManager.logW(TAG, "⚠️ 三维logits数组为空");
                            score = 0.5f;
                        }
                    } else if (logitsValue instanceof float[][]) {
                        // 二维输出：[batch_size, num_classes]
                        float[][] logits2D = (float[][]) logitsValue;
                        //LogManager.logI(TAG, "🔢 检测到二维logits输出");
                        //LogManager.logI(TAG, "  - 维度: [" + logits2D.length + ", " + 
                        //               (logits2D.length > 0 ? logits2D[0].length : 0) + "]");
                        //LogManager.logI(TAG, "  - 解释: [batch_size, num_classes]");
                        
                        if (logits2D.length > 0) {
                            //LogManager.logI(TAG, "📈 使用第一个batch计算得分");
                            //LogManager.logI(TAG, "  - 原始logits长度: " + logits2D[0].length);
                            if (logits2D[0].length > 0) {
                                //LogManager.logI(TAG, "  - 原始logits前5个值: " + java.util.Arrays.toString(java.util.Arrays.copyOf(logits2D[0], Math.min(5, logits2D[0].length))));
                            }
                            score = calculateRelevanceScore(logits2D[0]);
                        } else {
                            LogManager.logW(TAG, "⚠️ 二维logits数组为空");
                            score = 0.5f;
                        }
                    } else {
                        LogManager.logE(TAG, "❌ 不支持的logits输出类型: " + logitsValue.getClass().getName());
                        score = 0.5f;
                    }
                    
                    //LogManager.logI(TAG, "🎯 计算得到的相关性得分: " + score);
                    results.add(new RerankResult(document, score, startIndex + i));
                    
                    // 调用进度回调
                    if (progressCallback != null) {
                        progressCallback.onRerankProgress(i + 1, documents.size(), score);
                    }
                    
                    long outputProcessTime = System.currentTimeMillis() - outputProcessStartTime;
                    LogManager.logI(TAG, "✅ 文档 " + globalIndex + " 处理完成，得分: " + score + ", 输出处理耗时: " + outputProcessTime + "ms");
                } else {
                    LogManager.logE(TAG, "❌ 无法获取logits输出张量");
                    LogManager.logE(TAG, "可能的原因:");
                    LogManager.logE(TAG, "  1. 输出键名不匹配 (期望: " + OUTPUT_LOGITS + ")");
                    LogManager.logE(TAG, "  2. 模型输出格式不符合预期");
                    LogManager.logE(TAG, "  3. 推理过程中出现错误");
                }
                
                // 清理输出
                output.close();
                
                long totalDocTime = System.currentTimeMillis() - docStartTime;
                //LogManager.logI(TAG, "=== 文档 " + globalIndex + " 总耗时: " + totalDocTime + "ms ===");
                
            } finally {
                // 清理输入张量
                for (OnnxTensor tensor : inputs.values()) {
                    tensor.close();
                }
            }
        }
        
        return results;
    }
    
    /**
     * 使用TokenizerManager进行tokenization
     * 如果TokenizerManager不可用，直接抛出异常
     */
    private Map<String, OnnxTensor> tokenizeInput(String text) throws OrtException {
        long tokenizeStartTime = System.currentTimeMillis();
        
        // 检查TokenizerManager状态
        if (tokenizerManager == null || !tokenizerManager.isInitialized()) {
            throw new IllegalStateException("TokenizerManager未初始化或不可用，无法进行分词处理");
        }
        
        LogManager.logI(TAG, "✅ TokenizerManager已初始化，使用专业分词器");
        return tokenizeWithTokenizerManager(text, tokenizeStartTime);
    }
    
    /**
     * 使用TokenizerManager进行tokenization
     */
    private Map<String, OnnxTensor> tokenizeWithTokenizerManager(String text, long startTime) throws OrtException {
        try {
            LogManager.logI(TAG, "🔧 === 使用TokenizerManager进行专业分词 ===");
            
            // 使用TokenizerManager进行tokenization
            long tokenizeTime = System.currentTimeMillis();
            //LogManager.logI(TAG, "📞 调用TokenizerManager.tokenize()...");
            //LogManager.logI(TAG, "  - 输入文本长度: " + text.length());
            //LogManager.logI(TAG, "  - TokenizerManager实例: " + tokenizerManager.getClass().getSimpleName());
            
            long[][] tokenIds = tokenizerManager.tokenize(text);
            long tokenizeDuration = System.currentTimeMillis() - tokenizeTime;
            
            //LogManager.logI(TAG, "✅ TokenizerManager分词完成");
            //LogManager.logI(TAG, "📊 分词结果统计:");
            //LogManager.logI(TAG, "  - 分词耗时: " + tokenizeDuration + "ms");
            //LogManager.logI(TAG, "  - Token序列维度: [" + tokenIds.length + ", " + tokenIds[0].length + "]");
            //LogManager.logI(TAG, "  - 生成token数量: " + tokenIds[0].length);
            //LogManager.logI(TAG, "  - 压缩比: " + String.format("%.2f", (double)text.length() / tokenIds[0].length) + " 字符/token");
            //LogManager.logI(TAG, "  - 前10个token ID: " + java.util.Arrays.toString(
            //    java.util.Arrays.copyOf(tokenIds[0], Math.min(10, tokenIds[0].length))));
            if (tokenIds[0].length > 10) {
                //LogManager.logI(TAG, "  - 后5个token ID: " + java.util.Arrays.toString(
                //    java.util.Arrays.copyOfRange(tokenIds[0], Math.max(0, tokenIds[0].length - 5), tokenIds[0].length)));
            }
            
            // 处理序列长度
            //LogManager.logI(TAG, "🔍 检查序列长度限制...");
            int actualLength = tokenIds[0].length;
            if (actualLength > MAX_SEQUENCE_LENGTH) {
                LogManager.logW(TAG, "⚠️ 序列长度超出限制！");
                LogManager.logW(TAG, "  - 原始长度: " + actualLength);
                LogManager.logW(TAG, "  - 最大允许: " + MAX_SEQUENCE_LENGTH);
                LogManager.logW(TAG, "  - 执行截断操作...");
                long[][] truncatedTokenIds = new long[1][MAX_SEQUENCE_LENGTH];
                System.arraycopy(tokenIds[0], 0, truncatedTokenIds[0], 0, MAX_SEQUENCE_LENGTH);
                tokenIds = truncatedTokenIds;
                actualLength = MAX_SEQUENCE_LENGTH;
                LogManager.logI(TAG, "✅ 截断完成，新长度: " + actualLength);
            } else {
                LogManager.logI(TAG, "✅ 序列长度符合要求: " + actualLength + " <= " + MAX_SEQUENCE_LENGTH);
            }
            
            // 创建attention mask和token type ids
            long tensorCreateTime = System.currentTimeMillis();
            //LogManager.logI(TAG, "🎭 创建attention mask和token type ids...");
            
            long[][] attentionMask = new long[1][actualLength];
            long[][] tokenTypeIds = new long[1][actualLength];
            
            // 填充attention mask（所有位置都是1，表示有效token）
            for (int i = 0; i < actualLength; i++) {
                attentionMask[0][i] = 1;
                tokenTypeIds[0][i] = 0; // 对于标记输入格式，可能需要根据[Q]、[SEP]、[D]位置设置不同的token type
            }
            
            //LogManager.logI(TAG, "  - attention mask长度: " + actualLength);
            //LogManager.logI(TAG, "  - 所有值设为: 1 (表示所有token都需要注意)");
            //LogManager.logI(TAG, "  - token type IDs长度: " + actualLength);
            //LogManager.logI(TAG, "  - 所有值设为: 0 (表示单一句子类型)");
            
            // 创建ONNX张量
            LogManager.logI(TAG, "🔄 转换为ONNX张量...");
            Map<String, OnnxTensor> inputs = new HashMap<>();
            
            //LogManager.logI(TAG, "  - 创建input_ids张量: [1, " + actualLength + "]");
            inputs.put(INPUT_IDS, OnnxTensor.createTensor(environment, tokenIds));
            
            //LogManager.logI(TAG, "  - 创建attention_mask张量: [1, " + actualLength + "]");
            inputs.put(ATTENTION_MASK, OnnxTensor.createTensor(environment, attentionMask));
            
            //LogManager.logI(TAG, "  - 创建token_type_ids张量: [1, " + actualLength + "]");
            inputs.put(TOKEN_TYPE_IDS, OnnxTensor.createTensor(environment, tokenTypeIds));
            
            long tensorCreateDuration = System.currentTimeMillis() - tensorCreateTime;
            long totalDuration = System.currentTimeMillis() - startTime;
            
            //LogManager.logI(TAG, "✅ TokenizerManager分词流程完成");
            //LogManager.logI(TAG, "📈 性能统计:");
            //LogManager.logI(TAG, "  - 总耗时: " + totalDuration + "ms");
            //LogManager.logI(TAG, "  - 纯分词耗时: " + tokenizeDuration + "ms");
            //LogManager.logI(TAG, "  - 张量创建耗时: " + tensorCreateDuration + "ms");
            //LogManager.logI(TAG, "  - 输出张量数量: " + inputs.size());
            
            return inputs;
            
        } catch (Exception e) {
            LogManager.logE(TAG, "❌ TokenizerManager tokenization失败: " + e.getMessage(), e);
            throw new OrtException("分词处理失败: " + e.getMessage());
        }
    }
    

    
    /**
     * 计算相关性分数
     */
    private float calculateRelevanceScore(float[] logits) {
        LogManager.logI(TAG, "🧮 === 开始计算相关性得分 ===");
        // LogManager.logI(TAG, "📊 输入logits分析:");
        
        if (logits == null || logits.length == 0) {
            LogManager.logW(TAG, "⚠️ logits为空，返回默认得分 0.5");
            return 0.5f;
        }
        
        LogManager.logI(TAG, "  - logits长度: " + logits.length);
        // LogManager.logI(TAG, "  - 所有logits值: " + java.util.Arrays.toString(logits));
        
        // 对于二分类任务，通常使用sigmoid激活函数
        // 如果logits只有一个值，直接应用sigmoid
        if (logits.length == 1) {
            LogManager.logI(TAG, "🔢 检测到单值logits模式");
            LogManager.logI(TAG, "  - 原始logit值: " + logits[0]);
            float score = sigmoid(logits[0]);
            LogManager.logI(TAG, "  - 应用sigmoid函数: sigmoid(" + logits[0] + ") = " + score);
            LogManager.logI(TAG, "✅ 最终得分: " + score);
            return score;
        }
        
        // 如果有多个logits值，可能需要softmax或其他处理
        // 这里假设第二个值是正类的logits（根据具体模型调整）
        if (logits.length >= 2) {
            LogManager.logI(TAG, "🔢 检测到多值logits模式");
            LogManager.logI(TAG, "  - 负类logit (index 0): " + logits[0]);
            LogManager.logI(TAG, "  - 正类logit (index 1): " + logits[1]);
            
            float positiveLogit = logits[1];
            LogManager.logI(TAG, "  - 使用正类logit计算得分: " + positiveLogit);
            float score = sigmoid(positiveLogit);
            LogManager.logI(TAG, "  - 应用sigmoid函数: sigmoid(" + positiveLogit + ") = " + score);
            
            // 也计算softmax作为参考
            float[] softmaxScores = softmax(new float[]{logits[0], logits[1]});
            LogManager.logI(TAG, "  - 参考softmax得分: [" + softmaxScores[0] + ", " + softmaxScores[1] + "]");
            LogManager.logI(TAG, "  - softmax正类概率: " + softmaxScores[1]);
            
            LogManager.logI(TAG, "✅ 最终得分: " + score);
            return score;
        }
        
        LogManager.logW(TAG, "⚠️ 未知的logits格式，返回默认得分 0.5");
        return 0.5f;
    }
    
    /**
     * Sigmoid激活函数
     */
    private float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }
    
    /**
     * Softmax函数
     */
    private float[] softmax(float[] logits) {
        float[] result = new float[logits.length];
        float sum = 0.0f;
        
        // 计算exp值
        for (int i = 0; i < logits.length; i++) {
            result[i] = (float) Math.exp(logits[i]);
            sum += result[i];
        }
        
        // 归一化
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    /**
     * 将文档列表转换为RerankResult列表（保持原始顺序）
     */
    private List<RerankResult> convertToRerankResults(List<String> documents) {
        List<RerankResult> results = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) {
            // 使用递减分数保持原始顺序
            float score = 1.0f - (i * 0.01f);
            results.add(new RerankResult(documents.get(i), score, i));
        }
        return results;
    }
    
    /**
     * 检查模型是否已初始化
     */
    public boolean isInitialized() {
        return isInitialized.get();
    }
    
    /**
     * 获取模型路径
     */
    public String getModelPath() {
        return modelPath;
    }
    
    /**
     * 检查ONNX会话状态并尝试恢复
     * @return 会话是否可用
     */
    private boolean checkAndRecoverSession() {
        // 记录当前线程信息
        //LogManager.logD(TAG, "检查会话状态 [线程ID: " + Thread.currentThread().getId() + "]");
        
        synchronized (sessionLock) {
            // 记录上次检查时间
            lastSessionCheckTime = System.currentTimeMillis();
            
            // 检查会话是否可用
            if (session != null && sessionState == SESSION_STATE_READY) {
                //LogManager.logD(TAG, "会话状态正常，可以使用");
                return true;
            }
            
            // 如果会话正在加载中，等待一段时间
            if (sessionState == SESSION_STATE_LOADING) {
                LogManager.logD(TAG, "会话正在加载中，等待...");
                try {
                    // 等待最多3秒
                    for (int i = 0; i < 30; i++) {
                        // 每100毫秒检查一次
                        Thread.sleep(100);
                        if (session != null && sessionState == SESSION_STATE_READY) {
                            LogManager.logD(TAG, "会话已加载完成，可以使用");
                            return true;
                        }
                    }
                } catch (InterruptedException e) {
                    LogManager.logE(TAG, "等待会话加载被中断: " + e.getMessage());
                    Thread.currentThread().interrupt();
                }
            }
            
            // 如果会话不可用或处于错误状态，尝试恢复
            if (session == null || sessionState == SESSION_STATE_ERROR) {
                // 检查重试次数
                if (sessionRetryCount >= MAX_SESSION_RETRY) {
                    LogManager.logE(TAG, "会话恢复失败，已达到最大重试次数: " + sessionRetryCount);
                    return false;
                }
                
                LogManager.logD(TAG, "尝试恢复会话，当前重试次数: " + sessionRetryCount);
                
                // 更新会话状态
                int oldState = sessionState;
                sessionState = SESSION_STATE_LOADING;
                LogManager.logD(TAG, "会话状态变更: " + oldState + " -> " + sessionState + " (开始恢复)");
                
                // 增加重试计数
                sessionRetryCount++;
                
                try {
                    // 先关闭之前的会话
                    if (session != null) {
                        try {
                            session.close();
                            LogManager.logD(TAG, "已关闭旧的ONNX会话");
                        } catch (Exception e) {
                            LogManager.logE(TAG, "关闭旧的ONNX会话失败: " + e.getMessage(), e);
                        } finally {
                            session = null;
                        }
                    }
                    
                    // 重新初始化模型
                    try {
                        LogManager.logD(TAG, "重新初始化重排模型: " + modelPath);
                        
                        // 重置状态
                        isInitialized.set(false);
                        isLoading.set(false);
                        
                        // 调用初始化方法（但要避免无限递归）
                        boolean success = initializeInternal();
                        
                        if (success && session != null) {
                            LogManager.logD(TAG, "重排模型会话恢复成功");
                            sessionState = SESSION_STATE_READY;
                            return true;
                        } else {
                            LogManager.logE(TAG, "重排模型会话恢复失败，会话为null");
                            sessionState = SESSION_STATE_ERROR;
                            return false;
                        }
                    } catch (Exception e) {
                        LogManager.logE(TAG, "重新初始化重排模型失败: " + e.getMessage(), e);
                        sessionState = SESSION_STATE_ERROR;
                        return false;
                    }
                } finally {
                    // 如果会话仍然为null，确保状态为ERROR
                    if (session == null && sessionState != SESSION_STATE_ERROR) {
                        LogManager.logE(TAG, "会话为null但状态不是ERROR，更正状态");
                        sessionState = SESSION_STATE_ERROR;
                    }
                    
                    // 记录会话恢复结果
                    boolean success = session != null && sessionState == SESSION_STATE_READY;
                    LogManager.logD(TAG, "会话恢复" + (success ? "成功" : "失败") + 
                           "，最终状态: " + sessionState + 
                           "，重试次数: " + sessionRetryCount + "/" + MAX_SESSION_RETRY);
                }
            }
            
            // 会话仍然不可用
            LogManager.logE(TAG, "会话检查结束，会话仍然不可用，状态: " + sessionState);
            return false;
        }
    }
    
    /**
     * 内部初始化方法，避免递归调用
     */
    private boolean initializeInternal() {
        // 这里实现简化的初始化逻辑，避免调用checkAndRecoverSession
        try {
            // 检查模型文件是否存在
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                LogManager.logE(TAG, "重排模型文件不存在: " + modelPath);
                return false;
            }
            
            // 创建ONNX环境
            if (environment == null) {
                environment = OrtEnvironment.getEnvironment();
            }
            
            // 创建会话选项
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            int threadCount = ConfigManager.getThreads(context);
            sessionOptions.setIntraOpNumThreads(threadCount);
            sessionOptions.setInterOpNumThreads(threadCount);
            sessionOptions.setMemoryPatternOptimization(true);
            sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            
            // 加载模型
            session = environment.createSession(modelPath, sessionOptions);
            
            if (session != null) {
                isInitialized.set(true);
                return true;
            }
            
            return false;
        } catch (Exception e) {
            LogManager.logE(TAG, "内部初始化失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 清理资源
     */
    public void cleanup() {
        synchronized (sessionLock) {
            try {
                if (session != null) {
                    session.close();
                    session = null;
                }
                if (environment != null) {
                    environment.close();
                    environment = null;
                }
                isInitialized.set(false);
                sessionState = SESSION_STATE_NONE;
                sessionRetryCount = 0;
                lastSessionCheckTime = 0;
                LogManager.logI(TAG, "重排模型资源已清理");
            } catch (Exception e) {
                LogManager.logE(TAG, "清理重排模型资源时发生错误: " + e.getMessage(), e);
            }
        }
    }
    
    @Override
    protected void finalize() throws Throwable {
        cleanup();
        super.finalize();
    }
}