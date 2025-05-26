package com.example.starlocalrag;

import com.example.starlocalrag.api.TokenizerManager;
import android.content.Context;
import android.net.Uri;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 文本块处理器
 * 负责文本提取和向量化的两阶段处理
 */
public class TextChunkProcessor {
    private static final String TAG = "StarLocalRAG_TextChunk";
    
    // 最小文本块大小，从ConfigManager中获取
    private int minChunkSize;
    
    // 上下文
    private final Context context;
    
    // 文档解析器
    private final DocumentParser documentParser;
    
    // 任务取消标志
    private final AtomicBoolean isTaskCancelled;
    
    // 进度回调
    private ProgressCallback progressCallback;
    
    // 通知进度回调
    private NotificationProgressCallback notificationProgressCallback;
    
    // 中间文件名
    private static final String INTERMEDIATE_FILE_NAME = "intermediate_chunks.json";
    
    /**
     * 文本块类
     */
    public static class TextChunk {
        public String text;
        public String source;
        public int chunkIndex;
        public JSONObject metadata;
        
        public TextChunk(String text, String source, int chunkIndex, JSONObject metadata) {
            this.text = text;
            this.source = source;
            this.chunkIndex = chunkIndex;
            this.metadata = metadata;
        }
    }
    
    /**
     * 进度回调接口
     */
    public interface ProgressCallback {
        void onTextExtractionProgress(int processedFiles, int totalFiles, String currentFile);
        void onVectorizationProgress(int processedChunks, int totalChunks, float percentage);
        void onTextExtractionComplete(int totalChunks);
        void onVectorizationComplete(int totalVectors);
        void onError(String errorMessage);
        void onLog(String message);
    }
    
    /**
     * 通知进度回调接口
     */
    public interface NotificationProgressCallback {
        void onNotificationProgressUpdate(int processedChunks, int totalChunks, float percentage);
    }
    
    /**
     * 构造函数
     * @param context 上下文
     */
    public TextChunkProcessor(Context context) {
        this.context = context;
        this.minChunkSize = ConfigManager.getMinChunkSize(context);
        this.isTaskCancelled = new AtomicBoolean(false);
        this.documentParser = new DocumentParser(context);
    }
    
    /**
     * 构造函数
     * @param context 上下文
     */
    public TextChunkProcessor(Context context, AtomicBoolean isTaskCancelled) {
        this.context = context;
        this.minChunkSize = ConfigManager.getMinChunkSize(context);
        this.isTaskCancelled = isTaskCancelled;
        this.documentParser = new DocumentParser(context);
    }
    
    /**
     * 设置进度回调
     */
    public void setProgressCallback(ProgressCallback callback) {
        this.progressCallback = callback;
    }
    
    /**
     * 设置通知进度回调
     */
    public void setNotificationProgressCallback(NotificationProgressCallback callback) {
        this.notificationProgressCallback = callback;
    }
    
    /**
     * 处理文件列表
     * @param knowledgeBasePath 知识库路径
     * @param files 文件列表
     * @param chunkSize 分块大小
     * @param chunkOverlap 分块重叠大小
     * @param embeddingModel 词嵌入模型
     * @param vectorDB 向量数据库
     * @return 是否成功
     */
    public boolean processFiles(String knowledgeBasePath, List<Uri> files, int chunkSize, int chunkOverlap, 
                               EmbeddingModelHandler embeddingModel, SQLiteVectorDatabaseHandler vectorDB) {
        try {
            // 第一阶段：提取文本并分块
            List<TextChunk> chunks = extractTextFromFiles(knowledgeBasePath, files, chunkSize, chunkOverlap);
            if (chunks == null || chunks.isEmpty()) {
                logMessage("未能提取任何文本块");
                return false;
            }
            
            // 保存中间结果
            saveIntermediateChunks(knowledgeBasePath, chunks);
            
            // 通知文本提取完成
            if (progressCallback != null) {
                progressCallback.onTextExtractionComplete(chunks.size());
            }
            
            // 检查任务是否取消
            if (isTaskCancelled.get()) {
                logMessage("任务已取消");
                return false;
            }
            
            // 第二阶段：向量化
            boolean success = processChunksToVectors(chunks, embeddingModel, vectorDB);
            
            // 删除中间文件
            deleteIntermediateFile(knowledgeBasePath);
            
            return success;
        } catch (Exception e) {
            logError("处理文件失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 从文件中提取文本并分块
     */
    private List<TextChunk> extractTextFromFiles(String knowledgeBasePath, List<Uri> files, int chunkSize, int chunkOverlap) {
        List<TextChunk> allChunks = new ArrayList<>();
        int totalFiles = files.size();
        AtomicInteger processedFiles = new AtomicInteger(0);
        AtomicInteger totalChunks = new AtomicInteger(0);
        
        try {
            for (int i = 0; i < totalFiles; i++) {
                // 检查任务是否取消
                if (isTaskCancelled.get()) {
                    logMessage("文本提取已取消");
                    return allChunks;
                }
                
                Uri fileUri = files.get(i);
                String fileName = UriUtils.getFileName(context, fileUri);
                
                // 更新进度
                int currentProcessed = processedFiles.incrementAndGet();
                if (progressCallback != null) {
                    //progressCallback.onTextExtractionProgress(currentProcessed, totalFiles, fileName);
                }
                
                try {
                    // 提取文本
                    String text = documentParser.extractText(fileUri);
                    if (text == null || text.trim().isEmpty()) {
                        logMessage("警告: 文件 " + fileName + " 未能提取到文本");
                        continue;
                    }
                    
                    // 记录文件大小
                    int textLength = text.length();
                    logMessage("文件: " + fileName + " 提取的文本大小: " + (textLength / 1024) + "KB");
                    
                    // 检查是否是JSON内容
                    boolean isJson = false;
                    boolean isSpecialDataset = false;
                    try {
                        // 首先检查文件名是否包含json
                        boolean fileNameIndicatesJson = fileName.toLowerCase().endsWith(".json");
                        
                        // 检查是否为特定数据集
                        if (fileNameIndicatesJson && (
                            fileName.contains("datasets-sb") || 
                            fileName.contains("alpaca") || 
                            fileName.contains("STAR") || 
                            fileName.contains("star"))) {
                            isSpecialDataset = true;
                            logMessage("检测到特定数据集: " + fileName + "，将忽略最小块大小限制");
                        }
                        
                        if (fileNameIndicatesJson) {
                            logMessage("文件名表明这可能是JSON文件: " + fileName);
                            // 对于JSON文件，使用更严格的检测
                            isJson = JsonDatasetProcessor.isJsonContent(text);
                        } else {
                            // 对于非JSON文件，只有当内容明确是JSON格式时才识别为JSON
                            String trimmedText = text.trim();
                            boolean looksLikeJson = (trimmedText.startsWith("{") && trimmedText.endsWith("}")) || 
                                                  (trimmedText.startsWith("[") && trimmedText.endsWith("]"));
                            
                            if (looksLikeJson) {
                                isJson = JsonDatasetProcessor.isJsonContent(text);
                            }
                        }
                    } catch (Exception e) {
                        logError("检查JSON内容时出错: " + e.getMessage(), e);
                    }
                    
                    boolean jsonOptimizationEnabled = ConfigManager.isJsonDatasetSplittingEnabled(context);
                    
                    // 明确显示JSON格式识别结果
                    String jsonStatusMessage = "文件: " + fileName + (isJson ? " 是JSON格式" : " 不是JSON格式") + 
                                              (isSpecialDataset ? " (特定数据集，忽略最小块大小限制)" : "");
                    logMessage(jsonStatusMessage);
                    
                    if (isJson) {
                        String configStatusMessage = "JSON优化配置状态: " + (jsonOptimizationEnabled ? "已启用" : "已禁用");
                        logMessage(configStatusMessage);
                        
                        if (jsonOptimizationEnabled) {
                            // 尝试识别JSON格式
                            try {
                                // 获取JSON内容的前100个字符作为预览
                                String jsonPreview = text.length() > 100 ? text.substring(0, 100) + "..." : text;
                                logMessage("JSON内容预览: " + jsonPreview);
                                
                                // 使用JsonDatasetProcessor处理JSON内容
                                logMessage("开始使用JsonDatasetProcessor处理JSON内容...");
                                
                                // 确保没有异常阻止处理
                                List<String> jsonChunks = new ArrayList<>();
                                try {
                                    // 对于大型JSON文件，可能需要特殊处理
                                    if (text.length() > 1000000) { // 1MB以上的文件
                                        logMessage("检测到大型JSON文件 (" + (text.length() / 1024 / 1024) + "MB)，使用分段处理");
                                        
                                        // 尝试修复可能的JSON格式问题
                                        String processedText = text;
                                        if (text.trim().startsWith("[") && !text.trim().endsWith("]")) {
                                            processedText = text.trim() + "]";
                                            logMessage("JSON文件似乎不完整，尝试添加结束括号进行修复");
                                        }
                                        
                                        // 使用特定数据集处理标志
                                        jsonChunks = JsonDatasetProcessor.processJsonDataset(context, processedText, isSpecialDataset);
                                    } else {
                                        // 使用特定数据集处理标志
                                        jsonChunks = JsonDatasetProcessor.processJsonDataset(context, text, isSpecialDataset);
                                    }
                                    logMessage("JSON处理完成，返回了 " + jsonChunks.size() + " 个文本块");
                                } catch (Exception e) {
                                    logError("JSON处理过程中出错: " + e.getMessage(), e);
                                    // 尝试使用更宽容的方式处理JSON
                                    try {
                                        logMessage("尝试使用备用方法处理JSON...");
                                        // 检查是否为Alpaca格式
                                        if (text.contains("\"instruction\"") && 
                                            (text.contains("\"output\"") || text.contains("\"response\""))) {
                                            logMessage("检测到可能是Alpaca格式，尝试手动解析");
                                            
                                            // 简单的手动解析，提取instruction和output/response对
                                            String[] lines = text.split("\\n");
                                            StringBuilder currentItem = new StringBuilder();
                                            for (String line : lines) {
                                                line = line.trim();
                                                if (line.contains("\"instruction\"")) {
                                                    if (currentItem.length() > 0) {
                                                        // 处理上一个项目
                                                        String itemText = currentItem.toString();
                                                        if (itemText.length() >= minChunkSize) {
                                                            jsonChunks.add(itemText);
                                                        }
                                                        currentItem = new StringBuilder();
                                                    }
                                                    // 开始新项目
                                                    currentItem.append("指令: ").append(extractValue(line)).append("\n\n");
                                                } else if (line.contains("\"output\"") || line.contains("\"response\"")) {
                                                    currentItem.append("输出: ").append(extractValue(line));
                                                }
                                            }
                                            
                                            // 处理最后一个项目
                                            if (currentItem.length() > 0) {
                                                String itemText = currentItem.toString();
                                                if (itemText.length() >= minChunkSize) {
                                                    jsonChunks.add(itemText);
                                                }
                                            }
                                            
                                            logMessage("手动解析完成，提取了 " + jsonChunks.size() + " 个文本块");
                                        }
                                    } catch (Exception ex) {
                                        logError("备用JSON处理方法也失败: " + ex.getMessage(), ex);
                                    }
                                }
                                
                                if (!jsonChunks.isEmpty()) {
                                    logMessage("JSON处理成功，应用了优化，生成了 " + jsonChunks.size() + " 个文本块");
                                    
                                    // 创建文本块对象
                                    for (int j = 0; j < jsonChunks.size(); j++) {
                                        String chunkText = jsonChunks.get(j);
                                        
                                        // 注意：不需要在这里检查块大小是否合理
                                        // LangChainTextSplitter已经根据minChunkSize过滤了过小的文本块
                                        
                                        // 创建元数据
                                        JSONObject metadata = new JSONObject();
                                        try {
                                            metadata.put("fileName", fileName);
                                            metadata.put("fileIndex", i);
                                            metadata.put("chunkIndex", j);
                                            metadata.put("totalChunks", jsonChunks.size());
                                            metadata.put("extractionTime", System.currentTimeMillis());
                                            metadata.put("processingMethod", "JsonOptimized");
                                        } catch (JSONException e) {
                                            logError("创建元数据失败: " + e.getMessage(), e);
                                        }
                                        
                                        // 添加到文本块列表
                                        TextChunk chunk = new TextChunk(chunkText, fileName, j, metadata);
                                        allChunks.add(chunk);
                                    }
                                    
                                    // 为每个文件打印分块数量
                                    String fileProcessingSummary = "已处理JSON文件: " + fileName + ", 使用优化方式提取了 " + jsonChunks.size() + " 个文本块";
                                    logMessage(fileProcessingSummary);
                                    
                                    // 如果是进度回调，确保这个信息显示在UI上
                                    if (progressCallback != null) {
                                        progressCallback.onLog("文件: " + fileName + " -> JSON优化分块数量: " + jsonChunks.size());
                                    }
                                    
                                    totalChunks.addAndGet(jsonChunks.size());
                                    continue; // 跳过标准分块处理
                                } else {
                                    logMessage("JSON处理未生成任何文本块，将回退到标准分块处理");
                                }
                            } catch (Exception e) {
                                logError("JSON处理失败: " + e.getMessage() + "，将回退到标准分块处理", e);
                            }
                        } else {
                            logMessage("JSON优化已禁用，将使用标准分块处理");
                        }
                    } else {
                        logMessage("JSON优化已禁用，将使用标准分块处理");
                    }
                    
                    // 处理文本分块
                    List<String> chunks;
                    int fileChunkCount = 0;
                    
                    if (isJson && jsonOptimizationEnabled) {
                        // 使用JSON处理逻辑
                        List<String> jsonChunks = new ArrayList<>();
                        try {
                            // 对于大型JSON文件，可能需要特殊处理
                            if (text.length() > 1000000) { // 1MB以上的文件
                                logMessage("检测到大型JSON文件 (" + (text.length() / 1024 / 1024) + "MB)，使用分段处理");
                                
                                // 尝试修复可能的JSON格式问题
                                String processedText = text;
                                if (text.trim().startsWith("[") && !text.trim().endsWith("]")) {
                                    processedText = text.trim() + "]";
                                    logMessage("JSON文件似乎不完整，尝试添加结束括号进行修复");
                                }
                                
                                // 使用特定数据集处理标志
                                jsonChunks = JsonDatasetProcessor.processJsonDataset(context, processedText, isSpecialDataset);
                            } else {
                                // 使用特定数据集处理标志
                                jsonChunks = JsonDatasetProcessor.processJsonDataset(context, text, isSpecialDataset);
                            }
                            logMessage("JSON处理完成，返回了 " + jsonChunks.size() + " 个文本块");
                        } catch (Exception e) {
                            logError("JSON处理过程中出错: " + e.getMessage(), e);
                            // 如果JSON处理失败，使用标准分块处理
                            logMessage("JSON处理失败，回退到标准分块处理");
                            jsonChunks = splitTextIntoChunks(text, chunkSize, chunkOverlap);
                        }
                        
                        chunks = jsonChunks;
                        fileChunkCount = jsonChunks.size();
                    } else {
                        // 使用标准分块处理
                        logMessage("使用标准分块处理，分块大小: " + chunkSize + ", 重叠大小: " + chunkOverlap);
                        chunks = splitTextIntoChunks(text, chunkSize, chunkOverlap);
                        fileChunkCount = chunks.size();
                    }
                    
                    // 记录该文件生成的文本块数量
                    logMessage("文件: " + fileName + " 生成了 " + fileChunkCount + " 个文本块");
                    
                    // 添加到总块列表
                    int chunkIndex = 0;
                    for (String chunk : chunks) {
                        JSONObject metadata = new JSONObject();
                        try {
                            metadata.put("source", fileName);
                            metadata.put("chunkIndex", chunkIndex++);
                            metadata.put("extractionTime", System.currentTimeMillis());
                        } catch (JSONException e) {
                            logError("创建元数据时出错: " + e.getMessage(), e);
                        }
                        
                        allChunks.add(new TextChunk(chunk, fileName, chunkIndex - 1, metadata));
                        totalChunks.incrementAndGet();
                    }
                    
                    // 更新进度回调，包含当前文件的文本块数量
                    if (progressCallback != null) {
                        progressCallback.onTextExtractionProgress(currentProcessed, totalFiles, fileName + " (生成了 " + fileChunkCount + " 个文本块)");
                    }
                    
                } catch (Exception e) {
                    logError("处理文件失败: " + fileName + ", 错误: " + e.getMessage(), e);
                }
            }
            
            logMessage("文本提取完成，共 " + totalChunks.get() + " 个文本块");
            return allChunks;
        } catch (Exception e) {
            logError("文本提取过程失败: " + e.getMessage(), e);
            return allChunks;
        }
    }
    
    /**
     * 将文本分割成块
     * @param text 要分割的文本
     * @param chunkSize 块大小
     * @param chunkOverlap 块重叠大小
     * @return 分割后的文本块列表
     */
    private List<String> splitTextIntoChunks(String text, int chunkSize, int chunkOverlap) {
        if (text == null || text.isEmpty()) {
            return new ArrayList<>();
        }
        
        // 使用LangChainTextSplitter进行文本分割，确保与PC端一致
        Log.d(TAG, "使用LangChainTextSplitter进行文本分割");
        // 从ConfigManager获取最小分块大小，而不是使用硬编码值
        LangChainTextSplitter splitter = new LangChainTextSplitter(chunkSize, chunkOverlap, minChunkSize);
        List<String> chunks = splitter.splitText(text);
        
        Log.d(TAG, "文本分割完成，生成了" + chunks.size() + "个文本块");
        
        return chunks;
    }
    
    /**
     * 将文本块向量化并存入数据库
     */
    private boolean processChunksToVectors(List<TextChunk> chunks, EmbeddingModelHandler model, 
                                          SQLiteVectorDatabaseHandler vectorDB) {
        if (chunks == null || chunks.isEmpty()) {
            logError("没有文本块需要处理", null);
            return false;
        }
        
        try {
            int totalChunks = chunks.size();
            logMessage("开始向量化处理，共 " + totalChunks + " 个文本块");
            Log.d(TAG, "开始向量化处理，共 " + totalChunks + " 个文本块");
            
            // 获取EmbeddingModelManager实例，用于标记模型使用状态
            EmbeddingModelManager modelManager = EmbeddingModelManager.getInstance(context);
            
            try {
                // 标记模型开始使用
                modelManager.markModelInUse();
                Log.d(TAG, "标记模型开始使用");
                
                // 初始化进度日志
                StringBuilder progressLog = new StringBuilder("向量化进度");
                int lastPercentage = 0;
                
                // 首先发送初始进度日志
                if (progressCallback != null) {
                    progressCallback.onLog(progressLog.toString());
                }
                
                // 处理所有文本块
                for (int i = 0; i < totalChunks; i++) {
                    // 检查任务是否被取消
                    if (isTaskCancelled.get()) {
                        logMessage("任务已取消");
                        return false;
                    }
                    
                    TextChunk chunk = chunks.get(i);
                    String text = chunk.text;
                    String source = chunk.source;
                    JSONObject metadata = chunk.metadata;
                    
                    try {
                        // 每处理一个文本块，添加一个点
                        progressLog.append(".");
                        
                        // 计算当前百分比
                        int currentPercentage = (int)((i + 1) * 100 / totalChunks);
                        
                        // 检查是否需要显示百分比
                        boolean showPercentage = currentPercentage / 10 > lastPercentage / 10 || i == totalChunks - 1;
                        
                        if (showPercentage) {
                            // 打印百分比
                            progressLog.append(currentPercentage + "%");
                            lastPercentage = currentPercentage;
                        }
                        
                        // 每处理一个文本块，都更新UI显示
                        if (progressCallback != null) {
                            progressCallback.onLog(progressLog.toString());
                        }
                        
                        // 每100个文本块或最后一个文本块时，记录详细日志（仅在调试日志中显示）
                        if (i % 100 == 0 || i == totalChunks - 1) {
                            Log.d(TAG, "向量化详细进度: " + (i + 1) + "/" + totalChunks + 
                                  "，线程ID: " + Thread.currentThread().getId() + 
                                  "，源文件: " + source);
                        }
                        
                        // 生成向量
                        float[] embedding = model.generateEmbedding(text);
                        
                        // 添加到数据库
                        vectorDB.addVector(text, embedding, source, metadata.toString());
                        
                        // 更新进度
                        float percentage = (float) (i + 1) / totalChunks * 100;
                        if (progressCallback != null) {
                            progressCallback.onVectorizationProgress(i + 1, totalChunks, percentage);
                        }
                        
                        // 通知进度更新
                        if (notificationProgressCallback != null) {
                            notificationProgressCallback.onNotificationProgressUpdate(i + 1, totalChunks, percentage);
                        }
                    } catch (Exception e) {
                        logError("向量化失败: " + e.getMessage(), e);
                    }
                }
                
                // 保存数据库
                vectorDB.saveDatabase();
                logMessage("向量化处理完成");
                Log.d(TAG, "向量化处理全部完成，共处理 " + totalChunks + " 个文本块，线程ID: " + Thread.currentThread().getId());
                
                // 通知向量化处理完成
                if (progressCallback != null) {
                    progressCallback.onVectorizationComplete(totalChunks);
                }
            } finally {
                // 无论成功还是失败，最后标记模型为不再使用
                modelManager.markModelNotInUse();
                Log.d(TAG, "批量向量化处理完成，已标记模型为不再使用状态");
                
                // 关闭数据库
                vectorDB.close();
                Log.d(TAG, "向量数据库已关闭");
            }
            
            return !isTaskCancelled.get();
        } catch (Exception e) {
            logError("处理知识库失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 保存中间文本块到文件
     */
    private void saveIntermediateChunks(String knowledgeBasePath, List<TextChunk> chunks) {
        File intermediateFile = new File(knowledgeBasePath, INTERMEDIATE_FILE_NAME);
        
        try (FileWriter writer = new FileWriter(intermediateFile)) {
            JSONArray jsonArray = new JSONArray();
            
            for (TextChunk chunk : chunks) {
                JSONObject jsonChunk = new JSONObject();
                jsonChunk.put("text", chunk.text);
                jsonChunk.put("source", chunk.source);
                jsonChunk.put("chunkIndex", chunk.chunkIndex);
                jsonChunk.put("metadata", chunk.metadata.toString());
                jsonArray.put(jsonChunk);
            }
            
            writer.write(jsonArray.toString());
            logMessage("已保存中间文本块到: " + intermediateFile.getAbsolutePath());
        } catch (IOException | JSONException e) {
            logError("保存中间文本块失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 从文件加载中间文本块
     */
    public List<TextChunk> loadIntermediateChunks(String knowledgeBasePath) {
        File intermediateFile = new File(knowledgeBasePath, INTERMEDIATE_FILE_NAME);
        List<TextChunk> chunks = new ArrayList<>();
        
        if (!intermediateFile.exists()) {
            logMessage("中间文件不存在: " + intermediateFile.getAbsolutePath());
            return chunks;
        }
        
        try (BufferedReader reader = new BufferedReader(new FileReader(intermediateFile))) {
            StringBuilder content = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
            
            JSONArray jsonArray = new JSONArray(content.toString());
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject jsonChunk = jsonArray.getJSONObject(i);
                String text = jsonChunk.getString("text");
                String source = jsonChunk.getString("source");
                int chunkIndex = jsonChunk.getInt("chunkIndex");
                JSONObject metadata = new JSONObject(jsonChunk.getString("metadata"));
                
                chunks.add(new TextChunk(text, source, chunkIndex, metadata));
            }
            
            logMessage("已加载 " + chunks.size() + " 个中间文本块");
        } catch (IOException | JSONException e) {
            logError("加载中间文本块失败: " + e.getMessage(), e);
        }
        
        return chunks;
    }
    
    /**
     * 检查是否存在中间文件
     */
    public boolean hasIntermediateFile(String knowledgeBasePath) {
        File intermediateFile = new File(knowledgeBasePath, INTERMEDIATE_FILE_NAME);
        return intermediateFile.exists();
    }
    
    /**
     * 删除中间文件
     */
    public void deleteIntermediateFile(String knowledgeBasePath) {
        File intermediateFile = new File(knowledgeBasePath, INTERMEDIATE_FILE_NAME);
        if (intermediateFile.exists()) {
            boolean deleted = intermediateFile.delete();
            if (deleted) {
                logMessage("已删除中间文件: " + intermediateFile.getAbsolutePath());
            } else {
                logMessage("无法删除中间文件: " + intermediateFile.getAbsolutePath());
            }
        }
    }
    
    /**
     * 处理文件并构建知识库 - 一体化方法
     * 
     * @param knowledgeBaseName 知识库名称
     * @param embeddingModel 嵌入模型名称
     * @param files 要处理的文件列表
     * @param chunkSize 分块大小
     * @param chunkOverlap 重叠大小
     * @return 是否成功完成（未被取消）
     */
    public boolean processFilesAndBuildKnowledgeBase(String knowledgeBaseName, String embeddingModel, 
                                                  List<Uri> files, int chunkSize, int chunkOverlap) {
        try {
            Log.d(TAG, "开始处理文件并构建知识库，线程ID: " + Thread.currentThread().getId() + 
                  "，知识库: " + knowledgeBaseName + "，文件数: " + files.size());
            
            // 获取知识库目录路径
            String knowledgeBasePath = ConfigManager.getKnowledgeBasePath(context);
            String fullKnowledgeBasePath = knowledgeBasePath + File.separator + knowledgeBaseName;
            Log.d(TAG, "知识库目录: " + fullKnowledgeBasePath);
            logMessage("知识库目录: " + fullKnowledgeBasePath);
            
            // 创建知识库目录
            File knowledgeBaseDir = new File(fullKnowledgeBasePath);
            if (!knowledgeBaseDir.exists()) {
                boolean created = knowledgeBaseDir.mkdirs();
                if (!created) {
                    throw new Exception("无法创建知识库目录: " + fullKnowledgeBasePath);
                }
            }
            
            // 获取嵌入模型路径
            String embeddingModelPath = ConfigManager.getEmbeddingModelPath(context) + File.separator + embeddingModel;
            logMessage("使用嵌入模型: " + embeddingModelPath);
            
            // 使用模型管理器获取模型
            EmbeddingModelHandler model = EmbeddingModelManager.getInstance(context).getModel(embeddingModelPath);
            logMessage("已加载嵌入模型: " + model.getModelName());
            
            // 获取模型的向量维度
            int embeddingDimension = model.getEmbeddingDimension();
            Log.d(TAG, "模型向量维度: " + embeddingDimension);
            logMessage("模型向量维度: " + embeddingDimension);
            
            // 初始化TokenizerManager
            TokenizerManager tokenizerManager = TokenizerManager.getInstance(context);
            if (!tokenizerManager.isInitialized()) {
                logMessage("正在初始化全局分词器...");
                boolean initSuccess = tokenizerManager.initialize(model.getModelPath());
                if (initSuccess) {
                    logMessage("全局分词器初始化成功，将使用统一分词策略");
                    // 启用一致性分词
                    tokenizerManager.setUseConsistentTokenization(true);
                    
                    // 设置调试模式
                    boolean debugMode = ConfigManager.isDebugMode(context);
                    if (debugMode) {
                        tokenizerManager.setDebugMode(true);
                        logMessage("全局分词器已启用调试模式");
                    }
                } else {
                    logMessage("全局分词器初始化失败，将使用模型自带分词器");
                }
            } else {
                logMessage("使用已初始化的全局分词器");
                // 启用一致性分词
                tokenizerManager.setUseConsistentTokenization(true);
                
                // 设置调试模式
                boolean debugMode = ConfigManager.isDebugMode(context);
                if (debugMode) {
                    tokenizerManager.setDebugMode(true);
                }
            }
            
            // 启用一致性分词策略
            model.setUseConsistentProcessing(true);
            logMessage("启用一致性分词策略，确保知识库构建和查询使用相同的分词逻辑");
            
            // 是否启用调试模式
            boolean debugMode = ConfigManager.isDebugMode(context);
            if (debugMode) {
                model.setDebugMode(true);
                logMessage("启用调试模式，将输出详细的分词和处理日志");
            }
            
            // 初始化向量数据库
            SQLiteVectorDatabaseHandler vectorDB = new SQLiteVectorDatabaseHandler(
                    new File(fullKnowledgeBasePath), model.getEmbeddingModel(), embeddingDimension);
            
            // 提取文本并分块
            List<TextChunk> chunks = extractTextFromFiles(fullKnowledgeBasePath, files, chunkSize, chunkOverlap);
            
            // 检查是否被取消
            if (isTaskCancelled.get()) {
                logMessage("任务已取消");
                vectorDB.close();
                return false;
            }
            
            // 通知文本提取完成
            if (progressCallback != null) {
                progressCallback.onTextExtractionComplete(chunks.size());
            }
            
            // 获取模型管理器实例
            EmbeddingModelManager modelManager = EmbeddingModelManager.getInstance(context);
            
            // 向量化处理开始前标记模型为正在使用，防止在向量化过程中被自动卸载
            modelManager.markModelInUse();
            Log.d(TAG, "开始批量向量化处理，已标记模型为正在使用状态，防止自动卸载");
            
            try {
                // 生成向量并添加到数据库
                int totalChunks = chunks.size();
                logMessage("开始向量化处理，共 " + totalChunks + " 个文本块");
                
                // 向量化处理
                for (int i = 0; i < totalChunks; i++) {
                    // 检查任务是否取消
                    if (isTaskCancelled.get()) {
                        logMessage("任务已取消");
                        Log.d(TAG, "向量化处理中断：任务被取消，已处理 " + i + "/" + totalChunks + " 个文本块");
                        vectorDB.close();
                        return false;
                    }
                    
                    TextChunk chunk = chunks.get(i);
                    String text = chunk.text;
                    String source = chunk.source;
                    JSONObject metadata = chunk.metadata;
                    
                    try {
                        // 每100个文本块记录一次日志
                        if (i % 100 == 0 || i == totalChunks - 1) {
                            Log.d(TAG, "向量化进度: " + i + "/" + totalChunks + 
                                  "，线程ID: " + Thread.currentThread().getId() + 
                                  "，源文件: " + source);
                        }
                        
                        // 生成向量
                        float[] embedding = model.generateEmbedding(text);
                        
                        // 添加到数据库
                        vectorDB.addVector(text, embedding, source, metadata.toString());
                        
                        // 更新进度
                        float percentage = (float) (i + 1) / totalChunks * 100;
                        if (progressCallback != null) {
                            progressCallback.onVectorizationProgress(i + 1, totalChunks, percentage);
                        }
                        
                        // 通知进度更新
                        if (notificationProgressCallback != null) {
                            notificationProgressCallback.onNotificationProgressUpdate(i + 1, totalChunks, percentage);
                        }
                    } catch (Exception e) {
                        logError("向量化失败: " + e.getMessage(), e);
                    }
                }
                
                // 保存数据库
                vectorDB.saveDatabase();
                logMessage("向量化处理完成");
                Log.d(TAG, "向量化处理全部完成，共处理 " + totalChunks + " 个文本块，线程ID: " + Thread.currentThread().getId());
                
                // 通知向量化处理完成
                if (progressCallback != null) {
                    progressCallback.onVectorizationComplete(totalChunks);
                }
            } finally {
                // 无论成功还是失败，最后标记模型为不再使用
                modelManager.markModelNotInUse();
                Log.d(TAG, "批量向量化处理完成，已标记模型为不再使用状态");
                
                // 关闭数据库
                vectorDB.close();
                Log.d(TAG, "向量数据库已关闭");
            }
            
            return !isTaskCancelled.get();
        } catch (Exception e) {
            logError("处理知识库失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 从JSON字符串中提取值
     * 例如从 "instruction": "你好" 提取出 "你好"
     */
    private String extractValue(String jsonLine) {
        try {
            int colonPos = jsonLine.indexOf(':');
            if (colonPos < 0) return "";
            
            String valueStr = jsonLine.substring(colonPos + 1).trim();
            
            // 如果值是用引号括起来的
            if (valueStr.startsWith("\"") && valueStr.indexOf("\"", 1) > 0) {
                int endQuote = valueStr.lastIndexOf("\"");
                if (endQuote > 0) {
                    return valueStr.substring(1, endQuote);
                }
            }
            
            // 处理值后面可能有逗号的情况
            if (valueStr.endsWith(",")) {
                valueStr = valueStr.substring(0, valueStr.length() - 1);
            }
            
            // 去掉可能的引号
            if (valueStr.startsWith("\"") && valueStr.endsWith("\"")) {
                valueStr = valueStr.substring(1, valueStr.length() - 1);
            }
            
            return valueStr;
        } catch (Exception e) {
            logError("提取JSON值时出错: " + e.getMessage(), e);
            return "";
        }
    }
    
    /**
     * 记录日志消息
     */
    private void logMessage(String message) {
        Log.d(TAG, message);
        if (progressCallback != null) {
            progressCallback.onLog(message);
        }
    }
    
    /**
     * 记录错误消息
     */
    private void logError(String message, Exception e) {
        Log.e(TAG, message, e);
        if (progressCallback != null) {
            progressCallback.onError(message);
        }
    }
}
