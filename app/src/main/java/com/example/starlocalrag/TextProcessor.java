package com.example.starlocalrag;

import android.content.Context;
import android.net.Uri;
import android.util.Log;
import com.example.starlocalrag.LogManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import com.example.starlocalrag.ConfigManager;
import com.example.starlocalrag.JsonDatasetProcessor;
import com.example.starlocalrag.LangChainTextSplitter;

/**
 * 文本处理工具类，用于读取、分割和预处理文本
 */
public class TextProcessor {
    private static final String TAG = "StarLocalRAG_TextProc";
    
    // 恢复原始的分块大小和重叠大小
    private static final int DEFAULT_CHUNK_SIZE = 1000;
    private static final int DEFAULT_CHUNK_OVERLAP = 200;
    
    // 最大文件大小限制 (20MB)
    private static final int MAX_FILE_SIZE = 20 * 1024 * 1024;
    
    // 最大块数限制
    private static final int MAX_CHUNKS = 5000;
    
    private final Context context;
    private int chunkSize;
    private int chunkOverlap;
    private int minChunkSize;
    private final DocumentParser documentParser;
    
    /**
     * 构造函数
     * @param context 应用上下文
     */
    public TextProcessor(Context context) {
        this.context = context;
        // 从 ConfigManager 中获取分块大小和重叠大小
        this.chunkSize = ConfigManager.getChunkSize(context);
        this.chunkOverlap = ConfigManager.getInt(context, ConfigManager.KEY_OVERLAP_SIZE, ConfigManager.DEFAULT_OVERLAP_SIZE);
        this.minChunkSize = ConfigManager.getMinChunkSize(context); // 获取最小分块限制
        this.documentParser = new DocumentParser(context);
        
        LogManager.logD(TAG, "创建TextProcessor实例，从ConfigManager获取配置：" +
              "分块大小=" + chunkSize + ", 重叠大小=" + chunkOverlap + ", 最小分块大小=" + minChunkSize);
    }
    
    /**
     * 从Uri读取文本内容，使用文档解析器提取不同类型文件的文本
     * @param uri 文件Uri
     * @return 文件内容
     * @throws IOException 如果读取失败
     */
    public String readTextFromUri(Uri uri) throws IOException {
        try {
            // 使用DocumentParser提取文本，而不是直接读取原始内容
            String extractedText = documentParser.extractText(uri);
            
            // 清理提取的文本
            String cleanedText = documentParser.cleanText(extractedText);
            
            // 如果文本太长，截断处理
            if (cleanedText.length() > MAX_FILE_SIZE / 2) { // 字符数约为字节数的一半
                LogManager.logW(TAG, "文本过长: " + cleanedText.length() + " 字符，将被截断至 " + (MAX_FILE_SIZE / 2) + " 字符");
                cleanedText = cleanedText.substring(0, MAX_FILE_SIZE / 2) + "\n... [文本过长，已截断]";
            }
            
            // 记录文件类型信息
            String fileName = uri.getLastPathSegment();
            if (fileName != null) {
                LogManager.logD(TAG, "文件名: " + fileName + ", 大小: " + cleanedText.length() + " 字符");
                
                // 检查是否是JSON文件
                if (fileName.toLowerCase().endsWith(".json")) {
                    boolean jsonOptimizationEnabled = ConfigManager.isJsonDatasetSplittingEnabled(context);
                    LogManager.logD(TAG, "检测到JSON文件，JSON优化配置状态: " + (jsonOptimizationEnabled ? "启用" : "禁用"));
                    
                    if (jsonOptimizationEnabled) {
                        LogManager.logD(TAG, "将使用JsonDatasetProcessor处理JSON文件");
                        // 使用JsonDatasetProcessor处理JSON文件，但不在这里直接处理
                        // 只是记录日志，实际处理在splitTextIntoChunks中进行
                    } else {
                        LogManager.logD(TAG, "将按普通文本处理JSON文件");
                    }
                }
            }
            
            return cleanedText;
        } catch (Exception e) {
            LogManager.logE(TAG, "读取文件失败: " + e.getMessage(), e);
            throw new IOException("读取文件失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 将文本分割成块 - 优化版本，避免内存溢出
     * @param text 输入文本
     * @return 文本块列表
     */
    public List<String> splitTextIntoChunks(String text) {
        return splitTextIntoChunks(text, this.chunkSize, this.chunkOverlap);
    }
    
    /**
     * 将文本分割成块 - 使用指定的分块大小和重叠大小
     * @param text 输入文本
     * @param chunkSize 分块大小
     * @param chunkOverlap 重叠大小
     * @return 文本块列表
     */
    public List<String> splitTextIntoChunks(String text, int chunkSize, int chunkOverlap) {
        if (text == null || text.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<String> chunks = new ArrayList<>();
        
        // 记录分块参数
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        
        LogManager.logD(TAG, "分块参数：块大小=" + chunkSize + "，重叠大小=" + chunkOverlap + "，最小块大小=" + minChunkSize);
        
        // 规范化文本，使其更接近PC端的文本格式
        String originalText = text;
        text = normalizeText(text);
        
        // 检测是否为JSON内容
        if (JsonDatasetProcessor.isJsonContent(text)) {
            boolean jsonOptimizationEnabled = ConfigManager.isJsonDatasetSplittingEnabled(context);
            LogManager.logD(TAG, "检测到JSON内容，JSON优化配置状态: " + (jsonOptimizationEnabled ? "启用" : "禁用"));
            
            if (jsonOptimizationEnabled) {
                LogManager.logD(TAG, "使用JsonDatasetProcessor处理JSON内容");
                // 使用静态方法处理JSON内容
                chunks = JsonDatasetProcessor.processJsonDataset(context, text);
                
                LogManager.logD(TAG, "JSON处理完成，生成了" + chunks.size() + "个文本块");
                return chunks;
            }
        }
        
        // 使用LangChain4j进行文本分割
        LogManager.logD(TAG, "使用LangChainTextSplitter进行文本分割");
        // 从ConfigManager获取最小分块大小，而不是使用硬编码值
        LangChainTextSplitter splitter = new LangChainTextSplitter(chunkSize, chunkOverlap, minChunkSize);
        chunks = splitter.splitText(text);
        
        LogManager.logD(TAG, "文本分割完成，生成了" + chunks.size() + "个文本块");
        
        return chunks;
    }
    
    /**
     * 规范化文本，使其更接近PC端的文本格式
     * @param text 输入文本
     * @return 规范化后的文本
     */
    private String normalizeText(String text) {
        if (text == null) {
            return "";
        }
        
        // 保存原始文本长度用于日志记录
        int originalLength = text.length();
        
        // 1. 统一换行符（Windows CRLF -> Unix LF）
        text = text.replace("\r\n", "\n");
        
        // 2. 处理连续的换行符，保持与PC端LangChain处理一致
        // LangChain的RecursiveCharacterTextSplitter使用["\n\n", "\n", " ", ""]作为分隔符
        // 确保文本中的"\n\n"模式被保留，这对分块很重要
        text = text.replaceAll("\n{3,}", "\n\n"); // 3个以上连续换行符替换为2个
        
        // 3. 处理连续的空格，但保留段落格式
        text = text.replaceAll("[ \t]+", " "); // 连续空格或制表符替换为单个空格
        
        // 4. 移除不可见控制字符，但保留基本格式控制字符
        text = text.replaceAll("[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]", "");
        
        // 5. 规范化空行（确保空行只有一个\n字符）
        text = text.replaceAll(" *\n *", "\n");
        
        // 6. 移除文本开头和结尾的空白字符
        text = text.trim();
        
        // 记录规范化前后的文本长度差异
        int normalizedLength = text.length();
        LogManager.logD(TAG, "文本规范化：原始长度=" + originalLength + "，规范化后长度=" + normalizedLength + 
              "，差异=" + (normalizedLength - originalLength));
        
        return text;
    }
    
    /**
     * 查找字符串中最后一次出现的位置，但不超过指定的最大位置
     * @param text 文本
     * @param pattern 要查找的模式
     * @param maxPosition 最大位置
     * @return 最后出现的位置，如果未找到则返回-1
     */
    private int findLastOccurrence(String text, String pattern, int maxPosition) {
        int safeMaxPos = Math.min(maxPosition, text.length());
        int searchStartPos = Math.max(0, safeMaxPos - pattern.length());
        return text.substring(0, safeMaxPos).lastIndexOf(pattern, searchStartPos);
    }
    
    /**
     * 在文本中查找最近的句子结束位置
     * @param text 文本
     * @param position 起始位置
     * @return 最近的句子结束位置
     */
    private int findSentenceEnd(String text, int position) {
        // 安全检查
        if (position >= text.length()) {
            return text.length() - 1;
        }
        
        int maxSearchLength = Math.min(chunkSize / 4, 100); // 限制搜索范围
        int startPos = Math.max(0, position - maxSearchLength);
        int endPos = Math.min(position, text.length() - 1);
        
        // 从位置向前搜索句号、问号或感叹号
        for (int pos = endPos; pos >= startPos; pos--) {
            if (pos < text.length()) {
                char c = text.charAt(pos);
                if (c == '.' || c == '?' || c == '!') {
                    return pos;
                }
            }
        }
        
        return position;
    }
    
    /**
     * 预处理文本（删除多余空白、标准化等）
     * @param text 输入文本
     * @return 预处理后的文本
     */
    public String preprocessText(String text) {
        if (text == null) {
            return "";
        }
        
        // 如果文本过长，截断处理
        if (text.length() > chunkSize * 2) {
            // 只处理前两个块大小的文本，避免处理过长的文本
            text = text.substring(0, chunkSize * 2);
        }
        
        // 删除连续的空白字符
        text = text.replaceAll("\\s+", " ");
        
        // 删除开头和结尾的空白
        return text.trim();
    }
    
    /**
     * 设置分块大小
     * @param chunkSize 分块大小
     */
    public void setChunkSize(int chunkSize) {
        this.chunkSize = chunkSize;
    }
    
    /**
     * 设置重叠大小
     * @param chunkOverlap 重叠大小
     */
    public void setChunkOverlap(int chunkOverlap) {
        this.chunkOverlap = chunkOverlap;
    }
    
    /**
     * 获取当前分块大小
     * @return 分块大小
     */
    public int getChunkSize() {
        return this.chunkSize;
    }
    
    /**
     * 获取当前重叠大小
     * @return 重叠大小
     */
    public int getChunkOverlap() {
        return this.chunkOverlap;
    }
}
