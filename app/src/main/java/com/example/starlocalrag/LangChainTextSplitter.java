package com.example.starlocalrag;

import android.content.Context;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentByCharacterSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.document.Metadata;

/**
 * 使用LangChain4j实现的文本分割器
 * 与PC端使用相同的文本分割逻辑，确保分块结果一致
 */
public class LangChainTextSplitter {
    private static final String TAG = "LangChainTextSplitter";
    
    private final int chunkSize;
    private final int chunkOverlap;
    private final int minChunkSize;
    private final DocumentSplitter splitter;
    
    /**
     * 构造函数
     * @param context 上下文
     */
    public LangChainTextSplitter(Context context) {
        this.chunkSize = ConfigManager.getChunkSize(context);
        this.chunkOverlap = ConfigManager.getInt(context, ConfigManager.KEY_OVERLAP_SIZE, ConfigManager.DEFAULT_OVERLAP_SIZE);
        this.minChunkSize = ConfigManager.getMinChunkSize(context);
        
        // 创建LangChain4j的文档分割器
        // 注意：LangChain4j的Java版本API与Python版本不同，不支持直接传递分隔符列表
        this.splitter = new DocumentByCharacterSplitter(chunkSize, chunkOverlap);
        
        Log.d(TAG, "初始化LangChainTextSplitter，分块大小: " + chunkSize + 
                "，重叠大小: " + chunkOverlap + 
                "，最小分块大小: " + minChunkSize);
    }
    
    /**
     * 构造函数
     * @param chunkSize 分块大小
     * @param chunkOverlap 重叠大小
     * @param minChunkSize 最小分块大小
     */
    public LangChainTextSplitter(int chunkSize, int chunkOverlap, int minChunkSize) {
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        this.minChunkSize = minChunkSize;
        
        // 创建LangChain4j的文档分割器
        // 注意：LangChain4j的Java版本API与Python版本不同，不支持直接传递分隔符列表
        this.splitter = new DocumentByCharacterSplitter(chunkSize, chunkOverlap);
        
        Log.d(TAG, "初始化LangChainTextSplitter，分块大小: " + chunkSize + 
                "，重叠大小: " + chunkOverlap + 
                "，最小分块大小: " + minChunkSize);
    }
    
    /**
     * 分割文本
     * @param text 输入文本
     * @return 分割后的文本块列表
     */
    public List<String> splitText(String text) {
        if (text == null || text.isEmpty()) {
            return new ArrayList<>();
        }
        
        // 创建文档
        Document document = new Document(text, new Metadata());
        
        // 使用LangChain4j分割文档
        List<TextSegment> segments = splitter.split(document);
        
        // 转换为字符串列表
        List<String> chunks = new ArrayList<>();
        for (TextSegment segment : segments) {
            String chunk = segment.text();
            if (chunk.length() >= minChunkSize) {
                chunks.add(chunk);
            } else {
                Log.d(TAG, "过滤掉过小的文本块，长度: " + chunk.length());
            }
        }
        
        Log.d(TAG, "文本分割完成，共生成" + segments.size() + "个文本块，过滤后剩余" + chunks.size() + "个");
        return chunks;
    }
}
