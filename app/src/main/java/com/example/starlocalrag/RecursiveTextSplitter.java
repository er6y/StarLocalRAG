package com.example.starlocalrag;

import android.content.Context;
import android.util.Log;
import com.example.starlocalrag.LogManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 递归文本分割器，模拟LangChain的RecursiveCharacterTextSplitter
 * 实现与PC端相同的分块算法
 */
public class RecursiveTextSplitter {
    private static final String TAG = "StarLocalRAG_RecSplit";
    
    // 默认分隔符，按优先级排序（从高到低）
    private static final List<String> DEFAULT_SEPARATORS = Arrays.asList(
        "\n\n",  // 段落分隔符
        "\n",    // 换行符
        ". ",    // 句号后跟空格
        "! ",    // 感叹号后跟空格
        "? ",    // 问号后跟空格
        ";",     // 分号
        ",",     // 逗号
        " ",     // 空格
        ""       // 无分隔符，按字符分割
    );
    
    private final int chunkSize;
    private final int chunkOverlap;
    private final int minChunkSize;
    private final List<String> separators;
    private final Context context;
    
    /**
     * 构造函数
     * @param context 应用上下文
     */
    public RecursiveTextSplitter(Context context) {
        this.context = context;
        this.chunkSize = ConfigManager.getChunkSize(context);
        this.chunkOverlap = ConfigManager.getInt(context, ConfigManager.KEY_OVERLAP_SIZE, ConfigManager.DEFAULT_OVERLAP_SIZE);
        this.minChunkSize = ConfigManager.getMinChunkSize(context);
        this.separators = DEFAULT_SEPARATORS;
        
        LogManager.logD(TAG, "创建RecursiveTextSplitter实例，使用ConfigManager配置：" +
              "分块大小=" + chunkSize + ", 重叠大小=" + chunkOverlap + ", 最小分块大小=" + minChunkSize);
    }
    
    /**
     * 构造函数
     * @param chunkSize 分块大小
     * @param chunkOverlap 重叠大小
     */
    public RecursiveTextSplitter(int chunkSize, int chunkOverlap) {
        this(chunkSize, chunkOverlap, ConfigManager.DEFAULT_MIN_CHUNK_SIZE); // 使用默认最小分块大小
    }
    
    /**
     * 构造函数
     * @param chunkSize 分块大小
     * @param chunkOverlap 重叠大小
     * @param minChunkSize 最小分块大小
     */
    public RecursiveTextSplitter(int chunkSize, int chunkOverlap, int minChunkSize) {
        this(chunkSize, chunkOverlap, minChunkSize, DEFAULT_SEPARATORS);
    }
    
    /**
     * 构造函数
     * @param chunkSize 分块大小
     * @param chunkOverlap 重叠大小
     * @param minChunkSize 最小分块大小
     * @param separators 分隔符列表，按优先级排序
     */
    public RecursiveTextSplitter(int chunkSize, int chunkOverlap, int minChunkSize, List<String> separators) {
        this.context = null;
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        this.minChunkSize = minChunkSize;
        this.separators = separators;
        
        LogManager.logD(TAG, "创建RecursiveTextSplitter实例，使用指定参数：" +
              "分块大小=" + chunkSize + ", 重叠大小=" + chunkOverlap + ", 最小分块大小=" + minChunkSize);
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
        
        LogManager.logD(TAG, "开始分割文本，长度: " + text.length() + " 字符，分块大小: " + chunkSize + 
              "，重叠大小: " + chunkOverlap + "，最小分块大小: " + minChunkSize);
        
        List<String> finalChunks = new ArrayList<>();
        
        // 递归分割文本
        splitTextRecursive(text, finalChunks, 0);
        
        // 过滤掉过小的文本块
        List<String> filteredChunks = new ArrayList<>();
        for (String chunk : finalChunks) {
            if (chunk.length() >= minChunkSize) {
                filteredChunks.add(chunk);
            } else {
                LogManager.logD(TAG, "过滤掉过小的文本块，长度: " + chunk.length() + " 字符");
            }
        }
        
        LogManager.logD(TAG, "分割完成，共生成 " + finalChunks.size() + " 个文本块，过滤后剩余 " + 
              filteredChunks.size() + " 个");
        
        return filteredChunks;
    }
    
    /**
     * 递归分割文本
     * @param text 输入文本
     * @param chunks 存储结果的文本块列表
     * @param separatorIndex 当前使用的分隔符索引
     */
    private void splitTextRecursive(String text, List<String> chunks, int separatorIndex) {
        // 如果文本长度小于等于分块大小，直接添加
        if (text.length() <= chunkSize) {
            if (!text.trim().isEmpty()) {
                chunks.add(text.trim());
            }
            return;
        }
        
        // 如果已经尝试了所有分隔符，则强制分割
        if (separatorIndex >= separators.size()) {
            forceSplitText(text, chunks);
            return;
        }
        
        // 获取当前分隔符
        String separator = separators.get(separatorIndex);
        
        // 如果是空字符串分隔符，则按字符分割
        if (separator.isEmpty()) {
            forceSplitText(text, chunks);
            return;
        }
        
        // 使用当前分隔符分割文本
        String[] splits = text.split(Pattern.quote(separator));
        
        // 如果分割后只有一个段落，尝试使用下一个分隔符
        if (splits.length == 1) {
            splitTextRecursive(text, chunks, separatorIndex + 1);
            return;
        }
        
        // 处理分割后的段落
        List<String> segments = new ArrayList<>();
        
        // 重建文本，保留分隔符
        for (int i = 0; i < splits.length; i++) {
            segments.add(splits[i]);
            // 除了最后一个分割部分外，每个部分后面都加上分隔符
            if (i < splits.length - 1) {
                segments.add(separator);
            }
        }
        
        // 合并段落，确保不超过分块大小
        StringBuilder currentChunk = new StringBuilder();
        
        for (String segment : segments) {
            // 如果添加当前段落后不超过分块大小，则添加
            if (currentChunk.length() + segment.length() <= chunkSize) {
                currentChunk.append(segment);
            } else {
                // 如果当前块不为空，添加到结果中
                if (currentChunk.length() > 0) {
                    String chunk = currentChunk.toString().trim();
                    if (!chunk.isEmpty()) {
                        chunks.add(chunk);
                    }
                }
                
                // 如果当前段落本身就超过分块大小，递归处理
                if (segment.length() > chunkSize) {
                    splitTextRecursive(segment, chunks, separatorIndex + 1);
                } else {
                    // 否则开始新的块
                    currentChunk = new StringBuilder(segment);
                }
            }
        }
        
        // 添加最后一个块
        if (currentChunk.length() > 0) {
            String chunk = currentChunk.toString().trim();
            if (!chunk.isEmpty()) {
                chunks.add(chunk);
            }
        }
        
        // 处理重叠
        mergeWithOverlap(chunks);
    }
    
    /**
     * 强制分割文本（当所有分隔符都不适用时）
     * @param text 输入文本
     * @param chunks 存储结果的文本块列表
     */
    private void forceSplitText(String text, List<String> chunks) {
        LogManager.logD(TAG, "强制分割文本，长度: " + text.length() + " 字符");
        
        for (int i = 0; i < text.length(); i += chunkSize - chunkOverlap) {
            int end = Math.min(i + chunkSize, text.length());
            String chunk = text.substring(i, end).trim();
            if (!chunk.isEmpty()) {
                chunks.add(chunk);
            }
        }
    }
    
    /**
     * 合并文本块，考虑重叠
     * @param chunks 文本块列表
     */
    private void mergeWithOverlap(List<String> chunks) {
        if (chunks.size() <= 1 || chunkOverlap <= 0) {
            return;
        }
        
        List<String> mergedChunks = new ArrayList<>();
        String prevChunk = chunks.get(0);
        mergedChunks.add(prevChunk);
        
        for (int i = 1; i < chunks.size(); i++) {
            String currentChunk = chunks.get(i);
            
            // 如果前一个块的末尾与当前块的开头有重叠，合并它们
            if (prevChunk.length() >= chunkOverlap && currentChunk.length() >= chunkOverlap) {
                String prevEnd = prevChunk.substring(prevChunk.length() - chunkOverlap);
                String currentStart = currentChunk.substring(0, chunkOverlap);
                
                // 如果有重叠，合并
                if (prevEnd.equals(currentStart)) {
                    String mergedChunk = prevChunk + currentChunk.substring(chunkOverlap);
                    // 如果合并后的块不超过分块大小，替换前一个块
                    if (mergedChunk.length() <= chunkSize) {
                        mergedChunks.set(mergedChunks.size() - 1, mergedChunk);
                        prevChunk = mergedChunk;
                        continue;
                    }
                }
            }
            
            // 如果没有重叠或合并后超过分块大小，添加当前块
            mergedChunks.add(currentChunk);
            prevChunk = currentChunk;
        }
        
        // 更新原始列表
        chunks.clear();
        chunks.addAll(mergedChunks);
    }
}
