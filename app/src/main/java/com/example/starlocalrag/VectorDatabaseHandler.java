package com.example.starlocalrag;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * 向量数据库处理类，用于存储和检索文本块及其嵌入向量
 */
public class VectorDatabaseHandler {
    private static final String TAG = "StarLocalRAG_VectorDB";
    
    // 向量数据库文件名
    private static final String VECTOR_DB_FILENAME = "vector_database.dat";
    private static final String METADATA_FILENAME = "metadata.dat";
    
    // 数据库目录
    private final File databaseDir;
    private final Context context;
    
    /**
     * 文本块类，包含文本内容和对应的嵌入向量
     */
    public static class TextChunk implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private String text;
        private String source;
        private float[] embedding;
        
        public TextChunk(String text, String source, float[] embedding) {
            this.text = text;
            this.source = source;
            this.embedding = embedding;
        }
        
        public String getText() {
            return text;
        }
        
        public String getSource() {
            return source;
        }
        
        public float[] getEmbedding() {
            return embedding;
        }
    }
    
    /**
     * 数据库元数据类
     */
    public static class DatabaseMetadata implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private String embeddingModel;
        private int embeddingDimension;
        private int chunkCount;
        private List<String> sources;
        private long creationTimestamp;
        private long lastModifiedTimestamp;
        
        public DatabaseMetadata(String embeddingModel) {
            this.embeddingModel = embeddingModel;
            this.embeddingDimension = 0;
            this.chunkCount = 0;
            this.sources = new ArrayList<>();
            this.creationTimestamp = System.currentTimeMillis();
            this.lastModifiedTimestamp = this.creationTimestamp;
        }
        
        public String getEmbeddingModel() {
            return embeddingModel;
        }
        
        public int getEmbeddingDimension() {
            return embeddingDimension;
        }
        
        public void setEmbeddingDimension(int embeddingDimension) {
            this.embeddingDimension = embeddingDimension;
        }
        
        public int getChunkCount() {
            return chunkCount;
        }
        
        public void incrementChunkCount() {
            this.chunkCount++;
            this.lastModifiedTimestamp = System.currentTimeMillis();
        }
        
        public List<String> getSources() {
            return sources;
        }
        
        public void addSource(String source) {
            if (!sources.contains(source)) {
                sources.add(source);
                this.lastModifiedTimestamp = System.currentTimeMillis();
            }
        }
        
        public long getCreationTimestamp() {
            return creationTimestamp;
        }
        
        public long getLastModifiedTimestamp() {
            return lastModifiedTimestamp;
        }
    }
    
    // 数据库中的文本块列表
    private List<TextChunk> chunks;
    
    // 数据库元数据
    private DatabaseMetadata metadata;
    
    /**
     * 构造函数
     * @param databaseDir 数据库目录
     * @param embeddingModel 使用的嵌入模型名称
     */
    public VectorDatabaseHandler(File databaseDir, String embeddingModel) {
        this.databaseDir = databaseDir;
        this.context = null;
        this.chunks = new ArrayList<>();
        
        // 确保目录存在
        if (!databaseDir.exists()) {
            databaseDir.mkdirs();
        }
        
        // 尝试加载现有数据库，如果不存在则创建新的
        if (!loadDatabase()) {
            this.metadata = new DatabaseMetadata(embeddingModel);
        }
    }
    
    /**
     * 构造函数
     * @param context 上下文
     */
    public VectorDatabaseHandler(Context context) {
        this.context = context;
        this.databaseDir = null;
        this.chunks = new ArrayList<>();
        this.metadata = null;
    }
    
    /**
     * 获取知识库目录
     * @param knowledgeBaseName 知识库名称
     * @return 知识库目录
     */
    private File getKnowledgeBaseDir(String knowledgeBaseName) {
        if (context == null) {
            Log.e(TAG, "Context为null，无法获取知识库目录");
            return null;
        }
        File dir = new File(context.getFilesDir(), "knowledge_bases/" + knowledgeBaseName);
        Log.d(TAG, "知识库目录路径: " + dir.getAbsolutePath());
        return dir;
    }
    
    /**
     * 添加文本块及其嵌入向量到数据库
     * @param text 文本内容
     * @param source 文本来源（如文件名）
     * @param embedding 嵌入向量
     */
    public void addChunk(String text, String source, float[] embedding) {
        TextChunk chunk = new TextChunk(text, source, embedding);
        chunks.add(chunk);
        
        // 更新元数据
        if (metadata.getEmbeddingDimension() == 0 && embedding.length > 0) {
            metadata.setEmbeddingDimension(embedding.length);
        }
        metadata.incrementChunkCount();
        metadata.addSource(source);
        
        // 保存数据库
        saveDatabase();
    }
    
    /**
     * 保存数据库到文件
     * @return 是否保存成功
     */
    public boolean saveDatabase() {
        if (databaseDir == null) {
            Log.e(TAG, "数据库目录为null，无法保存数据库");
            return false;
        }
        
        boolean success = true;
        
        // 保存向量数据
        File vectorDbFile = new File(databaseDir, VECTOR_DB_FILENAME);
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(vectorDbFile))) {
            oos.writeObject(chunks);
            Log.d(TAG, "保存向量数据库成功，共 " + chunks.size() + " 个文本块");
        } catch (IOException e) {
            Log.e(TAG, "保存向量数据库失败: " + e.getMessage(), e);
            success = false;
        }
        
        // 保存元数据
        File metadataFile = new File(databaseDir, METADATA_FILENAME);
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(metadataFile))) {
            oos.writeObject(metadata);
            Log.d(TAG, "保存元数据成功");
        } catch (IOException e) {
            Log.e(TAG, "保存元数据失败: " + e.getMessage(), e);
            success = false;
        }
        
        return success;
    }
    
    /**
     * 从文件加载数据库
     * @return 是否加载成功
     */
    @SuppressWarnings("unchecked")
    public boolean loadDatabase() {
        if (databaseDir == null) {
            Log.e(TAG, "数据库目录为null，无法加载数据库");
            return false;
        }
        
        boolean success = true;
        
        // 加载向量数据
        File vectorDbFile = new File(databaseDir, VECTOR_DB_FILENAME);
        if (vectorDbFile.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(
                    new FileInputStream(vectorDbFile))) {
                chunks = (List<TextChunk>) ois.readObject();
                Log.d(TAG, "加载向量数据库成功，共 " + chunks.size() + " 个文本块");
            } catch (IOException | ClassNotFoundException e) {
                Log.e(TAG, "加载向量数据库失败: " + e.getMessage(), e);
                chunks = new ArrayList<>();
                success = false;
            }
        } else {
            Log.d(TAG, "向量数据库文件不存在，将创建新数据库");
            chunks = new ArrayList<>();
            success = false;
        }
        
        // 加载元数据
        File metadataFile = new File(databaseDir, METADATA_FILENAME);
        if (metadataFile.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(
                    new FileInputStream(metadataFile))) {
                metadata = (DatabaseMetadata) ois.readObject();
                Log.d(TAG, "加载元数据成功，嵌入模型: " + metadata.getEmbeddingModel());
            } catch (IOException | ClassNotFoundException e) {
                Log.e(TAG, "加载元数据失败: " + e.getMessage(), e);
                success = false;
            }
        } else {
            Log.d(TAG, "元数据文件不存在，将创建新元数据");
            success = false;
        }
        
        return success;
    }
    
    /**
     * 获取数据库中的文本块数量
     * @return 文本块数量
     */
    public int getChunkCount() {
        return chunks.size();
    }
    
    /**
     * 获取数据库元数据
     * @return 数据库元数据
     */
    public DatabaseMetadata getMetadata() {
        return metadata;
    }
    
    /**
     * 计算两个向量之间的余弦相似度
     * @param vec1 向量1
     * @param vec2 向量2
     * @return 余弦相似度值，范围[-1, 1]
     */
    private double cosineSimilarity(float[] vec1, float[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("向量维度不匹配");
        }
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        if (norm1 == 0 || norm2 == 0) {
            return 0;
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    /**
     * 根据查询向量搜索最相似的文本块
     * @param queryEmbedding 查询向量
     * @param topK 返回的最大结果数量
     * @return 按相似度排序的文本块列表
     */
    public List<Map<String, Object>> search(float[] queryEmbedding, int topK) {
        return search(queryEmbedding, topK, 0, 0);
    }
    
    /**
     * 根据查询向量搜索最相似的文本块，支持指定分块大小和重叠大小
     * @param queryEmbedding 查询向量
     * @param topK 返回的最大结果数量
     * @param chunkSize 分块大小，用于日志记录
     * @param overlapSize 重叠大小，用于日志记录
     * @return 按相似度排序的文本块列表
     */
    public List<Map<String, Object>> search(float[] queryEmbedding, int topK, int chunkSize, int overlapSize) {
        if (chunks == null || chunks.isEmpty()) {
            Log.w(TAG, "数据库为空，无法搜索");
            return new ArrayList<>();
        }
        
        if (queryEmbedding == null || queryEmbedding.length == 0) {
            Log.e(TAG, "查询向量为空");
            return new ArrayList<>();
        }
        
        // 记录搜索参数
        Log.d(TAG, "执行向量搜索: topK=" + topK + ", 分块大小=" + chunkSize + ", 重叠大小=" + overlapSize);
        
        // 计算每个文本块与查询向量的相似度
        List<Map<String, Object>> results = new ArrayList<>();
        for (TextChunk chunk : chunks) {
            float similarity = (float)cosineSimilarity(queryEmbedding, chunk.getEmbedding());
            
            Map<String, Object> result = new HashMap<>();
            result.put("text", chunk.getText());
            result.put("source", chunk.getSource());
            result.put("similarity", similarity);
            
            results.add(result);
        }
        
        // 按相似度降序排序
        results.sort((a, b) -> {
            float simA = (float) a.get("similarity");
            float simB = (float) b.get("similarity");
            return Float.compare(simB, simA); // 降序排序
        });
        
        // 限制返回结果数量
        if (results.size() > topK) {
            results = results.subList(0, topK);
        }
        
        Log.d(TAG, "搜索完成，找到 " + results.size() + " 个结果");
        return results;
    }
    
    /**
     * 搜索知识库
     * @param knowledgeBaseName 知识库名称
     * @param queryEmbedding 查询向量
     * @param topK 返回的最大结果数量
     * @return 按相似度排序的结果列表
     */
    public static List<Map<String, Object>> searchKnowledgeBase(String knowledgeBaseName, float[] queryEmbedding, int topK) {
        return searchKnowledgeBase(knowledgeBaseName, queryEmbedding, topK, 0, 0);
    }
    
    /**
     * 搜索知识库，支持指定分块大小和重叠大小
     * @param knowledgeBaseName 知识库名称
     * @param queryEmbedding 查询向量
     * @param topK 返回的最大结果数量
     * @param chunkSize 分块大小
     * @param overlapSize 重叠大小
     * @return 按相似度排序的结果列表
     */
    public static List<Map<String, Object>> searchKnowledgeBase(String knowledgeBaseName, float[] queryEmbedding, int topK, int chunkSize, int overlapSize) {
        // 获取知识库目录
        File kbDir = new VectorDatabaseHandler(null).getKnowledgeBaseDir(knowledgeBaseName);
        if (kbDir == null || !kbDir.exists()) {
            Log.e(TAG, "知识库目录不存在: " + knowledgeBaseName);
            return new ArrayList<>();
        }
        
        // 创建VectorDatabaseHandler实例
        VectorDatabaseHandler handler = new VectorDatabaseHandler(kbDir, "");
        
        // 加载数据库
        if (!handler.loadDatabase()) {
            Log.e(TAG, "加载知识库失败: " + knowledgeBaseName);
            return new ArrayList<>();
        }
        
        // 搜索
        return handler.search(queryEmbedding, topK, chunkSize, overlapSize);
    }
    
    /**
     * 添加笔记到知识库，使用提供的嵌入模型处理器生成嵌入向量
     * @param knowledgeBaseName 知识库名称
     * @param title 笔记标题
     * @param content 笔记内容
     * @param embeddingHandler 嵌入模型处理器
     * @return 是否添加成功
     */
    public boolean addNoteWithEmbedding(String knowledgeBaseName, String title, String content, 
                                       EmbeddingModelHandler embeddingHandler) {
        if (embeddingHandler == null) {
            Log.e(TAG, "嵌入模型处理器为空");
            return false;
        }
        
        File kbDir = getKnowledgeBaseDir(knowledgeBaseName);
        if (kbDir == null || !kbDir.exists()) {
            Log.e(TAG, "知识库目录不存在: " + knowledgeBaseName);
            return false;
        }
        
        Log.d(TAG, "正在向知识库添加笔记: " + knowledgeBaseName);
        Log.d(TAG, "标题: " + title);
        Log.d(TAG, "内容长度: " + content.length());
        
        try {
            // 创建VectorDatabaseHandler实例
            VectorDatabaseHandler handler = new VectorDatabaseHandler(kbDir, "");
            
            // 加载数据库
            if (!handler.loadDatabase()) {
                Log.e(TAG, "加载知识库失败: " + knowledgeBaseName);
                return false;
            }
            
            // 打印添加前的文本块数量和元数据
            int beforeChunkCount = handler.getChunkCount();
            Log.d(TAG, "添加笔记前文本块数量: " + beforeChunkCount);
            Log.d(TAG, "添加笔记前元数据 - 嵌入维度: " + handler.getMetadata().getEmbeddingDimension());
            Log.d(TAG, "添加笔记前元数据 - 文本块数量: " + handler.getMetadata().getChunkCount());
            Log.d(TAG, "添加笔记前元数据 - 来源数量: " + handler.getMetadata().getSources().size());
            
            // 生成内容的嵌入向量
            Log.d(TAG, "正在生成笔记内容的嵌入向量...");
            
            // 将标题和内容合并为一个文本
            String combinedText = "标题: " + title + "\n\n" + content;
            
            // 生成嵌入向量
            float[] embedding = embeddingHandler.generateEmbedding(combinedText);
            if (embedding == null || embedding.length == 0) {
                Log.e(TAG, "生成嵌入向量失败");
                return false;
            }
            Log.d(TAG, "嵌入向量生成成功，长度: " + embedding.length);
            
            // 设置笔记来源标识
            String source = "笔记: " + title;
            
            // 将笔记作为文本块添加到向量数据库
            handler.addChunk(combinedText, source, embedding);
            
            // 打印添加后的文本块数量和元数据
            int afterChunkCount = handler.getChunkCount();
            Log.d(TAG, "添加笔记后文本块数量: " + afterChunkCount);
            Log.d(TAG, "添加笔记后元数据 - 嵌入维度: " + handler.getMetadata().getEmbeddingDimension());
            Log.d(TAG, "添加笔记后元数据 - 文本块数量: " + handler.getMetadata().getChunkCount());
            Log.d(TAG, "添加笔记后元数据 - 来源数量: " + handler.getMetadata().getSources().size());
            Log.d(TAG, "添加笔记后元数据 - 来源列表: " + String.join(", ", handler.getMetadata().getSources()));
            
            // 保存成功
            Log.d(TAG, "添加笔记成功，文本块数量增加: " + (afterChunkCount - beforeChunkCount));
            return true;
        } catch (Exception e) {
            Log.e(TAG, "生成嵌入向量或添加笔记时发生异常", e);
            return false;
        }
    }
}
