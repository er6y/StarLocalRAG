package com.example.starlocalrag;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;
import android.util.Pair;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * SQLite向量数据库处理类，兼容Windows端Python生成的数据库格式
 */
public class SQLiteVectorDatabaseHandler {
    private static final String TAG = "StarLocalR...teVectorDB";
    
    // 数据库文件名
    private static final String DB_FILENAME = "vectorstore.db";
    private static final String METADATA_FILENAME = "metadata.json";
    
    // 数据库版本
    private static final int DATABASE_VERSION = 1;
    
    // 数据库表名
    private static final String TABLE_DOCUMENTS = "documents";
    
    // 数据库列名
    private static final String COLUMN_ID = "id";
    private static final String COLUMN_COLLECTION = "collection";
    private static final String COLUMN_CONTENT = "content";
    private static final String COLUMN_METADATA = "metadata";
    private static final String COLUMN_EMBEDDING = "embedding";
    
    // 数据库目录
    private final File databaseDir;
    private final Context context;
    
    // 数据库连接
    private SQLiteDatabase database;
    
    // 数据库元数据缓存
    private DatabaseMetadata metadata;
    
    /**
     * 数据库元数据类
     */
    public static class DatabaseMetadata {
        private String embeddingModel;
        private int embeddingDimension;
        private int chunkCount;
        private List<String> sources;
        private List<String> files;
        private int fileCount;
        private long creationTimestamp;
        private long lastModifiedTimestamp;
        private String collection;
        private String modelType;
        private String vectorStoreType;
        private String modeldir;
        
        public DatabaseMetadata(String embeddingModel) {
            this.embeddingModel = embeddingModel;
            this.embeddingDimension = 0;
            this.chunkCount = 0;
            this.sources = new ArrayList<>();
            this.files = new ArrayList<>();
            this.fileCount = 0;
            this.creationTimestamp = System.currentTimeMillis();
            this.lastModifiedTimestamp = this.creationTimestamp;
            this.collection = "";
            this.modelType = "local";
            this.vectorStoreType = "sqlite";
            this.modeldir = "";
        }
        
        public String getEmbeddingModel() {
            return embeddingModel;
        }
        
        public void setEmbeddingModel(String embeddingModel) {
            this.embeddingModel = embeddingModel;
            this.lastModifiedTimestamp = System.currentTimeMillis();
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
        
        public List<String> getFiles() {
            return files;
        }
        
        public void addFile(String file) {
            if (!files.contains(file)) {
                files.add(file);
                this.fileCount++;
                this.lastModifiedTimestamp = System.currentTimeMillis();
            }
        }
        
        public int getFileCount() {
            return fileCount;
        }
        
        public long getCreationTimestamp() {
            return creationTimestamp;
        }
        
        public long getLastModifiedTimestamp() {
            return lastModifiedTimestamp;
        }
        
        public String getCollection() {
            return collection;
        }
        
        public void setCollection(String collection) {
            this.collection = collection;
        }
        
        public String getModelType() {
            return modelType;
        }
        
        public void setModelType(String modelType) {
            this.modelType = modelType;
        }
        
        public String getVectorStoreType() {
            return vectorStoreType;
        }
        
        public void setVectorStoreType(String vectorStoreType) {
            this.vectorStoreType = vectorStoreType;
        }
        
        public String getModeldir() {
            return modeldir;
        }
        
        public void setModeldir(String modeldir) {
            this.modeldir = modeldir;
        }
    }
    
    /**
     * 搜索结果类
     */
    public static class SearchResult {
        public String text;
        public String source;
        public float similarity;
        
        public SearchResult(String text, String source, float similarity) {
            this.text = text;
            this.source = source;
            this.similarity = similarity;
        }
    }
    
    /**
     * SQLite数据库帮助类
     */
    private class VectorDatabaseHelper extends SQLiteOpenHelper {
        public VectorDatabaseHelper(Context context, String dbPath) {
            super(context, dbPath, null, DATABASE_VERSION);
            Log.d(TAG, "初始化VectorDatabaseHelper，数据库路径: " + dbPath);
        }
        
        @Override
        public void onCreate(SQLiteDatabase db) {
            Log.d(TAG, "正在创建数据库表...");
            try {
                // 创建文档表
                String createTableSql = "CREATE TABLE IF NOT EXISTS " + TABLE_DOCUMENTS + " (" +
                        COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                        COLUMN_COLLECTION + " TEXT, " +
                        COLUMN_CONTENT + " TEXT, " +
                        COLUMN_METADATA + " TEXT, " +
                        COLUMN_EMBEDDING + " BLOB)";
                db.execSQL(createTableSql);
                Log.d(TAG, "成功创建数据库表: " + TABLE_DOCUMENTS);
            } catch (Exception e) {
                Log.e(TAG, "创建数据库表失败: " + e.getMessage() + "\n堆栈: " + Log.getStackTraceString(e));
            }
        }
        
        @Override
        public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            Log.d(TAG, "正在升级数据库，旧版本: " + oldVersion + ", 新版本: " + newVersion);
            // 暂时不做任何操作，保留现有数据
        }
        
        @Override
        public void onOpen(SQLiteDatabase db) {
            super.onOpen(db);
            Log.d(TAG, "数据库已打开，路径: " + db.getPath() + ", 是否只读: " + db.isReadOnly());
        }
    }
    
    /**
     * 构造函数
     * @param databaseDir 数据库目录
     * @param embeddingModel 使用的嵌入模型名称
     */
    public SQLiteVectorDatabaseHandler(File databaseDir, String embeddingModel) {
        this.databaseDir = databaseDir;
        this.context = null;
        
        // 确保目录存在
        if (!databaseDir.exists()) {
            databaseDir.mkdirs();
        }
        
        // 打开或创建数据库
        openDatabase();
        
        // 加载或创建元数据
        if (!loadMetadata()) {
            this.metadata = new DatabaseMetadata(embeddingModel);
            this.metadata.setCollection(databaseDir.getName());
            saveMetadata();
        }
    }
    
    /**
     * 构造函数
     * @param databaseDir 数据库目录
     * @param embeddingModel 使用的嵌入模型名称
     * @param embeddingDimension 嵌入向量维度
     */
    public SQLiteVectorDatabaseHandler(File databaseDir, String embeddingModel, int embeddingDimension) {
        this.databaseDir = databaseDir;
        this.context = null;
        
        // 确保目录存在
        if (!databaseDir.exists()) {
            databaseDir.mkdirs();
        }
        
        // 打开或创建数据库
        openDatabase();
        
        // 加载或创建元数据
        if (!loadMetadata()) {
            this.metadata = new DatabaseMetadata(embeddingModel);
            this.metadata.setCollection(databaseDir.getName());
            this.metadata.setEmbeddingDimension(embeddingDimension);
            saveMetadata();
        } else if (this.metadata.getEmbeddingDimension() == 0) {
            // 如果已有元数据但维度为0，更新维度
            this.metadata.setEmbeddingDimension(embeddingDimension);
            saveMetadata();
        }
        
        Log.d(TAG, "初始化向量数据库，模型: " + embeddingModel + ", 向量维度: " + embeddingDimension);
    }
    
    /**
     * 构造函数
     * @param context 上下文
     */
    public SQLiteVectorDatabaseHandler(Context context) {
        this.context = context;
        this.databaseDir = null;
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
     * 打开数据库连接
     */
    private void openDatabase() {
        try {
            if (databaseDir == null) {
                Log.e(TAG, "数据库目录为null，无法打开数据库");
                return;
            }
            
            File dbFile = new File(databaseDir, DB_FILENAME);
            String dbPath = dbFile.getAbsolutePath();
            Log.i(TAG, "尝试打开数据库文件: " + dbPath + ", 目录是否存在: " + databaseDir.exists() + ", 文件是否存在: " + dbFile.exists());
            
            // 检查数据库文件大小
            if (dbFile.exists()) {
                Log.i(TAG, "数据库文件大小: " + dbFile.length() + " 字节");
            }
            
            // 检查数据库文件是否可读
            if (dbFile.exists() && !dbFile.canRead()) {
                Log.e(TAG, "数据库文件存在但无法读取: " + dbPath);
                return;
            }
            
            // 使用SQLiteOpenHelper打开数据库
            Context appContext = context != null ? context : GlobalApplication.getAppContext();
            if (appContext == null) {
                Log.e(TAG, "无法获取应用上下文，无法打开数据库");
                return;
            }
            
            Log.i(TAG, "使用上下文打开数据库: " + appContext.getPackageName());
            VectorDatabaseHelper helper = new VectorDatabaseHelper(appContext, dbPath);
            
            try {
                Log.i(TAG, "尝试以可写模式打开数据库...");
                database = helper.getWritableDatabase();
                Log.i(TAG, "数据库连接成功: " + dbPath);
            } catch (Exception e) {
                Log.e(TAG, "获取可写数据库失败: " + e.getMessage() + "\n堆栈: " + Log.getStackTraceString(e));
                // 尝试只读模式打开
                try {
                    Log.i(TAG, "尝试以只读模式打开数据库...");
                    database = helper.getReadableDatabase();
                    Log.i(TAG, "数据库以只读模式连接成功: " + dbPath);
                } catch (Exception e2) {
                    Log.e(TAG, "获取只读数据库也失败: " + e2.getMessage() + "\n堆栈: " + Log.getStackTraceString(e2));
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "打开数据库失败: " + e.getMessage() + "\n堆栈: " + Log.getStackTraceString(e));
        }
    }
    
    /**
     * 关闭数据库连接
     */
    public void close() {
        if (database != null && database.isOpen()) {
            database.close();
            Log.d(TAG, "数据库连接已关闭");
        }
    }
    
    /**
     * 保存数据库
     */
    public boolean saveDatabase() {
        // 保存元数据
        return saveMetadata();
    }
    
    /**
     * 加载数据库元数据
     * @return 是否加载成功
     */
    private boolean loadMetadata() {
        if (databaseDir == null) {
            Log.e(TAG, "数据库目录为null，无法加载元数据");
            return false;
        }
        
        try {
            // 从metadata.json文件加载元数据
            File metadataFile = new File(databaseDir, METADATA_FILENAME);
            if (!metadataFile.exists()) {
                Log.d(TAG, "元数据文件不存在，将创建新元数据");
                return false;
            }
            
            // 读取JSON文件
            StringBuilder jsonContent = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new FileReader(metadataFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    jsonContent.append(line);
                }
            }
            
            // 解析JSON
            JSONObject json = new JSONObject(jsonContent.toString());
            
            // 创建新的元数据对象
            metadata = new DatabaseMetadata("");
            
            // 从JSON中提取元数据
            if (json.has("embedding_model")) {
                metadata.embeddingModel = json.getString("embedding_model");
            }
            
            if (json.has("embedding_dimension")) {
                metadata.embeddingDimension = json.getInt("embedding_dimension");
            }
            
            if (json.has("chunk_count")) {
                metadata.chunkCount = json.getInt("chunk_count");
            }
            
            if (json.has("collection")) {
                metadata.collection = json.getString("collection");
            } else {
                metadata.collection = databaseDir.getName();
            }
            
            if (json.has("model_type")) {
                metadata.modelType = json.getString("model_type");
            }
            
            if (json.has("vector_store_type")) {
                metadata.vectorStoreType = json.getString("vector_store_type");
            }
            
            if (json.has("modeldir")) {
                metadata.modeldir = json.getString("modeldir");
            }
            
            // 解析创建时间
            if (json.has("created_at")) {
                String createdAtStr = json.getString("created_at");
                try {
                    // 尝试解析PC端格式的时间字符串
                    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSSSSS", Locale.getDefault());
                    Date createdDate = sdf.parse(createdAtStr);
                    if (createdDate != null) {
                        metadata.creationTimestamp = createdDate.getTime();
                    }
                } catch (Exception e) {
                    Log.w(TAG, "解析创建时间失败，使用当前时间: " + e.getMessage());
                    metadata.creationTimestamp = System.currentTimeMillis();
                }
            } else {
                metadata.creationTimestamp = System.currentTimeMillis();
            }
            
            if (json.has("last_modified_timestamp")) {
                metadata.lastModifiedTimestamp = json.getLong("last_modified_timestamp");
            } else {
                metadata.lastModifiedTimestamp = System.currentTimeMillis();
            }
            
            if (json.has("files")) {
                JSONArray filesArray = json.getJSONArray("files");
                for (int i = 0; i < filesArray.length(); i++) {
                    metadata.files.add(filesArray.getString(i));
                }
            }
            
            if (json.has("file_count")) {
                metadata.fileCount = json.getInt("file_count");
            } else if (metadata.files != null) {
                // 如果没有file_count但有files数组，使用数组长度
                metadata.fileCount = metadata.files.size();
            }
            
            // 加载来源列表
            if (json.has("sources")) {
                if (json.get("sources") instanceof JSONArray) {
                    JSONArray sourcesArray = json.getJSONArray("sources");
                    for (int i = 0; i < sourcesArray.length(); i++) {
                        metadata.sources.add(sourcesArray.getString(i));
                    }
                }
            }
            
            Log.d(TAG, "元数据加载成功，嵌入模型: " + metadata.getEmbeddingModel() +
                    ", 维度: " + metadata.getEmbeddingDimension() +
                    ", 文本块数量: " + metadata.getChunkCount());
            
            return true;
        } catch (Exception e) {
            Log.e(TAG, "加载元数据失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 保存数据库元数据
     * @return 是否保存成功
     */
    private boolean saveMetadata() {
        if (databaseDir == null) {
            Log.e(TAG, "数据库目录为null，无法保存元数据");
            return false;
        }
        
        if (metadata == null) {
            Log.e(TAG, "元数据为null，无法保存");
            return false;
        }
        
        try {
            // 创建JSON对象
            JSONObject json = new JSONObject();
            
            // 格式化创建时间
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSSSSS", Locale.getDefault());
            String formattedCreationTime = sdf.format(new Date(metadata.creationTimestamp));
            
            // 保存基本元数据，按照PC格式
            json.put("file_count", metadata.fileCount);
            json.put("chunk_count", metadata.chunkCount);
            json.put("files", new JSONArray(metadata.files));
            json.put("embedding_model", metadata.embeddingModel);
            json.put("model_type", metadata.modelType);
            json.put("created_at", formattedCreationTime);
            json.put("vector_store_type", metadata.vectorStoreType);
            json.put("modeldir", metadata.modeldir);
            
            // 添加嵌入维度信息，确保与PC端兼容
            if (metadata.getEmbeddingDimension() > 0) {
                json.put("embedding_dimension", metadata.getEmbeddingDimension());
            }
            
            // 添加来源信息，确保与PC端兼容
            if (metadata.sources != null && !metadata.sources.isEmpty()) {
                json.put("sources", new JSONArray(metadata.sources));
            }
            
            // 添加集合名称，确保与PC端兼容
            if (metadata.getCollection() != null && !metadata.getCollection().isEmpty()) {
                json.put("collection", metadata.getCollection());
            }
            
            // 写入JSON文件
            File metadataFile = new File(databaseDir, METADATA_FILENAME);
            try (FileWriter writer = new FileWriter(metadataFile)) {
                writer.write(json.toString(4)); // 使用4个空格缩进，美化输出
            }
            
            Log.d(TAG, "元数据保存成功");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "保存元数据失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 将嵌入向量转换为二进制数据
     * @param vector 向量
     * @return 二进制数据
     */
    private byte[] vectorToBlob(float[] vector) {
        // 使用小端序，与PC端保持一致
        byte[] embeddingBytes = new byte[vector.length * 4]; // 每个float占4字节
        ByteBuffer buffer = ByteBuffer.wrap(embeddingBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : vector) {
            buffer.putFloat(value);
        }
        return embeddingBytes;
    }
    
    /**
     * 将二进制数据转换为向量
     * @param blob 二进制数据
     * @return 向量
     */
    private float[] blobToVector(byte[] blob) {
        // 使用小端序，与PC端保持一致
        float[] vector = new float[blob.length / 4];
        ByteBuffer.wrap(blob)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
            .get(vector);
        return vector;
    }
    
    /**
     * 添加文本块到数据库
     * @param text 文本内容
     * @param source 来源信息
     * @param embedding 嵌入向量
     * @return 是否添加成功
     */
    public boolean addChunk(String text, String source, float[] embedding) {
        if (database == null || !database.isOpen()) {
            Log.e(TAG, "数据库未打开，无法添加文本块");
            return false;
        }
        
        if (text == null || text.isEmpty()) {
            Log.e(TAG, "文本内容为空，无法添加文本块");
            return false;
        }
        
        if (embedding == null || embedding.length == 0) {
            Log.e(TAG, "嵌入向量为空，无法添加文本块");
            return false;
        }
        
        try {
            // 开始事务
            database.beginTransaction();
            
            // 将嵌入向量转换为字节数组 - 使用小端序，与PC端保持一致
            byte[] embeddingBytes = vectorToBlob(embedding);
            
            // 创建元数据JSON
            JSONObject metadataJson = new JSONObject();
            metadataJson.put("source", source != null ? source : "unknown");
            metadataJson.put("created_at", System.currentTimeMillis());
            
            // 准备插入数据
            ContentValues values = new ContentValues();
            values.put(COLUMN_COLLECTION, metadata.getCollection());
            values.put(COLUMN_CONTENT, text);
            values.put(COLUMN_METADATA, metadataJson.toString());
            values.put(COLUMN_EMBEDDING, embeddingBytes);
            
            // 插入数据
            long rowId = database.insert(TABLE_DOCUMENTS, null, values);
            
            if (rowId == -1) {
                Log.e(TAG, "插入文本块失败");
                return false;
            }
            
            // 更新元数据
            if (metadata != null) {
                // 更新文本块数量
                metadata.incrementChunkCount();
                
                // 如果来源不在列表中，添加到来源列表
                if (source != null && !source.isEmpty() && !metadata.sources.contains(source)) {
                    metadata.addSource(source);
                }
                
                // 如果嵌入维度未设置，设置嵌入维度
                if (metadata.getEmbeddingDimension() == 0) {
                    metadata.setEmbeddingDimension(embedding.length);
                }
            }
            
            // 提交事务
            database.setTransactionSuccessful();
            
            Log.d(TAG, "成功添加文本块，ID: " + rowId);
            return true;
        } catch (Exception e) {
            Log.e(TAG, "添加文本块失败: " + e.getMessage(), e);
            return false;
        } finally {
            if (database.inTransaction()) {
                database.endTransaction();
            }
        }
    }
    
    /**
     * 从数据库加载所有文本块
     * @return 是否加载成功
     */
    public boolean loadDatabase() {
        if (databaseDir == null) {
            Log.e(TAG, "数据库目录为null，无法加载数据库");
            return false;
        }
        
        File dbFile = new File(databaseDir, DB_FILENAME);
        Log.i(TAG, "尝试加载数据库文件: " + dbFile.getAbsolutePath());
        
        if (!dbFile.exists()) {
            Log.e(TAG, "数据库文件不存在: " + dbFile.getAbsolutePath());
            return false;
        }
        
        // 检查数据库文件大小和权限
        Log.i(TAG, "数据库文件大小: " + dbFile.length() + " 字节, 可读: " + dbFile.canRead() + ", 可写: " + dbFile.canWrite());
        
        try {
            // 打开数据库
            Log.i(TAG, "开始打开数据库连接...");
            openDatabase();
            
            if (database == null || !database.isOpen()) {
                Log.e(TAG, "数据库打开失败，database对象为null或未打开");
                return false;
            }
            
            Log.i(TAG, "数据库连接成功，database对象: " + database + ", 路径: " + database.getPath() + ", 是否只读: " + database.isReadOnly());
            
            // 加载元数据
            Log.i(TAG, "开始加载元数据...");
            if (!loadMetadata()) {
                Log.e(TAG, "加载元数据失败");
                return false;
            }
            
            // 检查文档表是否存在
            Log.i(TAG, "检查文档表是否存在: " + TABLE_DOCUMENTS);
            Cursor cursor = database.rawQuery(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    new String[]{TABLE_DOCUMENTS});
            
            boolean tableExists = cursor.getCount() > 0;
            cursor.close();
            
            if (!tableExists) {
                Log.e(TAG, "文档表不存在: " + TABLE_DOCUMENTS);
                return false;
            }
            
            Log.i(TAG, "文档表存在: " + TABLE_DOCUMENTS);
            
            // 检查表结构
            try {
                Log.i(TAG, "检查表结构...");
                Cursor columnCursor = database.rawQuery("PRAGMA table_info(" + TABLE_DOCUMENTS + ")", null);
                StringBuilder columnsInfo = new StringBuilder("表结构信息:\n");
                while (columnCursor.moveToNext()) {
                    String columnName = columnCursor.getString(1);
                    String columnType = columnCursor.getString(2);
                    columnsInfo.append("列名: ").append(columnName).append(", 类型: ").append(columnType).append("\n");
                }
                columnCursor.close();
                Log.i(TAG, columnsInfo.toString());
            } catch (Exception e) {
                Log.w(TAG, "获取表结构信息失败: " + e.getMessage());
            }
            
            // 检查文本块数量
            Log.i(TAG, "检查文本块数量...");
            cursor = database.rawQuery("SELECT COUNT(*) FROM " + TABLE_DOCUMENTS, null);
            cursor.moveToFirst();
            int count = cursor.getInt(0);
            cursor.close();
            
            Log.i(TAG, "数据库加载成功，共 " + count + " 个文本块");
            
            return true;
        } catch (Exception e) {
            Log.e(TAG, "加载数据库失败: " + e.getMessage() + "\n堆栈: " + Log.getStackTraceString(e));
            return false;
        }
    }
    
    /**
     * 获取数据库中的文本块数量
     * @return 文本块数量
     */
    public int getChunkCount() {
        if (database == null || !database.isOpen()) {
            Log.e(TAG, "数据库未打开，无法获取文本块数量");
            return 0;
        }
        
        try {
            Cursor cursor = database.rawQuery("SELECT COUNT(*) FROM " + TABLE_DOCUMENTS, null);
            cursor.moveToFirst();
            int count = cursor.getInt(0);
            cursor.close();
            
            return count;
        } catch (Exception e) {
            Log.e(TAG, "获取文本块数量失败: " + e.getMessage(), e);
            return 0;
        }
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
    private float cosineSimilarity(float[] vec1, float[] vec2) {
        if (vec1.length != vec2.length) {
            Log.e(TAG, "向量维度不匹配: vec1长度=" + vec1.length + ", vec2长度=" + vec2.length);
            throw new IllegalArgumentException("向量维度不匹配");
        }
        
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            normA += vec1[i] * vec1[i];
            normB += vec2[i] * vec2[i];
        }
        
        if (normA <= 1e-6 || normB <= 1e-6) {
            Log.w(TAG, "向量范数接近零，返回零相似度");
            return 0;
        }
        
        float similarity = dotProduct / (float)(Math.sqrt(normA) * Math.sqrt(normB));
        return similarity;
    }
    
    /**
     * 搜索与查询向量最相似的文本块
     * @param queryVector 查询向量
     * @param topK 返回的最大结果数量
     * @return 搜索结果列表，按相似度降序排序
     */
    public List<SearchResult> searchSimilar(float[] queryVector, int topK) {
        List<SearchResult> results = new ArrayList<>();
        
        if (database == null || !database.isOpen()) {
            Log.e(TAG, "数据库未打开，无法搜索");
            return results;
        }
        
        if (queryVector == null || queryVector.length == 0) {
            Log.e(TAG, "查询向量为空，无法搜索");
            return results;
        }
        
        try {
            // 获取当前知识库的文档
            String collection = metadata.getCollection();
            Cursor cursor = database.query(
                    TABLE_DOCUMENTS,
                    new String[]{COLUMN_ID, COLUMN_CONTENT, COLUMN_METADATA, COLUMN_EMBEDDING},
                    COLUMN_COLLECTION + "=?",
                    new String[]{collection},
                    null, null, null);
            
            // 创建结果列表，用于存储相似度和对应的文本块
            List<Pair<Float, SearchResult>> similarityResults = new ArrayList<>();
            
            // 遍历所有文本块
            while (cursor.moveToNext()) {
                int contentIndex = cursor.getColumnIndex(COLUMN_CONTENT);
                int metadataIndex = cursor.getColumnIndex(COLUMN_METADATA);
                int embeddingIndex = cursor.getColumnIndex(COLUMN_EMBEDDING);
                
                // 检查列索引是否有效
                if (contentIndex == -1 || metadataIndex == -1 || embeddingIndex == -1) {
                    Log.e(TAG, "数据库表结构不匹配，缺少必要列");
                    continue;
                }
                
                String content = cursor.getString(contentIndex);
                String metadataJson = cursor.getString(metadataIndex);
                byte[] embeddingBlob = cursor.getBlob(embeddingIndex);
                
                // 解析元数据
                String source = "unknown";
                try {
                    JSONObject metadataObj = new JSONObject(metadataJson);
                    if (metadataObj.has("source")) {
                        source = metadataObj.getString("source");
                    }
                } catch (JSONException e) {
                    Log.w(TAG, "解析元数据失败: " + e.getMessage());
                }
                
                // 将字节数组转换为浮点数组 - 使用小端序，与PC端保持一致
                float[] embedding = blobToVector(embeddingBlob);
                
                // 计算余弦相似度
                float similarity = cosineSimilarity(queryVector, embedding);
                
                // 添加到结果列表
                similarityResults.add(new Pair<>(similarity, new SearchResult(content, source, similarity)));
            }
            
            cursor.close();
            
            // 按相似度降序排序
            Collections.sort(similarityResults, (a, b) -> Float.compare(b.first, a.first));
            
            // 取前topK个结果
            int resultCount = Math.min(topK, similarityResults.size());
            for (int i = 0; i < resultCount; i++) {
                results.add(similarityResults.get(i).second);
            }
            
            Log.d(TAG, "搜索完成，找到 " + results.size() + " 个相似文本块");
            return results;
        } catch (Exception e) {
            Log.e(TAG, "搜索相似文本块失败: " + e.getMessage(), e);
            return results;
        }
    }
    
    /**
     * 批量添加文本块到数据库
     * @param texts 文本内容列表
     * @param sources 来源信息列表
     * @param embeddings 嵌入向量列表
     * @return 是否添加成功
     */
    public boolean addChunks(List<String> texts, List<String> sources, List<float[]> embeddings) {
        if (database == null || !database.isOpen()) {
            Log.e(TAG, "数据库未打开，无法添加文本块");
            return false;
        }
        
        if (texts == null || texts.isEmpty()) {
            Log.e(TAG, "文本内容列表为空，无法添加文本块");
            return false;
        }
        
        if (embeddings == null || embeddings.isEmpty()) {
            Log.e(TAG, "嵌入向量列表为空，无法添加文本块");
            return false;
        }
        
        if (texts.size() != embeddings.size()) {
            Log.e(TAG, "文本内容列表和嵌入向量列表长度不匹配");
            return false;
        }
        
        if (sources != null && sources.size() != texts.size()) {
            Log.e(TAG, "来源信息列表和文本内容列表长度不匹配");
            return false;
        }
        
        try {
            // 开始事务
            database.beginTransaction();
            
            for (int i = 0; i < texts.size(); i++) {
                String text = texts.get(i);
                String source = (sources != null && i < sources.size()) ? sources.get(i) : "unknown";
                float[] embedding = embeddings.get(i);
                
                // 将嵌入向量转换为字节数组 - 使用小端序，与PC端保持一致
                byte[] embeddingBytes = vectorToBlob(embedding);
                
                // 创建元数据JSON
                JSONObject metadataJson = new JSONObject();
                metadataJson.put("source", source);
                metadataJson.put("created_at", System.currentTimeMillis());
                
                // 准备插入数据
                ContentValues values = new ContentValues();
                values.put(COLUMN_COLLECTION, metadata.getCollection());
                values.put(COLUMN_CONTENT, text);
                values.put(COLUMN_METADATA, metadataJson.toString());
                values.put(COLUMN_EMBEDDING, embeddingBytes);
                
                // 插入数据
                long rowId = database.insert(TABLE_DOCUMENTS, null, values);
                
                if (rowId == -1) {
                    Log.e(TAG, "插入文本块失败");
                    return false;
                }
                
                // 更新元数据
                if (metadata != null) {
                    // 更新文本块数量
                    metadata.incrementChunkCount();
                    
                    // 如果来源不在列表中，添加到来源列表
                    if (source != null && !source.isEmpty() && !metadata.sources.contains(source)) {
                        metadata.addSource(source);
                    }
                    
                    // 如果嵌入维度未设置，设置嵌入维度
                    if (metadata.getEmbeddingDimension() == 0 && embedding.length > 0) {
                        metadata.setEmbeddingDimension(embedding.length);
                    }
                }
            }
            
            // 提交事务
            database.setTransactionSuccessful();
            
            Log.d(TAG, "成功批量添加 " + texts.size() + " 个文本块");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "批量添加文本块失败: " + e.getMessage(), e);
            return false;
        } finally {
            if (database.inTransaction()) {
                database.endTransaction();
            }
        }
    }
    
    /**
     * 添加单个文本块和向量到数据库
     * @param text 文本内容
     * @param embedding 嵌入向量
     * @param source 来源信息
     * @param metadataStr 元数据（JSON字符串）
     * @return 是否添加成功
     */
    public boolean addVector(String text, float[] embedding, String source, String metadataStr) {
        if (database == null || !database.isOpen()) {
            Log.e(TAG, "数据库未打开，无法添加文本块");
            return false;
        }
        
        if (text == null || text.isEmpty()) {
            Log.e(TAG, "文本内容为空，无法添加文本块");
            return false;
        }
        
        if (embedding == null || embedding.length == 0) {
            Log.e(TAG, "嵌入向量为空，无法添加文本块");
            return false;
        }
        
        try {
            // 将嵌入向量转换为字节数组 - 使用小端序，与PC端保持一致
            byte[] embeddingBytes = vectorToBlob(embedding);
            
            // 创建元数据JSON
            JSONObject finalMetadata;
            try {
                // 尝试解析传入的元数据字符串
                finalMetadata = new JSONObject(metadataStr);
            } catch (JSONException e) {
                // 如果解析失败，创建新的元数据对象
                finalMetadata = new JSONObject();
                Log.w(TAG, "解析元数据字符串失败，创建新的元数据对象: " + e.getMessage());
            }
            
            // 添加基本元数据
            if (!finalMetadata.has("source") && source != null && !source.isEmpty()) {
                finalMetadata.put("source", source);
            }
            if (!finalMetadata.has("created_at")) {
                finalMetadata.put("created_at", System.currentTimeMillis());
            }
            
            // 准备插入数据
            ContentValues values = new ContentValues();
            values.put(COLUMN_COLLECTION, metadata.getCollection());
            values.put(COLUMN_CONTENT, text);
            values.put(COLUMN_METADATA, finalMetadata.toString());
            values.put(COLUMN_EMBEDDING, embeddingBytes);
            
            // 插入数据
            long rowId = database.insert(TABLE_DOCUMENTS, null, values);
            
            if (rowId == -1) {
                Log.e(TAG, "插入文本块失败");
                return false;
            }
            
            // 更新元数据
            if (metadata != null) {
                // 更新文本块数量
                metadata.incrementChunkCount();
                
                // 如果来源不在列表中，添加到来源列表
                if (source != null && !source.isEmpty() && !metadata.sources.contains(source)) {
                    metadata.addSource(source);
                }
                
                // 如果嵌入维度未设置，设置嵌入维度
                if (metadata.getEmbeddingDimension() == 0 && embedding.length > 0) {
                    metadata.setEmbeddingDimension(embedding.length);
                }
            }
            
            Log.d(TAG, "成功添加文本块: " + (text.length() > 30 ? text.substring(0, 30) + "..." : text));
            return true;
        } catch (Exception e) {
            Log.e(TAG, "添加文本块失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 添加文件到元数据
     * @param fileName 文件名
     */
    public void addFileToMetadata(String fileName) {
        if (metadata != null) {
            metadata.addFile(fileName);
            Log.d(TAG, "添加文件到元数据: " + fileName);
        }
    }
    
    /**
     * 更新嵌入模型路径
     * @param embeddingModelPath 新的嵌入模型路径
     * @return 是否更新成功
     */
    public boolean updateEmbeddingModel(String embeddingModelPath) {
        if (metadata == null) {
            Log.e(TAG, "元数据对象为空，无法更新嵌入模型路径");
            return false;
        }
        
        try {
            Log.d(TAG, "更新嵌入模型路径: " + embeddingModelPath);
            metadata.setEmbeddingModel(embeddingModelPath);
            return true;
        } catch (Exception e) {
            Log.e(TAG, "更新嵌入模型路径失败", e);
            return false;
        }
    }
}
