package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.starlocalrag.tokenizers.HuggingfaceTokenizer;

/**
 * 分词器管理类，提供全局统一的分词器实例
 * 确保RAG问答、知识库构建和知识笔记创建过程使用相同的分词策略
 * 内部使用 HuggingfaceTokenizer 开源分词器实现
 */
public class TokenizerManager implements AutoCloseable {
    private static final String TAG = "StarLocalRAG_TokenizerMgr";
    
    // 单例实例
    private static TokenizerManager instance;
    
    // 是否已初始化
    private boolean initialized = false;
    
    // 应用上下文
    private Context context;
    
    // 分词器实例
    private HuggingfaceTokenizer tokenizer = null;
    
    // 特殊token
    private String clsToken = HuggingfaceTokenizer.CLS_TOKEN;
    private String sepToken = HuggingfaceTokenizer.SEP_TOKEN;
    private String unkToken = HuggingfaceTokenizer.UNK_TOKEN;
    private String padToken = HuggingfaceTokenizer.PAD_TOKEN;
    private String maskToken = HuggingfaceTokenizer.MASK_TOKEN;
    
    // 词汇表 (token -> id)
    private Map<String, Integer> vocab = new HashMap<>();
    
    // 反向词汇表 (id -> token)
    private Map<Integer, String> vocabReverse = new HashMap<>();
    
    // 是否使用一致性分词策略
    private boolean useConsistentTokenization = false;
    
    // 是否启用调试模式
    private boolean debugMode = false;
    
    /**
     * 私有构造函数，防止外部实例化
     */
    private TokenizerManager(Context context) {
        this.context = context.getApplicationContext();
        
        // 先不创建分词器实例，等待初始化时创建
        Log.i(TAG, "分词器管理器已创建，等待初始化");
    }
    
    /**
     * 获取单例实例
     * @param context 应用上下文
     * @return TokenizerManager实例
     */
    public static synchronized TokenizerManager getInstance(Context context) {
        if (instance == null) {
            instance = new TokenizerManager(context);
            Log.d(TAG, "创建分词器管理器实例");
        }
        return instance;
    }
    
    /**
     * 检查分词器是否已初始化
     * @return 是否已初始化
     */
    public boolean isInitialized() {
        return initialized;
    }
    
    /**
     * 初始化分词器
     * @param modelDir 模型目录
     * @return 是否初始化成功
     */
    public boolean initialize(File modelDir) {
        if (initialized) {
            Log.d(TAG, "分词器已初始化");
            return true;
        }
        
        try {
            Log.d(TAG, "开始初始化分词器，模型目录: " + modelDir.getAbsolutePath());
            boolean success = loadFromDirectory(modelDir);
            
            if (success) {
                Log.d(TAG, "分词器初始化成功，词汇表大小: " + getVocabSize());
                
                // 默认启用一致性分词策略
                setUseConsistentTokenization(true);
                Log.d(TAG, "已启用一致性分词策略");
                
                initialized = true;
                return true;
            } else {
                Log.e(TAG, "分词器初始化失败");
                return false;
            }
        } catch (Exception e) {
            Log.e(TAG, "分词器初始化异常: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 初始化分词器（接受字符串路径）
     * @param modelPath 模型路径
     * @return 是否初始化成功
     */
    public boolean initialize(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            Log.e(TAG, "模型路径为空");
            return false;
        }
        
        try {
            File modelDir = new File(modelPath);
            return initialize(modelDir);
        } catch (Exception e) {
            Log.e(TAG, "初始化分词器失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 获取分词器实例
     * @return 分词器实例，如果未初始化则返回null
     */
    public HuggingfaceTokenizer getTokenizer() {
        if (!initialized) {
            Log.w(TAG, "分词器未初始化，无法获取实例");
            return null;
        }
        return tokenizer;
    }
    
    /**
     * 设置是否使用一致性分词策略
     * @param useConsistent 是否使用一致性分词
     */
    public void setUseConsistentTokenization(boolean useConsistent) {
        this.useConsistentTokenization = useConsistent;
        Log.d(TAG, "设置一致性分词策略: " + useConsistent);
    }
    
    /**
     * 检查是否使用一致性分词策略
     * @return 是否使用一致性分词
     */
    public boolean isUseConsistentTokenization() {
        return useConsistentTokenization;
    }
    
    /**
     * 设置是否启用调试模式
     * @param debug 是否启用调试
     */
    public void setDebugMode(boolean debug) {
        this.debugMode = debug;
        Log.d(TAG, "设置调试模式: " + debug);
    }
    
    /**
     * 从目录加载分词器
     * @param directory 包含tokenizer.json的目录
     * @return 是否加载成功
     */
    public boolean loadFromDirectory(File directory) {
        if (!directory.exists() || !directory.isDirectory()) {
            Log.e(TAG, "指定的目录不存在: " + directory.getAbsolutePath());
            return false;
        }
        
        // 尝试查找tokenizer.json文件
        File tokenizerFile = new File(directory, "tokenizer.json");
        if (!tokenizerFile.exists() || !tokenizerFile.isFile()) {
            Log.e(TAG, "在目录中找不到tokenizer.json文件: " + directory.getAbsolutePath());
            return false;
        }
        
        try {
            // 关闭现有分词器
            if (tokenizer != null) {
                tokenizer.close();
                tokenizer = null;
            }
            
            // 从文件创建新的分词器实例
            tokenizer = new HuggingfaceTokenizer(tokenizerFile.getAbsolutePath(), useConsistentTokenization);
            if (tokenizer != null) {
                Log.i(TAG, "成功从文件加载分词器: " + tokenizerFile.getAbsolutePath());
                
                // 加载词汇表
                loadVocabFromFile(tokenizerFile);
                
                return true;
            } else {
                Log.e(TAG, "从文件加载分词器失败: " + tokenizerFile.getAbsolutePath());
                return false;
            }
        } catch (Exception e) {
            Log.e(TAG, "加载分词器时出错: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 从文件加载词汇表
     * @param jsonFile tokenizer.json文件
     */
    private void loadVocabFromFile(File jsonFile) {
        try {
            // 读取文件内容
            StringBuilder content = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new FileReader(jsonFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    content.append(line);
                }
            }
            
            // 解析JSON
            JSONObject json = new JSONObject(content.toString());
            if (json.has("model") && json.getJSONObject("model").has("vocab")) {
                JSONObject vocabObj = json.getJSONObject("model").getJSONObject("vocab");
                
                // 清空现有词汇表
                vocab.clear();
                vocabReverse.clear();
                
                // 加载词汇表
                Iterator<String> keys = vocabObj.keys();
                while (keys.hasNext()) {
                    String token = keys.next();
                    int id = vocabObj.getInt(token);
                    vocab.put(token, id);
                    vocabReverse.put(id, token);
                }
                
                Log.i(TAG, "成功加载词汇表，大小: " + vocab.size());
            } else {
                Log.w(TAG, "未在tokenizer.json中找到词汇表");
            }
        } catch (Exception e) {
            Log.e(TAG, "加载词汇表失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 对文本进行分词
     * @param text 输入文本
     * @return 分词结果（token ID列表）
     */
    public long[][] tokenize(String text) {
        if (tokenizer == null) {
            Log.e(TAG, "分词器未初始化");
            return new long[1][0];
        }
        
        try {
            // 使用HuggingfaceTokenizer分词
            return tokenizer.tokenizeToLongArray(text);
        } catch (Exception e) {
            Log.e(TAG, "分词器异常: " + e.getMessage(), e);
            return new long[1][0];
        }
    }
    
    /**
     * 从JSONArray加载词汇表
     * @param vocabArray 词汇表数组
     * @throws JSONException 如果JSON解析失败
     */
    public void loadVocabFromArray(JSONArray vocabArray) throws JSONException {
        vocab.clear();
        vocabReverse.clear();
        for (int i = 0; i < vocabArray.length(); i++) {
            String token = vocabArray.getString(i);
            vocab.put(token, i);
            vocabReverse.put(i, token);
        }
        Log.i(TAG, "从JSONArray加载词汇表，大小: " + vocab.size());
    }
    
    /**
     * 获取词汇表大小
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return vocab.size();
    }
    
    /**
     * 释放资源
     */
    @Override
    public void close() {
        if (tokenizer != null) {
            try {
                tokenizer.close();
                tokenizer = null;
                Log.i(TAG, "释放分词器资源");
            } catch (Exception e) {
                Log.e(TAG, "关闭分词器时出错: " + e.getMessage(), e);
            }
        }
    }
    
    /**
     * 重置分词器管理器
     * 用于在应用重启或模型切换时重新初始化
     */
    public static synchronized void reset() {
        if (instance != null) {
            Log.d(TAG, "重置分词器管理器");
            
            // 释放资源
            try {
                instance.close();
            } catch (Exception e) {
                Log.e(TAG, "释放分词器资源失败: " + e.getMessage(), e);
            }
            
            instance = null;
        }
    }
}
