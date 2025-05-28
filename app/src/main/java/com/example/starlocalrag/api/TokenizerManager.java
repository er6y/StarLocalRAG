package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

import com.starlocalrag.tokenizers.HuggingfaceTokenizer;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
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
public class TokenizerManager implements TokenizerInterface {
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
    
    // 特殊token映射
    private Map<String, Integer> specialTokens = new HashMap<>();
    private Map<Integer, String> specialTokensReverse = new HashMap<>();
    
    // 模型类型
    private String modelType = "";
    
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
     * @param modelPath 模型目录或模型文件路径
     * @return 是否初始化成功
     */
    public boolean initialize(File modelPath) {
        if (initialized && tokenizer != null) {
            // 如果已经初始化且分词器实例存在，直接返回成功
            Log.d(TAG, "分词器已初始化，直接使用现有实例");
            return true;
        }
        
        // 如果已初始化但tokenizer为null，重置状态
        if (initialized && tokenizer == null) {
            Log.w(TAG, "分词器状态不一致，重新初始化");
            initialized = false;
        }
        
        try {
            // 判断传入的是文件还是目录
            File modelDir = modelPath;
            if (modelPath.isFile()) {
                // 如果是文件，使用其父目录
                Log.d(TAG, "检测到传入的是模型文件而非目录，将使用其父目录: " + modelPath.getAbsolutePath());
                modelDir = modelPath.getParentFile();
                if (modelDir == null || !modelDir.exists() || !modelDir.isDirectory()) {
                    Log.e(TAG, "无法获取模型文件的父目录，或父目录不存在: " + modelPath.getAbsolutePath());
                    return false;
                }
            }
            
            Log.d(TAG, "开始初始化分词器，模型目录: " + modelDir.getAbsolutePath());
            boolean success = loadFromDirectory(modelDir);
            
            if (success) {
                Log.d(TAG, "分词器初始化成功，词汇表大小: " + getVocabSize());
                
                // 默认启用一致性分词策略
                setUseConsistentTokenization(true);
                Log.d(TAG, "已启用一致性分词策略");
                
                // 从分词器中同步特殊token
                syncSpecialTokensFromTokenizer();
                Log.d(TAG, "已从分词器中同步特殊token，数量: " + specialTokens.size());
                
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
     * @param modelPath 模型路径（可以是目录或文件路径）
     * @return 是否初始化成功
     */
    public boolean initialize(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            Log.e(TAG, "模型路径为空");
            return false;
        }
        
        try {
            File modelFile = new File(modelPath);
            return initialize(modelFile);
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
     * 从指定目录加载分词器
     * @param directory 包含tokenizer.json的目录
     * @return 是否成功加载
     */
    public boolean loadFromDirectory(File directory) {
        if (directory == null) {
            Log.e(TAG, "目录对象为空");
            return false;
        }
        
        if (!directory.exists() || !directory.isDirectory()) {
            Log.e(TAG, "指定的目录不存在或不是目录: " + directory.getAbsolutePath());
            return false;
        }
        
        // 尝试查找tokenizer.json文件
        File tokenizerFile = new File(directory, "tokenizer.json");
        if (!tokenizerFile.exists() || !tokenizerFile.isFile()) {
            Log.e(TAG, "在目录中找不到tokenizer.json文件: " + directory.getAbsolutePath());
            return false;
        }
        
        // 检查文件大小和权限
        if (tokenizerFile.length() == 0) {
            Log.e(TAG, "tokenizer.json文件大小为0: " + tokenizerFile.getAbsolutePath());
            return false;
        }
        
        if (!tokenizerFile.canRead()) {
            Log.e(TAG, "tokenizer.json文件无法读取: " + tokenizerFile.getAbsolutePath());
            return false;
        }
        
        // 检查当前分词器是否已经加载了相同的文件
        if (tokenizer != null) {
            try {
                String currentModelPath = tokenizer.getModelPath();
                if (currentModelPath != null && currentModelPath.equals(tokenizerFile.getAbsolutePath())) {
                    Log.d(TAG, "当前分词器已经加载了相同的模型文件，无需重新加载");
                    return true;
                }
                
                // 如果是不同的模型文件，关闭现有分词器
                Log.d(TAG, "加载新的模型文件，关闭现有分词器");
                try {
                    tokenizer.close();
                } catch (Exception e) {
                    Log.w(TAG, "关闭现有分词器时出错", e);
                } finally {
                    tokenizer = null;
                }
            } catch (Exception e) {
                Log.w(TAG, "检查当前分词器模型路径时出错", e);
                // 关闭现有分词器
                try {
                    tokenizer.close();
                } catch (Exception ex) {
                    Log.w(TAG, "关闭现有分词器时出错", ex);
                } finally {
                    tokenizer = null;
                }
            }
        }
        
        try {
            // 读取tokenizer.json文件内容进行验证
            JSONObject tokenizerJson = null;
            try {
                String tokenizerContent = readFileContent(tokenizerFile);
                if (tokenizerContent.isEmpty()) {
                    Log.e(TAG, "tokenizer.json文件内容为空: " + tokenizerFile.getAbsolutePath());
                    return false;
                }
                
                // 验证JSON格式
                try {
                    tokenizerJson = new JSONObject(tokenizerContent);
                    Log.d(TAG, "tokenizer.json文件格式有效");
                } catch (JSONException e) {
                    Log.e(TAG, "tokenizer.json文件不是有效的JSON格式: " + e.getMessage(), e);
                    return false;
                }
            } catch (IOException e) {
                Log.e(TAG, "读取tokenizer.json文件内容失败: " + e.getMessage(), e);
                return false;
            }
            
            // 从文件创建新的分词器实例
            Log.d(TAG, "尝试创建分词器实例，模型路径: " + tokenizerFile.getAbsolutePath());
            try {
                // 正确传递参数，第二个参数应该是true，表示这是一个文件路径
                tokenizer = new HuggingfaceTokenizer(tokenizerFile.getAbsolutePath(), true);
                
                // 如果需要设置一致性分词，可以在这里设置
                if (tokenizer != null) {
                    Log.i(TAG, "成功从文件加载分词器: " + tokenizerFile.getAbsolutePath());
                    
                        // 直接从 HuggingfaceTokenizer 实例中获取特殊token信息
                    // 不再重复加载词汇表和特殊token，因为 HuggingfaceTokenizer 在创建时已经加载了
                    syncSpecialTokensFromTokenizer();
                    
                    // 记录模型类型，如果可用
                    if (tokenizerJson != null && tokenizerJson.has("model") && tokenizerJson.getJSONObject("model").has("type")) {
                        try {
                            String type = tokenizerJson.getJSONObject("model").getString("type");
                            setModelType(type);
                            Log.i(TAG, "识别到模型类型: " + type);
                        } catch (Exception e) {
                            Log.w(TAG, "获取模型类型失败");
                        }
                    }
                    
                    return true;
                } else {
                    Log.e(TAG, "分词器创建成功但实例为空");
                    return false;
                }
            } catch (IllegalArgumentException e) {
                Log.e(TAG, "创建分词器失败，参数错误: " + e.getMessage(), e);
                return false;
            } catch (UnsatisfiedLinkError e) {
                Log.e(TAG, "创建分词器失败，本地库加载错误: " + e.getMessage(), e);
                return false;
            } catch (Exception e) {
                Log.e(TAG, "创建分词器失败，未知错误: " + e.getMessage(), e);
                return false;
            }
        } catch (Exception e) {
            Log.e(TAG, "加载分词器过程中发生未捕获的异常: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 读取文件内容
     * @param file 要读取的文件
     * @return 文件内容字符串
     * @throws IOException 读取错误
     */
    private String readFileContent(File file) throws IOException {
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line).append("\n");
            }
        }
        return content.toString();
    }
    
    // loadVocabFromFile 方法已被移除，因为它已被 syncSpecialTokensFromTokenizer 方法替代
    
    /**
     * 从HuggingfaceTokenizer实例中同步特殊token信息
     * 这个方法替代了原来的loadVocabFromJson和loadSpecialTokensFromJson方法
     * 因为HuggingfaceTokenizer在创建时已经加载了特殊token
     */
    private void syncSpecialTokensFromTokenizer() {
        if (tokenizer == null) {
            Log.w(TAG, "分词器实例为空，无法同步特殊token");
            return;
        }
        
        try {
            // 清空现有特殊token和词汇表
            specialTokens.clear();
            specialTokensReverse.clear();
            vocab.clear();
            vocabReverse.clear();
            
            // 从分词器实例中获取特殊token
            Map<String, String> tokenizerSpecialTokens = tokenizer.getSpecialTokens();
            if (tokenizerSpecialTokens != null && !tokenizerSpecialTokens.isEmpty()) {
                // 记录词汇表大小
                int vocabSize = tokenizer.getVocabSize();
                Log.i(TAG, "分词器词汇表大小: " + vocabSize);
                
                // 获取关键特殊token
                for (Map.Entry<String, String> entry : tokenizerSpecialTokens.entrySet()) {
                    String tokenType = entry.getKey();
                    String tokenContent = entry.getValue();
                    
                    // 更新特殊token字段
                    if ("cls_token".equals(tokenType)) {
                        clsToken = tokenContent;
                    } else if ("sep_token".equals(tokenType)) {
                        sepToken = tokenContent;
                    } else if ("unk_token".equals(tokenType)) {
                        unkToken = tokenContent;
                    } else if ("pad_token".equals(tokenType)) {
                        padToken = tokenContent;
                    } else if ("mask_token".equals(tokenType)) {
                        maskToken = tokenContent;
                    }
                    
                    // 记录特殊token
                    // 注意：这里我们不知道特殊token的ID，但这不重要
                    // 因为实际的分词和解码操作是由HuggingfaceTokenizer实例处理的
                    if (debugMode) {
                        Log.d(TAG, "同步特殊token: " + tokenType + " -> " + tokenContent);
                    }
                }
                
                Log.i(TAG, "成功同步特殊token，数量: " + tokenizerSpecialTokens.size());
            } else {
                Log.w(TAG, "分词器实例中没有特殊token");
            }
        } catch (Exception e) {
            Log.w(TAG, "同步特殊token失败: " + e.getMessage());
            // 即使同步失败也不中断流程，因为分词器仍然可以使用默认特殊token
        }
    }
    
    // loadSpecialTokensFromJson 方法已被移除，因为它已被 syncSpecialTokensFromTokenizer 方法替代
    
    // loadModelSpecialTokens 方法已被移除，因为它已被 syncSpecialTokensFromTokenizer 方法替代
    
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
    public static synchronized void resetManager() {
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
    
    /**
     * 将token ID解码为文本
     * @param ids token ID数组
     * @return 解码后的文本
     */
    @Override
    public String decodeIds(int[] ids) {
        if (tokenizer == null) {
            Log.e(TAG, "分词器未初始化");
            return "";
        }
        
        try {
            // 使用原生的HuggingfaceTokenizer解码功能
            if (debugMode) {
                Log.d(TAG, "使用HuggingfaceTokenizer解码，ID数量: " + ids.length);
            }
            
            // 如果是特定模型，可以添加特殊处理
            if (modelType != null && !modelType.isEmpty()) {
                // 这里可以根据模型类型添加特殊处理逻辑
                if (debugMode) {
                    Log.d(TAG, "当前模型类型: " + modelType);
                }
            }
            
            // 回退到使用HuggingfaceTokenizer解码
            if (debugMode) {
                Log.d(TAG, "使用原始HuggingfaceTokenizer解码");
            }
            return tokenizer.decode(ids);
        } catch (Exception e) {
            Log.e(TAG, "解码异常: " + e.getMessage(), e);
            return "";
        }
    }
    
    /**
     * 根据ID获取token字符串
     * @param id token ID
     * @return token字符串，如果不存在则返回null
     */
    private String getTokenFromId(int id) {
        // 首先检查是否是特殊token
        String specialToken = getSpecialTokenContent(id);
        if (specialToken != null) {
            return specialToken;
        }
        
        // 然后检查词汇表
        if (vocabReverse.containsKey(id)) {
            return vocabReverse.get(id);
        }
        
        // 最后尝试从分词器获取
        try {
            if (tokenizer != null) {
                int[] singleId = {id};
                String token = tokenizer.decode(singleId).trim();
                if (!token.isEmpty()) {
                    return token;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "从分词器获取token失败: " + e.getMessage());
        }
        
        return null;
    }
    
    /**
     * 重置分词器
     */
    @Override
    public void reset() {
        // 清空词汇表和特殊token
        vocab.clear();
        vocabReverse.clear();
        specialTokens.clear();
        specialTokensReverse.clear();
        
        // 重置状态
        initialized = false;
        
        // 释放分词器资源
        if (tokenizer != null) {
            try {
                tokenizer.close();
                tokenizer = null;
            } catch (Exception e) {
                Log.e(TAG, "释放分词器资源失败: " + e.getMessage(), e);
            }
        }
        
        Log.d(TAG, "分词器已重置");
    }
    
    /**
     * 获取特殊token的ID
     * @param token 特殊token的内容
     * @return 特殊token的ID，如果不存在则返回-1
     */
    @Override
    public int getSpecialTokenId(String token) {
        Integer id = specialTokens.get(token);
        return id != null ? id : -1;
    }
    
    /**
     * 根据ID获取特殊token的内容
     * @param id token的ID
     * @return 特殊token的内容，如果不存在则返回null
     */
    @Override
    public String getSpecialTokenContent(int id) {
        return specialTokensReverse.get(id);
    }
    
    /**
     * 检查给定的ID是否为特殊token
     * @param id token的ID
     * @return 是否为特殊token
     */
    @Override
    public boolean isSpecialToken(int id) {
        return specialTokensReverse.containsKey(id);
    }
    
    /**
     * 获取所有特殊token的映射（内容到ID）
     * @return 特殊token映射
     */
    @Override
    public Map<String, Integer> getAllSpecialTokens() {
        return new HashMap<>(specialTokens);
    }
    
    /**
     * 添加一个特殊token
     * @param content token内容
     * @param id token ID，如果为-1，表示暂时不知道ID，由JNI层处理
     * @return 是否添加成功
     */
    @Override
    public boolean addSpecialToken(String content, int id) {
        if (content == null || content.isEmpty()) {
            Log.e(TAG, "特殊token内容不能为空");
            return false;
        }
        
        // 如果ID为-1，表示这是一个占位的ID，不添加到反向映射中
        // 实际的ID将由JNI层处理
        if (id != -1) {
            specialTokens.put(content, id);
            specialTokensReverse.put(id, content);
            
            // 只有当ID有效时，才添加到词汇表
            vocab.put(content, id);
            vocabReverse.put(id, content);
            
            if (debugMode) {
                Log.d(TAG, "添加特殊token: " + content + " (ID: " + id + ")");
            }
        } else {
            // 对于ID为-1的情况，只记录token内容，不记录ID
            // 这些特殊token将由JNI层处理
            if (debugMode) {
                Log.d(TAG, "记录特殊token内容（ID由JNI层处理）: " + content);
            }
        }
        
        return true;
    }
    
    /**
     * 应用聊天模板
     * @param messages 消息列表
     * @param addGenerationPrompt 是否添加生成提示
     * @return 应用模板后的文本
     */
    @Override
    public String applyChatTemplate(Object messages, boolean addGenerationPrompt) {
        if (tokenizer == null) {
            Log.e(TAG, "分词器未初始化，无法应用聊天模板");
            return "";
        }
        
        try {
            // 调用HuggingfaceTokenizer的applyChatTemplate方法
            if (messages instanceof JSONArray) {
                // 使用现有的JSONArray重载方法
                return tokenizer.applyChatTemplate((JSONArray)messages, addGenerationPrompt, false);
            } else {
                // 如果是其他类型，尝试转换为JSON字符串
                String jsonStr = messages.toString();
                try {
                    JSONArray jsonArray = new JSONArray(jsonStr);
                    return tokenizer.applyChatTemplate(jsonArray, addGenerationPrompt, false);
                } catch (JSONException je) {
                    Log.e(TAG, "无法将消息转换为JSONArray: " + je.getMessage(), je);
                    return "";
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "应用聊天模板失败: " + e.getMessage(), e);
            return "";
        }
    }
    
    /**
     * 获取模型类型
     * @return 模型类型名称
     */
    @Override
    public String getModelType() {
        return modelType;
    }
    
    /**
     * 设置模型类型
     * @param modelType 模型类型名称
     */
    @Override
    public void setModelType(String modelType) {
        this.modelType = modelType;
        if (debugMode) {
            Log.d(TAG, "设置模型类型: " + modelType);
        }
    }
    
    /**
     * 获取分词器配置信息
     * @return 配置信息的字符串表示
     */
    @Override
    public String getTokenizerInfo() {
        StringBuilder info = new StringBuilder();
        info.append("TokenizerManager配置信息:\n");
        info.append("- 初始化状态: ").append(initialized ? "已初始化" : "未初始化").append("\n");
        info.append("- 词汇表大小: ").append(getVocabSize()).append("\n");
        info.append("- 特殊token数量: ").append(specialTokens.size()).append("\n");
        info.append("- 模型类型: ").append(modelType.isEmpty() ? "未设置" : modelType).append("\n");
        info.append("- 一致性分词: ").append(useConsistentTokenization ? "启用" : "禁用").append("\n");
        info.append("- 调试模式: ").append(debugMode ? "启用" : "禁用").append("\n");
        
        // 添加部分特殊token信息
        info.append("- 常用特殊token:\n");
        info.append("  CLS: ").append(clsToken).append("\n");
        info.append("  SEP: ").append(sepToken).append("\n");
        info.append("  UNK: ").append(unkToken).append("\n");
        info.append("  PAD: ").append(padToken).append("\n");
        info.append("  MASK: ").append(maskToken).append("\n");
        
        return info.toString();
    }
}
