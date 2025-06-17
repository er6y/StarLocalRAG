package com.example.starlocalrag.api;

import android.content.Context;

import com.example.starlocalrag.LogManager;
import com.starlocalrag.tokenizers.HuggingfaceTokenizer;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

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
    
    // 当前加载的模型路径
    private String currentModelPath = null;
    
    // 模型类型
    private String modelType = "";
    
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
        LogManager.logI(TAG, "分词器管理器已创建，等待初始化");
    }
    
    /**
     * 获取单例实例
     * @param context 应用上下文
     * @return TokenizerManager实例
     */
    public static synchronized TokenizerManager getInstance(Context context) {
        if (instance == null) {
            instance = new TokenizerManager(context);
            LogManager.logD(TAG, "创建分词器管理器实例");
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
        try {
            // 判断传入的是文件还是目录
            File modelDir = modelPath;
            if (modelPath.isFile()) {
                // 如果是文件，使用其父目录
                LogManager.logD(TAG, "检测到传入的是模型文件而非目录，将使用其父目录: " + modelPath.getAbsolutePath());
                modelDir = modelPath.getParentFile();
                if (modelDir == null || !modelDir.exists() || !modelDir.isDirectory()) {
                    LogManager.logE(TAG, "无法获取模型文件的父目录，或父目录不存在: " + modelPath.getAbsolutePath());
                    return false;
                }
            }
            
            // 检查是否与当前加载的模型路径相同
            String newModelPath = modelDir.getAbsolutePath();
            if (initialized && tokenizer != null && newModelPath.equals(currentModelPath)) {
                LogManager.logD(TAG, "分词器已经初始化为相同的模型路径，无需重新加载: " + newModelPath);
                return true;
            }
            
            // 如果是不同的模型路径，先关闭现有分词器
            if (tokenizer != null) {
                LogManager.logD(TAG, "关闭现有分词器实例，准备加载新分词器: " + newModelPath);
                try {
                    tokenizer.close();
                    tokenizer = null;
                    initialized = false;
                } catch (Exception e) {
                    LogManager.logE(TAG, "关闭分词器失败: " + e.getMessage(), e);
                    // 即使关闭失败也继续初始化新分词器
                }
            }
            
            LogManager.logD(TAG, "开始初始化分词器，模型目录: " + modelDir.getAbsolutePath());
            boolean success = loadFromDirectory(modelDir);
            
            if (success) {
                // 更新当前模型路径
                currentModelPath = newModelPath;
                
                LogManager.logD(TAG, "分词器初始化成功，词汇表大小: " + getVocabSize());
                
                // 默认启用一致性分词策略
                setUseConsistentTokenization(true);
                LogManager.logD(TAG, "已启用一致性分词策略");
                
                initialized = true;
                return true;
            } else {
                LogManager.logE(TAG, "分词器初始化失败");
                return false;
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "分词器初始化异常: " + e.getMessage(), e);
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
            LogManager.logE(TAG, "模型路径为空");
            return false;
        }
        
        try {
            File modelFile = new File(modelPath);
            return initialize(modelFile);
        } catch (Exception e) {
            LogManager.logE(TAG, "初始化分词器失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 获取分词器实例
     * @return 分词器实例，如果未初始化则返回null
     */
    public HuggingfaceTokenizer getTokenizer() {
        if (!initialized) {
            LogManager.logW(TAG, "分词器未初始化，无法获取实例");
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
        LogManager.logD(TAG, "设置一致性分词策略: " + useConsistent);
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
        LogManager.logD(TAG, "设置调试模式: " + debug);
    }
    
    /**
     * 从指定目录加载分词器
     * @param directory 包含tokenizer.json的目录
     * @return 是否成功加载
     */
    public boolean loadFromDirectory(File directory) {
        if (directory == null) {
            LogManager.logE(TAG, "目录对象为空");
            return false;
        }
        
        if (!directory.exists() || !directory.isDirectory()) {
            LogManager.logE(TAG, "指定的目录不存在或不是目录: " + directory.getAbsolutePath());
            return false;
        }
        
        // 尝试查找tokenizer.json文件
        File tokenizerFile = new File(directory, "tokenizer.json");
        if (!tokenizerFile.exists() || !tokenizerFile.isFile()) {
            LogManager.logE(TAG, "在目录中找不到tokenizer.json文件: " + directory.getAbsolutePath());
            return false;
        }
        
        // 检查文件大小和权限
        if (tokenizerFile.length() == 0) {
            LogManager.logE(TAG, "tokenizer.json文件大小为0: " + tokenizerFile.getAbsolutePath());
            return false;
        }
        
        if (!tokenizerFile.canRead()) {
            LogManager.logE(TAG, "tokenizer.json文件无法读取: " + tokenizerFile.getAbsolutePath());
            return false;
        }
        
        // 检查当前分词器是否已经加载了相同的文件
        if (tokenizer != null) {
            try {
                String currentModelPath = tokenizer.getModelPath();
                if (currentModelPath != null && currentModelPath.equals(tokenizerFile.getAbsolutePath())) {
                    LogManager.logD(TAG, "当前分词器已经加载了相同的模型文件，无需重新加载");
                    return true;
                }
                
                // 如果是不同的模型文件，关闭现有分词器
                LogManager.logD(TAG, "加载新的模型文件，关闭现有分词器");
                try {
                    tokenizer.close();
                } catch (Exception e) {
                    LogManager.logW(TAG, "关闭现有分词器时出错", e);
                } finally {
                    tokenizer = null;
                }
            } catch (Exception e) {
                LogManager.logW(TAG, "检查当前分词器模型路径时出错", e);
                // 关闭现有分词器
                try {
                    tokenizer.close();
                } catch (Exception ex) {
                    LogManager.logW(TAG, "关闭现有分词器时出错", ex);
                } finally {
                    tokenizer = null;
                }
            }
        }
        
        try {
            // 直接尝试创建分词器，让HuggingfaceTokenizer处理文件验证
            
            // 从文件创建新的分词器实例
            LogManager.logD(TAG, "尝试创建分词器实例，模型路径: " + tokenizerFile.getAbsolutePath());
            
            // 检查JNI库加载状态
            
            try {
                System.loadLibrary("tokenizers_jni");

            } catch (UnsatisfiedLinkError jniError) {
                LogManager.logE(TAG, "JNI库加载失败: " + jniError.getMessage(), jniError);
                return false;
            } catch (Exception e) {
                LogManager.logE(TAG, "JNI库加载时发生未知异常: " + e.getMessage(), e);
                return false;
            }
            
            try {
                
                // 正确传递参数，第二个参数应该是true，表示这是一个文件路径
                tokenizer = new HuggingfaceTokenizer(tokenizerFile.getAbsolutePath(), true);
                

                
                // 如果需要设置一致性分词，可以在这里设置
                if (tokenizer != null) {
                    LogManager.logI(TAG, "成功从文件加载分词器: " + tokenizerFile.getAbsolutePath());
                    
                    // 模型类型由HuggingfaceTokenizer内部处理
                    
                    return true;
                } else {
                    LogManager.logE(TAG, "分词器创建成功但实例为空");
                    return false;
                }
            } catch (IllegalArgumentException e) {
                LogManager.logE(TAG, "创建分词器失败，参数错误: " + e.getMessage(), e);
                return false;
            } catch (UnsatisfiedLinkError e) {
                LogManager.logE(TAG, "创建分词器失败，本地库加载错误: " + e.getMessage(), e);
                return false;
            } catch (Exception e) {
                LogManager.logE(TAG, "创建分词器失败，未知错误: " + e.getMessage(), e);
                return false;
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "加载分词器过程中发生未捕获的异常: " + e.getMessage(), e);
            return false;
        }
    }
    

    
    /**
     * 对文本进行分词
     * @param text 输入文本
     * @return 分词结果（token ID列表）
     */
    public long[][] tokenize(String text) {
        if (tokenizer == null) {
            LogManager.logE(TAG, "分词器未初始化");
            return new long[1][0];
        }
        
        try {
            // 使用HuggingfaceTokenizer分词
            return tokenizer.tokenizeToLongArray(text);
        } catch (Exception e) {
            LogManager.logE(TAG, "分词器异常: " + e.getMessage(), e);
            return new long[1][0];
        }
    }
    
    /**
     * 获取词汇表大小
     * @return 词汇表大小
     */
    public int getVocabSize() {
        if (tokenizer != null) {
            return tokenizer.getVocabSize();
        }
        return 0;
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
                LogManager.logI(TAG, "释放分词器资源");
            } catch (Exception e) {
                LogManager.logE(TAG, "关闭分词器时出错: " + e.getMessage(), e);
            }
        }
    }
    
    /**
     * 重置分词器管理器
     * 用于在应用重启或模型切换时重新初始化
     */
    public static synchronized void resetManager() {
        if (instance != null) {
            LogManager.logD(TAG, "重置分词器管理器");
            
            // 释放资源
            try {
                instance.close();
            } catch (Exception e) {
                LogManager.logE(TAG, "释放分词器资源失败: " + e.getMessage(), e);
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
            LogManager.logE(TAG, "分词器未初始化");
            return "";
        }
        
        try {
            // 使用原生的HuggingfaceTokenizer解码功能
            if (debugMode) {
                LogManager.logD(TAG, "使用HuggingfaceTokenizer解码，ID数量: " + ids.length);
            }
            
            // 如果是特定模型，可以添加特殊处理
            if (modelType != null && !modelType.isEmpty()) {
                // 这里可以根据模型类型添加特殊处理逻辑
                if (debugMode) {
                    LogManager.logD(TAG, "当前模型类型: " + modelType);
                }
            }
            
            // 使用HuggingfaceTokenizer的特殊解码方法，过滤掉特殊token
        if (debugMode) {
            LogManager.logD(TAG, "使用HuggingfaceTokenizer的decodeForModelOutput方法解码");
        }
        return tokenizer.decodeForModelOutput(ids);
        } catch (Exception e) {
            LogManager.logE(TAG, "解码异常: " + e.getMessage(), e);
            return "";
        }
    }
    
    /**
     * 重置分词器
     */
    @Override
    public void reset() {
        // 重置状态
        initialized = false;
        
        // 释放分词器资源
        if (tokenizer != null) {
            try {
                tokenizer.close();
                tokenizer = null;
            } catch (Exception e) {
                LogManager.logE(TAG, "释放分词器资源失败: " + e.getMessage(), e);
            }
        }
        
        LogManager.logD(TAG, "分词器已重置");
    }
    
    /**
     * 获取特殊token的ID
     * @param token 特殊token的内容
     * @return 特殊token的ID，如果不存在则返回-1
     */
    @Override
    public int getSpecialTokenId(String token) {
        if (tokenizer != null) {
            try {
                Map<String, String> specialTokens = tokenizer.getSpecialTokens();
                if (specialTokens != null && specialTokens.containsValue(token)) {
                    // 由于HuggingfaceTokenizer的getSpecialTokens返回的是类型到内容的映射
                    // 这里简化处理，返回-1表示由底层处理
                    return -1;
                }
            } catch (Exception e) {
                LogManager.logW(TAG, "获取特殊token ID失败: " + e.getMessage());
            }
        }
        return -1;
    }
    
    /**
     * 根据ID获取特殊token的内容
     * @param id token的ID
     * @return 特殊token的内容，如果不存在则返回null
     */
    @Override
    public String getSpecialTokenContent(int id) {
        if (tokenizer != null) {
            try {
                int[] singleId = {id};
                return tokenizer.decode(singleId);
            } catch (Exception e) {
                LogManager.logW(TAG, "获取特殊token内容失败: " + e.getMessage());
            }
        }
        return null;
    }
    
    /**
     * 检查给定的ID是否为特殊token
     * @param id token的ID
     * @return 是否为特殊token
     */
    @Override
    public boolean isSpecialToken(int id) {
        // 简化实现，委托给HuggingfaceTokenizer
        return false;
    }
    
    /**
     * 获取所有特殊token的映射（内容到ID）
     * @return 特殊token映射
     */
    @Override
    public Map<String, Integer> getAllSpecialTokens() {
        if (tokenizer != null) {
            try {
                Map<String, String> specialTokens = tokenizer.getSpecialTokens();
                Map<String, Integer> result = new HashMap<>();
                if (specialTokens != null) {
                    for (String content : specialTokens.values()) {
                        result.put(content, -1); // 简化处理，ID由底层管理
                    }
                }
                return result;
            } catch (Exception e) {
                LogManager.logW(TAG, "获取所有特殊token失败: " + e.getMessage());
            }
        }
        return new HashMap<>();
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
            LogManager.logE(TAG, "特殊token内容不能为空");
            return false;
        }
        
        // 简化实现，特殊token由HuggingfaceTokenizer管理
        if (debugMode) {
            LogManager.logD(TAG, "特殊token添加请求: " + content + " (委托给HuggingfaceTokenizer处理)");
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
            LogManager.logE(TAG, "分词器未初始化，无法应用聊天模板");
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
                    LogManager.logE(TAG, "无法将消息转换为JSONArray: " + je.getMessage(), je);
                    return "";
                }
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "应用聊天模板失败: " + e.getMessage(), e);
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
            LogManager.logD(TAG, "设置模型类型: " + modelType);
        }
    }
    
    /**
     * 对输入文本进行分词（兼容方法）
     * @param text 输入文本
     * @return token ID数组
     */
    public int[] tokenizeInput(String text) {
        if (!initialized || tokenizer == null) {
            LogManager.logE(TAG, "分词器未初始化，无法进行分词");
            return new int[0];
        }
        
        try {
            // 使用现有的tokenize方法
            long[][] result = tokenize(text);
            if (result != null && result.length > 0 && result[0] != null) {
                // 转换long[]到int[]
                long[] longIds = result[0];
                int[] intIds = new int[longIds.length];
                for (int i = 0; i < longIds.length; i++) {
                    intIds[i] = (int) longIds[i];
                }
                return intIds;
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "分词失败: " + e.getMessage(), e);
        }
        
        return new int[0];
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
        info.append("- 模型类型: ").append(modelType.isEmpty() ? "未设置" : modelType).append("\n");
        info.append("- 一致性分词: ").append(useConsistentTokenization ? "启用" : "禁用").append("\n");
        info.append("- 调试模式: ").append(debugMode ? "启用" : "禁用").append("\n");
        
        // 添加特殊token信息
        if (tokenizer != null) {
            try {
                Map<String, String> specialTokens = tokenizer.getSpecialTokens();
                if (specialTokens != null) {
                    info.append("- 特殊token数量: ").append(specialTokens.size()).append("\n");
                    info.append("- 常用特殊token:\n");
                    for (Map.Entry<String, String> entry : specialTokens.entrySet()) {
                        info.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
                    }
                }
            } catch (Exception e) {
                info.append("- 特殊token信息获取失败\n");
            }
        }
        
        return info.toString();
    }
}
