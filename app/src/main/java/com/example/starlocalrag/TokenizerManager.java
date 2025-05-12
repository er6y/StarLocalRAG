package com.example.starlocalrag;

import android.content.Context;
import android.util.Log;

import java.io.File;

/**
 * 分词器管理类，提供全局统一的分词器实例
 * 确保RAG问答、知识库构建和知识笔记创建过程使用相同的分词策略
 */
public class TokenizerManager {
    private static final String TAG = "StarLocalRAG_TokenizerMgr";
    
    // 单例实例
    private static TokenizerManager instance;
    
    // 分词器实例
    private BertTokenizer tokenizer;
    
    // 是否已初始化
    private boolean initialized = false;
    
    // 应用上下文
    private Context context;
    
    /**
     * 私有构造函数，防止外部实例化
     */
    private TokenizerManager(Context context) {
        this.context = context.getApplicationContext();
        this.tokenizer = new BertTokenizer();
    }
    
    /**
     * 获取TokenizerManager单例实例
     * @param context 应用上下文
     * @return TokenizerManager实例
     */
    public static synchronized TokenizerManager getInstance(Context context) {
        if (instance == null) {
            instance = new TokenizerManager(context);
        }
        return instance;
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
            boolean success = tokenizer.loadFromDirectory(modelDir);
            
            if (success) {
                Log.d(TAG, "分词器初始化成功，词汇表大小: " + tokenizer.getVocabSize());
                
                // 默认启用一致性分词策略
                tokenizer.setUseConsistentTokenization(true);
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
            if (!modelDir.exists()) {
                Log.e(TAG, "模型路径不存在: " + modelPath);
                return false;
            }
            
            // 如果是文件而不是目录，则使用其父目录
            if (modelDir.isFile()) {
                Log.d(TAG, "模型路径是文件，使用其父目录: " + modelDir.getParent());
                modelDir = modelDir.getParentFile();
            }
            
            return initialize(modelDir);
        } catch (Exception e) {
            Log.e(TAG, "初始化分词器异常: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 获取分词器实例
     * @return 分词器实例，如果未初始化则返回null
     */
    public BertTokenizer getTokenizer() {
        if (!initialized) {
            Log.w(TAG, "分词器尚未初始化");
            return null;
        }
        return tokenizer;
    }
    
    /**
     * 检查分词器是否已初始化
     * @return 是否已初始化
     */
    public boolean isInitialized() {
        return initialized;
    }
    
    /**
     * 设置是否使用一致性分词策略
     * @param useConsistent 是否使用一致性分词
     */
    public void setUseConsistentTokenization(boolean useConsistent) {
        if (initialized) {
            tokenizer.setUseConsistentTokenization(useConsistent);
            Log.d(TAG, "设置一致性分词策略: " + useConsistent);
        } else {
            Log.w(TAG, "分词器尚未初始化，无法设置一致性分词策略");
        }
    }
    
    /**
     * 检查是否使用一致性分词策略
     * @return 是否使用一致性分词
     */
    public boolean isUseConsistentTokenization() {
        if (initialized) {
            return tokenizer.isUseConsistentTokenization();
        } else {
            Log.w(TAG, "分词器尚未初始化，无法获取一致性分词策略状态");
            return false;
        }
    }
    
    /**
     * 设置是否启用调试模式
     * @param debug 是否启用调试
     */
    public void setDebugMode(boolean debug) {
        if (initialized) {
            tokenizer.setDebugMode(debug);
            Log.d(TAG, "设置调试模式: " + debug);
        } else {
            Log.w(TAG, "分词器尚未初始化，无法设置调试模式");
        }
    }
    
    /**
     * 重置分词器管理器
     * 用于在应用重启或模型切换时重新初始化
     */
    public static synchronized void reset() {
        if (instance != null) {
            Log.d(TAG, "重置分词器管理器");
            instance = null;
        }
    }
}
