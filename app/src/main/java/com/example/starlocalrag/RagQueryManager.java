package com.example.starlocalrag;

import android.content.Context;
import android.util.Log;

import com.example.starlocalrag.api.LlmApiAdapter;
import com.example.starlocalrag.api.LlmModelFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * RAG查询管理器
 * 负责处理知识库查询和大模型调用的整合
 */
public class RagQueryManager {
    private static final String TAG = "RagQueryManager";
    
    private final Context context;
    private final LlmModelFactory modelFactory;
    
    public RagQueryManager(Context context) {
        this.context = context;
        this.modelFactory = LlmModelFactory.getInstance(context);
    }
    
    /**
     * 执行RAG查询
     * @param apiUrl API地址
     * @param apiKey API密钥
     * @param model 模型名称
     * @param knowledgeBase 知识库名称
     * @param systemPrompt 系统提示词
     * @param userPrompt 用户提问
     * @param callback 回调接口
     */
    public void executeRagQuery(String apiUrl, String apiKey, String model, String knowledgeBase, 
                               String systemPrompt, String userPrompt, RagQueryCallback callback) {
        // 检查是否使用知识库
        if ("无".equals(knowledgeBase) || "无可用知识库".equals(knowledgeBase)) {
            // 不使用知识库，直接构建提示词
            String fullPrompt = buildDirectPrompt(systemPrompt, userPrompt);
            Log.d(TAG, "不使用知识库，直接构建提示词: " + fullPrompt);
            
            // 回调进度更新
            callback.onProgressUpdate("正在构建提示词...", "不使用知识库，直接构建提示词");
            
            // 调用大模型API
            callback.onProgressUpdate("正在调用大模型API...", "API类型: " + modelFactory.getProviderByUrl(apiUrl).getName());
            
            modelFactory.callModel(apiUrl, apiKey, model, fullPrompt, new LlmApiAdapter.ApiCallback() {
                @Override
                public void onSuccess(String response) {
                    callback.onQueryCompleted(response);
                }
                
                @Override
                public void onStreamingData(String chunk) {
                    callback.onStreamingData(chunk);
                }
                
                @Override
                public void onError(String errorMessage) {
                    callback.onQueryError(errorMessage);
                }
            });
        } else {
            // 使用知识库，先查询知识库
            callback.onProgressUpdate("正在查询知识库...", "知识库: " + knowledgeBase);
            
            // 在后台线程中执行知识库查询
            new Thread(() -> {
                try {
                    // 查询知识库获取相关内容
                    String relevantContent = queryKnowledgeBase(knowledgeBase, userPrompt);
                    Log.d(TAG, "知识库查询结果(前200字符): " + 
                          (relevantContent.length() > 200 ? relevantContent.substring(0, 200) + "..." : relevantContent));
                    
                    // 回调进度更新
                    callback.onProgressUpdate("正在查询知识库...", 
                                            "已从知识库获取" + relevantContent.length() + "字符的内容");
                    
                    // 构建完整的提示词
                    callback.onProgressUpdate("正在构建提示词...", "构建包含知识库内容的完整提示词");
                    String fullPrompt = buildFullPrompt(systemPrompt, userPrompt, relevantContent);
                    
                    // 调用大模型API
                    callback.onProgressUpdate("正在调用大模型API...", 
                                            "API类型: " + modelFactory.getProviderByUrl(apiUrl).getName());
                    
                    modelFactory.callModel(apiUrl, apiKey, model, fullPrompt, new LlmApiAdapter.ApiCallback() {
                        @Override
                        public void onSuccess(String response) {
                            callback.onQueryCompleted(response);
                        }
                        
                        @Override
                        public void onStreamingData(String chunk) {
                            callback.onStreamingData(chunk);
                        }
                        
                        @Override
                        public void onError(String errorMessage) {
                            callback.onQueryError(errorMessage);
                        }
                    });
                    
                } catch (Exception e) {
                    Log.e(TAG, "RAG查询异常", e);
                    callback.onQueryError("RAG查询错误: " + e.getMessage());
                }
            }).start();
        }
    }
    
    /**
     * 查询知识库获取相关内容
     */
    private String queryKnowledgeBase(String knowledgeBase, String query) {
        if ("无".equals(knowledgeBase) || "无可用知识库".equals(knowledgeBase)) {
            Log.d(TAG, "未选择知识库，跳过知识库查询");
            return ""; // 如果选择了"无"
        }
        
        Log.d(TAG, "开始查询知识库: " + knowledgeBase);
        
        try {
            // 获取知识库目录
            File knowledgeBaseDir = new File(context.getExternalFilesDir(null), "knowledge_bases/" + knowledgeBase);
            if (!knowledgeBaseDir.exists()) {
                return "知识库 '" + knowledgeBase + "' 不存在";
            }
            
            // 实际实现中，这里应该使用向量数据库进行相似度查询
            // 这里简化为读取知识库中的文件内容
            StringBuilder relevantContent = new StringBuilder();
            File[] files = knowledgeBaseDir.listFiles();
            
            if (files != null && files.length > 0) {
                // 简单起见，只读取前3个文件的内容
                int count = Math.min(files.length, 3);
                
                for (int i = 0; i < count; i++) {
                    if (files[i].isFile() && files[i].getName().endsWith(".txt")) {
                        try (BufferedReader reader = new BufferedReader(new FileReader(files[i]))) {
                            Log.d(TAG, "读取知识库文件: " + files[i].getName());
                            
                            String line;
                            while ((line = reader.readLine()) != null) {
                                relevantContent.append(line).append("\n");
                            }
                        }
                    }
                }
            }
            
            return relevantContent.toString();
        } catch (IOException e) {
            Log.e(TAG, "查询知识库错误", e);
            return "查询知识库错误: " + e.getMessage();
        }
    }
    
    /**
     * 构建不包含知识库内容的直接提示词
     */
    private String buildDirectPrompt(String systemPrompt, String userPrompt) {
        StringBuilder fullPrompt = new StringBuilder();
        if (!systemPrompt.isEmpty()) {
            fullPrompt.append("系统: ").append(systemPrompt).append("\n\n");
        }
        fullPrompt.append("用户: ").append(userPrompt);
        return fullPrompt.toString();
    }
    
    /**
     * 构建包含知识库内容的完整提示词
     */
    private String buildFullPrompt(String systemPrompt, String userPrompt, String relevantContent) {
        StringBuilder fullPrompt = new StringBuilder();
        if (!systemPrompt.isEmpty()) {
            fullPrompt.append("系统: ").append(systemPrompt).append("\n\n");
        }
        fullPrompt.append("相关知识: ").append(relevantContent).append("\n\n");
        fullPrompt.append("用户: ").append(userPrompt);
        return fullPrompt.toString();
    }
    
    /**
     * RAG查询回调接口
     */
    public interface RagQueryCallback {
        /**
         * 查询进度更新
         * @param progress 进度信息
         * @param debugInfo 调试信息
         */
        void onProgressUpdate(String progress, String debugInfo);
        
        /**
         * 查询完成
         * @param result 查询结果
         */
        void onQueryCompleted(String result);
        
        /**
         * 查询错误
         * @param errorMessage 错误信息
         */
        void onQueryError(String errorMessage);
        
        /**
         * 接收流式数据
         * @param chunk 流式数据块
         */
        void onStreamingData(String chunk);
    }
}