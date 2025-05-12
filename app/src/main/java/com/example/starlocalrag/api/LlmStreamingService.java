package com.example.starlocalrag.api;

import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Header;
import retrofit2.http.POST;
import retrofit2.http.Streaming;
import retrofit2.http.Url;

/**
 * 流式API服务接口
 * 支持多种LLM供应商的流式响应
 */
public interface LlmStreamingService {
    /**
     * OpenAI兼容的流式聊天接口
     */
    @POST("chat/completions")
    @Streaming
    Call<ResponseBody> streamChatCompletion(
        @Header("Authorization") String authorization,
        @Header("Content-Type") String contentType,
        @Header("Accept") String accept,
        @Body ChatCompletionRequest request
    );
    
    /**
     * 使用完整URL的流式聊天接口
     */
    @POST
    @Streaming
    Call<ResponseBody> streamChatCompletionWithUrl(
        @Url String url,
        @Header("Authorization") String authorization,
        @Header("Content-Type") String contentType,
        @Header("Accept") String accept,
        @Body ChatCompletionRequest request
    );
}

/**
 * 聊天完成请求模型
 */
class ChatCompletionRequest {
    String model;
    Object messages;
    boolean stream = true;

    public ChatCompletionRequest(String model, Object messages) {
        this.model = model;
        this.messages = messages;
    }
}
