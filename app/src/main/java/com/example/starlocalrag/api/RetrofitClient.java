package com.example.starlocalrag.api;

import android.content.Context;
import android.util.Log;

import com.google.gson.Gson;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

/**
 * Retrofit客户端工具类
 * 用于创建流式API服务实例
 */
public class RetrofitClient {
    private static final String TAG = "RetrofitClient";
    private static Retrofit retrofit;
    private static String currentBaseUrl = "";
    
    /**
     * 获取流式API服务实例
     * @param context 上下文
     * @param baseUrl API基础URL
     * @return LlmStreamingService实例
     */
    public static LlmStreamingService getStreamingService(Context context, String baseUrl) {
        // 如果baseUrl发生变化，重新创建Retrofit实例
        if (retrofit == null || !currentBaseUrl.equals(baseUrl)) {
            Log.d(TAG, "创建新的Retrofit实例，baseUrl: " + baseUrl);
            currentBaseUrl = baseUrl;
            
            // 创建日志拦截器
            HttpLoggingInterceptor logging = new HttpLoggingInterceptor(message -> Log.d("API_DEBUG", message));
            logging.setLevel(HttpLoggingInterceptor.Level.BODY);
            
            // 创建OkHttp客户端
            OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(60, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .addInterceptor(logging)
                .build();
            
            // 确保baseUrl以/结尾
            if (!baseUrl.endsWith("/")) {
                baseUrl = baseUrl + "/";
            }
            
            // 创建Retrofit实例
            retrofit = new Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        }
        
        return retrofit.create(LlmStreamingService.class);
    }
    
    /**
     * 使用默认baseUrl获取流式API服务实例
     * @param context 上下文
     * @return LlmStreamingService实例
     */
    public static LlmStreamingService getStreamingService(Context context) {
        return getStreamingService(context, "https://api.deepseek.com/");
    }
}
