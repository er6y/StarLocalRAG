package com.example.starlocalrag;

import android.app.Application;
import android.content.Context;

/**
 * 全局应用类，用于提供应用上下文
 */
public class GlobalApplication extends Application {
    private static Context appContext;
    
    @Override
    public void onCreate() {
        super.onCreate();
        appContext = getApplicationContext();
    }
    
    /**
     * 获取应用上下文
     * @return 应用上下文
     */
    public static Context getAppContext() {
        return appContext;
    }
}
