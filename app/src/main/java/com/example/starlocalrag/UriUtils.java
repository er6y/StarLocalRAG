package com.example.starlocalrag;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.util.Log;

/**
 * Uri 工具类
 * 用于处理 Uri 相关操作
 */
public class UriUtils {
    private static final String TAG = "UriUtils";
    
    /**
     * 从 Uri 获取文件名
     * @param context 上下文
     * @param uri 文件 Uri
     * @return 文件名，如果获取失败则返回 Uri 的字符串表示
     */
    public static String getFileName(Context context, Uri uri) {
        if (uri == null) return "";
        
        String result = null;
        
        try {
            if (uri.getScheme().equals("content")) {
                try (Cursor cursor = context.getContentResolver().query(uri, null, null, null, null)) {
                    if (cursor != null && cursor.moveToFirst()) {
                        int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                        if (nameIndex != -1) {
                            result = cursor.getString(nameIndex);
                        }
                    }
                }
            }
            
            if (result == null) {
                result = uri.getPath();
                int cut = result.lastIndexOf('/');
                if (cut != -1) {
                    result = result.substring(cut + 1);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "获取文件名失败: " + e.getMessage(), e);
            result = uri.toString();
        }
        
        return result;
    }
}
