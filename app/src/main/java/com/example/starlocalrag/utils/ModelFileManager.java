package com.example.starlocalrag.utils;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.FileChannel;

/**
 * 模型文件管理器
 * 负责将模型文件从assets或下载目录复制到应用的私有目录
 */
public class ModelFileManager {
    private static final String TAG = "ModelFileManager";
    
    // 模型文件在assets中的路径
    private static final String ASSETS_MODEL_DIR = "models";
    
    // 应用私有目录中的模型路径
    private static final String PRIVATE_MODEL_DIR = "models";
    
    private final Context context;
    
    public ModelFileManager(Context context) {
        this.context = context.getApplicationContext();
    }
    
    /**
     * 确保模型文件已复制到私有目录
     * @param modelName 模型名称
     * @return 模型目录的绝对路径，如果失败则返回null
     */
    public String ensureModelFiles(String modelName) {
        // 检查模型是否已存在于私有目录
        File privateModelDir = new File(context.getFilesDir(), PRIVATE_MODEL_DIR + "/" + modelName);
        if (privateModelDir.exists() && privateModelDir.isDirectory()) {
            // 检查必要的文件是否存在
            File modelFile = new File(privateModelDir, "model.onnx");
            File configFile = new File(privateModelDir, "config.json");
            File tokenizerFile = new File(privateModelDir, "tokenizer.json");
            
            if (modelFile.exists() && configFile.exists() && tokenizerFile.exists()) {
                Log.d(TAG, "模型文件已存在: " + privateModelDir.getAbsolutePath());
                return privateModelDir.getAbsolutePath();
            }
        }
        
        // 如果模型文件不存在，尝试从assets复制
        return copyModelFromAssets(modelName);
    }
    
    /**
     * 从assets复制模型文件到私有目录
     * @param modelName 模型名称
     * @return 模型目录的绝对路径，如果失败则返回null
     */
    private String copyModelFromAssets(String modelName) {
        AssetManager assetManager = context.getAssets();
        String sourceDir = ASSETS_MODEL_DIR + "/" + modelName;
        File targetDir = new File(context.getFilesDir(), PRIVATE_MODEL_DIR + "/" + modelName);
        
        try {
            // 确保目标目录存在
            if (!targetDir.exists()) {
                if (!targetDir.mkdirs()) {
                    Log.e(TAG, "无法创建目标目录: " + targetDir.getAbsolutePath());
                    return null;
                }
            }
            
            // 列出assets中的文件
            String[] files = assetManager.list(sourceDir);
            if (files == null || files.length == 0) {
                Log.e(TAG, "在assets中找不到模型文件: " + sourceDir);
                return null;
            }
            
            // 复制文件
            for (String filename : files) {
                try (InputStream in = assetManager.open(sourceDir + "/" + filename);
                     OutputStream out = new FileOutputStream(new File(targetDir, filename))) {
                    
                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = in.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                    Log.d(TAG, "已复制文件: " + filename);
                }
            }
            
            return targetDir.getAbsolutePath();
            
        } catch (IOException e) {
            Log.e(TAG, "复制模型文件失败: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * 从下载目录复制模型文件到私有目录
     * @param sourcePath 源文件路径
     * @param modelName 模型名称
     * @return 模型目录的绝对路径，如果失败则返回null
     */
    public String copyModelFromDownload(String sourcePath, String modelName) {
        File sourceFile = new File(sourcePath);
        if (!sourceFile.exists()) {
            Log.e(TAG, "源文件不存在: " + sourcePath);
            return null;
        }
        
        File targetDir = new File(context.getFilesDir(), PRIVATE_MODEL_DIR + "/" + modelName);
        
        // 确保目标目录存在
        if (!targetDir.exists() && !targetDir.mkdirs()) {
            Log.e(TAG, "无法创建目标目录: " + targetDir.getAbsolutePath());
            return null;
        }
        
        // 如果源是文件，复制到目标目录
        if (sourceFile.isFile()) {
            File targetFile = new File(targetDir, sourceFile.getName());
            if (copyFile(sourceFile, targetFile)) {
                return targetDir.getAbsolutePath();
            }
            return null;
        }
        
        // 如果源是目录，复制整个目录
        if (sourceFile.isDirectory()) {
            if (copyDirectory(sourceFile, targetDir)) {
                return targetDir.getAbsolutePath();
            }
            return null;
        }
        
        return null;
    }
    
    /**
     * 复制文件
     */
    private boolean copyFile(File source, File target) {
        try (FileInputStream inStream = new FileInputStream(source);
             FileOutputStream outStream = new FileOutputStream(target);
             FileChannel inChannel = inStream.getChannel();
             FileChannel outChannel = outStream.getChannel()) {
            
            inChannel.transferTo(0, inChannel.size(), outChannel);
            Log.d(TAG, "已复制文件: " + source.getAbsolutePath() + " -> " + target.getAbsolutePath());
            return true;
            
        } catch (IOException e) {
            Log.e(TAG, "复制文件失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 复制目录
     */
    private boolean copyDirectory(File sourceDir, File targetDir) {
        if (!sourceDir.isDirectory()) {
            return false;
        }
        
        // 创建目标目录
        if (!targetDir.exists() && !targetDir.mkdirs()) {
            Log.e(TAG, "无法创建目标目录: " + targetDir.getAbsolutePath());
            return false;
        }
        
        // 列出源目录中的所有文件和子目录
        File[] files = sourceDir.listFiles();
        if (files == null) {
            return true; // 空目录
        }
        
        // 复制每个文件或子目录
        boolean success = true;
        for (File file : files) {
            File targetFile = new File(targetDir, file.getName());
            
            if (file.isDirectory()) {
                success &= copyDirectory(file, targetFile);
            } else {
                success &= copyFile(file, targetFile);
            }
            
            if (!success) {
                Log.e(TAG, "复制目录时出错: " + sourceDir.getAbsolutePath());
                break;
            }
        }
        
        return success;
    }
}
