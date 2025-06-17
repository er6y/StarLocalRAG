package com.example.starlocalrag;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.CheckBox;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * 词嵌入模型工具类，提供模型检测和查找的公共方法
 */
public class EmbeddingModelUtils {
    private static final String TAG = "EmbeddingModelUtils";
    
    /**
     * 检查并加载词嵌入模型
     * @param context 上下文
     * @param vectorDb 向量数据库处理器
     * @param callback 回调函数，参数为找到的模型路径，如果未找到则为null
     * @param modelSelectedCallback 模型选择回调，当用户选择了模型后调用
     */
    public static void checkAndLoadEmbeddingModel(
            Context context,
            SQLiteVectorDatabaseHandler vectorDb,
            Consumer<String> callback,
            ModelSelectedCallback modelSelectedCallback) {
        
        // 获取嵌入模型
        String embeddingModel = vectorDb.getMetadata().getEmbeddingModel();
        String embeddingModelPath = ConfigManager.getEmbeddingModelPath(context);
        String modelPath = null;
        boolean needModelSelection = false;
        
        // 检查元数据中是否有modeldir配置
        String modeldir = vectorDb.getMetadata().getModeldir();
        LogManager.logD(TAG, "元数据中的modeldir: " + (modeldir != null ? modeldir : "null"));
        
        if (modeldir != null && !modeldir.isEmpty()) {
            // 使用modeldir指定的目录
            File modeldirFile = new File(embeddingModelPath, modeldir);
            if (modeldirFile.exists() && modeldirFile.isDirectory()) {
                // 在modeldir中查找模型文件
                File[] files = modeldirFile.listFiles();
                if (files != null) {
                    for (File file : files) {
                        if (file.isFile() && isModelFile(file)) {
                            modelPath = file.getAbsolutePath();
                            LogManager.logD(TAG, "使用modeldir中的模型: " + modelPath);
                            break;
                        }
                    }
                }
                
                if (modelPath == null) {
                    LogManager.logD(TAG, "在指定的modeldir中未找到模型文件: " + modeldirFile.getAbsolutePath());
                    needModelSelection = true;
                }
            } else {
                LogManager.logD(TAG, "指定的modeldir不存在或不是目录: " + modeldirFile.getAbsolutePath());
                needModelSelection = true;
            }
        } else {
            LogManager.logD(TAG, "元数据中没有modeldir配置或为空");
            needModelSelection = true;
        }
        
        // 如果modeldir中没有找到模型，尝试直接使用embeddingModel
        if (modelPath == null) {
            modelPath = new File(embeddingModelPath, embeddingModel).getAbsolutePath();
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                LogManager.logD(TAG, "模型文件不存在: " + modelPath);
                needModelSelection = true;
            }
        }

        // 如果需要选择模型
        if (needModelSelection) {
            LogManager.logD(TAG, "需要选择模型，将尝试在嵌入模型目录中查找");
            
            // 尝试在嵌入模型目录中查找模型文件
            File embeddingModelDir = new File(embeddingModelPath);
            if (embeddingModelDir.exists() && embeddingModelDir.isDirectory()) {
                // 获取所有子目录，用于模型选择
                List<String> availableModels = new ArrayList<>();
                File[] directories = embeddingModelDir.listFiles(File::isDirectory);
                if (directories != null) {
                    for (File dir : directories) {
                        // 检查目录中是否有模型文件
                        File[] modelFiles = dir.listFiles(file -> isModelFile(file));
                        if (modelFiles != null && modelFiles.length > 0) {
                            availableModels.add(dir.getName());
                        }
                    }
                }
                
                // 也检查根目录中的模型文件
                File[] rootModelFiles = embeddingModelDir.listFiles(file -> isModelFile(file));
                if (rootModelFiles != null && rootModelFiles.length > 0) {
                    availableModels.add("根目录");
                }
                
                if (!availableModels.isEmpty()) {
                    // 弹出模型选择对话框
                    showModelSelectionDialog(context, embeddingModel, availableModels, embeddingModelPath, vectorDb, modelSelectedCallback);
                    callback.accept(null); // 返回null，表示需要等待用户选择
                } else {
                    LogManager.logE(TAG, "在嵌入模型目录中未找到可用的模型文件");
                    callback.accept(null); // 返回null，表示没有找到模型
                }
            } else {
                callback.accept(null); // 返回null，表示没有找到模型
            }
        } else {
            // 模型文件存在，直接返回路径
            callback.accept(modelPath);
        }
    }
    
    /**
     * 显示模型选择对话框
     */
    private static void showModelSelectionDialog(
            Context context,
            String originalModel,
            List<String> availableModels,
            String embeddingModelPath,
            SQLiteVectorDatabaseHandler vectorDb,
            ModelSelectedCallback callback) {
        
        // 确保在主线程中执行UI操作
        Handler mainHandler = new Handler(Looper.getMainLooper());
        mainHandler.post(() -> {
            try {
                LogManager.logD(TAG, "开始创建模型选择对话框");
                long startTime = System.currentTimeMillis();
                
                // 使用自定义布局创建对话框
                View dialogView = LayoutInflater.from(context).inflate(R.layout.dialog_model_selection, null);
                
                // 获取控件
                TextView textViewInfo = dialogView.findViewById(R.id.textViewInfo);
                Spinner spinnerModels = dialogView.findViewById(R.id.spinnerModels);
                CheckBox checkBoxRemember = dialogView.findViewById(R.id.checkBoxRemember);
                
                // 设置提示信息
                textViewInfo.setText("知识库使用的词嵌入模型 '" + originalModel + "' 不存在，请从以下可用模型目录中选择一个：");
                
                // 创建模型列表适配器
                ArrayAdapter<String> adapter = new ArrayAdapter<>(context, android.R.layout.simple_spinner_item, availableModels);
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                spinnerModels.setAdapter(adapter);
                
                // 查找原始模型在列表中的位置
                int originalModelIndex = availableModels.indexOf(originalModel);
                if (originalModelIndex >= 0) {
                    spinnerModels.setSelection(originalModelIndex);
                }
                
                LogManager.logD(TAG, "准备创建对话框，耗时: " + (System.currentTimeMillis() - startTime) + "ms");
                
                // 创建对话框 - 使用原生按钮
                AlertDialog.Builder builder = new AlertDialog.Builder(context)
                        .setTitle("选择词嵌入模型目录")
                        .setView(dialogView)
                        .setCancelable(false) // 防止用户通过点击外部或返回键关闭对话框
                        .setPositiveButton("确定", (dialog, which) -> {
                            LogManager.logD(TAG, "确定按钮被点击");
                            long processStartTime = System.currentTimeMillis();
                            
                            String selectedModel = (String) spinnerModels.getSelectedItem();
                            boolean rememberChoice = checkBoxRemember.isChecked();
                            
                            if (selectedModel != null) {
                                LogManager.logD(TAG, "用户选择了模型: " + selectedModel + ", 记住选择: " + rememberChoice + ", 处理耗时: " + (System.currentTimeMillis() - processStartTime) + "ms");
                                
                                // 在后台线程中处理选定的模型，避免阻塞UI线程
                                new Thread(() -> {
                                    LogManager.logD(TAG, "开始在后台线程处理选定的模型");
                                    long threadStartTime = System.currentTimeMillis();
                                    
                                    // 处理选定的模型
                                    processSelectedModel(context, selectedModel, embeddingModelPath, vectorDb, callback);
                                    
                                    LogManager.logD(TAG, "模型处理完成，耗时: " + (System.currentTimeMillis() - threadStartTime) + "ms");
                                }).start();
                            }
                        })
                        .setNegativeButton("取消", (dialog, which) -> {
                            LogManager.logD(TAG, "取消按钮被点击");
                            LogManager.logD(TAG, "用户取消了模型选择");
                            
                            // 调用回调函数，传递null表示用户取消了选择
                            if (callback != null) {
                                callback.onModelSelected(null, null);
                            }
                        });
                
                // 创建并显示对话框
                AlertDialog dialog = builder.create();
                dialog.show();
                
                LogManager.logD(TAG, "已显示模型选择对话框，总耗时: " + (System.currentTimeMillis() - startTime) + "ms");
            } catch (Exception e) {
                LogManager.logE(TAG, "显示模型选择对话框失败: " + e.getMessage(), e);
                // 出错时调用回调函数，传递null表示选择失败
                if (callback != null) {
                    callback.onModelSelected(null, null);
                }
            }
        });
    }
    
    /**
     * 处理用户选择的模型
     */
    private static void processSelectedModel(
            Context context,
            String selectedModel,
            String embeddingModelPath,
            SQLiteVectorDatabaseHandler vectorDb,
            ModelSelectedCallback callback) {
        
        String modelPath = null;
        boolean modelFound = false;
        
        if (selectedModel.equals("根目录")) {
            // 在根目录中查找模型文件
            File embeddingModelDir = new File(embeddingModelPath);
            File[] files = embeddingModelDir.listFiles(file -> isModelFile(file));
            if (files != null && files.length > 0) {
                modelPath = files[0].getAbsolutePath();
                modelFound = true;
                
                // 确保元数据中存在modeldir项，并设置为空字符串（表示使用根目录）
                SQLiteVectorDatabaseHandler.DatabaseMetadata metadata = vectorDb.getMetadata();
                metadata.setModeldir("");
                vectorDb.saveDatabase();
                LogManager.logD(TAG, "已更新元数据，modeldir设置为空（使用根目录）");
            }
        } else {
            // 使用选定的目录
            File selectedDir = new File(embeddingModelPath, selectedModel);
            if (selectedDir.exists() && selectedDir.isDirectory()) {
                File[] files = selectedDir.listFiles(file -> isModelFile(file));
                if (files != null && files.length > 0) {
                    modelPath = files[0].getAbsolutePath();
                    modelFound = true;
                    
                    // 确保元数据中存在modeldir项，并设置为选定的目录
                    SQLiteVectorDatabaseHandler.DatabaseMetadata metadata = vectorDb.getMetadata();
                    metadata.setModeldir(selectedModel);
                    vectorDb.saveDatabase();
                    LogManager.logD(TAG, "已更新元数据，modeldir设置为: " + selectedModel);
                }
            }
        }
        
        if (modelFound && callback != null) {
            // 保存模型映射
            ConfigManager.setModelMapping(context, "model_" + vectorDb.getMetadata().getEmbeddingModel(), selectedModel);
            
            // 调用回调函数
            callback.onModelSelected(selectedModel, modelPath);
        } else {
            LogManager.logE(TAG, "在选定的模型目录中未找到模型文件");
            if (callback != null) {
                callback.onModelSelected(null, null);
            }
        }
    }
    
    /**
     * 判断文件是否为模型文件
     */
    private static boolean isModelFile(File file) {
        return EmbeddingModelHandler.isModelFile(file);
    }
    
    /**
     * 模型选择回调接口
     */
    public interface ModelSelectedCallback {
        void onModelSelected(String selectedModel, String modelPath);
    }
}
