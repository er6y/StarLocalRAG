package com.example.starlocalrag;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.ServiceConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.text.InputType;
import android.util.Log;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import com.example.starlocalrag.ConfigManager;
import com.example.starlocalrag.EmbeddingModelHandler;
import com.example.starlocalrag.EmbeddingModelManager.ModelCallback;
import com.example.starlocalrag.SQLiteVectorDatabaseHandler;
import com.example.starlocalrag.SettingsFragment;
import com.example.starlocalrag.TextChunkProcessor.ProgressCallback;
import com.example.starlocalrag.TextProcessor;
import com.example.starlocalrag.api.TokenizerManager;
import com.example.starlocalrag.Utils;

public class BuildKnowledgeBaseFragment extends Fragment {

    private static final String TAG = "StarLocalRAG_Build";
    private static final int REQUEST_OPEN_DOCUMENT = 1;

    // ActivityResultLauncher替代startActivityForResult
    private ActivityResultLauncher<Intent> documentPickerLauncher;

    private Button buttonBrowseFiles;
    private Button buttonClearFiles;
    private Spinner spinnerEmbeddingModel;
    private TextView textViewFileList;
    private TextView textViewProgress;
    private Button buttonCreateKnowledgeBase;
    private Button buttonNewKnowledgeBase;

    private List<Uri> selectedFiles = new ArrayList<>();
    private List<String> knowledgeBaseNames = new ArrayList<>();
    
    // 用于执行后台任务的线程池
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    // 知识库名称下拉框
    private Spinner spinnerKnowledgeBaseName;
    
    // 进度显示相关变量
    private TextView textViewProgressLabel;
    private int totalFiles = 0;
    private int processedFiles = 0;
    private int processedFilesCount = 0;
    
    // 向量化阶段进度变量
    private int totalChunks = 0;
    private int processedChunks = 0;
    private float vectorizationPercentage = 0;
    
    private long startTime = 0;
    private boolean isProcessing = false;
    private final Handler timerHandler = new Handler(Looper.getMainLooper());
    private final Runnable timerRunnable = new Runnable() {
        @Override
        public void run() {
            if (isProcessing && startTime > 0) {
                updateProgressStatus();
                timerHandler.postDelayed(this, 1000); // 每秒更新一次
            }
        }
    };

    // 任务取消标志
    private boolean isTaskCancelled = false;
    private final AtomicBoolean isTaskCancelledAtomic = new AtomicBoolean(false);
    
    // 防锁屏标志
    private boolean isKeepScreenOn = false;

    // 文本块处理器
    private TextChunkProcessor textChunkProcessor;
    
    // 当前处理阶段
    private enum ProcessingStage {
        IDLE,
        TEXT_EXTRACTION,
        VECTORIZATION,
        COMPLETED
    }
    
    private ProcessingStage currentStage = ProcessingStage.IDLE;
    
    // 跟踪电池优化状态
    private boolean batteryOptimizationDisabled = false;
    
    // 知识库构建服务
    private KnowledgeBaseBuilderService builderService;
    private boolean isServiceBound = false;
    
    // 服务连接
    private final ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            KnowledgeBaseBuilderService.LocalBinder binder = (KnowledgeBaseBuilderService.LocalBinder) service;
            builderService = binder.getService();
            isServiceBound = true;
            
            // 设置进度回调
            builderService.setProgressCallback(new KnowledgeBaseBuilderService.ProgressCallback() {
                @Override
                public void onProgressUpdate(int progress, String status) {
                    // 在UI线程更新进度
                    mainHandler.post(() -> {
                        // 如果状态不为空，则更新进度标签和处理进度信息
                        if (status != null) {
                            // 更新进度标签
                            textViewProgressLabel.setText(status);
                            
                            // 根据状态信息更新内部进度变量
                            if (status.contains("正在提取文本")) {
                                // 确保startTime已初始化
                                if (startTime <= 0) {
                                    startTime = System.currentTimeMillis();
                                    isProcessing = true;
                                }
                                
                                // 文本提取阶段
                                currentStage = ProcessingStage.TEXT_EXTRACTION;
                                
                                // 解析文件计数信息，新格式：正在提取文本 (N/M): xxx
                                try {
                                    // 检查是否包含文件块信息
                                    if (status.contains("(生成了")) {
                                        // 包含文本块信息，尝试提取
                                        int blocksStart = status.indexOf("(生成了");
                                        int blocksEnd = status.indexOf("个文本块)");
                                        if (blocksStart > 0 && blocksEnd > blocksStart) {
                                            String blocksStr = status.substring(blocksStart + 4, blocksEnd).trim();
                                            try {
                                                int extractedTextBlocks = Integer.parseInt(blocksStr);
                                                LogManager.logD(TAG, "提取到文本块数量: " + extractedTextBlocks);
                                            } catch (NumberFormatException e) {
                                                LogManager.logE(TAG, "解析文本块数量失败: " + e.getMessage());
                                            }
                                        }
                                    }
                                    
                                    // 解析处理文件数量信息
                                    // 格式: 正在提取文本 (N/M): xxx
                                    int startIdx = status.indexOf("(") + 1;
                                    int endIdx = status.indexOf(")");
                                    if (startIdx > 0 && endIdx > startIdx) {
                                        String countPart = status.substring(startIdx, endIdx);
                                        String[] parts = countPart.split("/");
                                        if (parts.length == 2) {
                                            int processed = Integer.parseInt(parts[0].trim());
                                            int total = Integer.parseInt(parts[1].trim());
                                            
                                            // 更新进度变量
                                            processedFilesCount = processed;
                                            totalFiles = total;
                                            
                                            LogManager.logD(TAG, "文本提取进度变量: " + processedFilesCount + "/" + totalFiles + " " + formatElapsedTime());
                                        }
                                    }
                                } catch (Exception e) {
                                    LogManager.logE(TAG, "解析文本提取进度信息失败: " + status, e);
                                }
                                updateProgressUI(progress, status);
                            } else if (status.contains("正在生成向量")) {
                                // 确保startTime已初始化
                                if (startTime <= 0) {
                                    startTime = System.currentTimeMillis();
                                    isProcessing = true;
                                }
                                
                                // 向量化阶段
                                currentStage = ProcessingStage.VECTORIZATION;

                                // 解析向量化进度信息，格式：正在生成向量 (N/M)
                                try {
                                    int startIndex = status.indexOf("(") + 1;
                                    int endIndex = status.indexOf(")");
                                    if (startIndex > 0 && endIndex > startIndex) {
                                        String countPart = status.substring(startIndex, endIndex);
                                        String[] parts = countPart.split("/");
                                        if (parts.length == 2) {
                                            int processed = Integer.parseInt(parts[0].trim());
                                            int total = Integer.parseInt(parts[1].trim());
                                            
                                            // 更新进度变量
                                            processedChunks = processed;
                                            totalChunks = total;
                                            vectorizationPercentage = (float) processed / total * 100;
                                            
                                        }
                                    }
                                } catch (Exception e) {
                                    LogManager.logE(TAG, "向量化进度信息失败: " + status, e);
                                }
                                updateProgressLabel();
                                //updateProgressUI(progress, status);
                            } else if (status.contains("文本提取完成")) {
                                // 文本提取完成
                                currentStage = ProcessingStage.VECTORIZATION;
                                
                                // 解析总块数
                                try {
                                    String totalStr = status.substring(status.indexOf("共") + 1, status.indexOf("个文本块")).trim();
                                    totalChunks = Integer.parseInt(totalStr);
                                    processedChunks = 0;
                                    vectorizationPercentage = 0;
                                    
                                    LogManager.logD(TAG, "文本提取完成，获取到总块数: " + totalChunks + " " + formatElapsedTime());
                                } catch (Exception e) {
                                    LogManager.logE(TAG, "解析文本块总数失败: " + status, e);
                                }
                                updateProgressUI(progress, status);
                            } else if (status.contains("向量化处理完成")) {
                                // 向量化完成
                                currentStage = ProcessingStage.COMPLETED;
                                // 确保显示100%完成
                                processedChunks = totalChunks;
                                totalChunks = totalChunks;
                                vectorizationPercentage = 100;
                                
                                LogManager.logD(TAG, "向量化处理完成 " + formatElapsedTime());
                                updateProgressUI(progress, status + " Total:" + totalChunks + " " + formatElapsedTime());
                            }
                            else
                            {
                                // 更新UI，只有当status不为null时才更新文本内容
                                updateProgressUI(progress, status);
                            }
                        } else {
                            // 即使status为null，也要更新进度标签显示当前进度
                            // 这里使用内部进度变量来构造进度标签

                            if (currentStage == ProcessingStage.VECTORIZATION && totalChunks > 0) {
                                //String progressText = String.format(Locale.getDefault(), 
                                //    "正在生成向量 (%d/%d): %.1f%%", 
                                //    processedChunks, totalChunks, vectorizationPercentage);
                                //textViewProgressLabel.setText(progressText);
                            }
                            
                            // 只更新进度标签，不更新文本内容
                            updateProgressLabel();

                        }
                    });
                }
                
                @Override
                public void onBuildCompleted(boolean success) {
                    // 在UI线程处理知识库构建完成事件
                    mainHandler.post(() -> {
                        // 恢复电池优化设置
                        if (batteryOptimizationDisabled && getActivity() instanceof MainActivity) {
                            ((MainActivity) getActivity()).restoreBatteryOptimization();
                            batteryOptimizationDisabled = false;
                            LogManager.logD(TAG, "已恢复电池优化设置（构建" + (success ? "完成" : "取消") + "时）");
                        }
                        
                        // 关闭防锁屏
                        if (isKeepScreenOn) {
                            enableKeepScreenOn(false);
                            LogManager.logD(TAG, "已关闭防锁屏（构建" + (success ? "完成" : "取消") + "时）");
                        }
                    });
                }
                
                @Override
                public void onTaskCompleted(boolean success, String message) {
                    // 在UI线程处理任务完成
                    mainHandler.post(() -> {
                        handleTaskCompletion(success, message);
                    });
                }
            });
            
            LogManager.logD(TAG, "已连接到知识库构建服务");
        }
        
        @Override
        public void onServiceDisconnected(ComponentName name) {
            builderService = null;
            isServiceBound = false;
            LogManager.logD(TAG, "与知识库构建服务的连接已断开");
        }
    };

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_build_knowledge_base, container, false);
        
        // 初始化ActivityResultLauncher
        initializeActivityResultLauncher();
        
        // 初始化UI元素
        buttonBrowseFiles = view.findViewById(R.id.buttonBrowseFiles);
        buttonClearFiles = view.findViewById(R.id.buttonClearFiles);
        spinnerEmbeddingModel = view.findViewById(R.id.spinnerEmbeddingModel);
        textViewFileList = view.findViewById(R.id.textViewFileList);
        textViewProgress = view.findViewById(R.id.textViewProgress);
        buttonCreateKnowledgeBase = view.findViewById(R.id.buttonCreateKnowledgeBase);
        spinnerKnowledgeBaseName = view.findViewById(R.id.knowledge_base_name_spinner);
        textViewProgressLabel = view.findViewById(R.id.textViewProgressLabel);
        buttonNewKnowledgeBase = view.findViewById(R.id.buttonNewKnowledgeBase);
        
        // 初始化进度窗口 - 不再需要设置初始文本，已在布局中设置
        // textViewProgress.setText("准备就绪，等待创建知识库...");
        // textViewProgressLabel.setText("0/0 00:00:00");
        
        // 初始化按钮点击事件
        buttonBrowseFiles.setOnClickListener(v -> browseFiles());
        buttonClearFiles.setOnClickListener(v -> clearFiles());
        buttonCreateKnowledgeBase.setOnClickListener(v -> {
            if (isProcessing) {
                // 显示确认对话框
                new AlertDialog.Builder(requireContext())
                    .setTitle("确认中断")
                    .setMessage("确定要中断当前知识库创建过程吗？已处理的数据将被保留。")
                    .setPositiveButton("确定", (dialog, which) -> {
                        cancelProcessing();
                    })
                    .setNegativeButton("取消", null)
                    .show();
            } else {
                createKnowledgeBase();
            }
        });
        buttonNewKnowledgeBase.setOnClickListener(v -> showNewKnowledgeBaseDialog());
        
        // 设置词嵌入模型下拉框监听器
        spinnerEmbeddingModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                LogManager.logD(TAG, "选择了词嵌入模型: " + parent.getItemAtPosition(position).toString());
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                LogManager.logD(TAG, "未选择词嵌入模型");
            }
        });
        
        // 加载词嵌入模型
        loadEmbeddingModels();
        
        // 加载知识库名称列表
        loadKnowledgeBaseNames();
        
        // 应用全局字体大小
        applyGlobalTextSize();
        
        return view;
    }
    
    // 浏览文件
    private void browseFiles() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        documentPickerLauncher.launch(intent);
    }
    

    
    // 更新文件列表显示
    private void updateFileListDisplay() {
        StringBuilder sb = new StringBuilder();
        for (Uri uri : selectedFiles) {
            String fileName = getFileNameFromUri(uri);
            sb.append(fileName).append("\n");
        }
        textViewFileList.setText(sb.toString());
    }
    
    // 从Uri获取文件名
    private String getFileNameFromUri(Uri uri) {
        String fileName = "未知文件";
        try {
            if (uri.getScheme().equals("content")) {
                android.database.Cursor cursor = requireContext().getContentResolver().query(uri, null, null, null, null);
                if (cursor != null && cursor.moveToFirst()) {
                    int nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
                    if (nameIndex != -1) {
                        fileName = cursor.getString(nameIndex);
                    }
                    cursor.close();
                }
            } else if (uri.getScheme().equals("file")) {
                fileName = new File(uri.getPath()).getName();
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "获取文件名失败", e);
        }
        return fileName;
    }
    
    // 清空文件列表
    private void clearFiles() {
        selectedFiles.clear();
        textViewFileList.setText("");
    }
    
    // 加载词嵌入模型
    private void loadEmbeddingModels() {
        LogManager.logD(TAG, "开始加载词嵌入模型");
        
        // 显示加载状态
        String[] loadingState = {"加载中..."};
        final ArrayAdapter<String> initialAdapter = new ArrayAdapter<>(requireContext(), android.R.layout.simple_spinner_item, loadingState);
        initialAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerEmbeddingModel.setAdapter(initialAdapter);
        
        // 在后台线程加载模型列表
        executor.execute(() -> {
            // 从ConfigManager获取嵌入模型目录
            String embeddingModelPath = ConfigManager.getEmbeddingModelPath(requireContext());
            LogManager.logD(TAG, "嵌入模型目录路径: " + embeddingModelPath);
            
            // 获取embeddings目录
            File embeddingsDir = new File(embeddingModelPath);
            if (!embeddingsDir.exists()) {
                LogManager.logD(TAG, "嵌入模型目录不存在，尝试创建: " + embeddingModelPath);
                boolean created = embeddingsDir.mkdirs();
                LogManager.logD(TAG, "创建目录结果: " + created);
                
                // 如果外部目录创建失败，回退到内部存储
                if (!created) {
                    LogManager.logD(TAG, "回退到应用内部存储");
                    embeddingsDir = new File(requireContext().getFilesDir(), "embeddings");
                    if (!embeddingsDir.exists()) {
                        embeddingsDir.mkdirs();
                    }
                }
            }
            
            // 获取所有文件作为模型
            File[] files = embeddingsDir.listFiles();
            LogManager.logD(TAG, "发现模型文件数量: " + (files != null ? files.length : 0));
            
            // 在主线程更新UI
            mainHandler.post(() -> {
                if (files != null && files.length > 0) {
                    String[] models = new String[files.length];
                    for (int i = 0; i < files.length; i++) {
                        models[i] = files[i].getName();
                    }
                    ArrayAdapter<String> modelsAdapter = new ArrayAdapter<>(requireContext(), android.R.layout.simple_spinner_item, models);
                    modelsAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                    spinnerEmbeddingModel.setAdapter(modelsAdapter);
                    LogManager.logD(TAG, "已加载 " + models.length + " 个词嵌入模型");
                } else {
                    String[] noModels = {"无可用模型"};
                    ArrayAdapter<String> noModelsAdapter = new ArrayAdapter<>(requireContext(), android.R.layout.simple_spinner_item, noModels);
                    noModelsAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                    spinnerEmbeddingModel.setAdapter(noModelsAdapter);
                    LogManager.logD(TAG, "未找到可用的词嵌入模型");
                }
            });
        });
    }
    
    // 加载知识库名称列表
    private void loadKnowledgeBaseNames() {
        LogManager.logD(TAG, "开始加载知识库名称列表");
        
        knowledgeBaseNames.clear();
        
        // 从ConfigManager获取知识库目录
        String knowledgeBasePath = ConfigManager.getKnowledgeBasePath(requireContext());
        LogManager.logD(TAG, "知识库目录路径: " + knowledgeBasePath);
        
        // 获取知识库目录
        File knowledgeBaseDir = new File(knowledgeBasePath);
        if (!knowledgeBaseDir.exists()) {
            LogManager.logD(TAG, "知识库目录不存在，尝试创建: " + knowledgeBasePath);
            boolean created = knowledgeBaseDir.mkdirs();
            LogManager.logD(TAG, "创建目录结果: " + created);
            
            // 如果外部目录创建失败，回退到内部存储
            if (!created) {
                LogManager.logD(TAG, "回退到应用内部存储");
                knowledgeBaseDir = new File(requireContext().getFilesDir(), "knowledge_bases");
                if (!knowledgeBaseDir.exists()) {
                    knowledgeBaseDir.mkdirs();
                }
            }
        }
        
        // 获取所有子目录作为知识库
        File[] directories = knowledgeBaseDir.listFiles(File::isDirectory);
        if (directories != null) {
            LogManager.logD(TAG, "发现知识库数量: " + directories.length);
            for (File dir : directories) {
                if (!dir.getName().equals("无")) {
                    knowledgeBaseNames.add(dir.getName());
                }
            }
        }
        
        // 设置下拉框适配器
        ArrayAdapter<String> adapter = new ArrayAdapter<>(requireContext(), 
                android.R.layout.simple_spinner_item, knowledgeBaseNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerKnowledgeBaseName.setAdapter(adapter);
        
        // 设置项目选择监听器
        spinnerKnowledgeBaseName.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position >= 0 && position < knowledgeBaseNames.size()) {
                    String selected = knowledgeBaseNames.get(position);
                    if (!selected.isEmpty()) {
                        LogManager.logD(TAG, "已选择知识库: " + selected);
                        // 保存选择到ConfigManager
                        ConfigManager.setString(requireContext(), ConfigManager.KEY_KNOWLEDGE_BASE, selected);
                        LogManager.logD(TAG, "已保存知识库选择到ConfigManager: " + selected);
                    }
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // 不做任何处理
            }
        });
        
        // 从ConfigManager加载上次选择的知识库
        loadLastSelectedKnowledgeBase();
    }
    
    // 加载上次选择的知识库
    private void loadLastSelectedKnowledgeBase() {
        try {
            // 使用ConfigManager获取上次选择的知识库
            String lastKnowledgeBase = ConfigManager.getString(requireContext(), 
                    ConfigManager.KEY_KNOWLEDGE_BASE, "");
            
            LogManager.logD(TAG, "从ConfigManager加载上次选择的知识库: " + 
                    (lastKnowledgeBase.isEmpty() ? "[空]" : lastKnowledgeBase));
            
            if (!lastKnowledgeBase.isEmpty()) {
                // 在下拉列表中查找并选择上次使用的知识库
                for (int i = 0; i < knowledgeBaseNames.size(); i++) {
                    if (knowledgeBaseNames.get(i).equals(lastKnowledgeBase)) {
                        spinnerKnowledgeBaseName.setSelection(i);
                        LogManager.logD(TAG, "已选择上次使用的知识库: " + lastKnowledgeBase);
                        break;
                    }
                }
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "加载知识库选择失败: " + e.getMessage(), e);
        }
    }
    
    // 显示新建知识库对话框
    private void showNewKnowledgeBaseDialog() {
        // 创建一个输入框
        final EditText input = new EditText(requireContext());
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        input.setHint("请输入新知识库名称");
        
        // 设置输入框的布局参数
        FrameLayout container = new FrameLayout(requireContext());
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 
                ViewGroup.LayoutParams.WRAP_CONTENT);
        params.leftMargin = getResources().getDimensionPixelSize(R.dimen.dialog_margin);
        params.rightMargin = getResources().getDimensionPixelSize(R.dimen.dialog_margin);
        input.setLayoutParams(params);
        container.addView(input);
        
        // 创建并显示对话框
        new AlertDialog.Builder(requireContext())
            .setTitle("新建知识库")
            .setView(container)
            .setPositiveButton("确定", (dialog, which) -> {
                String newKnowledgeBaseName = input.getText().toString().trim();
                if (!newKnowledgeBaseName.isEmpty()) {
                    // 将新知识库名称添加到下拉框并选中
                    if (!knowledgeBaseNames.contains(newKnowledgeBaseName)) {
                        knowledgeBaseNames.add(newKnowledgeBaseName);
                        ArrayAdapter<String> adapter = new ArrayAdapter<>(requireContext(), 
                                android.R.layout.simple_spinner_item, knowledgeBaseNames);
                        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                        spinnerKnowledgeBaseName.setAdapter(adapter);
                    }
                    
                    // 选中新添加的知识库名称
                    int position = knowledgeBaseNames.indexOf(newKnowledgeBaseName);
                    if (position >= 0) {
                        spinnerKnowledgeBaseName.setSelection(position);
                    }
                    
                    Utils.showToastSafely(requireContext(), "已添加新知识库名称: " + newKnowledgeBaseName, Toast.LENGTH_SHORT);
                } else {
                    Utils.showToastSafely(requireContext(), "知识库名称不能为空", Toast.LENGTH_SHORT);
                }
            })
            .setNegativeButton("取消", null)
            .show();
    }
    
    // 创建知识库
    private void createKnowledgeBase() {
        // 基本验证
        if (selectedFiles.isEmpty()) {
            Utils.showToastSafely(requireContext(), "请先选择文件", Toast.LENGTH_SHORT);
            return;
        }
        
        String embeddingModel = spinnerEmbeddingModel.getSelectedItem().toString();
        if (embeddingModel.equals("加载中...") || embeddingModel.equals("无可用模型")) {
            Utils.showToastSafely(requireContext(), "请先选择有效的词嵌入模型", Toast.LENGTH_SHORT);
            return;
        }
        
        String knowledgeBaseName = "";
        if (spinnerKnowledgeBaseName.getSelectedItem() != null) {
            knowledgeBaseName = spinnerKnowledgeBaseName.getSelectedItem().toString();
        }
        
        if (knowledgeBaseName.isEmpty()) {
            // 如果没有选择知识库，弹出对话框让用户输入新的知识库名称
            final EditText input = new EditText(requireContext());
            input.setInputType(InputType.TYPE_CLASS_TEXT);
            
            new AlertDialog.Builder(requireContext())
                .setTitle("新建知识库")
                .setMessage("请输入知识库名称")
                .setView(input)
                .setPositiveButton("确定", (dialog, which) -> {
                    String newName = input.getText().toString().trim();
                    if (!newName.isEmpty()) {
                        checkAndProcessKnowledgeBase(newName, embeddingModel);
                    } else {
                        Utils.showToastSafely(requireContext(), "知识库名称不能为空", Toast.LENGTH_SHORT);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
        } else {
            checkAndProcessKnowledgeBase(knowledgeBaseName, embeddingModel);
        }
    }
    
    // 检查知识库是否存在并处理
    private void checkAndProcessKnowledgeBase(String knowledgeBaseName, String embeddingModel) {
        // 从ConfigManager获取知识库目录路径
        String knowledgeBasePath = ConfigManager.getKnowledgeBasePath(requireContext());
        File knowledgeBaseParentDir = new File(knowledgeBasePath);
        
        // 确保父目录存在
        if (!knowledgeBaseParentDir.exists()) {
            if (!knowledgeBaseParentDir.mkdirs()) {
                LogManager.logE(TAG, "无法创建知识库父目录: " + knowledgeBasePath);
                // 回退到应用内部存储
                knowledgeBaseParentDir = new File(requireContext().getFilesDir(), "knowledge_bases");
                if (!knowledgeBaseParentDir.exists()) {
                    knowledgeBaseParentDir.mkdirs();
                }
            }
        }
        
        // 检查知识库是否已存在
        File knowledgeBaseDir = new File(knowledgeBaseParentDir, knowledgeBaseName);
        LogManager.logD(TAG, "检查知识库是否存在: " + knowledgeBaseDir.getAbsolutePath());
        
        if (knowledgeBaseDir.exists()) {
            // 显示确认对话框
            new AlertDialog.Builder(requireContext())
                .setTitle("知识库已存在")
                .setMessage("知识库 '" + knowledgeBaseName + "' 已存在，您想要如何处理？")
                .setPositiveButton("覆盖", (dialog, which) -> {
                    processKnowledgeBase(knowledgeBaseName, embeddingModel, true);
                })
                .setNeutralButton("追加", (dialog, which) -> {
                    processKnowledgeBase(knowledgeBaseName, embeddingModel, false);
                })
                .setNegativeButton("取消", null)
                .show();
        } else {
            // 直接创建新知识库
            processKnowledgeBase(knowledgeBaseName, embeddingModel, false);
        }
    }
    
    /**
     * 处理知识库创建/更新
     * @param knowledgeBaseName 知识库名称
     * @param embeddingModel 嵌入模型路径
     * @param overwrite 是否覆盖现有知识库
     */
    private void processKnowledgeBase(String knowledgeBaseName, String embeddingModel, boolean overwrite) {
        // 检查是否已选择文件
        if (selectedFiles.isEmpty()) {
            Utils.showToastSafely(requireContext(), "请先选择文件", Toast.LENGTH_SHORT);
            return;
        }
        
        // 检查是否已选择知识库名称
        if (knowledgeBaseName == null || knowledgeBaseName.trim().isEmpty()) {
            Utils.showToastSafely(requireContext(), "请输入或选择知识库名称", Toast.LENGTH_SHORT);
            return;
        }
        
        // 检查是否已选择嵌入模型
        if (embeddingModel == null || embeddingModel.trim().isEmpty()) {
            Utils.showToastSafely(requireContext(), "请选择嵌入模型", Toast.LENGTH_SHORT);
            return;
        }
        
        // 如果已经在处理中，则取消当前任务
        if (isProcessing) {
            // 更改按钮文本
            buttonCreateKnowledgeBase.setText("创建知识库");
            
            // 设置取消标志
            isTaskCancelledAtomic.set(true);
            
            // 如果服务已绑定，通知服务取消任务
            if (isServiceBound && builderService != null) {
                builderService.cancelTask();
            }
            
            // 如果之前禁用了电池优化，现在恢复
            if (batteryOptimizationDisabled) {
                MainActivity activity = (MainActivity) getActivity();
                if (activity != null) {
                    activity.restoreBatteryOptimization();
                    batteryOptimizationDisabled = false;
                }
            }
            
            Utils.showToastSafely(requireContext(), "已取消知识库构建", Toast.LENGTH_SHORT);
            isProcessing = false;
            
            return;
        }
        
        // 设置处理标志
        isProcessing = true;
        isTaskCancelledAtomic.set(false);
        currentStage = ProcessingStage.IDLE;
        
        // 更改按钮文本
        buttonCreateKnowledgeBase.setText("取消");
        
        // 记录开始时间
        startTime = System.currentTimeMillis();
        
        // 启动定时器更新UI
        timerHandler.postDelayed(timerRunnable, 1000);
        
        // 获取分块配置信息
        int chunkSize = ConfigManager.getChunkSize(requireContext());
        int chunkOverlap = ConfigManager.getInt(requireContext(), ConfigManager.KEY_OVERLAP_SIZE, ConfigManager.DEFAULT_OVERLAP_SIZE);
        int minChunkSize = ConfigManager.getInt(requireContext(), ConfigManager.KEY_MIN_CHUNK_SIZE, ConfigManager.DEFAULT_MIN_CHUNK_SIZE);
        
        // 清空进度显示
        textViewProgress.setText("正在准备构建知识库...\n" +
                "分块大小: " + chunkSize + ", 重叠大小: " + chunkOverlap + ", 最小分块限制: " + minChunkSize);
        
        // 保存最后选择的知识库名称
        ConfigManager.setString(requireContext(), ConfigManager.KEY_LAST_SELECTED_KB, knowledgeBaseName);
        
        // 保存最后选择的嵌入模型
        ConfigManager.setString(requireContext(), ConfigManager.KEY_LAST_SELECTED_EMBEDDING_MODEL, embeddingModel);
        
        // 标记嵌入模型正在使用中，防止被卸载
        EmbeddingModelManager.getInstance(requireContext()).markModelInUse();
        
        // 获取知识库目录路径
        String knowledgeBasePath = ConfigManager.getKnowledgeBasePath(requireContext());
        File knowledgeBaseDir = new File(knowledgeBasePath, knowledgeBaseName);
        
        // 获取嵌入模型路径
        String embeddingModelPath = ConfigManager.getEmbeddingModelPath(requireContext()) + File.separator + embeddingModel;
        
        try {
            // 加载模型以获取维度信息
            EmbeddingModelHandler model = EmbeddingModelManager.getInstance(requireContext()).getModel(embeddingModelPath);
            int modelDimension = model.getEmbeddingDimension();
            LogManager.logD(TAG, "当前模型向量维度: " + modelDimension);
            
            // 检查知识库是否已存在
            File vectorDbFile = new File(knowledgeBaseDir, "vectorstore.db");
            if (!overwrite && vectorDbFile.exists()) {
                // 检查现有知识库的向量维度和模型
                SQLiteVectorDatabaseHandler existingDb = new SQLiteVectorDatabaseHandler(knowledgeBaseDir, embeddingModel);
                int existingDimension = existingDb.getMetadata().getEmbeddingDimension();
                String existingModel = existingDb.getMetadata().getEmbeddingModel();
                existingDb.close();
                
                if (existingDimension > 0 && existingDimension != modelDimension) {
                    // 在追加模式下，如果维度不匹配，显示错误并停止
                    String errorMsg = "错误: 向量维度不匹配! 现有知识库维度: " + existingDimension + 
                                     ", 当前模型维度: " + modelDimension;
                    LogManager.logE(TAG, errorMsg);
                    textViewProgress.append("\n" + errorMsg + "\n追加模式下维度必须匹配，请使用覆盖模式或选择匹配维度的模型。");
                    
                    // 恢复UI状态
                    isProcessing = false;
                    buttonCreateKnowledgeBase.setText("创建知识库");
                    
                    // 标记模型不再使用
                    EmbeddingModelManager.getInstance(requireContext()).markModelNotInUse();
                    
                    return;
                } else if (!existingModel.equals(embeddingModel)) {
                    // 模型不匹配，弹出警告对话框
                    new AlertDialog.Builder(requireContext())
                        .setTitle("模型不匹配警告")
                        .setMessage("现有知识库使用的模型(" + existingModel + ")与当前选择的模型(" + embeddingModel + ")不匹配。\n\n继续使用当前模型可能会导致检索结果不准确。")
                        .setPositiveButton("继续", (dialog, which) -> {
                            // 用户确认继续，不做任何处理
                        })
                        .setNegativeButton("取消", (dialog, which) -> {
                            // 恢复UI状态
                            isProcessing = false;
                            buttonCreateKnowledgeBase.setText("创建知识库");
                            EmbeddingModelManager.getInstance(requireContext()).markModelNotInUse();
                        })
                        .show();
                }
            } else if (overwrite && vectorDbFile.exists()) {
                // 在覆盖模式下，删除现有数据库文件
                LogManager.logD(TAG, "覆盖模式: 删除现有数据库文件");
                vectorDbFile.delete();
                
                // 删除元数据文件
                File metadataFile = new File(knowledgeBaseDir, "metadata.json");
                if (metadataFile.exists()) {
                    metadataFile.delete();
                }
                
                textViewProgress.append("\n覆盖模式: 已删除现有数据库文件");
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "检查向量维度时出错: " + e.getMessage(), e);
            textViewProgress.append("\n检查向量维度时出错: " + e.getMessage());
        }
        
        // 请求忽略电池优化
        MainActivity activity = (MainActivity) getActivity();
        if (activity != null) {
            batteryOptimizationDisabled = activity.requestIgnoreBatteryOptimizationIfNeeded();
            if (batteryOptimizationDisabled) {
                Utils.showToastSafely(requireContext(), "已请求忽略电池优化，构建过程将在后台继续", Toast.LENGTH_SHORT);
            }
        }
        
        // 启用防锁屏
        enableKeepScreenOn(true);
        
        // 启动和绑定前台服务
        startBuilderService();
        bindBuilderService();
        
        // 等待服务绑定完成
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            if (isServiceBound && builderService != null) {
                // 使用前台服务开始构建知识库
                builderService.startBuildKnowledgeBase(knowledgeBaseName, embeddingModel, selectedFiles);
                
                LogManager.logD(TAG, "已通过前台服务开始构建知识库: " + knowledgeBaseName);
            } else {
                // 服务绑定失败，回退到传统方式
                LogManager.logE(TAG, "服务绑定失败，回退到传统方式构建知识库");
                Utils.showToastSafely(requireContext(), "无法启动前台服务，构建过程可能会在应用切换到后台时暂停", Toast.LENGTH_LONG);
                
                // 使用传统方式构建知识库（在UI线程中执行）
                processKnowledgeBaseTraditional(knowledgeBaseName, embeddingModel, overwrite);
            }
        }, 500); // 给服务绑定一些时间
    }
    
    /**
     * 使用传统方式处理知识库创建/更新（在后台线程中执行，但没有前台服务保护）
     */
    private void processKnowledgeBaseTraditional(String knowledgeBaseName, String embeddingModel, boolean overwrite) {
        // 在后台线程中执行
        executor.execute(() -> {
            try {
                // 获取分块大小和重叠大小
                int chunkSize = ConfigManager.getChunkSize(requireContext());
                int chunkOverlap = ConfigManager.getInt(requireContext(), ConfigManager.KEY_OVERLAP_SIZE, ConfigManager.DEFAULT_OVERLAP_SIZE);
                
                // 创建文本块处理器
                textChunkProcessor = new TextChunkProcessor(requireContext(), isTaskCancelledAtomic);
                
                // 设置进度回调
                textChunkProcessor.setProgressCallback(new ProgressCallback() {
                    @Override
                    public void onTextExtractionProgress(int processedFiles, int totalFiles, String currentFile) {
                        // 记录阶段变更
                        ProcessingStage oldStage = currentStage;
                        
                        // 确保设置当前阶段为文本提取阶段
                        currentStage = ProcessingStage.TEXT_EXTRACTION;
                        
                        // 如果阶段发生变化，记录状态转换
                        if (oldStage != currentStage) {
                            recordProcessingStageChange(oldStage, currentStage);
                        }
                        
                        // 更新文件计数
                        BuildKnowledgeBaseFragment.this.processedFiles = processedFiles;
                        BuildKnowledgeBaseFragment.this.totalFiles = totalFiles;
                        processedFilesCount = processedFiles;
                        
                        // 更新UI
                        mainHandler.post(() -> {
                            // 构建进度信息
                            String progressInfo = String.format(Locale.getDefault(), 
                                                "正在提取文本 (%d/%d): %s", 
                                                processedFiles, totalFiles, currentFile);
                            
                            // 更新进度UI，文本提取阶段进度条显示为0
                            updateProgressUI(0, progressInfo);
                        });
                    }
                    
                    @Override
                    public void onVectorizationProgress(int processedChunks, int totalChunks, float percentage) {
                        // 记录阶段变更
                        ProcessingStage oldStage = currentStage;
                        
                        // 确保设置当前阶段为向量化阶段
                        currentStage = ProcessingStage.VECTORIZATION;
                        
                        // 如果阶段发生变化，记录状态转换
                        if (oldStage != currentStage) {
                            recordProcessingStageChange(oldStage, currentStage);
                        }
                        
                        // 更新向量化进度变量
                        BuildKnowledgeBaseFragment.this.processedChunks = processedChunks;
                        BuildKnowledgeBaseFragment.this.totalChunks = totalChunks;
                        BuildKnowledgeBaseFragment.this.vectorizationPercentage = percentage;
                        
                        // 更新UI
                        mainHandler.post(() -> {
                            // 构建进度信息，显示处理的块数和百分比
                            //String progressInfo = String.format(Locale.getDefault(), 
                            //                    "正在生成向量 (%d/%d): %.1f%%", 
                            //                    processedChunks, totalChunks, percentage);
                            
                            // 添加详细日志，记录向量化进度
                            //LogManager.logD(TAG, "向量化进度: " + processedChunks + "/" + totalChunks + " (" + percentage + "%)");
                            
                            // 更新进度UI，直接传递百分比值
                            //updateProgressUI(Math.round(percentage), progressInfo);
                        });
                    }
                    
                    @Override
                    public void onTextExtractionComplete(int totalChunks) {
                        mainHandler.post(() -> {
                            currentStage = ProcessingStage.VECTORIZATION;
                            appendToProgress("文本提取完成，共生成 " + totalChunks + " 个文本块，开始向量化处理...");
                        });
                    }
                    
                    @Override
                    public void onVectorizationComplete(int vectorCount) {
                        mainHandler.post(() -> {
                            appendToProgress("向量化处理完成，共生成 " + vectorCount + " 个向量");
                        });
                    }
                    
                    @Override
                    public void onError(String errorMessage) {
                        // 处理错误
                        LogManager.logE(TAG, "错误: " + errorMessage);
                        mainHandler.post(() -> {
                            appendToProgress("错误: " + errorMessage);
                        });
                    }
                    
                    @Override
                    public void onLog(String message) {
                        // 记录日志
                        LogManager.logD(TAG, message);
                        mainHandler.post(() -> {
                            // 检查是否是向量化进度信息
                            if (message != null && message.startsWith("向量化进度")) {
                                // 使用专门的方法处理向量化进度
                                appendVectorizationProgress(message);
                            } else {
                                // 其他日志信息正常追加
                                appendToProgress(message);
                            }
                        });
                    }
                });
                
                // 处理文件并构建知识库
                boolean success = textChunkProcessor.processFilesAndBuildKnowledgeBase(
                    knowledgeBaseName, 
                    embeddingModel, 
                    selectedFiles,
                    chunkSize,
                    chunkOverlap
                );
                
                // 处理完成后的操作
                mainHandler.post(() -> {
                    // 标记嵌入模型不再使用
                    EmbeddingModelManager.getInstance(requireContext()).markModelNotInUse();
                    
                    isProcessing = false;
                    isTaskCancelledAtomic.set(false);
                    currentStage = ProcessingStage.COMPLETED;
                    
                    // 停止计时器
                    timerHandler.removeCallbacks(timerRunnable);
                    
                    // 更新UI
                    if (success) {
                        appendToProgress("知识库构建完成: " + knowledgeBaseName);
                        Utils.showToastSafely(requireContext(), "知识库构建完成", Toast.LENGTH_SHORT);
                    } else {
                        appendToProgress("知识库构建已取消");
                        Utils.showToastSafely(requireContext(), "知识库构建已取消", Toast.LENGTH_SHORT);
                    }
                    
                    // 恢复按钮状态
                    buttonCreateKnowledgeBase.setText("创建知识库");
                    
                    // 刷新知识库列表
                    loadKnowledgeBaseNames();
                });
                
            } catch (Exception e) {
                LogManager.logE(TAG, "知识库构建失败", e);
                
                // 在UI线程中处理异常
                mainHandler.post(() -> {
                    // 标记嵌入模型不再使用
                    EmbeddingModelManager.getInstance(requireContext()).markModelNotInUse();
                    
                    isProcessing = false;
                    isTaskCancelledAtomic.set(false);
                    
                    // 停止计时器
                    timerHandler.removeCallbacks(timerRunnable);
                    
                    // 更新UI
                    appendToProgress("知识库构建失败: " + e.getMessage());
                    Utils.showToastSafely(requireContext(), "知识库构建失败: " + e.getMessage(), Toast.LENGTH_LONG);
                    
                    // 恢复按钮状态
                    buttonCreateKnowledgeBase.setText("创建知识库");
                    
                    // 恢复电池优化设置
                    if (batteryOptimizationDisabled) {
                        MainActivity activity = (MainActivity) getActivity();
                        if (activity != null) {
                            activity.restoreBatteryOptimization();
                            batteryOptimizationDisabled = false;
                        }
                    }
                    
                    // 关闭防锁屏
                    if (isKeepScreenOn) {
                        enableKeepScreenOn(false);
                    }
                });
            }
        });
    }
    
    /**
     * 更新进度状态标签
     */
    private void updateProgressStatus() {
        if (startTime <= 0) {
            return;
        }
        
        // 更新进度标签
        updateProgressLabel();
    }
    
    // 中断正在进行的处理
    private void cancelProcessing() {
        if (isProcessing) {
            isTaskCancelled = true;
            isTaskCancelledAtomic.set(true);
            appendToProgress("正在中断处理...");
            
            // 恢复电池优化设置
            if (batteryOptimizationDisabled) {
                MainActivity activity = (MainActivity) getActivity();
                if (activity != null) {
                    activity.restoreBatteryOptimization();
                    batteryOptimizationDisabled = false;
                }
            }
            
            // 关闭防锁屏
            if (isKeepScreenOn) {
                enableKeepScreenOn(false);
            }
        }
    }
    
    // 重置处理状态
    private void resetProcessingState() {
        mainHandler.post(() -> {
            // 检查Fragment是否仍然附加到Context
            if (isAdded() && getActivity() != null) {
                buttonCreateKnowledgeBase.setText("新建知识库");
                Utils.showToastSafely(getActivity(), isTaskCancelled ? "任务已中断" : "知识库创建完成", Toast.LENGTH_SHORT);
            } else {
                LogManager.logW(TAG, "Fragment未附加到Context，无法更新UI");
            }
        });
    }
    
    /**
     * 更新进度UI
     */
    private void updateProgressUI(int progress, String status) {
        if (textViewProgress != null && status != null) {
            // 追加日志而不是替换，确保日志连续
            String currentText = textViewProgress.getText().toString();
            
            // 追加新消息
            if (currentText.isEmpty()) {
                textViewProgress.setText(status);
            } else {
                textViewProgress.append("\n" + status);
                
                // 滚动到底部
                if (textViewProgress.getLayout() != null) {
                    try {
                        final int scrollAmount = textViewProgress.getLayout().getLineTop(textViewProgress.getLineCount()) - textViewProgress.getHeight();
                        if (scrollAmount > 0) {
                            textViewProgress.scrollTo(0, scrollAmount);
                        } else {
                            textViewProgress.scrollTo(0, 0);
                        }
                    } catch (Exception e) {
                        // 如果滚动失败，至少确保文本被添加
                        LogManager.logE(TAG, "滚动到底部失败", e);
                    }
                }
            }
        }
        
        // 更新进度标签
        updateProgressLabel();
    }
    
    /**
     * 更新进度标签
     */
    private void updateProgressLabel() {
        if (textViewProgressLabel != null) {
            // 检查startTime，避免在初始化之前更新
            if (startTime <= 0) {
                return;
            }
            
            String progressText;
            
            // 根据当前阶段显示不同的进度格式
            switch (currentStage) {
                case TEXT_EXTRACTION:
                    // 文本提取阶段：显示 [已经提取文件计数]/[总计数] + 构建时间
                    // 确保分母不为0，避免显示0/0
                    int displayTotal = totalFiles > 0 ? totalFiles : 1;
                    progressText = String.format(Locale.getDefault(), "[%d/%d] %s", 
                            processedFilesCount, displayTotal, formatElapsedTime());
                    break;
                    
                case VECTORIZATION:
                case COMPLETED :
                    // 向量化阶段：显示已处理块数/总块数、百分比和构建时间，保留一位小数
                    progressText = String.format(Locale.getDefault(), "[%d/%d] %.1f%% %s", 
                            processedChunks, totalChunks, vectorizationPercentage, formatElapsedTime());
                    break;
                    
                default:
                    // 其他阶段
                    progressText = "0% 完成";
                    break;
            }
            
            textViewProgressLabel.setText(progressText);
            
            // 添加详细日志，记录进度状态
            if (LogManager.logIsLoggable(TAG, LogManager.LOG_LEVEL_DEBUG)) {
                LogManager.logD(TAG, "更新进度标签 - 阶段: " + currentStage + ", 文件进度: " + processedFilesCount + "/" + totalFiles + 
                      ", 向量化进度: " + processedChunks + "/" + totalChunks + " (" + vectorizationPercentage + "%)");
            }
        }
    }
    
    /**
     * 格式化已经过的时间
     */
    private String formatElapsedTime() {
        long currentTime = System.currentTimeMillis();
        long elapsedTime = currentTime - startTime;
        return formatTime(elapsedTime);
    }
    
    // 格式化时间
    private String formatTime(long timeInMillis) {
        long seconds = timeInMillis / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;
        
        seconds %= 60;
        minutes %= 60;
        
        return String.format(Locale.getDefault(), "%02d:%02d:%02d", hours, minutes, seconds);
    }
    
    /**
     * 向进度框中追加消息
     * @param message 要追加的消息
     */
    private void appendToProgress(String message) {
        mainHandler.post(() -> {
            if (textViewProgress != null) {
                // 获取当前文本
                String currentText = textViewProgress.getText().toString();
                
                // 追加新消息
                if (currentText.isEmpty()) {
                    textViewProgress.setText(message);
                } else {
                    textViewProgress.append("\n" + message);
                    
                    // 滚动TextView到底部
                    if (textViewProgress.getLayout() != null) {
                        try {
                            final int scrollAmount = textViewProgress.getLayout().getLineTop(textViewProgress.getLineCount()) - textViewProgress.getHeight();
                            if (scrollAmount > 0) {
                                textViewProgress.scrollTo(0, scrollAmount);
                            } else {
                                textViewProgress.scrollTo(0, 0);
                            }
                        } catch (Exception e) {
                            // 如果滚动失败，至少确保文本被添加
                            LogManager.logE(TAG, "滚动到底部失败", e);
                        }
                    }
                }
            }
        });
    }
    
    /**
     * 向进度框中追加向量化进度信息
     * 格式为：向量化进度..........10%..........20%...
     * 每个点代表一个处理完成的文本块
     * @param message 向量化进度信息
     */
    private void appendVectorizationProgress(String message) {
        mainHandler.post(() -> {
            if (textViewProgress != null) {
                // 检查当前文本是否已经包含"向量化进度"
                String currentText = textViewProgress.getText().toString();
                
                // 如果当前文本为空，则直接设置
                if (currentText.isEmpty()) {
                    //textViewProgress.setText(message);
                } else {
                    // 其他情况，总是追加新行
                    //textViewProgress.append("\n" + message);
                }
                
                // 滚动TextView到底部
                if (textViewProgress.getLayout() != null) {
                    try {
                        final int scrollAmount = textViewProgress.getLayout().getLineTop(textViewProgress.getLineCount()) - textViewProgress.getHeight();
                        if (scrollAmount > 0) {
                            textViewProgress.scrollTo(0, scrollAmount);
                        } else {
                            textViewProgress.scrollTo(0, 0);
                        }
                    } catch (Exception e) {
                        // 如果滚动失败，至少确保文本被添加
                        LogManager.logE(TAG, "滚动到底部失败", e);
                    }
                }
            }
        });
    }
    
    /**
     * 启用或禁用屏幕常亮
     * @param enable 是否启用屏幕常亮
     */
    private void enableKeepScreenOn(boolean enable) {
        if (getActivity() == null) {
            return;
        }
        
        try {
            if (enable) {
                // 启用屏幕常亮
                getActivity().getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                isKeepScreenOn = true;
                LogManager.logD(TAG, "已启用屏幕常亮");
            } else {
                // 禁用屏幕常亮
                getActivity().getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                isKeepScreenOn = false;
                LogManager.logD(TAG, "已禁用屏幕常亮");
            }
        } catch (Exception e) {
            LogManager.logE(TAG, "设置屏幕常亮状态失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 应用全局字体大小设置
     */
    private void applyGlobalTextSize() {
        if (textViewFileList != null && textViewProgress != null) {
            float fontSize = ConfigManager.getGlobalTextSize(requireContext());
            textViewFileList.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSize);
            textViewProgress.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSize);
            LogManager.logD(TAG, "已应用全局字体大小: " + fontSize + "sp");
        }
    }
    
    @Override
    public void onResume() {
        super.onResume();
        // 在页面恢复时重新应用字体大小，以便在设置页面修改后能够立即生效
        applyGlobalTextSize();
    }
    
    /**
     * 重命名知识库
     */
    private void renameKnowledgeBase(String oldName, String newName) {
        // 从ConfigManager获取知识库目录路径
        String knowledgeBasePath = ConfigManager.getKnowledgeBasePath(requireContext());
        File knowledgeBaseParentDir = new File(knowledgeBasePath);
        
        // 如果外部目录不存在，尝试使用内部存储
        if (!knowledgeBaseParentDir.exists()) {
            knowledgeBaseParentDir = new File(requireContext().getFilesDir(), "knowledge_bases");
        }
        
        File oldDir = new File(knowledgeBaseParentDir, oldName);
        File newDir = new File(knowledgeBaseParentDir, newName);
        
        LogManager.logD(TAG, "尝试重命名知识库: " + oldDir.getAbsolutePath() + " -> " + newDir.getAbsolutePath());
        
        if (oldDir.exists()) {
            if (oldDir.renameTo(newDir)) {
                // 重命名成功，更新UI
                Utils.showToastSafely(requireContext(), "知识库重命名成功", Toast.LENGTH_SHORT);
                
                // 更新下拉列表
                loadKnowledgeBaseNames();
                
                // 选择新名称
                @SuppressWarnings("unchecked")
                ArrayAdapter<String> adapter = (ArrayAdapter<String>) spinnerKnowledgeBaseName.getAdapter();
                int position = -1;
                for (int i = 0; i < adapter.getCount(); i++) {
                    if (adapter.getItem(i).equals(newName)) {
                        position = i;
                        break;
                    }
                }
                
                if (position >= 0) {
                    spinnerKnowledgeBaseName.setSelection(position);
                }
                
                // 记录日志
                LogManager.logD(TAG, "知识库重命名成功: " + oldName + " -> " + newName);
            } else {
                // 重命名失败
                Utils.showToastSafely(requireContext(), "知识库重命名失败", Toast.LENGTH_SHORT);
                LogManager.logE(TAG, "知识库重命名失败: " + oldName + " -> " + newName);
            }
        } else {
            // 原目录不存在
            Utils.showToastSafely(requireContext(), "知识库不存在: " + oldName, Toast.LENGTH_SHORT);
            LogManager.logE(TAG, "知识库不存在: " + oldDir.getAbsolutePath());
        }
    }

    /**
     * 递归删除目录
     */
    private void deleteDirectory(File directory) {
        if (directory.exists()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        deleteDirectory(file);
                    } else {
                        file.delete();
                    }
                }
            }
            // 不删除目录本身，只清空内容
        }
    }
    
    /**
     * 绑定知识库构建服务
     */
    private void bindBuilderService() {
        if (!isServiceBound) {
            Intent intent = new Intent(requireContext(), KnowledgeBaseBuilderService.class);
            requireContext().bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE);
            LogManager.logD(TAG, "正在绑定知识库构建服务");
        }
    }
    
    /**
     * 解绑知识库构建服务
     */
    private void unbindBuilderService() {
        if (isServiceBound) {
            requireContext().unbindService(serviceConnection);
            isServiceBound = false;
            LogManager.logD(TAG, "已解绑知识库构建服务");
        }
    }
    
    /**
     * 启动知识库构建服务
     */
    private void startBuilderService() {
        Intent intent = new Intent(requireContext(), KnowledgeBaseBuilderService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            requireContext().startForegroundService(intent);
        } else {
            requireContext().startService(intent);
        }
        LogManager.logD(TAG, "已启动知识库构建服务");
    }
    
    /**
     * 停止知识库构建服务
     */
    private void stopBuilderService() {
        Intent intent = new Intent(requireContext(), KnowledgeBaseBuilderService.class);
        requireContext().stopService(intent);
        LogManager.logD(TAG, "已停止知识库构建服务");
    }
    
    @Override
    public void onDestroy() {
        super.onDestroy();
        // 解绑服务
        unbindBuilderService();
        executor.shutdown();
    }

    /**
     * 处理任务完成
     */
    private void handleTaskCompletion(boolean success, String message) {
        isProcessing = false;
        isTaskCancelledAtomic.set(false);
        
        // 停止计时器
        timerHandler.removeCallbacks(timerRunnable);
        
        // 更新UI
        if (textViewProgress != null) {
            appendToProgress(message);
        }
        
        // 恢复按钮状态
        if (buttonCreateKnowledgeBase != null) {
            buttonCreateKnowledgeBase.setText("创建知识库");
            buttonCreateKnowledgeBase.setEnabled(true);
        }
        
        // 显示完成消息
        Utils.showToastSafely(requireContext(), message, Toast.LENGTH_LONG);
        
        // 刷新知识库列表
        loadKnowledgeBaseNames();
        
        // 解绑并停止服务
        unbindBuilderService();
        stopBuilderService();
        
        // 恢复电池优化设置
        if (batteryOptimizationDisabled) {
            MainActivity activity = (MainActivity) getActivity();
            if (activity != null) {
                activity.restoreBatteryOptimization();
                batteryOptimizationDisabled = false;
            }
        }
        
        // 关闭防锁屏
        if (isKeepScreenOn) {
            enableKeepScreenOn(false);
        }
        
        LogManager.logD(TAG, "任务完成: " + message + ", 成功: " + success);
    }
    
    // 重置进度计数和计时器
    private void resetProgress() {
        processedFiles = 0;
        totalFiles = 0;
        processedFilesCount = 0;
        
        // 重置向量化进度变量
        processedChunks = 0;
        totalChunks = 0;
        vectorizationPercentage = 0;
        
        startTime = 0;
        isProcessing = false;
        timerHandler.removeCallbacks(timerRunnable);
        
        // 重置阶段
        currentStage = ProcessingStage.IDLE;
        
        // 重置UI显示
        mainHandler.post(() -> {
            textViewProgressLabel.setText("0/0 00:00:00");
            textViewProgress.setText("准备就绪，等待创建知识库...");
        });
    }

    // 在开始处理前初始化进度
    private void initProgress(int total) {
        // 重置所有进度变量
        processedFiles = 0;
        totalFiles = total;
        processedFilesCount = 0;
        
        // 重置向量化进度变量
        processedChunks = 0;
        totalChunks = 0;
        vectorizationPercentage = 0;
        
        // 设置开始时间和处理标志
        startTime = System.currentTimeMillis();
        isProcessing = true;
        
        // 设置初始阶段
        currentStage = ProcessingStage.TEXT_EXTRACTION;
        
        // 更新UI并启动计时器
        mainHandler.post(() -> {
            // 避免显示0/0
            int displayTotal = total > 0 ? total : 1;
            textViewProgressLabel.setText("0/" + displayTotal + " " + formatTime(0));
            timerHandler.post(timerRunnable);
        });
    }
    
    /**
     * 记录处理阶段变更
     * @param oldStage 旧阶段
     * @param newStage 新阶段
     */
    private void recordProcessingStageChange(ProcessingStage oldStage, ProcessingStage newStage) {
        LogManager.logD(TAG, "处理阶段变更: " + oldStage + " -> " + newStage 
               + ", 文件处理: " + processedFilesCount + "/" + totalFiles
               + ", 向量化: " + processedChunks + "/" + totalChunks + " (" + vectorizationPercentage + "%)");
    }
    
    /**
     * 初始化ActivityResultLauncher
     */
    private void initializeActivityResultLauncher() {
        documentPickerLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    Intent data = result.getData();
                    if (data != null) {
                        // 处理多选文件
                        if (data.getClipData() != null) {
                            int count = data.getClipData().getItemCount();
                            for (int i = 0; i < count; i++) {
                                Uri uri = data.getClipData().getItemAt(i).getUri();
                                if (!selectedFiles.contains(uri)) {
                                    selectedFiles.add(uri);
                                }
                            }
                        } else if (data.getData() != null) {
                            // 处理单选文件
                            Uri uri = data.getData();
                            if (!selectedFiles.contains(uri)) {
                                selectedFiles.add(uri);
                            }
                        }
                        updateFileListDisplay();
                    }
                }
            }
        );
    }
}