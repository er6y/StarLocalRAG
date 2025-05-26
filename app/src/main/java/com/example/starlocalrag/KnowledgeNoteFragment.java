package com.example.starlocalrag;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface; // 添加DialogInterface的导入语句
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.text.Editable;
import android.util.Log;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import org.json.JSONException;
import org.json.JSONObject;

import com.example.starlocalrag.ConfigManager; // 添加 ConfigManager 的导入
import com.example.starlocalrag.EmbeddingModelManager; // 导入 EmbeddingModelManager
import com.example.starlocalrag.EmbeddingModelHandler; // 导入 EmbeddingModelHandler
import com.example.starlocalrag.SQLiteVectorDatabaseHandler; // 导入 SQLiteVectorDatabaseHandler
import com.example.starlocalrag.EmbeddingModelUtils; // 导入词嵌入模型工具类
import com.example.starlocalrag.api.TokenizerManager; // 导入分词器管理器
import com.example.starlocalrag.TextProcessor; // 导入文本处理器

public class KnowledgeNoteFragment extends Fragment {
    private static final String TAG = "KnowledgeNoteFragment";

    private EditText editTextTitle;
    private EditText editTextContent;
    private TextView textViewProgress;
    private Spinner spinnerKnowledgeBase;
    private Button buttonAddToKnowledgeBase;
    private ExecutorService executorService;
    private List<String> knowledgeBaseNames = new ArrayList<>();
    private Handler mainHandler = new Handler(Looper.getMainLooper()); // 主线程Handler

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_knowledge_note, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 初始化UI元素
        editTextTitle = view.findViewById(R.id.editTextTitle);
        editTextContent = view.findViewById(R.id.editTextContent);
        textViewProgress = view.findViewById(R.id.textViewProgress);
        spinnerKnowledgeBase = view.findViewById(R.id.spinnerNoteKnowledgeBase);
        buttonAddToKnowledgeBase = view.findViewById(R.id.buttonAddToKnowledgeBase);
        
        // 设置文本框可长按选择
        editTextContent.setLongClickable(true);
        editTextContent.setTextIsSelectable(true);
        
        // 创建线程池
        executorService = Executors.newSingleThreadExecutor();
        
        // 加载知识库列表
        loadKnowledgeBaseNames();
        
        // 设置添加到知识库按钮的点击事件
        buttonAddToKnowledgeBase.setOnClickListener(v -> addToKnowledgeBase());
        
        // 应用全局字体大小
        applyGlobalTextSize();
    }

    @Override
    public void onResume() {
        super.onResume();
        // 每次页面恢复时刷新知识库列表
        loadKnowledgeBaseNames();
        // 在页面恢复时重新应用字体大小，以便在设置页面修改后能够立即生效
        applyGlobalTextSize();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // 关闭线程池
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
    }

    // 加载知识库名称列表
    private void loadKnowledgeBaseNames() {
        Log.d(TAG, "开始加载知识库名称列表");
        knowledgeBaseNames.clear();

        // 获取设置中的知识库路径
        String knowledgeBasePath = ConfigManager.getKnowledgeBasePath(requireContext());
        Log.d(TAG, "从设置中获取知识库路径: " + knowledgeBasePath);

        // 获取知识库目录
        File knowledgeBaseDir = new File(knowledgeBasePath);
        if (!knowledgeBaseDir.exists()) {
            Log.d(TAG, "知识库目录不存在，尝试创建: " + knowledgeBaseDir.getAbsolutePath());
            knowledgeBaseDir.mkdirs();
        }

        // 获取所有子目录作为知识库
        File[] directories = knowledgeBaseDir.listFiles(File::isDirectory);
        if (directories != null && directories.length > 0) {
            for (File dir : directories) {
                // 确保不添加名为"无"的知识库
                if (!dir.getName().equals("无")) {
                    knowledgeBaseNames.add(dir.getName());
                }
            }
            Log.d(TAG, "找到 " + knowledgeBaseNames.size() + " 个知识库");
            buttonAddToKnowledgeBase.setEnabled(true);
        } else {
            // 如果没有知识库，添加提示
            knowledgeBaseNames.add("请先创建知识库");
            buttonAddToKnowledgeBase.setEnabled(false);
            Log.d(TAG, "未找到知识库，已禁用添加按钮");
        }

        // 设置适配器
        ArrayAdapter<String> adapter = new ArrayAdapter<>(requireContext(), android.R.layout.simple_spinner_item, knowledgeBaseNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerKnowledgeBase.setAdapter(adapter);
        
        // 设置选择监听器
        spinnerKnowledgeBase.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String selectedKnowledgeBase = knowledgeBaseNames.get(position);
                Log.d(TAG, "已选择知识库: " + selectedKnowledgeBase);
                
                // 如果选择了"请先创建知识库"，禁用添加按钮
                buttonAddToKnowledgeBase.setEnabled(!selectedKnowledgeBase.equals("请先创建知识库"));
                
                // 保存选择到ConfigManager（如果不是提示信息）
                if (!selectedKnowledgeBase.equals("请先创建知识库")) {
                    ConfigManager.setString(requireContext(), ConfigManager.KEY_KNOWLEDGE_BASE, selectedKnowledgeBase);
                    Log.d(TAG, "已保存知识库选择到ConfigManager: " + selectedKnowledgeBase);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // 不做任何操作
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
            
            Log.d(TAG, "从ConfigManager加载上次选择的知识库: " + 
                    (lastKnowledgeBase.isEmpty() ? "[空]" : lastKnowledgeBase));
            
            if (!lastKnowledgeBase.isEmpty() && !knowledgeBaseNames.isEmpty() && !knowledgeBaseNames.get(0).equals("请先创建知识库")) {
                // 在下拉列表中查找并选择上次使用的知识库
                for (int i = 0; i < knowledgeBaseNames.size(); i++) {
                    if (knowledgeBaseNames.get(i).equals(lastKnowledgeBase)) {
                        spinnerKnowledgeBase.setSelection(i);
                        Log.d(TAG, "已选择上次使用的知识库: " + lastKnowledgeBase);
                        break;
                    }
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "加载知识库选择失败: " + e.getMessage(), e);
        }
    }

    // 添加到知识库
    private void addToKnowledgeBase() {
        // 获取标题和内容
        String title = editTextTitle.getText().toString().trim();
        String content = editTextContent.getText().toString().trim();
        
        // 检查标题和内容是否为空
        if (title.isEmpty()) {
            Toast.makeText(requireContext(), "请输入标题", Toast.LENGTH_SHORT).show();
            return;
        }
        
        if (content.isEmpty()) {
            Toast.makeText(requireContext(), "请输入内容", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // 获取选中的知识库
        if (spinnerKnowledgeBase.getSelectedItemPosition() < 0 || 
            knowledgeBaseNames.isEmpty() || 
            "请先创建知识库".equals(knowledgeBaseNames.get(0))) {
            Toast.makeText(requireContext(), "请先创建知识库", Toast.LENGTH_SHORT).show();
            return;
        }
        
        String selectedKnowledgeBase = knowledgeBaseNames.get(spinnerKnowledgeBase.getSelectedItemPosition());
        
        // 显示进度
        textViewProgress.setText("正在处理...\n");
        
        // 禁用按钮，防止重复点击
        buttonAddToKnowledgeBase.setEnabled(false);
        
        // 在后台线程中处理
        executorService.execute(() -> {
            try {
                // 获取知识库目录
                String knowledgeBasePath = ConfigManager.getKnowledgeBasePath(requireContext());
                File knowledgeBaseDir = new File(knowledgeBasePath, selectedKnowledgeBase);
                if (!knowledgeBaseDir.exists()) {
                    updateProgress("错误：知识库不存在");
                    enableAddButton();
                    return;
                }
                
                // 优先从SQLite数据库中获取嵌入模型信息
                String embeddingModelPath = null;
                SQLiteVectorDatabaseHandler vectorDb = null;
                try {
                    vectorDb = new SQLiteVectorDatabaseHandler(knowledgeBaseDir, "unknown");
                    if (vectorDb.loadDatabase()) {
                        SQLiteVectorDatabaseHandler.DatabaseMetadata metadata = vectorDb.getMetadata();
                        if (metadata != null) {
                            String embeddingModel = metadata.getEmbeddingModel();
                            Log.d(TAG, "从SQLite数据库中读取到嵌入模型: " + embeddingModel);
                            
                            // 使用EmbeddingModelUtils检查并加载嵌入模型
                            CountDownLatch modelLatch = new CountDownLatch(1);
                            final AtomicReference<String> modelPathRef = new AtomicReference<>();
                            final AtomicBoolean modelFoundRef = new AtomicBoolean(false);
                            
                            // 创建一个副本以在lambda表达式中使用
                            final SQLiteVectorDatabaseHandler finalVectorDb = vectorDb;
                            
                            requireActivity().runOnUiThread(() -> {
                                EmbeddingModelUtils.checkAndLoadEmbeddingModel(
                                    requireContext(),
                                    finalVectorDb,
                                    modelPath -> {
                                        if (modelPath != null) {
                                            modelPathRef.set(modelPath);
                                            modelFoundRef.set(true);
                                        }
                                        modelLatch.countDown();
                                    },
                                    (selectedModel, selectedModelPath) -> {
                                        if (selectedModelPath != null) {
                                            modelPathRef.set(selectedModelPath);
                                            modelFoundRef.set(true);
                                        }
                                        modelLatch.countDown();
                                    }
                                );
                            });
                            
                            // 等待模型检查完成
                            try {
                                boolean modelCheckCompleted = modelLatch.await(60, TimeUnit.SECONDS);
                                if (!modelCheckCompleted) {
                                    updateProgress("错误：检查嵌入模型超时");
                                    enableAddButton();
                                    return;
                                }
                            } catch (InterruptedException e) {
                                updateProgress("错误：等待模型检查被中断: " + e.getMessage());
                                enableAddButton();
                                return;
                            }
                            
                            // 获取模型路径
                            embeddingModelPath = modelPathRef.get();
                            if (!modelFoundRef.get() || embeddingModelPath == null) {
                                updateProgress("错误：未找到可用的嵌入模型");
                                enableAddButton();
                                return;
                            }
                        }
                    }
                } catch (Exception e) {
                    Log.e(TAG, "读取SQLite数据库失败", e);
                } finally {
                    if (vectorDb != null) {
                        try {
                            vectorDb.close();
                        } catch (Exception e) {
                            Log.e(TAG, "关闭SQLite数据库失败", e);
                        }
                    }
                }
                
                // 如果从SQLite数据库中获取失败，则尝试从ConfigManager中获取（兼容旧版本）
                if (embeddingModelPath == null || embeddingModelPath.isEmpty()) {
                    Log.d(TAG, "从SQLite数据库中获取嵌入模型失败，尝试从ConfigManager中获取");
                    embeddingModelPath = ConfigManager.getKnowledgeBaseEmbeddingModel(requireContext(), selectedKnowledgeBase);
                    
                    // 如果从ConfigManager中获取到了模型名称，检查模型文件是否存在
                    if (embeddingModelPath != null && !embeddingModelPath.isEmpty()) {
                        File modelFile = new File(embeddingModelPath);
                        if (!modelFile.exists()) {
                            Log.d(TAG, "模型文件不存在: " + modelFile.getAbsolutePath());
                            embeddingModelPath = null;
                        }
                    }
                }
                
                // 如果模型路径为空或模型文件不存在，显示模型选择对话框
                if (embeddingModelPath == null || embeddingModelPath.isEmpty() || !new File(embeddingModelPath).exists()) {
                    updateProgress("知识库的嵌入模型不存在，需要选择新的模型");
                    
                    // 获取可用的嵌入模型列表
                    List<String> availableModels = getAvailableEmbeddingModels();
                    
                    if (availableModels.isEmpty()) {
                        updateProgress("错误：未找到可用的嵌入模型，请在设置中配置嵌入模型路径");
                        enableAddButton();
                        return;
                    }
                    
                    // 在UI线程中显示模型选择对话框
                    CountDownLatch dialogLatch = new CountDownLatch(1);
                    final AtomicReference<String> selectedModelRef = new AtomicReference<>();
                    final AtomicBoolean rememberChoiceRef = new AtomicBoolean(true);
                    
                    requireActivity().runOnUiThread(() -> {
                        showModelSelectionDialog(availableModels, (selectedModel, rememberChoice) -> {
                            selectedModelRef.set(selectedModel);
                            rememberChoiceRef.set(rememberChoice);
                            dialogLatch.countDown();
                        });
                    });
                    
                    // 等待用户选择模型
                    try {
                        boolean dialogCompleted = dialogLatch.await(60, TimeUnit.SECONDS);
                        if (!dialogCompleted) {
                            updateProgress("错误：选择模型超时");
                            enableAddButton();
                            return;
                        }
                    } catch (InterruptedException e) {
                        updateProgress("错误：等待模型选择被中断: " + e.getMessage());
                        enableAddButton();
                        return;
                    }
                    
                    // 获取用户选择的模型
                    embeddingModelPath = selectedModelRef.get();
                    boolean rememberChoice = rememberChoiceRef.get();
                    
                    if (embeddingModelPath == null || embeddingModelPath.isEmpty()) {
                        updateProgress("错误：未选择嵌入模型");
                        enableAddButton();
                        return;
                    }
                    
                    updateProgress("已选择新的嵌入模型：" + embeddingModelPath);
                    
                    // 如果用户选择记住选择，则更新知识库元数据中的模型信息
                    if (rememberChoice) {
                        updateProgress("正在更新知识库元数据...");
                        updateKnowledgeBaseModelMetadata(knowledgeBaseDir, embeddingModelPath);
                    } else {
                        updateProgress("仅使用所选模型进行此次操作，不更新知识库元数据");
                    }
                }
                
                updateProgress("找到嵌入模型：" + embeddingModelPath);
                updateProgress("正在加载模型（可能需要几秒钟）...");
                
                // 使用EmbeddingModelManager异步加载模型
                EmbeddingModelManager modelManager = EmbeddingModelManager.getInstance(requireContext());
                
                // 创建一个CountDownLatch来等待异步加载完成
                final CountDownLatch modelLoadLatch = new CountDownLatch(1);
                final AtomicReference<EmbeddingModelHandler> modelHandlerRef = new AtomicReference<>();
                final AtomicReference<Exception> modelErrorRef = new AtomicReference<>();
                
                modelManager.getModelAsync(embeddingModelPath, new EmbeddingModelManager.ModelCallback() {
                    @Override
                    public void onModelReady(EmbeddingModelHandler model) {
                        modelHandlerRef.set(model);
                        modelLoadLatch.countDown();
                    }
                    
                    @Override
                    public void onModelError(Exception e) {
                        modelErrorRef.set(e);
                        modelLoadLatch.countDown();
                    }
                });
                
                // 等待模型加载完成，设置超时
                try {
                    boolean modelLoaded = modelLoadLatch.await(60, TimeUnit.SECONDS);
                    if (!modelLoaded) {
                        updateProgress("错误：加载嵌入模型超时");
                        enableAddButton();
                        return;
                    }
                } catch (InterruptedException e) {
                    updateProgress("错误：等待模型加载被中断: " + e.getMessage());
                    enableAddButton();
                    return;
                }
                
                // 检查是否有错误
                if (modelErrorRef.get() != null) {
                    updateProgress("错误：加载嵌入模型失败: " + modelErrorRef.get().getMessage());
                    enableAddButton();
                    return;
                }
                
                // 获取加载好的模型
                EmbeddingModelHandler embeddingHandler = modelHandlerRef.get();
                if (embeddingHandler == null) {
                    updateProgress("错误：无法加载嵌入模型，可能是模型文件不兼容或损坏");
                    enableAddButton();
                    return;
                }
                
                // 确保TokenizerManager已初始化
                TokenizerManager tokenizerManager = TokenizerManager.getInstance(requireContext());
                if (!tokenizerManager.isInitialized()) {
                    updateProgress("正在初始化分词器...");
                    boolean initSuccess = tokenizerManager.initialize(embeddingModelPath);
                    if (initSuccess) {
                        updateProgress("分词器初始化成功，将使用统一分词策略");
                        // 启用一致性分词
                        tokenizerManager.setUseConsistentTokenization(true);
                    } else {
                        updateProgress("分词器初始化失败，将使用模型自带分词器");
                    }
                } else {
                    updateProgress("使用已初始化的全局分词器");
                    // 启用一致性分词
                    tokenizerManager.setUseConsistentTokenization(true);
                }
                
                updateProgress("模型加载成功，类型：" + embeddingHandler.getModelType());
                
                // 标记模型开始使用
                modelManager.markModelInUse();
                updateProgress("标记模型开始使用，防止自动卸载");
                
                // 使用SQLiteVectorDatabaseHandler添加笔记
                updateProgress("正在将笔记添加到知识库...");
                SQLiteVectorDatabaseHandler noteVectorDb = new SQLiteVectorDatabaseHandler(knowledgeBaseDir, "note");
                
                try {
                    // 加载数据库
                    if (!noteVectorDb.loadDatabase()) {
                        updateProgress("错误：无法加载SQLite向量数据库");
                        enableAddButton();
                        
                        // 标记模型使用结束（即使发生错误）
                        modelManager.markModelNotInUse();
                        updateProgress("标记模型使用结束，允许自动卸载");
                        
                        return;
                    }
                    
                    // 使用TextProcessor进行文本处理，确保与构建知识库使用相同的分块算法
                    TextProcessor textProcessor = new TextProcessor(requireContext());
                    
                    // 获取分块大小和重叠大小
                    int chunkSize = ConfigManager.getChunkSize(requireContext());
                    int chunkOverlap = ConfigManager.getInt(requireContext(), ConfigManager.KEY_OVERLAP_SIZE, ConfigManager.DEFAULT_OVERLAP_SIZE);
                    int minChunkSize = ConfigManager.getMinChunkSize(requireContext());
                    
                    updateProgress("使用分块参数：块大小=" + chunkSize + "，重叠大小=" + chunkOverlap + "，最小块大小=" + minChunkSize);
                    
                    // 如果内容长度超过分块大小，需要进行分块处理
                    if (content.length() > chunkSize) {
                        updateProgress("内容长度(" + content.length() + ")超过分块大小(" + chunkSize + ")，进行分块处理...");
                        List<String> chunks = textProcessor.splitTextIntoChunks(content, chunkSize, chunkOverlap);
                        updateProgress("分块完成，共生成" + chunks.size() + "个文本块");
                        
                        // 获取添加前的文本块数量
                        int beforeChunkCount = noteVectorDb.getChunkCount();
                        updateProgress("添加前数据库文本块数量: " + beforeChunkCount);
                        
                        boolean success = true;
                        int addedCount = 0;
                        
                        // 处理每个文本块
                        for (int i = 0; i < chunks.size(); i++) {
                            String chunk = chunks.get(i);
                            updateProgress("处理文本块 " + (i+1) + "/" + chunks.size() + "，长度: " + chunk.length());
                            
                            // 生成嵌入向量
                            float[] chunkEmbedding = embeddingHandler.generateEmbedding(chunk);
                            
                            // 添加文本块到数据库
                            String chunkTitle = title + " (块 " + (i+1) + "/" + chunks.size() + ")";
                            boolean chunkSuccess = noteVectorDb.addChunk(chunk, chunkTitle, chunkEmbedding);
                            
                            if (chunkSuccess) {
                                addedCount++;
                            } else {
                                success = false;
                                updateProgress("添加文本块 " + (i+1) + " 失败");
                            }
                        }
                        
                        // 保存数据库
                        if (success) {
                            success = noteVectorDb.saveDatabase();
                        }
                        
                        // 获取添加后的文本块数量
                        int afterChunkCount = noteVectorDb.getChunkCount();
                        updateProgress("添加后数据库文本块数量: " + afterChunkCount);
                        updateProgress("成功添加文本块数量: " + addedCount);
                    } else {
                        // 直接生成嵌入向量
                        float[] contentEmbedding = embeddingHandler.generateEmbedding(content);
                        
                        // 记录向量调试信息
                        String vectorDebugInfo = embeddingHandler.getVectorDebugInfo(content, contentEmbedding, System.currentTimeMillis());
                        updateProgress(vectorDebugInfo);
                        
                        // 获取添加前的文本块数量
                        int beforeChunkCount = noteVectorDb.getChunkCount();
                        updateProgress("添加前数据库文本块数量: " + beforeChunkCount);
                        
                        // 添加文本块到数据库
                        boolean success = noteVectorDb.addChunk(content, title, contentEmbedding);
                        
                        // 保存数据库
                        if (success) {
                            success = noteVectorDb.saveDatabase();
                        }
                        
                        // 获取添加后的文本块数量
                        int afterChunkCount = noteVectorDb.getChunkCount();
                        updateProgress("添加后数据库文本块数量: " + afterChunkCount);
                        updateProgress("新增文本块数量: " + (afterChunkCount - beforeChunkCount));
                    }
                    
                    // 关闭数据库
                    noteVectorDb.close();
                    
                    // 标记模型使用结束
                    modelManager.markModelNotInUse();
                    updateProgress("标记模型使用结束，允许自动卸载");
                    
                    updateProgress("笔记已成功添加到知识库");
                    
                    // 清空输入框
                    requireActivity().runOnUiThread(() -> {
                        editTextTitle.setText("");
                        editTextContent.setText("");
                    });
                } catch (Exception e) {
                    Log.e(TAG, "生成嵌入向量或添加笔记失败", e);
                    updateProgress("错误：" + e.getMessage());
                    if (noteVectorDb != null) {
                        noteVectorDb.close();
                    }
                    modelManager.markModelNotInUse();
                    updateProgress("标记模型使用结束（发生错误），允许自动卸载");
                }
            } catch (Exception e) {
                Log.e(TAG, "添加到知识库失败", e);
                updateProgress("错误：" + e.getMessage());
            } finally {
                enableAddButton();
            }
        });
    }
    
    // 启用添加按钮
    private void enableAddButton() {
        requireActivity().runOnUiThread(() -> {
            buttonAddToKnowledgeBase.setEnabled(true);
        });
    }
    
    // 更新进度显示
    private void updateProgress(String message) {
        requireActivity().runOnUiThread(() -> {
            textViewProgress.append(message + "\n");
            // 滚动到底部
            int scrollAmount = textViewProgress.getLayout().getLineTop(textViewProgress.getLineCount()) - textViewProgress.getHeight();
            if (scrollAmount > 0) {
                textViewProgress.scrollTo(0, scrollAmount);
            } else {
                textViewProgress.scrollTo(0, 0);
            }
        });
    }

    // 获取可用的嵌入模型列表
    private List<String> getAvailableEmbeddingModels() {
        List<String> availableModels = new ArrayList<>();
        
        // 获取设置中的嵌入模型路径
        String embeddingModelPath = ConfigManager.getEmbeddingModelPath(requireContext());
        Log.d(TAG, "嵌入模型路径: " + embeddingModelPath);
        
        // 获取嵌入模型目录
        File embeddingModelDir = new File(embeddingModelPath);
        if (embeddingModelDir.exists() && embeddingModelDir.isDirectory()) {
            // 获取所有模型文件
            File[] files = embeddingModelDir.listFiles();
            if (files != null) {
                for (File file : files) {
                    String name = file.getName().toLowerCase();
                    // 查找.pt、.pth、.onnx文件或包含model的目录
                    if ((file.isFile() && (name.endsWith(".pt") || name.endsWith(".pth") || name.endsWith(".onnx"))) ||
                        (file.isDirectory() && name.contains("model"))) {
                        availableModels.add(file.getName()); // 只添加文件名，而不是完整路径
                        Log.d(TAG, "找到可用模型: " + file.getName());
                    }
                }
            }
        }
        
        if (availableModels.isEmpty()) {
            Log.w(TAG, "未找到可用的嵌入模型");
        } else {
            Log.d(TAG, "共找到 " + availableModels.size() + " 个可用模型");
        }
        
        return availableModels;
    }

    // 获取所有可能的模型路径
    private List<String> getPossibleModelPaths() {
        List<String> possiblePaths = new ArrayList<>();
        
        // 获取配置中的嵌入模型路径
        String configPath = ConfigManager.getEmbeddingModelPath(requireContext());
        possiblePaths.add(configPath);
        
        // 添加可能的替代路径
        File externalStorageDir = Environment.getExternalStorageDirectory();
        File downloadDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        
        possiblePaths.add(new File(externalStorageDir, "starragdata/embeddings").getAbsolutePath());
        possiblePaths.add(new File(downloadDir, "starragdata/embeddings").getAbsolutePath());
        possiblePaths.add(new File(externalStorageDir, "Download/starragdata/embeddings").getAbsolutePath());
        possiblePaths.add("/storage/emulated/0/Download/starragdata/embeddings");
        possiblePaths.add("/sdcard/Download/starragdata/embeddings");
        
        return possiblePaths;
    }

    // 显示模型选择对话框
    private void showModelSelectionDialog(List<String> availableModels, ModelSelectionCallback callback) {
        // 创建对话框
        ModelSelectionDialog dialog = new ModelSelectionDialog(requireContext(), availableModels, callback);
        dialog.show();
    }

    // 更新知识库元数据中的模型信息
    private void updateKnowledgeBaseModelMetadata(File knowledgeBaseDir, String embeddingModelPath) {
        try {
            Log.d(TAG, "更新知识库元数据中的模型信息: " + embeddingModelPath);
            
            // 使用 SQLiteVectorDatabaseHandler 更新元数据
            SQLiteVectorDatabaseHandler vectorDb = null;
            try {
                // 创建SQLite向量数据库处理器
                Log.i(TAG, "开始创建SQLite向量数据库处理器，知识库目录: " + knowledgeBaseDir.getAbsolutePath());
                
                vectorDb = new SQLiteVectorDatabaseHandler(knowledgeBaseDir, "note");
                Log.i(TAG, "正在加载SQLite向量数据库...");
                
                if (vectorDb.loadDatabase()) {
                    // 更新嵌入模型路径
                    vectorDb.updateEmbeddingModel(embeddingModelPath);
                    Log.d(TAG, "成功更新元数据");
                    updateProgress("已更新知识库元数据中的模型信息");
                } else {
                    Log.e(TAG, "加载SQLite向量数据库失败");
                    updateProgress("警告：加载SQLite向量数据库失败");
                    
                    // 尝试创建新的元数据
                    vectorDb.updateEmbeddingModel(embeddingModelPath);
                    Log.d(TAG, "创建了新的数据库元数据");
                    updateProgress("已创建新的数据库元数据");
                }
            } catch (Exception e) {
                Log.e(TAG, "使用 SQLiteVectorDatabaseHandler 更新元数据失败", e);
                updateProgress("警告：更新数据库元数据失败: " + e.getMessage());
            } finally {
                // 确保关闭数据库
                if (vectorDb != null) {
                    try {
                        vectorDb.close();
                        Log.d(TAG, "已关闭向量数据库");
                    } catch (Exception e) {
                        Log.e(TAG, "关闭向量数据库失败", e);
                    }
                }
            }
            
            // 同时更新 metadata.json 文件以保持兼容性
            File jsonMetadataFile = new File(knowledgeBaseDir, "metadata.json");
            try {
                JSONObject metadata;
                if (jsonMetadataFile.exists()) {
                    String metadataJson = FileUtil.readFile(jsonMetadataFile);
                    if (!metadataJson.isEmpty()) {
                        metadata = new JSONObject(metadataJson);
                    } else {
                        metadata = new JSONObject();
                    }
                } else {
                    metadata = new JSONObject();
                }
                
                metadata.put("embeddingModel", embeddingModelPath);
                metadata.put("updated", System.currentTimeMillis());
                
                boolean success = FileUtil.writeFile(jsonMetadataFile, metadata.toString());
                if (success) {
                    Log.d(TAG, "成功更新 metadata.json 文件");
                } else {
                    Log.e(TAG, "更新 metadata.json 文件失败");
                }
            } catch (Exception e) {
                Log.e(TAG, "处理 metadata.json 文件失败", e);
            }
        } catch (Exception e) {
            Log.e(TAG, "更新知识库元数据时发生错误", e);
            updateProgress("警告：更新知识库元数据时发生错误: " + e.getMessage());
        }
    }

    // 模型选择回调接口
    private interface ModelSelectionCallback {
        void onModelSelected(String selectedModel, boolean rememberChoice);
    }

    // 模型选择对话框
    private static class ModelSelectionDialog extends AlertDialog {
        private List<String> availableModels;
        private ModelSelectionCallback callback;
        private boolean rememberChoice = true;

        public ModelSelectionDialog(Context context, List<String> availableModels, ModelSelectionCallback callback) {
            super(context);
            this.availableModels = availableModels;
            this.callback = callback;
        }

        @Override
        protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.dialog_model_selection);
            setTitle("选择嵌入模型");

            // 获取控件
            TextView textViewInfo = findViewById(R.id.textViewInfo);
            Spinner spinnerModels = findViewById(R.id.spinnerModels);
            CheckBox checkBoxRemember = findViewById(R.id.checkBoxRemember);

            // 设置提示信息
            textViewInfo.setText("请选择一个嵌入模型用于此知识库。\n" +
                    "如果选择\"记住选择\"，将更新知识库元数据。");

            // 创建模型列表适配器
            ArrayAdapter<String> adapter = new ArrayAdapter<>(getContext(), android.R.layout.simple_spinner_item, availableModels);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            spinnerModels.setAdapter(adapter);

            // 设置记住选择的复选框监听器
            checkBoxRemember.setOnCheckedChangeListener((buttonView, isChecked) -> {
                rememberChoice = isChecked;
            });

            // 设置确认和取消按钮
            setButton(DialogInterface.BUTTON_POSITIVE, "确定", (dialog, which) -> {
                int selectedPosition = spinnerModels.getSelectedItemPosition();
                if (selectedPosition >= 0 && selectedPosition < availableModels.size()) {
                    String selectedModel = availableModels.get(selectedPosition);
                    callback.onModelSelected(selectedModel, rememberChoice);
                }
            });

            setButton(DialogInterface.BUTTON_NEGATIVE, "取消", (dialog, which) -> {
                callback.onModelSelected(null, false);
            });
        }
    }

    /**
     * 将文本插入到内容编辑框中
     * @param text 要插入的文本
     */
    public void insertTextToContentEditor(String text) {
        if (editTextContent == null) {
            Log.e(TAG, "内容编辑框为空，无法插入文本");
            return;
        }
        
        try {
            // 获取当前光标位置
            int cursorPosition = editTextContent.getSelectionStart();
            
            // 获取当前内容
            Editable editable = editTextContent.getText();
            
            // 在光标位置插入文本
            if (cursorPosition != -1) {
                editable.insert(cursorPosition, text);
            } else {
                // 如果没有光标位置，则追加到末尾
                editable.append(text);
                // 将光标移动到末尾
                editTextContent.setSelection(editable.length());
            }
            
            // 显示提示
            Toast.makeText(requireContext(), "已将文本插入到笔记中", Toast.LENGTH_SHORT).show();
            
            // 如果标题为空，自动生成标题（使用文本的前10个字符）
            if (editTextTitle.getText().toString().trim().isEmpty()) {
                String title = text.trim();
                if (title.length() > 20) {
                    title = title.substring(0, 20) + "...";
                }
                editTextTitle.setText(title);
            }
            
            // 确保内容编辑框获得焦点
            editTextContent.requestFocus();
            
            Log.d(TAG, "成功将文本插入到内容编辑框");
        } catch (Exception e) {
            Log.e(TAG, "插入文本到内容编辑框失败", e);
            Toast.makeText(requireContext(), "插入文本失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
    
    /**
     * 应用全局字体大小设置
     */
    private void applyGlobalTextSize() {
        if (editTextContent != null) {
            float fontSize = ConfigManager.getGlobalTextSize(requireContext());
            editTextContent.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSize);
            Log.d(TAG, "已应用全局字体大小: " + fontSize + "sp");
        }
    }
}