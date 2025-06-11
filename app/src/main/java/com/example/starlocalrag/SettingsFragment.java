package com.example.starlocalrag;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import com.example.starlocalrag.LogManager;
import com.example.starlocalrag.api.LocalLlmHandler;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.view.MenuProvider;
import androidx.documentfile.provider.DocumentFile;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Lifecycle;
import androidx.navigation.Navigation;

import org.json.JSONObject;

import java.io.File;

public class SettingsFragment extends Fragment {
    private static final String TAG = "SettingsFragment";
    
    // UI组件
    private EditText editTextChunkSize;
    private EditText editTextOverlapSize;
    private EditText editTextMinChunkSize; // 添加最小分块限制输入框
    private EditText editTextModelPath;
    private EditText editTextEmbeddingModelPath;
    private EditText editTextKnowledgeBasePath;
    private Button buttonSelectModelPath;
    private Button buttonSelectEmbeddingModelPath;
    private Button buttonSelectKnowledgeBasePath;
    private Button buttonSaveSettings;
    private SwitchCompat switchDebugMode;
    private SwitchCompat switchUseGpu;
    private SwitchCompat switchUseOnnxGenAI; // OnnxRuntimeGenAI引擎开关
    private SwitchCompat switchJsonDatasetSplitting; // JSON训练集分块优化开关
    private SeekBar seekBarFontSize; // 字体大小拖动条
    private TextView textViewFontSizeValue; // 字体大小值显示
    
    // LLM 推理设置相关UI组件
    private EditText editTextMaxSequenceLength; // 最大序列长度
    private EditText editTextThreads; // 推理线程数
    private EditText editTextMaxNewTokens; // 最大输出token数
    
    // Activity Result Launchers
    private ActivityResultLauncher<Intent> modelPathLauncher;
    private ActivityResultLauncher<Intent> embeddingModelPathLauncher;
    private ActivityResultLauncher<Intent> knowledgeBasePathLauncher;
    // 思考模式开关已移动到RAG问答界面
    
    // 设置变更监听器
    private SettingsChangeListener settingsChangeListener;
    
    // 请求码
    private static final int REQUEST_CODE_MODEL_PATH = 1001;
    private static final int REQUEST_CODE_EMBEDDING_MODEL_PATH = 1002;
    private static final int REQUEST_CODE_KNOWLEDGE_BASE_PATH = 1003;
    
    // 设置监听器接口
    public interface SettingsChangeListener {
        void onSettingsChanged();
    }
    
    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
        try {
            settingsChangeListener = (SettingsChangeListener) context;
        } catch (ClassCastException e) {
            LogManager.logE(TAG, "Activity must implement SettingsChangeListener", e);
        }
    }
    
    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // 初始化Activity Result Launchers
        modelPathLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                    handleDirectorySelection(result.getData().getData(), editTextModelPath);
                }
            }
        );
        
        embeddingModelPathLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                    handleDirectorySelection(result.getData().getData(), editTextEmbeddingModelPath);
                }
            }
        );
        
        knowledgeBasePathLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                    handleDirectorySelection(result.getData().getData(), editTextKnowledgeBasePath);
                }
            }
        );
    }
    
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_settings, container, false);
        
        // 初始化UI控件
        editTextChunkSize = view.findViewById(R.id.editTextChunkSize);
        editTextOverlapSize = view.findViewById(R.id.editTextOverlapSize);
        editTextMinChunkSize = view.findViewById(R.id.editTextMinChunkSize); // 初始化最小分块限制输入框
        editTextModelPath = view.findViewById(R.id.editTextModelPath);
        editTextEmbeddingModelPath = view.findViewById(R.id.editTextEmbeddingModelPath);
        editTextKnowledgeBasePath = view.findViewById(R.id.editTextKnowledgeBasePath);
        buttonSelectModelPath = view.findViewById(R.id.buttonSelectModelPath);
        buttonSelectEmbeddingModelPath = view.findViewById(R.id.buttonSelectEmbeddingModelPath);
        buttonSelectKnowledgeBasePath = view.findViewById(R.id.buttonSelectKnowledgeBasePath);
        buttonSaveSettings = view.findViewById(R.id.buttonSaveSettings);
        switchDebugMode = view.findViewById(R.id.switchDebugMode);
        switchUseGpu = view.findViewById(R.id.switchUseGpu);
        switchUseOnnxGenAI = view.findViewById(R.id.switchUseOnnxGenAI); // OnnxRuntimeGenAI引擎开关
        switchJsonDatasetSplitting = view.findViewById(R.id.switchJsonDatasetSplitting); // JSON训练集分块优化开关
        
        // 初始化 LLM 推理设置相关UI组件
        editTextMaxSequenceLength = view.findViewById(R.id.editTextMaxSequenceLength);
        editTextThreads = view.findViewById(R.id.editTextThreads);
        editTextMaxNewTokens = view.findViewById(R.id.editTextKvCacheSize); // UI控件ID保持不变以兼容现有布局
        // switchNoThinking已移动到RAG问答界面
        seekBarFontSize = view.findViewById(R.id.seekBarFontSize); // 字体大小拖动条
        textViewFontSizeValue = view.findViewById(R.id.textViewFontSizeValue); // 字体大小值显示
        
        // 加载当前设置
        loadSettings();
        
        // 设置按钮点击事件
        setupListeners();
        
        // 设置字体大小拖动条变化监听器
        setupFontSizeSeekBar();
        
        return view;
    }
    
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 添加MenuProvider来处理菜单
        requireActivity().addMenuProvider(new MenuProvider() {
            @Override
            public void onCreateMenu(@NonNull Menu menu, @NonNull MenuInflater menuInflater) {
                menuInflater.inflate(R.menu.settings_menu, menu);
            }
            
            @Override
            public boolean onMenuItemSelected(@NonNull MenuItem menuItem) {
                int id = menuItem.getItemId();
                
                if (id == R.id.action_close_settings) {
                    // 关闭设置页面 - 使用Navigation组件的返回操作
                    requireActivity().getOnBackPressedDispatcher().onBackPressed();
                    return true;
                }
                
                return false;
            }
        }, getViewLifecycleOwner(), Lifecycle.State.RESUMED);
    }
    
    private void setupListeners() {
        // 选择模型目录
        buttonSelectModelPath.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            modelPathLauncher.launch(intent);
        });
        
        // 选择嵌入模型目录
        buttonSelectEmbeddingModelPath.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            embeddingModelPathLauncher.launch(intent);
        });
        
        // 选择知识库目录
        buttonSelectKnowledgeBasePath.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            knowledgeBasePathLauncher.launch(intent);
        });
        
        // 保存设置
        buttonSaveSettings.setOnClickListener(v -> saveSettings());
    }
    
    private void setupFontSizeSeekBar() {
        // 设置初始值
        float currentFontSize = ConfigManager.getGlobalTextSize(requireContext());
        // 将字体大小转换为进度值（10-24sp对应0-14的进度）
        int progress = Math.round(currentFontSize) - 10;
        if (progress < 0) progress = 0;
        if (progress > 14) progress = 14;
        seekBarFontSize.setProgress(progress);
        updateFontSizeText(progress);
        
        seekBarFontSize.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                // 更新字体大小值显示
                updateFontSizeText(progress);
            }
            
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // 不需要处理
            }
            
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // 停止拖动时，应用字体大小变化
                int progress = seekBar.getProgress();
                float fontSize = progress + 10;
                // 立即应用字体大小变化，让用户可以预览效果
                ConfigManager.setGlobalTextSize(requireContext(), fontSize);
                // 通知设置已变更
                if (settingsChangeListener != null) {
                    settingsChangeListener.onSettingsChanged();
                }
            }
        });
    }
    
    private void updateFontSizeText(int progress) {
        float fontSize = progress + 10;
        textViewFontSizeValue.setText(String.format("字体大小: %.0fsp", fontSize));
        // 应用字体大小到预览文本
        textViewFontSizeValue.setTextSize(fontSize);
    }
    
    private void loadSettings() {
        try {
            // 从ConfigManager加载设置
            Context context = requireContext();
            
            // 加载分块设置
            int chunkSize = ConfigManager.getChunkSize(context);
            int overlapSize = ConfigManager.getInt(context, ConfigManager.KEY_OVERLAP_SIZE, ConfigManager.DEFAULT_OVERLAP_SIZE);
            int minChunkSize = ConfigManager.getMinChunkSize(context); // 获取最小分块限制
            
            // 加载模型路径
            String modelPath = ConfigManager.getModelPath(context);
            String embeddingModelPath = ConfigManager.getEmbeddingModelPath(context);
            String knowledgeBasePath = ConfigManager.getKnowledgeBasePath(context);
            
            // 加载调试模式设置
            boolean debugMode = ConfigManager.getBoolean(context, ConfigManager.KEY_DEBUG_MODE, false);
            
            // 加载GPU加速设置
            boolean useGpu = ConfigManager.getBoolean(context, ConfigManager.KEY_USE_GPU, false);
            
            // 加载OnnxRuntimeGenAI引擎设置
            boolean useOnnxGenAI = ConfigManager.getBoolean(context, ConfigManager.KEY_USE_ONNX_GENAI, true);
            
            // 加载JSON训练集分块优化开关
            boolean jsonDatasetSplitting = ConfigManager.isJsonDatasetSplittingEnabled(context);
            
            // 加载字体大小
            float fontSize = ConfigManager.getGlobalTextSize(context);
            
            // 加载 LLM 推理设置
            int maxSequenceLength = ConfigManager.getMaxSequenceLength(context);
            int threads = ConfigManager.getThreads(context);
            int maxNewTokens = ConfigManager.getMaxNewTokens(context);
            boolean noThinking = ConfigManager.getNoThinking(context);
            
            // 设置UI
            editTextChunkSize.setText(String.valueOf(chunkSize));
            editTextOverlapSize.setText(String.valueOf(overlapSize));
            editTextMinChunkSize.setText(String.valueOf(minChunkSize)); // 设置最小分块限制UI
            editTextModelPath.setText(modelPath);
            editTextEmbeddingModelPath.setText(embeddingModelPath);
            editTextKnowledgeBasePath.setText(knowledgeBasePath);
            switchDebugMode.setChecked(debugMode);
            switchUseGpu.setChecked(useGpu);
            switchUseOnnxGenAI.setChecked(useOnnxGenAI);
            switchJsonDatasetSplitting.setChecked(jsonDatasetSplitting);
            seekBarFontSize.setProgress(Math.round(fontSize) - 10);
            updateFontSizeText(Math.round(fontSize) - 10);
            
            // 设置 LLM 推理设置UI
            editTextMaxSequenceLength.setText(String.valueOf(maxSequenceLength));
            editTextThreads.setText(String.valueOf(threads));
            editTextMaxNewTokens.setText(String.valueOf(maxNewTokens));
            // switchNoThinking已移动到RAG问答界面
            
            LogManager.logD(TAG, "设置加载完成");
        } catch (Exception e) {
            LogManager.logE(TAG, "加载设置失败: " + e.getMessage(), e);
            Toast.makeText(requireContext(), "加载设置失败", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void saveSettings() {
        Context context = requireContext();
        
        try {
            // 获取用户输入
            String chunkSizeStr = editTextChunkSize.getText().toString().trim();
            String overlapSizeStr = editTextOverlapSize.getText().toString().trim();
            String minChunkSizeStr = editTextMinChunkSize.getText().toString().trim(); // 获取最小分块限制输入
            String modelPathStr = editTextModelPath.getText().toString().trim();
            String embeddingModelPathStr = editTextEmbeddingModelPath.getText().toString().trim();
            String knowledgeBasePathStr = editTextKnowledgeBasePath.getText().toString().trim();
            
            // 验证输入
            if (chunkSizeStr.isEmpty() || overlapSizeStr.isEmpty() || minChunkSizeStr.isEmpty()) { // 验证最小分块限制
                Toast.makeText(context, "请填写所有分块设置", Toast.LENGTH_SHORT).show();
                return;
            }
            
            // 转换为整数
            int chunkSize = Integer.parseInt(chunkSizeStr);
            int overlapSize = Integer.parseInt(overlapSizeStr);
            int minChunkSize = Integer.parseInt(minChunkSizeStr); // 转换最小分块限制
            
            // 验证值范围
            if (chunkSize < 100 || chunkSize > 4000) {
                Toast.makeText(context, "分块大小应在100-4000之间", Toast.LENGTH_SHORT).show();
                return;
            }
            
            if (overlapSize < 20 || overlapSize > 800 || overlapSize >= chunkSize) {
                Toast.makeText(context, "重叠大小应在20-800之间且小于分块大小", Toast.LENGTH_SHORT).show();
                return;
            }
            
            if (minChunkSize < 10 || minChunkSize > 200 || minChunkSize >= chunkSize) {
                Toast.makeText(context, "最小分块限制应在10-200之间且小于分块大小", Toast.LENGTH_SHORT).show();
                return;
            }
            
            // 获取路径设置
            String modelPath = editTextModelPath.getText().toString().trim();
            String embeddingModelPath = editTextEmbeddingModelPath.getText().toString().trim();
            String knowledgeBasePath = editTextKnowledgeBasePath.getText().toString().trim();
            
            // 验证路径
            if (modelPath.isEmpty() || embeddingModelPath.isEmpty() || knowledgeBasePath.isEmpty()) {
                Toast.makeText(context, "请设置所有路径", Toast.LENGTH_SHORT).show();
                return;
            }
            
            // 获取调试模式设置
            boolean debugMode = switchDebugMode.isChecked();
            
            // 获取GPU加速设置
            boolean useGpu = switchUseGpu.isChecked();
            
            // 获取OnnxRuntimeGenAI引擎设置
            boolean useOnnxGenAI = switchUseOnnxGenAI.isChecked();
            
            // 获取JSON训练集分块优化开关
            boolean jsonDatasetSplitting = switchJsonDatasetSplitting.isChecked();
            
            // 获取字体大小
            int progress = seekBarFontSize.getProgress();
            float fontSize = progress + 10;
            
            // 获取 LLM 推理设置
            String maxSequenceLengthStr = editTextMaxSequenceLength.getText().toString().trim();
            String threadsStr = editTextThreads.getText().toString().trim();
            String maxNewTokensStr = editTextMaxNewTokens.getText().toString().trim();
            // noThinking已移动到RAG问答界面
            
            // 验证 LLM 推理设置
            if (maxSequenceLengthStr.isEmpty() || threadsStr.isEmpty() || maxNewTokensStr.isEmpty()) {
                Toast.makeText(context, "请填写所有 LLM 推理设置", Toast.LENGTH_SHORT).show();
                return;
            }
            
            // 转换为整数
            int maxSequenceLength = Integer.parseInt(maxSequenceLengthStr);
            int threads = Integer.parseInt(threadsStr);
            int maxNewTokens = Integer.parseInt(maxNewTokensStr);
            
            // 验证值范围
            if (maxSequenceLength < 100 || maxSequenceLength > 8192) {
                Toast.makeText(context, "最大序列长度应在100-8192之间", Toast.LENGTH_SHORT).show();
                return;
            }
            
            if (threads < 1 || threads > 16) {
                Toast.makeText(context, "推理线程数应在1-16之间", Toast.LENGTH_SHORT).show();
                return;
            }
            
            if (maxNewTokens < 512 || maxNewTokens > 4096) {
                Toast.makeText(context, "最大输出token数应在512-4096之间", Toast.LENGTH_SHORT).show();
                return;
            }
            
            // 验证最大序列长度与最大输出token数的关系
            // maxSequenceLength是总的上下文长度，maxNewTokens是最大输出token数
        // 输入token数 = maxSequenceLength - maxNewTokens，需要预留一定缓冲区
        if (maxSequenceLength <= maxNewTokens + 256) {
            Toast.makeText(context, "最大序列长度必须大于最大输出token数加上256 (当前需要大于" + (maxNewTokens + 256) + ")", Toast.LENGTH_LONG).show();
                return;
            }
            
            // 保存设置到ConfigManager
            ConfigManager.setChunkSize(context, chunkSize);
            ConfigManager.setInt(context, ConfigManager.KEY_OVERLAP_SIZE, overlapSize);
            ConfigManager.setMinChunkSize(context, minChunkSize); // 保存最小分块限制
            ConfigManager.setString(context, ConfigManager.KEY_MODEL_PATH, modelPath);
            ConfigManager.setString(context, ConfigManager.KEY_EMBEDDING_MODEL_PATH, embeddingModelPath);
            ConfigManager.setString(context, ConfigManager.KEY_KNOWLEDGE_BASE_PATH, knowledgeBasePath);
            ConfigManager.setBoolean(context, ConfigManager.KEY_DEBUG_MODE, debugMode);
            ConfigManager.setBoolean(context, ConfigManager.KEY_USE_GPU, useGpu);
            ConfigManager.setBoolean(context, ConfigManager.KEY_USE_ONNX_GENAI, useOnnxGenAI);
            ConfigManager.setJsonDatasetSplittingEnabled(context, jsonDatasetSplitting);
            ConfigManager.setGlobalTextSize(context, fontSize);
            
            // 保存 LLM 推理设置
            ConfigManager.setMaxSequenceLength(context, maxSequenceLength);
            ConfigManager.setThreads(context, threads);
            ConfigManager.setMaxNewTokens(context, maxNewTokens);
            // noThinking已移动到RAG问答界面
            
            // 创建JSON格式的设置摘要
            JSONObject settingsSummary = new JSONObject();
            settingsSummary.put("chunkSize", chunkSize);
            settingsSummary.put("overlapSize", overlapSize);
            settingsSummary.put("minChunkSize", minChunkSize); // 添加最小分块限制
            settingsSummary.put("modelPath", modelPath);
            settingsSummary.put("embeddingModelPath", embeddingModelPath);
            settingsSummary.put("knowledgeBasePath", knowledgeBasePath);
            settingsSummary.put("debugMode", debugMode);
            settingsSummary.put("useGpu", useGpu);
            settingsSummary.put("useOnnxGenAI", useOnnxGenAI);
            settingsSummary.put("jsonDatasetSplitting", jsonDatasetSplitting);
            
            // 添加 LLM 推理设置信息
            settingsSummary.put("maxSequenceLength", maxSequenceLength);
            settingsSummary.put("threads", threads);
            // noThinking已移动到RAG问答界面
            
            LogManager.logD(TAG, "设置已保存: " + settingsSummary.toString());
            
            // 显示成功消息
            Toast.makeText(context, "设置已保存", Toast.LENGTH_SHORT).show();
            
            // 更新LocalLlmHandler的推理引擎配置
            try {
                LocalLlmHandler localLlmHandler = LocalLlmHandler.getInstance(context);
                localLlmHandler.updateEngineFromConfig();
                LogManager.logI(TAG, "推理引擎配置已更新");
            } catch (Exception e) {
                LogManager.logE(TAG, "更新推理引擎配置失败: " + e.getMessage(), e);
            }
            
            // 通知监听器
            if (settingsChangeListener != null) {
                settingsChangeListener.onSettingsChanged();
            }
        } catch (NumberFormatException e) {
            LogManager.logE(TAG, "解析数字失败: " + e.getMessage(), e);
            Toast.makeText(context, "请输入有效的数字", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            LogManager.logE(TAG, "保存设置失败: " + e.getMessage(), e);
            Toast.makeText(context, "保存设置失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
    
    private void handleDirectorySelection(Uri uri, EditText targetEditText) {
        if (uri != null) {
            try {
                // 将 URI 转换为实际可用的文件路径
                String realPath = getPathFromUri(uri);
                LogManager.logD(TAG, "选择的路径 URI: " + uri);
                LogManager.logD(TAG, "转换后的实际路径: " + realPath);
                
                targetEditText.setText(realPath);
            } catch (Exception e) {
                LogManager.logE(TAG, "转换路径失败: " + e.getMessage(), e);
                Toast.makeText(requireContext(), "无法获取选择的路径", Toast.LENGTH_SHORT).show();
            }
        }
    }
    
    /**
     * 将 URI 转换为实际可用的文件路径
     * @param uri 文件 URI
     * @return 实际文件路径
     */
    private String getPathFromUri(Uri uri) {
        String path = uri.getPath();
        
        // 检查是否是 content:// 类型的 URI
        if (uri.getScheme() != null && uri.getScheme().equals("content")) {
            // 处理 /tree/primary: 格式的路径
            if (path != null && path.contains("primary:")) {
                // 提取 primary: 后面的部分
                String[] segments = path.split("primary:");
                if (segments.length > 1) {
                    // 转换为实际的存储路径
                    String relativePath = segments[1];
                    // 替换可能存在的重复路径
                    if (relativePath.contains("/document/primary:")) {
                        relativePath = relativePath.split("/document/primary:")[0];
                    }
                    
                    // 构建实际路径
                    File externalStorage = Environment.getExternalStorageDirectory();
                    path = new File(externalStorage, relativePath).getAbsolutePath();
                    LogManager.logD(TAG, "转换 primary: 路径: " + path);
                }
            }
        }
        
        return path;
    }
    

    
    // 静态方法，用于获取设置值
    public static int getChunkSize(Context context) {
        return ConfigManager.getChunkSize(context);
    }
    
    public static int getOverlapSize(Context context) {
        return ConfigManager.getInt(context, ConfigManager.KEY_OVERLAP_SIZE, ConfigManager.DEFAULT_OVERLAP_SIZE);
    }
    
    /**
     * 获取最小分块限制
     * @param context 上下文
     * @return 最小分块限制
     */
    public static int getMinChunkSize(Context context) {
        return ConfigManager.getMinChunkSize(context);
    }
    
    public static String getModelPath(Context context) {
        return ConfigManager.getModelPath(context);
    }
    
    public static String getEmbeddingModelPath(Context context) {
        return ConfigManager.getEmbeddingModelPath(context);
    }
    
    public static String getKnowledgeBasePath(Context context) {
        return ConfigManager.getKnowledgeBasePath(context);
    }
    
    /**
     * 获取是否启用调试模式
     * @param context 上下文
     * @return 是否启用调试模式
     */
    public static boolean isDebugModeEnabled(Context context) {
        return ConfigManager.getBoolean(context, ConfigManager.KEY_DEBUG_MODE, false);
    }
    
    /**
     * 获取是否使用GPU加速
     * @param context 上下文
     * @return 是否使用GPU加速
     */
    public static boolean getUseGpu(Context context) {
        return ConfigManager.getBoolean(context, ConfigManager.KEY_USE_GPU, false);
    }
    
    /**
     * 获取是否启用JSON训练集分块优化
     * @param context 上下文
     * @return 是否启用JSON训练集分块优化
     */
    public static boolean isJsonDatasetSplittingEnabled(Context context) {
        return ConfigManager.isJsonDatasetSplittingEnabled(context);
    }
}
