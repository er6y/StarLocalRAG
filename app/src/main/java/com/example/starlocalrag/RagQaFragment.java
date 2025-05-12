package com.example.starlocalrag;

import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.Selection;
import android.text.Spannable;
import android.text.TextUtils;
import android.util.Log;
import android.util.TypedValue;
import android.view.ActionMode;
import android.view.ContextMenu;
import android.view.ContextMenu.ContextMenuInfo;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.SpinnerAdapter;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;

import com.android.volley.Request;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import com.example.starlocalrag.VectorDatabaseHandler;
import com.example.starlocalrag.EmbeddingModelHandler;
import com.example.starlocalrag.EmbeddingModelManager;
import com.example.starlocalrag.EmbeddingModelUtils;
import com.example.starlocalrag.SQLiteVectorDatabaseHandler;
import com.example.starlocalrag.ConfigManager;
import com.example.starlocalrag.ApiUrlAdapter;
import com.example.starlocalrag.TokenizerManager;
import io.noties.markwon.Markwon;
import io.noties.markwon.html.HtmlPlugin;
import io.noties.markwon.AbstractMarkwonPlugin;
import io.noties.markwon.ext.tables.TablePlugin;
import io.noties.markwon.ext.tasklist.TaskListPlugin;
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin;
import io.noties.markwon.image.ImagesPlugin;
import io.noties.markwon.core.MarkwonTheme;
import io.noties.markwon.movement.MovementMethodPlugin;
import io.noties.markwon.linkify.LinkifyPlugin;
import android.graphics.Color;
import android.text.Spanned;
import android.text.method.LinkMovementMethod;

public class RagQaFragment extends Fragment {

    private static final String TAG = "StarLocalRAG_RagQa"; // 添加TAG用于日志打印
    private static final String LOG_FILE = "api_log.txt"; // 日志文件名

    private Spinner spinnerApiUrl;
    private EditText editTextApiKey;
    private Spinner spinnerApiModel;
    private Spinner spinnerKnowledgeBase;
    private EditText editTextSystemPrompt;
    private EditText editTextUserPrompt;
    private Button buttonSendStop;
    private Button buttonNewChat;
    private EditText editTextSearchDepth; // 近似深度输入框
    private TextView textViewResponse; // 回答文本框
    
    // Markdown渲染器
    private Markwon markwon;
    private final StringBuilder answerBuilder = new StringBuilder();
    private final StringBuilder debugBuilder = new StringBuilder();

    private boolean isSending = false; // Track the state of the send/stop button
    private static final String CONFIG_FILE = ".config"; // 配置文件名
    private List<String> systemPromptHistory = new ArrayList<>(); // 系统提示词历史记录
    private Map<String, String> apiKeyMap = new HashMap<>(); // API Key映射
    
    // 当前是否有正在执行的RAG查询任务
    private boolean isTaskRunning = false;
    private boolean isTaskCancelled = false;
    
    // 用于执行后台任务的线程池
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    // 主线程Handler
    private Handler mainHandler;

    // 搜索结果文档
    private List<String> relevantDocuments;
    private String similarityInfo;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_rag_qa, container, false);
        
        // 初始化UI元素
        spinnerApiUrl = view.findViewById(R.id.spinnerApiUrl);
        editTextApiKey = view.findViewById(R.id.editTextApiKey);
        spinnerApiModel = view.findViewById(R.id.spinnerApiModel);
        spinnerKnowledgeBase = view.findViewById(R.id.spinnerKnowledgeBase);
        editTextSystemPrompt = view.findViewById(R.id.editTextSystemPrompt);
        editTextUserPrompt = view.findViewById(R.id.editTextUserPrompt);
        buttonSendStop = view.findViewById(R.id.buttonSendStop);
        buttonNewChat = view.findViewById(R.id.buttonNewChat);
        editTextSearchDepth = view.findViewById(R.id.editTextSearchDepth); // 初始化近似深度输入框
        
        // 为用户提问文本框添加回车键监听
        editTextUserPrompt.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND) {
                handleSendStopClick();
                return true;
            }
            return false;
        });
        
        // 加载配置和设置UI
        loadConfig();
        
        // 加载API URL列表，包括从配置中获取的自定义URL
        loadApiUrlList();
        
        // 设置其他Spinner的初始数据
        setupSpinner(spinnerApiModel, new String[]{"加载中..."});
        setupSpinner(spinnerKnowledgeBase, new String[]{"加载中..."});
        
        // 为API URL Spinner添加选择监听器，自动加载对应的API Key
        spinnerApiUrl.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String selectedApiUrl = parent.getItemAtPosition(position).toString();
                
                // 检查是否选择了"新建..."选项
                if (selectedApiUrl.equals("新建...")) {
                    showAddApiUrlDialog();
                    return;
                }
                
                loadApiKeyForUrl(selectedApiUrl);
                fetchModelsForApi(); // 自动获取模型列表
                
                // 保存API URL设置
                ConfigManager.setString(requireContext(), ConfigManager.KEY_API_URL, selectedApiUrl);
                Log.d(TAG, "已保存API URL: " + selectedApiUrl);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // 不做任何操作
            }
        });
        
        // 为API Key添加焦点变化监听器，当失去焦点时保存API Key
        editTextApiKey.setOnFocusChangeListener((v, hasFocus) -> {
            if (!hasFocus) {
                String apiKey = editTextApiKey.getText().toString();
                String apiUrl = spinnerApiUrl.getSelectedItem().toString();
                if (!apiKey.isEmpty()) {
                    ConfigManager.saveApiKey(requireContext(), apiUrl, apiKey);
                    Log.d(TAG, "已保存API Key到URL: " + apiUrl);
                }
            }
        });

        // 添加触摸监听器，当点击模型下拉框时获取模型列表
        spinnerApiModel.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_UP) {
                fetchModelsForApi();
            }
            return false; // 允许正常的spinner行为
        });
        
        // 为模型Spinner添加选择监听器，保存选择的模型
        spinnerApiModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String selectedModel = parent.getItemAtPosition(position).toString();
                if (!selectedModel.equals("加载中...")) {
                    ConfigManager.setString(requireContext(), ConfigManager.KEY_MODEL_NAME, selectedModel);
                    Log.d(TAG, "已保存模型名称: " + selectedModel);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // 不做任何操作
            }
        });
        
        // 添加知识库下拉框触摸监听器
        spinnerKnowledgeBase.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_UP) {
                loadKnowledgeBases();
            }
            return false;
        });
        
        // 为知识库Spinner添加选择监听器，保存选择的知识库
        spinnerKnowledgeBase.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String selectedKnowledgeBase = parent.getItemAtPosition(position).toString();
                if (!selectedKnowledgeBase.equals("加载中...")) {
                    ConfigManager.setString(requireContext(), ConfigManager.KEY_KNOWLEDGE_BASE, selectedKnowledgeBase);
                    Log.d(TAG, "已保存知识库名称: " + selectedKnowledgeBase);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // 不做任何操作
            }
        });
        
        // 为系统提示词添加焦点变化监听器，当失去焦点时保存系统提示词
        editTextSystemPrompt.setOnFocusChangeListener((v, hasFocus) -> {
            if (!hasFocus) {
                String systemPrompt = editTextSystemPrompt.getText().toString();
                if (!systemPrompt.isEmpty()) {
                    ConfigManager.setSystemPrompt(requireContext(), systemPrompt);
                    Log.d(TAG, "已保存系统提示词");
                }
            }
        });
        
        // 为近似深度添加焦点变化监听器，当失去焦点时保存近似深度
        editTextSearchDepth.setOnFocusChangeListener((v, hasFocus) -> {
            if (!hasFocus) {
                String searchDepthStr = editTextSearchDepth.getText().toString();
                if (!searchDepthStr.isEmpty()) {
                    try {
                        int searchDepth = Integer.parseInt(searchDepthStr);
                        ConfigManager.setInt(requireContext(), ConfigManager.KEY_SEARCH_DEPTH, searchDepth);
                        Log.d(TAG, "已保存近似深度: " + searchDepth);
                    } catch (NumberFormatException e) {
                        Log.e(TAG, "近似深度格式错误: " + searchDepthStr, e);
                    }
                }
            }
        });
        
        // 设置按钮监听器
        buttonSendStop.setOnClickListener(v -> handleSendStopClick());
        buttonNewChat.setOnClickListener(v -> handleNewChatClick());
        
        // 加载知识库列表
        loadKnowledgeBases();
        
        return view;
    }
    
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 初始化主线程Handler
        mainHandler = new Handler(Looper.getMainLooper());
        
        // 初始Markdown渲染器，使用全功能插件支持
        Log.d(TAG, "初始化Markwon渲染器");
        markwon = Markwon.builder(requireContext())
                // 添加HTML支持
                .usePlugin(HtmlPlugin.create())
                // 添加表格支持
                .usePlugin(TablePlugin.create(requireContext()))
                // 添加任务列表支持
                .usePlugin(TaskListPlugin.create(requireContext()))
                // 添加删除线支持
                .usePlugin(StrikethroughPlugin.create())
                // 添加图片支持
                .usePlugin(ImagesPlugin.create())
                // 添加链接支持
                .usePlugin(LinkifyPlugin.create())
                // 添加移动方法插件，支持链接点击
                .usePlugin(MovementMethodPlugin.create(LinkMovementMethod.getInstance()))
                // 添加自定义插件，使行内代码使用等宽字体且不使用背景色
                .usePlugin(new AbstractMarkwonPlugin() {
                    @Override
                    public void beforeSetText(@NonNull TextView textView, @NonNull Spanned markdown) {
                        // 设置文本选择可用
                        textView.setTextIsSelectable(true);
                        // 设置文本颜色为黑色，提高可读性
                        textView.setTextColor(Color.BLACK);
                    }
                    
                    @Override
                    public void configureTheme(@NonNull MarkwonTheme.Builder builder) {
                        // 配置代码块样式 - 使用现代IDE风格的深色主题
                        builder
                            // 代码块使用深色背景，类似VSCode的One Dark主题
                            .codeBlockBackgroundColor(Color.parseColor("#282c34"))
                            // 代码块使用浅色文本，提供高对比度
                            .codeBlockTextColor(Color.parseColor("#abb2bf"))
                            // 行内代码不使用背景色，设置为透明
                            .codeBackgroundColor(Color.TRANSPARENT)
                            // 行内代码使用粗体显示，增强视觉效果
                            .codeTextColor(Color.parseColor("#000000"))
                            // 增加代码块内边距
                            .codeBlockMargin(16)
                            // 增加块间距
                            .blockMargin(12)
                            // 设置引用块样式
                            .blockQuoteColor(Color.parseColor("#5c6bc0"));
                    }
                })
                .build();
                
        Log.d(TAG, "Markwon渲染器初始化完成");
        
        // 初始化文本缩放辅助类
        textViewResponse = view.findViewById(R.id.textViewResponse);
        
        // 应用全局字体大小
        applyGlobalTextSize();
        
        // 初始化日志管理器
        LogManager.getInstance(requireContext());
        
        // 设置自定义文本选择菜单
        setupCustomTextSelectionMenu();
    }
    
    // 以下方法从MainActivity中复制，并根据Fragment的需要进行了调整
    
    private void setupSpinner(Spinner spinner, String[] items) {
        ArrayAdapter<String> adapter = new ArrayAdapter<>(requireContext(), android.R.layout.simple_spinner_item, items);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
    }
    
    // 加载配置文件
    private void loadConfig() {
        try {
            // 使用ConfigManager加载配置
            
            // 加载API URL
            String apiUrl = ConfigManager.getString(requireContext(), ConfigManager.KEY_API_URL, "");
            if (!apiUrl.isEmpty()) {
                setSpinnerSelection(spinnerApiUrl, apiUrl);
            }
            
            // 加载模型名称
            String modelName = ConfigManager.getString(requireContext(), ConfigManager.KEY_MODEL_NAME, "");
            if (!modelName.isEmpty()) {
                setSpinnerSelection(spinnerApiModel, modelName);
            }
            
            // 加载知识库名称
            String knowledgeBase = ConfigManager.getString(requireContext(), ConfigManager.KEY_KNOWLEDGE_BASE, "");
            if (!knowledgeBase.isEmpty()) {
                setSpinnerSelection(spinnerKnowledgeBase, knowledgeBase);
            }
            
            // 加载系统提示词
            String systemPrompt = ConfigManager.getSystemPrompt(requireContext());
            if (!systemPrompt.isEmpty()) {
                editTextSystemPrompt.setText(systemPrompt);
            }
            
            // 加载所有API Keys
            Map<String, String> apiKeys = ConfigManager.getAllApiKeys(requireContext());
            if (!apiKeys.isEmpty()) {
                apiKeyMap.putAll(apiKeys);
                Log.d(TAG, "已加载 " + apiKeys.size() + " 个API Keys");
                
                // 根据当前选择的API URL加载对应的API Key
                if (!apiUrl.isEmpty()) {
                    loadApiKeyForUrl(apiUrl);
                }
            }
            
            // 加载近似深度
            int searchDepth = ConfigManager.getSearchDepth(requireContext());
            editTextSearchDepth.setText(String.valueOf(searchDepth));
            Log.d(TAG, "已加载近似深度: " + searchDepth);
            
            Log.d(TAG, "配置加载完成");
        } catch (Exception e) {
            Log.e(TAG, "加载配置失败", e);
        }
    }
    
    // 保存配置到文件
    private void saveConfig() {
        try {
            // 获取当前选择的值
            String apiUrl = spinnerApiUrl.getSelectedItem().toString();
            String apiKey = editTextApiKey.getText().toString();
            String model = spinnerApiModel.getSelectedItem().toString();
            String knowledgeBase = spinnerKnowledgeBase.getSelectedItem().toString();
            String systemPrompt = editTextSystemPrompt.getText().toString();
            
            // 直接保存到一级配置
            ConfigManager.setString(requireContext(), ConfigManager.KEY_API_URL, apiUrl);
            ConfigManager.setString(requireContext(), ConfigManager.KEY_MODEL_NAME, model);
            ConfigManager.setString(requireContext(), ConfigManager.KEY_KNOWLEDGE_BASE, knowledgeBase);
            
            // 保存API Key到对应的URL
            if (!apiKey.isEmpty()) {
                ConfigManager.saveApiKey(requireContext(), apiUrl, apiKey);
                Log.d(TAG, "已保存API Key到URL: " + apiUrl);
            }
            
            // 保存系统提示词（使用一级项）
            // 无论是否为空都保存，确保用户清空系统提示词时能正确保存
            ConfigManager.setSystemPrompt(requireContext(), systemPrompt);
            Log.d(TAG, "已保存系统提示词: " + (systemPrompt.isEmpty() ? "[空]" : systemPrompt));
            
            // 保存近似深度
            String searchDepthStr = editTextSearchDepth.getText().toString();
            if (!searchDepthStr.isEmpty()) {
                try {
                    int searchDepth = Integer.parseInt(searchDepthStr);
                    ConfigManager.setInt(requireContext(), ConfigManager.KEY_SEARCH_DEPTH, searchDepth);
                    Log.d(TAG, "已保存近似深度: " + searchDepth);
                } catch (NumberFormatException e) {
                    Log.e(TAG, "近似深度格式错误: " + searchDepthStr, e);
                }
            }
            
            Log.d(TAG, "配置已保存到 .config 文件");
            Toast.makeText(requireContext(), "设置已保存", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.e(TAG, "保存配置失败", e);
            Toast.makeText(requireContext(), "保存设置失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
    
    // 根据API URL加载对应的API Key
    private void loadApiKeyForUrl(String apiUrl) {
        if (apiUrl == null || apiUrl.isEmpty()) {
            return;
        }
        
        // 使用ConfigManager获取API Key
        String apiKey = ConfigManager.getApiKey(requireContext(), apiUrl);
        if (apiKey != null && !apiKey.isEmpty()) {
            editTextApiKey.setText(apiKey);
            Log.d(TAG, "已加载API URL对应的Key: " + apiUrl);
        } else {
            // 如果没有找到对应的API Key，清空输入框
            editTextApiKey.setText("");
            Log.d(TAG, "未找到API URL对应的Key: " + apiUrl);
        }
    }
    
    // 设置Spinner的选中项
    private void setSpinnerSelection(Spinner spinner, String value) {
        if (spinner == null || value == null || value.isEmpty()) {
            return;
        }
        
        // 获取适配器，不进行类型转换
        SpinnerAdapter adapter = spinner.getAdapter();
        if (adapter != null) {
            for (int i = 0; i < adapter.getCount(); i++) {
                Object item = adapter.getItem(i);
                if (item != null && item.toString().equals(value)) {
                    spinner.setSelection(i);
                    break;
                }
            }
        }
    }
    
    // 加载API URL列表，包括从配置中获取的自定义URL
    private void loadApiUrlList() {
        Log.d(TAG, "开始加载API URL列表");
        
        // 获取预定义的API URL列表
        String[] predefinedApiUrls = getResources().getStringArray(R.array.api_urls);
        
        // 获取设置中的API URL列表
        String[] customApiUrls = ConfigManager.getApiUrls(requireContext());
        
        // 合并预定义和自定义的API URL列表
        List<String> apiUrlsList = new ArrayList<>();
        
        // 添加"新建..."选项作为第一项
        apiUrlsList.add("新建...");
        
        // 添加预定义的API URL
        for (String apiUrl : predefinedApiUrls) {
            if (!apiUrl.equals("新建...") && !apiUrlsList.contains(apiUrl)) {
                apiUrlsList.add(apiUrl);
            }
        }
        
        // 添加自定义的API URL
        if (customApiUrls != null && customApiUrls.length > 0) {
            for (String apiUrl : customApiUrls) {
                if (!apiUrlsList.contains(apiUrl)) {
                    apiUrlsList.add(apiUrl);
                }
            }
        }
        
        // 创建并设置适配器
        ApiUrlAdapter adapter = new ApiUrlAdapter(
                requireContext(),
                apiUrlsList,
                this::deleteApiUrl,
                (apiUrl, position) -> {
                    // 处理API URL选择事件
                    if (apiUrl.equals("新建...")) {
                        showAddApiUrlDialog();
                    } else {
                        // 加载对应的API Key
                        loadApiKeyForUrl(apiUrl);
                        // 保存当前选择的API URL
                        ConfigManager.setString(requireContext(), ConfigManager.KEY_API_URL, apiUrl);
                    }
                },
                spinnerApiUrl
        );
        
        spinnerApiUrl.setAdapter(adapter);
        
        // 设置当前选中的API URL
        String currentApiUrl = ConfigManager.getString(requireContext(), ConfigManager.KEY_API_URL, "");
        if (!currentApiUrl.isEmpty()) {
            // 查找当前API URL的位置
            for (int i = 0; i < apiUrlsList.size(); i++) {
                if (apiUrlsList.get(i).equals(currentApiUrl)) {
                    spinnerApiUrl.setSelection(i);
                    adapter.setSelectedPosition(i);
                    break;
                }
            }
        }
        
        Log.d(TAG, "已加载 " + apiUrlsList.size() + " 个API URL");
    }
    
    /**
     * 删除API URL
     * @param apiUrl 要删除的API URL
     * @param position 位置
     */
    private void deleteApiUrl(String apiUrl, int position) {
        Log.d(TAG, "删除API URL: " + apiUrl);
        
        // 显示确认对话框
        android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(requireContext());
        builder.setTitle("删除API地址")
               .setMessage("确定要删除API地址 \"" + apiUrl + "\" 吗？")
               .setPositiveButton("删除", (dialog, which) -> {
                   // 从配置中删除API URL
                   ConfigManager.removeApiUrl(requireContext(), apiUrl);
                   
                   // 重新加载API URL列表
                   loadApiUrlList();
                   
                   // 提示用户
                   Toast.makeText(requireContext(), "已删除API地址", Toast.LENGTH_SHORT).show();
               })
               .setNegativeButton("取消", null)
               .show();
    }
    
    // 显示添加API URL对话框
    private void showAddApiUrlDialog() {
        Log.d(TAG, "显示添加API URL对话框");
        
        // 创建对话框布局
        View dialogView = getLayoutInflater().inflate(R.layout.dialog_add_api_url, null);
        EditText editTextNewApiUrl = dialogView.findViewById(R.id.editTextNewApiUrl);
        EditText editTextNewApiKey = dialogView.findViewById(R.id.editTextNewApiKey);
        
        // 创建对话框
        android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(requireContext());
        builder.setTitle("添加新的API地址")
               .setView(dialogView)
               .setPositiveButton("添加", (dialog, which) -> {
                   // 获取输入的API URL和Key
                   String newApiUrl = editTextNewApiUrl.getText().toString().trim();
                   String newApiKey = editTextNewApiKey.getText().toString().trim();
                   
                   // 验证输入
                   if (newApiUrl.isEmpty()) {
                       Toast.makeText(requireContext(), "API地址不能为空", Toast.LENGTH_SHORT).show();
                       return;
                   }
                   
                   // 添加新的API URL和Key
                   ConfigManager.addApiUrl(requireContext(), newApiUrl, newApiKey);
                   
                   // 重新加载API URL列表
                   loadApiUrlList();
                   
                   // 选择新添加的API URL
                   setSpinnerSelection(spinnerApiUrl, newApiUrl);
                   
                   Toast.makeText(requireContext(), "已添加新的API地址", Toast.LENGTH_SHORT).show();
               })
               .setNegativeButton("取消", null);
        
        // 显示对话框
        android.app.AlertDialog dialog = builder.create();
        dialog.show();
    }
    
    // 加载知识库列表
    private void loadKnowledgeBases() {
        Log.d(TAG, "开始加载知识库列表");
        // 显示加载状态
        setupSpinner(spinnerKnowledgeBase, new String[]{"加载中..."});
        
        // 获取设置中的知识库路径
        String knowledgeBasePath = ConfigManager.getString(requireContext(), ConfigManager.KEY_KNOWLEDGE_BASE_PATH, ConfigManager.DEFAULT_KNOWLEDGE_BASE_PATH);
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
            // 添加一个额外的选项 "无"
            String[] knowledgeBases = new String[directories.length + 1];
            knowledgeBases[0] = "无"; // 第一个选项为"无"
            for (int i = 0; i < directories.length; i++) {
                knowledgeBases[i + 1] = directories[i].getName();
            }
            setupSpinner(spinnerKnowledgeBase, knowledgeBases);
            
            // 从配置文件加载上次选择的知识库
            loadLastSelectedKnowledgeBase();
            
            Log.d(TAG, "已加载 " + directories.length + " 个知识库");
        } else {
            setupSpinner(spinnerKnowledgeBase, new String[]{"无", "无可用知识库"});
            Log.d(TAG, "未找到可用知识库");
        }
    }
    
    // 加载上次选择的知识库
    private void loadLastSelectedKnowledgeBase() {
        try {
            // 使用 ConfigManager 获取上次选择的知识库
            String lastKnowledgeBase = ConfigManager.getString(requireContext(), 
                    ConfigManager.KEY_KNOWLEDGE_BASE, "");
            
            Log.d(TAG, "从 ConfigManager 加载上次选择的知识库: " + 
                    (lastKnowledgeBase.isEmpty() ? "[空]" : lastKnowledgeBase));
            
            if (!lastKnowledgeBase.isEmpty()) {
                setSpinnerSelection(spinnerKnowledgeBase, lastKnowledgeBase);
            }
        } catch (Exception e) {
            Log.e(TAG, "加载知识库选择失败: " + e.getMessage(), e);
            Toast.makeText(requireContext(), "加载知识库选择失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void handleSendStopClick() {
        if (!isSending) {
            // --- 开始发送 --- 
            String apiUrl = spinnerApiUrl.getSelectedItem().toString();
            String apiKey = editTextApiKey.getText().toString();
            String model = spinnerApiModel.getSelectedItem().toString();
            String knowledgeBase = spinnerKnowledgeBase.getSelectedItem().toString();
            String systemPrompt = editTextSystemPrompt.getText().toString();
            String userPrompt = editTextUserPrompt.getText().toString();
            
            Log.d(TAG, "用户点击发送按钮，准备发送请求");
            Log.d(TAG, "请求参数: API URL=" + apiUrl + ", 模型=" + model + ", 知识库=" + knowledgeBase);

            // 基本验证
            if (userPrompt.trim().isEmpty()) {
                Toast.makeText(requireContext(), "请输入用户提问", Toast.LENGTH_SHORT).show();
                return;
            }
            
            if (apiUrl.trim().isEmpty() || apiKey.trim().isEmpty() || 
                model.equals("加载中...") || model.equals("获取模型失败")) {
                Toast.makeText(requireContext(), "请确保API地址、API Key和模型都已正确设置", Toast.LENGTH_SHORT).show();
                return;
            }

            // 保存当前配置
            saveConfig();
            
            // 更新按钮状态
            buttonSendStop.setText("停止 ■");
            isSending = true;
            
            // 显示正在处理的消息，不显示调试信息
            //updateProgressOnUiThread("正在查询知识库...");

            // 执行RAG查询任务
            executeRagQuery(apiUrl, apiKey, model, knowledgeBase, systemPrompt, userPrompt);

        } else {
            // --- 停止发送 --- 
            if (isTaskRunning) {
                isTaskCancelled = true;
            }
            Toast.makeText(requireContext(), "已停止请求", Toast.LENGTH_SHORT).show();
            appendToResponse("\n请求已停止。");
            buttonSendStop.setText("发送 ▶");
            isSending = false;
        }
    }
    
    // 将日志写入文件
    private void logToFile(String message) {
        // 安全检查：确保Fragment已附加到Context
        if (!isAdded()) {
            Log.e(TAG, "无法写入日志：Fragment未附加到Context");
            return;
        }
        
        try {
            Context context = getContext();
            if (context == null) {
                Log.e(TAG, "无法写入日志：Context为空");
                return;
            }
            
            File logFile = new File(context.getFilesDir(), LOG_FILE);
            if (!logFile.exists()) {
                logFile.createNewFile();
            }
            // 使用追加模式写入日志
            BufferedWriter writer = new BufferedWriter(new FileWriter(logFile, true));
            String timestamp = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
            writer.append(timestamp + ": " + message + "\n");
            writer.close();
        } catch (IOException e) {
            Log.e(TAG, "写入日志文件失败", e);
        }
    }
    
    // 获取日志文件路径
    private String getLogFilePath() {
        return requireContext().getFilesDir() + "/" + LOG_FILE;
    }
    
    // 执行RAG查询任务
    private void executeRagQuery(String apiUrl, String apiKey, String model, String knowledgeBase, String systemPrompt, String userPrompt) {
        isTaskRunning = true;
        isTaskCancelled = false;
        
        // 初始化相关文档列表
        synchronized (this) {
            relevantDocuments = new ArrayList<>();
            similarityInfo = "";
        }
        
        // 记录开始时间
        final long startTime = System.currentTimeMillis();
        
        // 获取近似深度
        final int searchDepth = Integer.parseInt(editTextSearchDepth.getText().toString());
        
        // 更新UI，显示开始查询
        mainHandler.post(() -> {
            buttonSendStop.setText("停止 ■");
            isSending = true;
            isTaskRunning = true;
            isTaskCancelled = false;
            
            // 清空响应区域
            //updateProgressOnUiThread("正在查询知识库...");
        });
        
        // 在后台线程中执行查询
        executor.execute(() -> {
            try {
                // 记录查询信息到日志
                String logMessage = "执行RAG查询:\n" +
                        "API URL: " + apiUrl + "\n" +
                        "模型: " + model + "\n" +
                        "知识库: " + knowledgeBase + "\n" +
                        "近似深度: " + searchDepth + "\n" +
                        "系统提示词: " + systemPrompt + "\n" +
                        "用户提问: " + userPrompt;
                Log.d(TAG, logMessage);
                logToFile(logMessage);
                
                // 更新UI，显示查询日志
                mainHandler.post(() -> {
                    //updateProgressOnUiThread("开始查询知识库...");
                    //updateProgressOnUiThread("知识库: " + knowledgeBase);
                    //updateProgressOnUiThread("近似深度: " + searchDepth);
                    updateProgressOnUiThread("\n ##--- 调试信息 ---\n\n用户提问: " + userPrompt);
                });
                
                // 检查是否需要查询知识库
                if (!knowledgeBase.equals("无") && !knowledgeBase.equals("无可用知识库")) {
                    String kbInfo = "使用知识库进行查询: " + knowledgeBase;
                    Log.d(TAG, kbInfo);
                    logToFile(kbInfo);
                    //updateProgressOnUiThread(kbInfo);
                    
                    // 查询知识库获取相关内容 - 只调用queryKnowledgeBase，不使用返回值
                    queryKnowledgeBase(knowledgeBase, userPrompt);
                    
                    // 等待查询结果 - 从relevantDocuments成员变量中获取
                    int waitCount = 0;
                    List<String> relevantDocs = new ArrayList<>();
                    
                    while (waitCount < 300) { // 最多等待30秒
                        if (isTaskCancelled) {
                            String cancelMsg = "RAG查询被用户取消";
                            Log.d(TAG, cancelMsg);
                            logToFile(cancelMsg);
                            updateProgressOnUiThread(cancelMsg);
                            return;
                        }
                        
                        // 检查是否有查询结果
                        synchronized (this) {
                            if (relevantDocuments != null && !relevantDocuments.isEmpty()) {
                                relevantDocs = new ArrayList<>(relevantDocuments);
                                break;
                            }
                        }
                        
                        // 如果等待超过25秒，退出循环
                        if (waitCount >= 250) {
                            Log.w(TAG, "等待查询结果超时");
                            logToFile("等待查询结果超时");
                            break;
                        }
                        
                        // 等待100毫秒
                        try {
                            Thread.sleep(100);
                            waitCount++;
                        } catch (InterruptedException e) {
                            Log.e(TAG, "等待查询结果时被中断", e);
                        }
                    }
                    
                    // 检查查询结果
                    if (relevantDocs.isEmpty()) {
                        String warnMsg = "警告: 知识库查询未返回相关文档";
                        Log.w(TAG, warnMsg);
                        logToFile(warnMsg);
                        updateProgressOnUiThread(warnMsg);
                        
                        // 直接构建不包含知识库内容的提示词
                        String fullPrompt = buildPromptWithoutKnowledgeBase(systemPrompt, userPrompt);
                        
                        // 记录提示词信息
                        int promptLength = fullPrompt.length();
                        String promptInfo = "提示词长度: " + promptLength + " 字符";
                        Log.d(TAG, promptInfo);
                        logToFile(promptInfo);
                        updateProgressOnUiThread(promptInfo);
                        
                        // 如果提示词太长，记录警告
                        if (promptLength > 4000) {
                            String warnMsg2 = "警告: 提示词长度超过4000字符，可能被模型截断";
                            Log.w(TAG, warnMsg2);
                            logToFile(warnMsg2);
                            updateProgressOnUiThread(warnMsg2);
                        }
                        
                        // 计算查询耗时
                        long queryTime = System.currentTimeMillis() - startTime;
                        String timeMsg = "知识库查询耗时: " + queryTime + "ms";
                        Log.d(TAG, timeMsg);
                        logToFile(timeMsg);
                        updateProgressOnUiThread(timeMsg);
                        
                        // 调用大模型API获取回答
                        updateProgressOnUiThread("正在调用大模型API...");
                        callLLMApi(apiUrl, apiKey, model, fullPrompt);
                    } else {
                        // 获取相似度信息
                        String simInfo = "";
                        synchronized (this) {
                            simInfo = this.similarityInfo;
                        }
                        
                        // 显示相似度信息（无论是否为调试模式）
                        if (!TextUtils.isEmpty(simInfo)) {
                            updateProgressOnUiThread("相似度信息: " + simInfo);
                        }
                        
                        updateProgressOnUiThread("已找到 " + relevantDocs.size() + " 个相关内容...");
                        
                        // 构建包含知识库内容的提示词
                        //updateProgressOnUiThread("建提示词");
                        String fullPrompt = buildPromptWithKnowledgeBase(systemPrompt, userPrompt, relevantDocs);
                        
                        // 记录提示词信息 - 只显示长度，不显示内容
                        int promptLength = fullPrompt.length();
                        String promptInfo = "建提示词长度: " + promptLength + " 字符";
                        Log.d(TAG, promptInfo);
                        logToFile(promptInfo);
                        updateProgressOnUiThread(promptInfo);
                        
                        // 如果提示词太长，记录警告
                        if (promptLength > 4000) {
                            String warnMsg = "警告: 提示词长度超过4000字符，可能被模型截断";
                            Log.w(TAG, warnMsg);
                            logToFile(warnMsg);
                            updateProgressOnUiThread(warnMsg);
                        }
                        
                        // 计算查询耗时
                        long queryTime = System.currentTimeMillis() - startTime;
                        String timeMsg = "知识库查询耗时: " + queryTime + "ms";
                        Log.d(TAG, timeMsg);
                        logToFile(timeMsg);
                        updateProgressOnUiThread(timeMsg);
                        
                        // 调用大模型API获取回答
                        updateProgressOnUiThread("正在调用大模型API...");
                        callLLMApi(apiUrl, apiKey, model, fullPrompt);
                    }
                } else {
                    // 不使用知识库，直接调用大模型API
                    String directMsg = "不使用知识库，直接调用大模型";
                    Log.d(TAG, directMsg);
                    logToFile(directMsg);
                    updateProgressOnUiThread(directMsg);
                    updateProgressOnUiThread("正在生成回答...");
                    
                    // 构建不包含知识库内容的提示词
                    String fullPrompt = buildPromptWithoutKnowledgeBase(systemPrompt, userPrompt);
                    
                    // 记录提示词信息 - 只显示长度，不显示内容
                    int promptLength = fullPrompt.length();
                    String promptInfo = "提示词长度: " + promptLength + " 字符";
                    Log.d(TAG, promptInfo);
                    logToFile(promptInfo);
                    updateProgressOnUiThread(promptInfo);
                    
                    // 如果提示词太长，记录警告
                    if (promptLength > 4000) {
                        String warnMsg = "警告: 提示词长度超过4000字符，可能被模型截断";
                        Log.w(TAG, warnMsg);
                        logToFile(warnMsg);
                        updateProgressOnUiThread(warnMsg);
                    }
                    
                    // 调用大模型API获取回答
                    updateProgressOnUiThread("正在调用大模型API...");
                    callLLMApi(apiUrl, apiKey, model, fullPrompt);
                }
            } catch (Exception e) {
                String errorMsg = "执行RAG查询任务失败: " + e.getMessage();
                Log.e(TAG, errorMsg, e);
                logToFile(errorMsg);
                logToFile("异常堆栈: " + android.util.Log.getStackTraceString(e));
                logToFile("========== 结束RAG查询(异常) ==========\n");
                
                updateResultOnUiThread("查询失败: " + e.getMessage());
                mainHandler.post(() -> {
                    buttonSendStop.setText("发送 ▶");
                    isSending = false;
                });
            } finally {
                isTaskRunning = false;
                
                // 如果任务被取消，记录日志
                if (isTaskCancelled) {
                    logToFile("RAG查询被用户取消");
                    logToFile("========== 结束RAG查询(已取消) ==========\n");
                }
            }
        });
    }
    
    // 查询知识库获取相关内容
    private List<String> queryKnowledgeBase(String knowledgeBase, String query) {
        List<String> relevantDocs = new ArrayList<>();

        try {
            Log.d(TAG, "开始查询知识库: " + knowledgeBase + ", 查询关键词: " + query);
            logToFile("开始查询知识库: " + knowledgeBase + ", 查询关键词: " + query);
            //updateProgressOnUiThread("开始查询知识库: " + knowledgeBase);

            // 获取近似深度（从界面输入框获取）
            int searchDepth = Integer.parseInt(editTextSearchDepth.getText().toString());
            Log.d(TAG, "使用界面设置的近似深度: " + searchDepth);
            logToFile("使用界面设置的近似深度: " + searchDepth);
            
            // 检查知识库名称是否有效
            if (knowledgeBase == null || knowledgeBase.trim().isEmpty()) {
                String errorMsg = "错误: 知识库名称为空";
                Log.e(TAG, errorMsg);
                logToFile(errorMsg);
                updateProgressOnUiThread(errorMsg);
                return relevantDocs;
            }

            // 检查上下文是否可用
            if (!isAdded()) {
                String errorMsg = "错误: Fragment未附加到Activity";
                Log.e(TAG, errorMsg);
                logToFile(errorMsg);
                return relevantDocs;
            }

            // 获取知识库目录 - 使用配置中的知识库路径
            String knowledgeBasePath = ConfigManager.getString(requireContext(), ConfigManager.KEY_KNOWLEDGE_BASE_PATH, ConfigManager.DEFAULT_KNOWLEDGE_BASE_PATH);
            Log.d(TAG, "从设置中获取知识库路径: " + knowledgeBasePath);

            // 获取知识库目录
            File knowledgeBaseDir = new File(knowledgeBasePath, knowledgeBase);
            String pathInfo = "知识库目录路径: " + knowledgeBaseDir.getAbsolutePath();
            Log.d(TAG, pathInfo);
            logToFile(pathInfo);
            updateProgressOnUiThread(pathInfo);

            // 检查知识库目录是否存在
            if (!knowledgeBaseDir.exists()) {
                String errorMsg = "错误: 知识库目录不存在: " + knowledgeBaseDir.getAbsolutePath();
                Log.e(TAG, errorMsg);
                logToFile(errorMsg);
                updateProgressOnUiThread(errorMsg);
                return relevantDocs;
            }

            // 检查SQLite数据库文件
            File vectorDbFile = new File(knowledgeBaseDir, "vectorstore.db");
            if (!vectorDbFile.exists()) {
                String errorMsg = "错误: SQLite向量数据库文件不存在: " + vectorDbFile.getAbsolutePath();
                Log.e(TAG, errorMsg);
                logToFile(errorMsg);
                updateProgressOnUiThread(errorMsg);
                return relevantDocs;
            } else {
                String fileInfo = "SQLite数据库文件存在: " + vectorDbFile.getAbsolutePath() + 
                                 ", 大小: " + (vectorDbFile.length() / 1024) + "KB, " +
                                 "可读: " + vectorDbFile.canRead();
                Log.i(TAG, fileInfo);
                logToFile(fileInfo);
            }

            // 检查元数据文件
            File metadataFile = new File(knowledgeBaseDir, "metadata.json");
            if (!metadataFile.exists()) {
                String errorMsg = "错误: 元数据文件不存在: " + metadataFile.getAbsolutePath();
                Log.e(TAG, errorMsg);
                logToFile(errorMsg);
                updateProgressOnUiThread(errorMsg);
                return relevantDocs;
            } else {
                String fileInfo = "元数据文件存在: " + metadataFile.getAbsolutePath() + 
                                 ", 大小: " + (metadataFile.length() / 1024) + "KB, " +
                                 "可读: " + metadataFile.canRead();
                Log.i(TAG, fileInfo);
                logToFile(fileInfo);
                
                // 读取元数据文件内容并记录 - 使用单独线程避免阻塞
                try {
                    // 在后台线程中读取文件
                    ExecutorService readExecutor = Executors.newSingleThreadExecutor();
                    Future<String> metadataContentFuture = readExecutor.submit(() -> {
                        try {
                            StringBuilder content = new StringBuilder();
                            try (BufferedReader reader = new BufferedReader(new FileReader(metadataFile))) {
                                String line;
                                while ((line = reader.readLine()) != null) {
                                    content.append(line);
                                }
                            }
                            return "元数据文件内容: " + content.toString();
                        } catch (Exception e) {
                            return "读取元数据文件失败: " + e.getMessage();
                        }
                    });
                    
                    String metadataContent;
                    try {
                        metadataContent = metadataContentFuture.get(30, TimeUnit.SECONDS);
                        Log.i(TAG, metadataContent);
                        logToFile(metadataContent);
                    } catch (Exception e) {
                        String readError = "读取元数据文件超时或失败: " + e.getMessage();
                        Log.e(TAG, readError);
                        logToFile(readError);
                        updateProgressOnUiThread(readError);
                        return relevantDocs;
                    } finally {
                        readExecutor.shutdownNow();
                    }
                } catch (Exception e) {
                    String readError = "启动读取元数据文件线程失败: " + e.getMessage();
                    Log.e(TAG, readError);
                    logToFile(readError);
                    updateProgressOnUiThread(readError);
                    return relevantDocs;
                }
            }
            
            // 查询向量数据库
            try {
                // 创建SQLite向量数据库处理器
                Log.i(TAG, "开始创建SQLite向量数据库处理器，知识库目录: " + knowledgeBaseDir.getAbsolutePath());
                logToFile("开始创建SQLite向量数据库处理器，知识库目录: " + knowledgeBaseDir.getAbsolutePath());
                
                SQLiteVectorDatabaseHandler vectorDb = new SQLiteVectorDatabaseHandler(knowledgeBaseDir, "unknown");
                //updateProgressOnUiThread("正在加载SQLite向量数据库...");

                // 加载向量数据库
                Log.i(TAG, "开始加载SQLite向量数据库...");
                logToFile("开始加载SQLite向量数据库...");
                
                if (!vectorDb.loadDatabase()) {
                    String errorMsg = "错误: 加载SQLite向量数据库失败";
                    Log.e(TAG, errorMsg);
                    logToFile(errorMsg);
                    updateProgressOnUiThread(errorMsg);
                    return relevantDocs;
                }

                // 获取数据库统计信息
                int totalChunks = vectorDb.getChunkCount();
                String dbInfo = "SQLite向量数据库加载成功，共包含 " + totalChunks + " 个文本块";
                Log.d(TAG, dbInfo);
                logToFile(dbInfo);
                updateProgressOnUiThread(dbInfo);

                // 获取嵌入模型
                String embModelName = vectorDb.getMetadata().getEmbeddingModel();
                String embeddingModelPath = ConfigManager.getEmbeddingModelPath(requireContext());
                String foundModelPath = null;
                
                // 检查元数据中是否有modeldir配置
                String modeldir = vectorDb.getMetadata().getModeldir();
                if (modeldir != null && !modeldir.isEmpty()) {
                    // 使用modeldir指定的目录
                    File modeldirFile = new File(embeddingModelPath, modeldir);
                    if (modeldirFile.exists() && modeldirFile.isDirectory()) {
                        // 在modeldir中查找模型文件
                        File[] files = modeldirFile.listFiles();
                        if (files != null) {
                            for (File file : files) {
                                if (file.isFile() && (file.getName().endsWith(".pt") || 
                                                     file.getName().endsWith(".pth") || 
                                                     file.getName().endsWith(".onnx"))) {
                                    foundModelPath = file.getAbsolutePath();
                                    Log.d(TAG, "使用modeldir中的模型: " + foundModelPath);
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // 如果modeldir中没有找到模型，尝试直接使用embeddingModel
                if (foundModelPath == null) {
                    foundModelPath = new File(embeddingModelPath, embModelName).getAbsolutePath();
                }

                // 检查模型文件是否存在
                File modelFile = new File(foundModelPath);
                if (!modelFile.exists()) {
                    Log.d(TAG, "模型文件不存在: " + foundModelPath + "，将尝试在嵌入模型目录中查找");
                    
                    // 尝试在嵌入模型目录中查找模型文件
                    File embeddingModelDir = new File(embeddingModelPath);
                    if (embeddingModelDir.exists() && embeddingModelDir.isDirectory()) {
                        // 获取所有子目录，用于模型选择
                        List<String> availableModels = new ArrayList<>();
                        File[] directories = embeddingModelDir.listFiles(File::isDirectory);
                        if (directories != null) {
                            for (File dir : directories) {
                                // 检查目录中是否有模型文件
                                File[] modelFiles = dir.listFiles(file -> 
                                    file.isFile() && (file.getName().endsWith(".pt") || 
                                                     file.getName().endsWith(".pth") || 
                                                     file.getName().endsWith(".onnx")));
                                if (modelFiles != null && modelFiles.length > 0) {
                                    availableModels.add(dir.getName());
                                }
                            }
                        }
                        
                        // 也检查根目录中的模型文件
                        File[] rootModelFiles = embeddingModelDir.listFiles(file -> 
                            file.isFile() && (file.getName().endsWith(".pt") || 
                                             file.getName().endsWith(".pth") || 
                                             file.getName().endsWith(".onnx")));
                        if (rootModelFiles != null && rootModelFiles.length > 0) {
                            availableModels.add("根目录");
                        }
                        
                        if (!availableModels.isEmpty()) {
                            // 弹出模型选择对话框
                            selectModelAndContinueQuery(embModelName, availableModels, knowledgeBase, embeddingModelPath, vectorDb);
                            return relevantDocs; // 提前返回，等待用户选择模型
                        } else {
                            Log.e(TAG, "在嵌入模型目录中未找到可用的模型文件");
                            updateProgressOnUiThread("错误: 在嵌入模型目录中未找到可用的模型文件");
                            return relevantDocs; // 提前返回，因为没有可用的模型
                        }
                    }
                }
                
                // 使用工具类检查并加载词嵌入模型
                EmbeddingModelUtils.checkAndLoadEmbeddingModel(
                    requireContext(),
                    vectorDb,
                    modelFoundPath -> {
                        if (modelFoundPath == null) {
                            // 模型不存在或需要用户选择，已由工具类处理
                            return;
                        }
                        
                        // 模型存在，继续处理
                        String modelInfo = "使用嵌入模型: " + embModelName + ", 路径: " + modelFoundPath;
                        Log.d(TAG, modelInfo);
                        logToFile(modelInfo);
                        updateProgressOnUiThread("正在使用嵌入模型: " + embModelName);
                        
                        // 加载嵌入模型
                        loadModelAndProcessQuery(modelFoundPath, query, vectorDb);
                    },
                    (selectedModel, selectedModelPath) -> {
                        // 用户选择了模型，继续处理
                        String modelInfo = "使用选定的嵌入模型: " + selectedModel + ", 路径: " + selectedModelPath;
                        Log.d(TAG, modelInfo);
                        logToFile(modelInfo);
                        updateProgressOnUiThread("正在使用选定的嵌入模型: " + selectedModel);
                        
                        // 加载嵌入模型
                        loadModelAndProcessQuery(selectedModelPath, query, vectorDb);
                    }
                );
                
                return relevantDocs;
            } catch (Exception e) {
                String errorMsg = "查询向量数据库时发生错误: " + e.getMessage();
                Log.e(TAG, errorMsg, e);
                if (isAdded()) {
                    logToFile(errorMsg);
                    updateProgressOnUiThread(errorMsg);
                }
                
                // 确保在异常情况下也释放模型资源
                EmbeddingModelManager modelManager = EmbeddingModelManager.getInstance(requireContext());
                modelManager.markModelNotInUse();
                Log.d(TAG, "异常情况下标记模型使用结束，允许自动卸载");
                
                return relevantDocs;
            }
        } catch (Exception e) {
            String errorMsg = "查询知识库时发生错误: " + e.getMessage();
            Log.e(TAG, errorMsg, e);
            if (isAdded()) {
                logToFile(errorMsg);
                updateProgressOnUiThread(errorMsg);
            }
            return relevantDocs;
        }
    }
    
    // 构建包含知识库内容的提示词
    private String buildPromptWithKnowledgeBase(String systemPrompt, String userPrompt, List<String> relevantDocs) {
        StringBuilder fullPrompt = new StringBuilder();
        
        Log.d(TAG, "构建包含知识库内容的提示词，找到 " + relevantDocs.size() + " 个相关文档");
        logToFile("构建包含知识库内容的提示词，找到 " + relevantDocs.size() + " 个相关文档");
        
        // 添加系统提示词
        if (!systemPrompt.isEmpty()) {
            fullPrompt.append(systemPrompt).append("\n\n");
            Log.d(TAG, "添加系统提示词，长度: " + systemPrompt.length());
        } else {
            Log.d(TAG, "系统提示词为空");
        }
        
        // 添加知识库内容
        if (!relevantDocs.isEmpty()) {
            fullPrompt.append("以下是与问题相关的信息：\n");
            
            for (int i = 0; i < relevantDocs.size(); i++) {
                String docContent = relevantDocs.get(i);
                if (docContent == null || docContent.trim().isEmpty()) {
                    Log.w(TAG, "文档 #" + (i + 1) + " 内容为空，已跳过");
                    logToFile("警告: 文档 #" + (i + 1) + " 内容为空，已跳过");
                    continue;
                }
                
                // 不再限制文本长度，显示完整内容
                fullPrompt.append("文档").append(i + 1).append(":\n").append(docContent).append("\n\n");
                Log.d(TAG, "添加文档 #" + (i + 1) + "，长度: " + docContent.length());
            }
        } else {
            fullPrompt.append("未找到与问题相关的信息。\n\n");
            Log.w(TAG, "未找到相关文档，提示模型无相关信息");
            logToFile("警告: 未找到相关文档，提示模型无相关信息");
        }
        
        // 添加用户问题
        fullPrompt.append("用户问题: ").append(userPrompt);
        
        // 记录最终提示词长度
        int promptLength = fullPrompt.length();
        Log.d(TAG, "最终提示词长度: " + promptLength + " 字符");
        logToFile("最终提示词长度: " + promptLength + " 字符");
        
        return fullPrompt.toString();
    }
    
    // 构建不包含知识库内容的提示词
    private String buildPromptWithoutKnowledgeBase(String systemPrompt, String userPrompt) {
        StringBuilder fullPrompt = new StringBuilder();
        
        // 添加系统提示词
        if (!systemPrompt.isEmpty()) {
            fullPrompt.append(systemPrompt).append("\n\n");
        }
        
        // 添加用户问题
        fullPrompt.append("用户问题: ").append(userPrompt);
        
        return fullPrompt.toString();
    }
    
    // 调用大模型API获取回答
    private void callLLMApi(String apiUrl, String apiKey, String model, String prompt) {
        try {
            Log.d(TAG, "开始调用大模型API: " + apiUrl);
            Log.d(TAG, "使用模型: " + model);
            Log.d(TAG, "提示词长度: " + prompt.length() + " 字符");
            
            // 记录请求信息到日志
            logToFile("========== 开始API请求 ==========");
            logToFile("API地址: " + apiUrl);
            logToFile("模型: " + model);
            logToFile("API密钥: " + (apiKey.isEmpty() ? "未提供" : "已提供，长度: " + apiKey.length()));
            logToFile("提示词长度: " + prompt.length() + " 字符");
            
            // 记录提示词的前200个字符和后200个字符，避免日志过大
            if (prompt.length() > 400) {
                logToFile("提示词开头: " + prompt.substring(0, 200) + "...");
                logToFile("提示词结尾: ..." + prompt.substring(prompt.length() - 200));
            } else {
                logToFile("提示词: " + prompt);
            }
            
            // 添加连接信息，但不清空之前的调试信息
            //appendToResponse("正在连接API服务器...");
            
            // 安全检查：确保Fragment已附加到Context
            if (!isAdded()) {
                String errorMsg = "错误: Fragment未附加到Context，无法调用API";
                Log.e(TAG, errorMsg);
                updateResultOnUiThread(errorMsg);
                return;
            }
            
            Context context = getContext();
            if (context == null) {
                String errorMsg = "错误: Context为空，无法调用API";
                Log.e(TAG, errorMsg);
                updateResultOnUiThread(errorMsg);
                return;
            }
            
            // 创建LlmApiAdapter实例
            com.example.starlocalrag.api.LlmApiAdapter apiAdapter = new com.example.starlocalrag.api.LlmApiAdapter(context);
            
            // 记录开始时间
            final long startTime = System.currentTimeMillis();
            
            // 调用API并处理流式响应
            apiAdapter.callLlmApi(apiUrl, apiKey, model, prompt, new com.example.starlocalrag.api.LlmApiAdapter.ApiCallback() {
                // 在onSuccess方法中，进行一次完整的Markdown渲染
                @Override
                public void onSuccess(String response) {
                    // 处理完整响应
                    Log.d(TAG, "API调用成功，耗时: " + (System.currentTimeMillis() - startTime) + "ms");
                    Log.d(TAG, "响应长度: " + response.length() + " 字符");
                    logToFile("API调用成功，耗时: " + (System.currentTimeMillis() - startTime) + "ms");
                    logToFile("响应长度: " + response.length() + " 字符");
                    
                    // 记录响应的前200个字符，避免日志过大
                    if (response.length() > 200) {
                        logToFile("响应开头: " + response.substring(0, 200) + "...");
                    } else {
                        logToFile("响应: " + response);
                    }
                    logToFile("========== 结束API请求 ==========\n");
                    
                    // 在UI线程中进行最终的Markdown渲染
                    mainHandler.post(() -> {
                        try {
                            TextView textViewResponse = getView().findViewById(R.id.textViewResponse);
                            if (textViewResponse != null) {
                                // 获取当前显示的内容
                                String currentText = textViewResponse.getText().toString();
                                
                                // 检查并修复代码块
                                if (hasIncompleteCodeBlock(currentText)) {
                                    currentText = fixCodeBlocks(currentText);
                                }
                                
                                // 使用Markwon进行最终渲染
                                markwon.setMarkdown(textViewResponse, currentText);
                                
                                // 确保文本可选择
                                textViewResponse.setTextIsSelectable(true);
                                
                                // 确保链接可点击
                                textViewResponse.setMovementMethod(LinkMovementMethod.getInstance());
                                
                                Log.d(TAG, "最终Markdown渲染完成");
                            }
                            
                            // 恢复按钮状态
                            buttonSendStop.setText("发送 ▶");
                            isSending = false;
                        } catch (Exception e) {
                            Log.e(TAG, "最终Markdown渲染失败", e);
                            // 恢复按钮状态
                            buttonSendStop.setText("发送 ▶");
                            isSending = false;
                        }
                    });
                }
                
                // 用于累积流式响应的StringBuilder
                private final StringBuilder responseBuilder = new StringBuilder();
                // 记录是否已添加模型回答标题
                private final boolean[] modelTitleAdded = {false};
                // 上次显示的响应内容
                private final String[] lastDisplayedResponse = {""};
                // 字符变化阈值，小于这个值的变化不触发UI更新
                private static final int MIN_CHAR_CHANGE = 5;
                // 上次更新UI的时间
                private long lastUpdateTime = System.currentTimeMillis();
                // 更新间隔时间（毫秒）
                private static final long UPDATE_INTERVAL = 100;

                // 在onStreamingData方法中，使用简单的setText方法
                @Override
                public void onStreamingData(final String chunk) {
                    if (getActivity() == null) return;
                    
                    // 记录收到的数据块
                    Log.d(TAG, "收到数据块: [" + chunk + "]");
                    
                    // 累积响应内容
                    responseBuilder.append(chunk);
                    final String fullContent = responseBuilder.toString();
                    
                    // 在UI线程中更新内容，使用纯文本方式
                    getActivity().runOnUiThread(() -> {
                        try {
                            // 获取文本视图和滚动视图
                            TextView textViewResponse = getView().findViewById(R.id.textViewResponse);
                            ScrollView scrollView = getView().findViewById(R.id.scrollViewResponse);
                            if (textViewResponse == null || scrollView == null) return;
                            
                            // 检查当前滚动位置
                            boolean wasAtBottom = isScrolledToBottom(scrollView);
                            
                            // 准备要显示的完整内容
                            String displayContent;
                            long currentTime = System.currentTimeMillis();
                            
                            // 如果是第一次收到数据，添加模型回答标题
                            if (!modelTitleAdded[0]) {
                                modelTitleAdded[0] = true;
                                String currentText = textViewResponse.getText().toString();
                                displayContent = currentText.isEmpty() 
                                    ? "\n\n---\n\n## 模型回答\n\n" + fullContent 
                                    : currentText + "\n\n---\n\n## 模型回答\n\n" + fullContent;
                            } else {
                                // 检查内容变化是否足够大或时间间隔足够长
                                int charDiff = fullContent.length() - lastDisplayedResponse[0].length();
                                long timeDiff = currentTime - lastUpdateTime;
                                
                                // 如果变化不够大且时间间隔不够长，不更新UI
                                if (charDiff < MIN_CHAR_CHANGE && timeDiff < UPDATE_INTERVAL) {
                                    return;
                                }
                                
                                // 获取当前内容，并追加新内容
                                String currentText = textViewResponse.getText().toString();
                                
                                // 找到最后一次显示内容的位置，并替换为新的完整内容
                                int lastResponseIndex = currentText.lastIndexOf(lastDisplayedResponse[0]);
                                if (lastResponseIndex >= 0) {
                                    displayContent = currentText.substring(0, lastResponseIndex) + fullContent;
                                } else {
                                    // 如果找不到上次内容，则直接追加新内容
                                    String incrementalContent = fullContent.substring(lastDisplayedResponse[0].length());
                                    displayContent = currentText + incrementalContent;
                                }
                            }
                            
                            // 更新最后显示的内容和时间
                            lastDisplayedResponse[0] = fullContent;
                            lastUpdateTime = currentTime;
                            
                            // 使用纯文本方式更新内容
                            textViewResponse.setText(displayContent);
                            
                            // 如果之前在底部，则滚动到底部
                            if (wasAtBottom) {
                                scrollToBottom(scrollView);
                            }
                        } catch (Exception e) {
                            Log.e(TAG, "更新流式响应UI失败", e);
                        }
                    });
                }
                




                // 这些变量已不再使用，但保留以便其他方法可能引用
                

                
                
                /**
                 * 记录内容中的Markdown标记
                 * @param content 要检查的内容
                 */
                private void logMarkdownMarkers(String content) {
                    if (content == null || content.isEmpty()) return;
                    
                    // 检查常见的Markdown标记
                    if (content.contains("```")) {
                        Log.d(TAG, "检测到代码块标记: ``` 在内容中");
                    }
                    if (content.contains("`")) {
                        Log.d(TAG, "检测到行内代码标记: ` 在内容中");
                    }
                    if (content.contains("**")) {
                        Log.d(TAG, "检测到粗体标记: ** 在内容中");
                    }
                    if (content.contains("#")) {
                        Log.d(TAG, "检测到标题标记: # 在内容中");
                    }
                }
                
                /**
                 * 检查内容中是否有未完成的代码块
                 * @param content 要检查的内容
                 * @return 如果有未完成的代码块返回true，否则返回false
                 */
                private boolean hasIncompleteCodeBlock(String content) {
                    if (content == null || content.isEmpty()) return false;
                    
                    // 计算代码块标记的数量
                    int count = 0;
                    int index = -1;
                    
                    // 使用更精确的方法检测代码块标记
                    while ((index = content.indexOf("```", index + 1)) != -1) {
                        // 检查是否是真正的代码块开始/结束标记，而不是嵌套在其他代码块中的文本
                        boolean isRealCodeBlockMarker = true;
                        
                        // 检查这个标记是否在行首或前面是换行符
                        if (index > 0) {
                            char prevChar = content.charAt(index - 1);
                            // 如果前一个字符不是换行符或空格，可能不是真正的代码块标记
                            if (prevChar != '\n' && prevChar != ' ' && prevChar != '\t') {
                                // 进一步检查，如果前面有换行符，则可能是真正的代码块标记
                                int prevNewlineIndex = content.lastIndexOf('\n', index - 1);
                                if (prevNewlineIndex == -1 || index - prevNewlineIndex > 4) { // 允许有少量缩进
                                    isRealCodeBlockMarker = false;
                                }
                            }
                        }
                        
                        if (isRealCodeBlockMarker) {
                            count++;
                        }
                    }
                    
                    // 如果代码块标记数量为奇数，说明有未完成的代码块
                    return count % 2 != 0;
                }
                
                /**
                 * 计算字符串中指定模式出现的次数
                 * @param content 要检查的内容
                 * @param pattern 要查找的模式
                 * @return 模式出现的次数
                 */
                private int countOccurrences(String content, String pattern) {
                    if (content == null || content.isEmpty() || pattern == null || pattern.isEmpty()) {
                        return 0;
                    }
                    
                    int count = 0;
                    int index = 0;
                    while ((index = content.indexOf(pattern, index)) != -1) {
                        count++;
                        index += pattern.length();
                    }
                    
                    return count;
                }
                
                /**
                 * 修复内容中的代码块标记
                 * @param content 要修复的内容
                 * @return 修复后的内容
                 */
                private String fixCodeBlocks(String content) {
                    if (content == null || content.isEmpty()) return content;
                    
                    // 记录原始内容长度
                    int originalLength = content.length();
                    
                    // 检查并修复代码块标记
                    StringBuilder sb = new StringBuilder(content);
                    
                    // 计算代码块标记的数量和位置
                    List<Integer> positions = new ArrayList<>();
                    int index = -1;
                    while ((index = content.indexOf("```", index + 1)) != -1) {
                        positions.add(index);
                    }
                    
                    // 如果代码块标记数量为奇数，添加一个结束标记
                    if (positions.size() % 2 != 0) {
                        Log.d(TAG, "检测到未完成的代码块，添加结束标记");
                        sb.append("\n```");
                    }
                    
                    // 检查行内代码标记的数量
                    int inlineCount = 0;
                    index = -1;
                    while ((index = content.indexOf("`", index + 1)) != -1) {
                        // 跳过代码块标记
                        boolean isCodeBlockMarker = false;
                        for (int pos : positions) {
                            if (Math.abs(index - pos) < 3) { // 允许小误差
                                isCodeBlockMarker = true;
                                break;
                            }
                        }
                        if (!isCodeBlockMarker) {
                            inlineCount++;
                        }
                    }
                    
                    // 如果行内代码标记数量为奇数，添加一个结束标记
                    if (inlineCount % 2 != 0) {
                        Log.d(TAG, "检测到未完成的行内代码标记，添加结束标记");
                        sb.append("`");
                    }
                    
                    String result = sb.toString();
                    if (result.length() > originalLength) {
                        Log.d(TAG, "内容已修复，原始长度: " + originalLength + ", 新长度: " + result.length());
                    }
                    
                    return result;
                }
                
                /**
                 * 检查滚动视图是否已经滚动到底部
                 * @param scrollView 要检查的滚动视图
                 * @return 如果滚动到底部返回true，否则返回false
                 */
                private boolean isScrolledToBottom(ScrollView scrollView) {
                    if (scrollView == null) return false;
                    int scrollY = scrollView.getScrollY();
                    int height = scrollView.getHeight();
                    int scrollViewBottom = scrollY + height;
                    int contentHeight = scrollView.getChildAt(0).getHeight();
                    // 允许20像素的误差，以便更可靠地检测底部
                    return (scrollViewBottom >= contentHeight - 20);
                }
                
                /**
                 * 将滚动视图滚动到底部
                 * @param scrollView 要滚动的视图
                 */
                private void scrollToBottom(ScrollView scrollView) {
                    if (scrollView == null) return;
                    scrollView.post(() -> {
                        scrollView.fullScroll(ScrollView.FOCUS_DOWN);
                    });
                }
                
                @Override
                public void onError(String errorMessage) {
                    // 处理错误
                    Log.e(TAG, "API调用失败，耗时: " + (System.currentTimeMillis() - startTime) + "ms, 错误: " + errorMessage);
                    logToFile("API调用失败，耗时: " + (System.currentTimeMillis() - startTime) + "ms");
                    logToFile("错误: " + errorMessage);
                    logToFile("========== 结束API请求 ==========\n");
                    
                    // 显示错误信息
                    updateResultOnUiThread("调用API失败: " + errorMessage);
                    
                    // 恢复按钮状态
                    mainHandler.post(() -> {
                        buttonSendStop.setText("发送 ▶");
                        isSending = false;
                    });
                    
                    // 记录错误到日志
                    logToFile("API错误: " + errorMessage);
                }
            });
            
        } catch (Exception e) {
            Log.e(TAG, "调用大模型API失败", e);
            updateResultOnUiThread("调用API失败: " + e.getMessage());
            mainHandler.post(() -> {
                buttonSendStop.setText("发送 ▶");
                isSending = false;
            });
        }
    }
    
    // 在UI线程上更新进度信息
    private void updateProgressOnUiThread(String progress) {
        getActivity().runOnUiThread(() -> {
            appendToResponse(progress);
        });
    }
    
    // 完全重写的追加内容方法，解决滚动和Markdown渲染问题
    private void appendToResponse(String text) {
        if (getActivity() == null || getView() == null) return;
        
        getActivity().runOnUiThread(() -> {
            try {
                // 获取文本视图和滚动视图
                TextView textViewResponse = getView().findViewById(R.id.textViewResponse);
                ScrollView scrollView = getView().findViewById(R.id.scrollViewResponse);
                if (textViewResponse == null || scrollView == null) return;
                
                // 获取当前滚动状态
                int scrollY = scrollView.getScrollY();
                int scrollViewHeight = scrollView.getHeight();
                int contentHeight = scrollView.getChildAt(0).getHeight();
                
                // 判断是否在底部或接近底部（允许30px的误差）
                boolean wasAtBottom = (scrollY + scrollViewHeight >= contentHeight - 30);
                
                // 保存当前文本
                CharSequence currentText = textViewResponse.getText();
                
                // 准备新文本
                String newText;
                if (currentText.length() == 0) {
                    newText = text;
                } else {
                    newText = currentText + "\n" + text;
                }
                
                // 添加调试日志，查看文本内容和Markdown渲染过程
                //Log.d(TAG, "DEBUG: 要渲染的文本内容: " + newText);
                //Log.d(TAG, "DEBUG: Markwon实例是否为空: " + (markwon == null ? "是" : "否"));
                
                try {
                    // 尝试使用不同的方式渲染Markdown
                    // 方式1: 直接设置文本，然后使用setMarkdown
                    textViewResponse.setText(newText);
                    markwon.setMarkdown(textViewResponse, newText);
                    
                    // 调试日志
                    //Log.d(TAG, "DEBUG: 已尝试使用setMarkdown渲染");
                    
                    // 检查TextView的属性
                    //Log.d(TAG, "DEBUG: TextView的文本选择状态: " + textViewResponse.isTextSelectable());
                    //Log.d(TAG, "DEBUG: TextView的MovementMethod: " + textViewResponse.getMovementMethod());
                } catch (Exception e) {
                    Log.e(TAG, "DEBUG: Markdown渲染失败", e);
                }
                
                // 不再使用自动滚动，由用户自行控制滚动位置
            } catch (Exception e) {
                Log.e(TAG, "追加内容失败", e);
            }
        });
    }
    
    // 在UI线程上更新结果（替换全部内容）
    private void updateResultOnUiThread(String result) {
        mainHandler.post(() -> {
            try {
                // 获取结果文本视图
                TextView textViewResult = getView().findViewById(R.id.textViewResponse);
                if (textViewResult == null) return;
                
                // 获取滚动视图
                ScrollView scrollView = getView().findViewById(R.id.scrollViewResponse);
                if (scrollView == null) return;
                
                // 添加调试日志，查看文本内容和Markdown渲染过程
                //Log.d(TAG, "DEBUG-updateResult: 要渲染的文本内容: " + result);
                //Log.d(TAG, "DEBUG-updateResult: Markwon实例是否为空: " + (markwon == null ? "是" : "否"));
                
                try {
                    // 尝试使用不同的方式渲染Markdown
                    // 先设置纯文本，再尝试渲染
                    textViewResult.setText(result);
                    
                    // 优化的Markdown渲染逻辑
                    // 始终进行Markdown渲染，确保格式正确显示
                    try {
                        // 使用完整的Markdown渲染
                        Spanned spanned = markwon.toMarkdown(result);
                        markwon.setParsedMarkdown(textViewResult, spanned);
                        //Log.d(TAG, "DEBUG-updateResult: 使用完整Markdown渲染");
                        
                        // 确保链接可点击
                        if (textViewResult.getMovementMethod() == null) {
                            textViewResult.setMovementMethod(LinkMovementMethod.getInstance());
                        }
                    } catch (Exception e) {
                        // 如果渲染失败，回退到简单文本设置
                        textViewResult.setText(result);
                        //Log.e(TAG, "DEBUG-updateResult: Markdown渲染失败，回退到纯文本", e);
                    }
                    
                    // 检查TextView的属性
                    //Log.d(TAG, "DEBUG-updateResult: TextView的文本选择状态: " + textViewResult.isTextSelectable());
                    //Log.d(TAG, "DEBUG-updateResult: TextView的MovementMethod: " + textViewResult.getMovementMethod());
                } catch (Exception e) {
                    Log.e(TAG, "DEBUG-updateResult: Markdown渲染失败", e);
                    // 如果高级API失败，尝试使用基本方法
                    markwon.setMarkdown(textViewResult, result);
                }
                
                // 使用多级延迟确保滚动到底部
                scrollView.post(() -> {
                    scrollView.fullScroll(View.FOCUS_DOWN);
                    
                    scrollView.postDelayed(() -> {
                        scrollView.fullScroll(View.FOCUS_DOWN);
                    }, 100);
                    
                    scrollView.postDelayed(() -> {
                        scrollView.fullScroll(View.FOCUS_DOWN);
                    }, 300);
                });
            } catch (Exception e) {
                Log.e(TAG, "更新结果失败", e);
            }
        });
    }
    
    // 获取模型列表
    
    // 获取模型列表
    private void fetchModelsForApi() {
        String apiUrl = spinnerApiUrl.getSelectedItem().toString();
        String apiKey = editTextApiKey.getText().toString();
        
        if (apiUrl.isEmpty() || apiKey.isEmpty()) {
            Toast.makeText(requireContext(), "请先设置API地址和API Key", Toast.LENGTH_SHORT).show();
            setupSpinner(spinnerApiModel, new String[]{"获取模型失败"});
            return;
        }
        
        // 显示加载状态
        setupSpinner(spinnerApiModel, new String[]{"加载中..."});
        
        // 构建请求URL，根据不同API调整
        String modelsUrl = apiUrl;
        if (!modelsUrl.endsWith("/")) {
            modelsUrl += "/";
        }
        modelsUrl += "models";
        
        // 创建请求头
        Map<String, String> headers = new HashMap<>();
        headers.put("Authorization", "Bearer " + apiKey);
        
        // 使用Volley发送请求
        JsonObjectRequest request = new JsonObjectRequest(Request.Method.GET, modelsUrl, null,
            response -> {
                try {
                    // 解析响应，提取模型列表
                    JSONArray modelsArray = response.getJSONArray("data");
                    List<String> modelsList = new ArrayList<>();
                    
                    for (int i = 0; i < modelsArray.length(); i++) {
                        JSONObject modelObj = modelsArray.getJSONObject(i);
                        String modelId = modelObj.getString("id");
                        modelsList.add(modelId);
                    }
                    
                    // 更新UI
                    if (modelsList.isEmpty()) {
                        setupSpinner(spinnerApiModel, new String[]{"无可用模型"});
                    } else {
                        setupSpinner(spinnerApiModel, modelsList.toArray(new String[0]));
                    }
                    
                    Log.d(TAG, "成功获取模型列表: " + modelsList.size() + "个模型");
                } catch (JSONException e) {
                    Log.e(TAG, "解析模型列表失败", e);
                    setupSpinner(spinnerApiModel, new String[]{"获取模型失败"});
                    Toast.makeText(requireContext(), "解析模型列表失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                }
            },
            error -> {
                Log.e(TAG, "获取模型列表失败", error);
                setupSpinner(spinnerApiModel, new String[]{"获取模型失败"});
                Toast.makeText(requireContext(), "获取模型列表失败: " + error.getMessage(), Toast.LENGTH_SHORT).show();
            }
        ) {
            @Override
            public Map<String, String> getHeaders() {
                return headers;
            }
        };
        
        // 添加请求到队列
        Volley.newRequestQueue(requireContext()).add(request);
    }
    
    // 处理新对话按钮点击
    private void handleNewChatClick() {
        updateProgressOnUiThread("");
        editTextUserPrompt.setText("");
        
        // 清空回答框
        if (textViewResponse != null) {
            textViewResponse.setText("");
        }
        
        // 重置发送/停止按钮状态
        if (isSending) {
            buttonSendStop.setText("发送 ▶");
            isSending = false;
            if (isTaskRunning) {
                isTaskCancelled = true;
            }
        }
    }
    
    // 创建上下文菜单
    @Override
    public void onCreateContextMenu(ContextMenu menu, View v, ContextMenuInfo menuInfo) {
        super.onCreateContextMenu(menu, v, menuInfo);
    }
    
    // 处理上下文菜单项点击
    @Override
    public boolean onContextItemSelected(MenuItem item) {
        return super.onContextItemSelected(item);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        executor.shutdown();
    }
    
    // 加载模型并处理查询
    private void loadModelAndProcessQuery(String foundModelPath, String query, SQLiteVectorDatabaseHandler vectorDb) {
        try {
            // 更新进度
            updateProgressOnUiThread("正在加载嵌入模型...");
            
            // 初始化TokenizerManager
            TokenizerManager tokenizerManager = TokenizerManager.getInstance(requireContext());
            if (!tokenizerManager.isInitialized()) {
                //updateProgressOnUiThread("正在初始化全局分词器...");
                boolean initSuccess = tokenizerManager.initialize(foundModelPath);
                if (initSuccess) {
                    //updateProgressOnUiThread("全局分词器初始化成功，将使用统一分词策略");
                    // 启用一致性分词
                    tokenizerManager.setUseConsistentTokenization(true);
                    
                    // 设置调试模式
                    boolean debugMode = SettingsFragment.isDebugModeEnabled(requireContext());
                    if (debugMode) {
                        tokenizerManager.setDebugMode(true);
                        updateProgressOnUiThread("全局分词器已启用调试模式");
                    }
                } else {
                    updateProgressOnUiThread("全局分词器初始化失败，将使用模型自带分词器");
                }
            } else {
                updateProgressOnUiThread("使用已初始化的全局分词器");
                // 启用一致性分词
                tokenizerManager.setUseConsistentTokenization(true);
                
                // 设置调试模式
                boolean debugMode = SettingsFragment.isDebugModeEnabled(requireContext());
                if (debugMode) {
                    tokenizerManager.setDebugMode(true);
                }
            }
            
            // 创建模型处理器
            EmbeddingModelHandler embeddingHandler = new EmbeddingModelHandler(requireContext(), foundModelPath, false);
            
            // 获取模型的向量维度
            int embeddingDimension = embeddingHandler.getEmbeddingDimension();
            Log.d(TAG, "模型向量维度: " + embeddingDimension);
            updateProgressOnUiThread("模型向量维度: " + embeddingDimension);

            // 检查向量维度是否与知识库匹配
            int dbDimension = vectorDb.getMetadata().getEmbeddingDimension();
            Log.d(TAG, "知识库向量维度: " + dbDimension + ", 模型向量维度: " + embeddingDimension);
            updateProgressOnUiThread("知识库向量维度: " + dbDimension);

            if (dbDimension > 0 && dbDimension != embeddingDimension) {
                String warningMsg = "警告: 向量维度不匹配! 知识库维度: " + dbDimension + ", 模型维度: " + embeddingDimension;
                Log.w(TAG, warningMsg);
                updateProgressOnUiThread(warningMsg);
                updateProgressOnUiThread("这可能导致搜索失败，建议重新构建知识库或使用匹配的模型");
            }


            // 标记模型开始使用
            EmbeddingModelManager modelManager = EmbeddingModelManager.getInstance(requireContext());
            modelManager.markModelInUse();
            Log.d(TAG, "标记模型开始使用，防止自动卸载");
            //updateProgressOnUiThread("标记模型开始使用，防止自动卸载");
            
            try {
                // 生成查询向量
                updateProgressOnUiThread("正在生成查询向量...");
                
                // 获取用户查询
                String userQuery = editTextUserPrompt.getText().toString().trim();
                
                // 生成向量
                float[] queryVector = embeddingHandler.generateEmbedding(userQuery);
                
                // 记录向量调试信息
                String vectorDebugInfo = embeddingHandler.getVectorDebugInfo(userQuery, queryVector, System.currentTimeMillis());
                
                // 只在非调试模式下显示基本向量信息
                boolean isDebugMode = ConfigManager.getBoolean(requireContext(), ConfigManager.KEY_DEBUG_MODE, false);
                
                updateProgressOnUiThread(vectorDebugInfo);

                
                // 搜索相似文本块
                updateProgressOnUiThread("正在搜索相似文本块...");
                
                // 获取检索数量设置
                int retrievalCount = Integer.parseInt(editTextSearchDepth.getText().toString());
                
                // 搜索相似文本块
                List<SQLiteVectorDatabaseHandler.SearchResult> searchResults = vectorDb.searchSimilar(queryVector, retrievalCount);
                
                // 提取相关文档
                List<String> relevantDocs = new ArrayList<>();
                StringBuilder similarityInfoBuilder = new StringBuilder();
                
                for (int i = 0; i < searchResults.size(); i++) {
                    SQLiteVectorDatabaseHandler.SearchResult result = searchResults.get(i);
                    relevantDocs.add(result.text);
                    
                    // 记录详细信息到日志
                    String resultInfo = "相似度: " + result.similarity + ", 文本: " + result.text.substring(0, Math.min(50, result.text.length())) + "...";
                    Log.d(TAG, resultInfo);
                    logToFile(resultInfo);

                    // 添加到进度显示 - 只显示匹配序号和相似度值，不显示文本内容
                    similarityInfoBuilder.append("匹配").append(i + 1).append(": ").append(String.format("%.4f", result.similarity));
                    if (i < searchResults.size() - 1) {
                        similarityInfoBuilder.append(", ");
                    }
                    
                    // 记录详细信息到日志
                    String resultInfo2 = "相似度: " + result.similarity + ", 文本: " + result.text.substring(0, Math.min(50, result.text.length())) + "...";
                    Log.d(TAG, resultInfo2);
                    logToFile(resultInfo2);
                }

                // 保存相似度信息
                synchronized (this) {
                    this.similarityInfo = similarityInfoBuilder.toString();
                    this.relevantDocuments = relevantDocs;
                }

                // 标记模型使用结束
                modelManager.markModelNotInUse();
                Log.d(TAG, "标记模型使用结束，允许自动卸载");
                //updateProgressOnUiThread("标记模型使用结束，允许自动卸载");
                
                // 获取API信息
                String apiUrl = spinnerApiUrl.getSelectedItem().toString();
                String apiKey = editTextApiKey.getText().toString();
                String apiModel = spinnerApiModel.getSelectedItem().toString();
                
                // 不再直接调用API，而是让executeRagQuery方法处理
                // 这样可以避免重复调用API
                // callLLMApi(apiUrl, apiKey, apiModel, buildPromptWithKnowledgeBase(editTextSystemPrompt.getText().toString(), userQuery, relevantDocs));
            } catch (Exception e) {
                String errorMsg = "处理查询失败: " + e.getMessage();
                Log.e(TAG, errorMsg, e);
                updateProgressOnUiThread(errorMsg);
            }
        } catch (Exception e) {
            String errorMsg = "加载模型失败: " + e.getMessage();
            Log.e(TAG, errorMsg, e);
            updateProgressOnUiThread(errorMsg);
            
            // 关闭数据库
            try {
                vectorDb.close();
                Log.d(TAG, "向量数据库已关闭");
            } catch (Exception ex) {
                Log.e(TAG, "关闭向量数据库失败: " + ex.getMessage(), ex);
            }
        }
    }
    
    // 显示模型选择对话框
    private void selectModelAndContinueQuery(String originalModel, List<String> availableModels, String knowledgeBase, String embeddingModelPath, SQLiteVectorDatabaseHandler vectorDb) {
        // 确保在UI线程中运行
        if (Looper.myLooper() != Looper.getMainLooper()) {
            // 如果不在UI线程，切换到UI线程
            Handler mainHandler = new Handler(Looper.getMainLooper());
            mainHandler.post(() -> selectModelAndContinueQuery(originalModel, availableModels, knowledgeBase, embeddingModelPath, vectorDb));
            return;
        }
        
        // 如果没有可用模型，显示错误信息并提示用户添加模型
        if (availableModels.isEmpty()) {
            android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(requireContext());
            builder.setTitle("找不到嵌入模型")
                   .setMessage("在 " + embeddingModelPath + " 目录中找不到任何嵌入模型文件。\n\n请将模型文件（.bin、.onnx、.pt 或 .model 格式）复制到该目录，然后重试。\n\n" +
                               "原始模型: " + originalModel)
                   .setPositiveButton("确定", null)
                   .show();
            return;
        }
        
        // 创建对话框布局
        View dialogView = getLayoutInflater().inflate(R.layout.dialog_model_selection, null);
        Spinner spinnerModels = dialogView.findViewById(R.id.spinnerModels);
        CheckBox checkBoxRemember = dialogView.findViewById(R.id.checkBoxRemember);
        TextView textViewInfo = dialogView.findViewById(R.id.textViewInfo);
        
        // 设置提示信息
        String infoText = "找不到原始模型: " + originalModel + "\n请从下列可用模型中选择一个替代模型:";
        textViewInfo.setText(infoText);
        
        // 设置模型列表
        ArrayAdapter<String> adapter = new ArrayAdapter<>(requireContext(), android.R.layout.simple_spinner_item, availableModels);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerModels.setAdapter(adapter);
        
        // 如果原始模型在列表中，选中它
        int originalModelIndex = availableModels.indexOf(originalModel);
        if (originalModelIndex >= 0) {
            spinnerModels.setSelection(originalModelIndex);
        }
        
        // 检查是否有保存的模型映射
        String savedMapping = null;
        
        // 检查是否是特定的模型，使用ConfigManager获取映射
        if (originalModel.equals("bge-m3")) {
            savedMapping = ConfigManager.getModelMapping(requireContext(), "model_bge-m3", null);
        } else if (originalModel.equals("SBKNBaseV1.0")) {
            savedMapping = ConfigManager.getModelMapping(requireContext(), "kb_SBKNBaseV1.0", null);
        } else {
            // 对于其他模型，使用通用映射格式
            savedMapping = ConfigManager.getModelMapping(requireContext(), "model_" + originalModel, null);
        }
        
        if (savedMapping != null && !savedMapping.isEmpty()) {
            // 找到保存的映射模型在列表中的位置
            int savedModelIndex = availableModels.indexOf(savedMapping);
            if (savedModelIndex >= 0) {
                spinnerModels.setSelection(savedModelIndex);
                checkBoxRemember.setChecked(true);
            }
        }
        
        // 创建对话框
        android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(requireContext());
        builder.setTitle("选择嵌入模型")
               .setView(dialogView)
               .setCancelable(false)
               .setPositiveButton("确定", (dialog, which) -> {
                   // 获取选中的模型
                   String selectedModel = (String) spinnerModels.getSelectedItem();
                   
                   // 如果选中了"记住此选择"，保存映射
                   if (checkBoxRemember.isChecked()) {
                       ConfigManager.setModelMapping(requireContext(), "model_" + originalModel, selectedModel);
                   }
                   
                   // 显示加载中提示
                   updateProgressOnUiThread("正在准备模型...");
                   
                   // 在后台线程中执行耗时操作，避免UI卡顿
                   new Thread(() -> {
                       try {
                           // 继续执行RAG查询任务
                           continueQueryWithSelectedModel(selectedModel, knowledgeBase, embeddingModelPath, vectorDb);
                       } catch (Exception e) {
                           Log.e(TAG, "处理模型选择时出错", e);
                           updateProgressOnUiThread("错误: 处理模型选择时出错: " + e.getMessage());
                       }
                   }).start();
               })
               .setNegativeButton("取消", (dialog, which) -> {
                   // 用户取消了模型选择，显示提示
                   updateProgressOnUiThread("已取消模型选择");
               });
        
        // 显示对话框
        android.app.AlertDialog dialog = builder.create();
        dialog.show();
    }
    
    // 继续执行RAG查询任务
    private void continueQueryWithSelectedModel(String selectedModel, String knowledgeBase, String embeddingModelPath, SQLiteVectorDatabaseHandler vectorDb) {
        // 获取嵌入模型路径
        String foundModelPath = null;
        boolean modelFound = false;
        
        if (selectedModel.equals("根目录")) {
            // 在根目录中查找模型文件
            File embeddingModelDir = new File(embeddingModelPath);
            File[] files = embeddingModelDir.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile() && (file.getName().endsWith(".pt") || 
                                         file.getName().endsWith(".pth") || 
                                         file.getName().endsWith(".onnx"))) {
                        foundModelPath = file.getAbsolutePath();
                        modelFound = true;
                        
                        // 更新元数据中的modeldir为空字符串（表示使用根目录）
                        vectorDb.getMetadata().setModeldir("");
                        vectorDb.saveDatabase();
                        Log.d(TAG, "已更新元数据，modeldir设置为空（使用根目录）");
                        break;
                    }
                }
            }
        } else {
            // 使用选定的目录
            File selectedDir = new File(embeddingModelPath, selectedModel);
            if (selectedDir.exists() && selectedDir.isDirectory()) {
                File[] files = selectedDir.listFiles();
                if (files != null) {
                    for (File file : files) {
                        if (file.isFile() && (file.getName().endsWith(".pt") || 
                                             file.getName().endsWith(".pth") || 
                                             file.getName().endsWith(".onnx"))) {
                            foundModelPath = file.getAbsolutePath();
                            modelFound = true;
                            
                            // 更新元数据中的modeldir为选定的目录
                            vectorDb.getMetadata().setModeldir(selectedModel);
                            vectorDb.saveDatabase();
                            Log.d(TAG, "已更新元数据，modeldir设置为: " + selectedModel);
                            break;
                        }
                    }
                }
            }
        }
        
        if (!modelFound) {
            updateProgressOnUiThread("错误: 无法在选定的目录中找到模型文件");
            return;
        }
        
        // 保存模型映射
        ConfigManager.setModelMapping(requireContext(), "model_" + vectorDb.getMetadata().getEmbeddingModel(), selectedModel);
        
        // 显示模型信息
        String modelInfo = "使用嵌入模型: " + selectedModel + ", 路径: " + foundModelPath;
        Log.d(TAG, modelInfo);
        logToFile(modelInfo);
        updateProgressOnUiThread("正在使用嵌入模型: " + selectedModel);
        
        // 加载嵌入模型
        updateProgressOnUiThread("正在加载嵌入模型...");
        
        // 使用EmbeddingModelManager异步加载模型
        EmbeddingModelManager modelManager = EmbeddingModelManager.getInstance(requireContext());
        
        // 创建一个CountDownLatch来等待异步加载完成
        final CountDownLatch modelLoadLatch = new CountDownLatch(1);
        final AtomicReference<EmbeddingModelHandler> modelHandlerRef = new AtomicReference<>();
        final AtomicReference<Exception> modelErrorRef = new AtomicReference<>();
        
        modelManager.getModelAsync(foundModelPath, new EmbeddingModelManager.ModelCallback() {
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
                String errorMsg = "错误: 加载嵌入模型超时";
                Log.e(TAG, errorMsg);
                updateProgressOnUiThread(errorMsg);
                return;
            }
        } catch (InterruptedException e) {
            String errorMsg = "错误: 等待模型加载被中断: " + e.getMessage();
            Log.e(TAG, errorMsg, e);
            updateProgressOnUiThread(errorMsg);
            return;
        }
        
        // 检查是否有错误
        if (modelErrorRef.get() != null) {
            String errorMsg = "错误: 加载嵌入模型失败: " + modelErrorRef.get().getMessage();
            Log.e(TAG, errorMsg, modelErrorRef.get());
            updateProgressOnUiThread(errorMsg);
            return;
        }
        
        // 获取加载好的模型
        EmbeddingModelHandler modelHandler = modelHandlerRef.get();
        if (modelHandler == null) {
            String errorMsg = "错误: 创建嵌入模型处理器失败";
            Log.e(TAG, errorMsg);
            updateProgressOnUiThread(errorMsg);
            return;
        }
        updateProgressOnUiThread("嵌入模型加载成功: " + modelHandler.getModelName());

        // 标记模型开始使用
        modelManager.markModelInUse();
        Log.d(TAG, "标记模型开始使用，防止自动卸载");
        updateProgressOnUiThread("标记模型开始使用，防止自动卸载");

        // 生成查询向量
        try {
            updateProgressOnUiThread("正在生成查询向量...");
            
            // 获取用户查询
            String userQuery = editTextUserPrompt.getText().toString().trim();
            
            // 生成向量
            float[] queryVector = modelHandler.generateEmbedding(userQuery);
            
            // 记录向量调试信息
            String vectorDebugInfo = modelHandler.getVectorDebugInfo(userQuery, queryVector, System.currentTimeMillis());
            updateProgressOnUiThread(vectorDebugInfo);
            
            // 搜索相似文本块
            updateProgressOnUiThread("正在搜索相似文本块...");
            
            // 获取检索数量设置
            int retrievalCount = Integer.parseInt(editTextSearchDepth.getText().toString());
            
            // 搜索相似文本块
            List<SQLiteVectorDatabaseHandler.SearchResult> searchResults = vectorDb.searchSimilar(queryVector, retrievalCount);
            
            // 提取相关文档
            List<String> relevantDocs = new ArrayList<>();
            StringBuilder similarityInfoBuilder = new StringBuilder("找到的相似文本块：\n");
            for (int i = 0; i < searchResults.size(); i++) {
                SQLiteVectorDatabaseHandler.SearchResult result = searchResults.get(i);
                relevantDocs.add(result.text);
                
                // 记录详细信息到日志
                String resultInfo = "相似度: " + result.similarity + ", 文本: " + result.text.substring(0, Math.min(50, result.text.length())) + "...";
                Log.d(TAG, resultInfo);
                logToFile(resultInfo);

                // 添加到进度显示 - 只显示匹配序号和相似度值，不显示文本内容
                similarityInfoBuilder.append("匹配").append(i + 1).append(": ").append(String.format("%.4f", result.similarity));
                similarityInfoBuilder.append("\n");
            }

            // 显示相似度信息
            if (!searchResults.isEmpty()) {
                updateProgressOnUiThread(similarityInfoBuilder.toString());
            } else {
                updateProgressOnUiThread("警告: 知识库查询未返回相关文档");
            }

            // 标记模型使用结束
            modelManager.markModelNotInUse();
            Log.d(TAG, "标记模型使用结束，允许自动卸载");
            updateProgressOnUiThread("标记模型使用结束，允许自动卸载");
            
            // 获取API信息
            String apiUrl = spinnerApiUrl.getSelectedItem().toString();
            String apiKey = editTextApiKey.getText().toString();
            String apiModel = spinnerApiModel.getSelectedItem().toString();
            
            // 不再直接调用API，而是让executeRagQuery方法处理
            // 这样可以避免重复调用API
            // callLLMApi(apiUrl, apiKey, apiModel, buildPromptWithKnowledgeBase(editTextSystemPrompt.getText().toString(), userQuery, relevantDocs));
            
        } catch (Exception e) {
            String errorMsg = "处理查询失败: " + e.getMessage();
            Log.e(TAG, errorMsg, e);
            updateProgressOnUiThread(errorMsg);
        }
    }
    
    // 获取保存的模型映射
    private String getModelMapping(String originalModel) {
        try {
            if (originalModel == null || originalModel.isEmpty()) {
                return null;
            }
            
            // 从ConfigManager获取模型映射
            return ConfigManager.getModelMapping(requireContext(), "model_" + originalModel, null);
        } catch (Exception e) {
            Log.e(TAG, "获取模型映射失败", e);
            return null;
        }
    }
    
    // 保存模型映射到数据库元数据
    private void saveModelMapping(String originalModel, String selectedModel, String knowledgeBase) {
        try {
            if (originalModel == null || originalModel.isEmpty() || selectedModel == null || selectedModel.isEmpty() || knowledgeBase == null || knowledgeBase.isEmpty()) {
                return;
            }
            
            // 获取知识库目录
            String knowledgeBasePath = ConfigManager.getString(requireContext(), ConfigManager.KEY_KNOWLEDGE_BASE_PATH, ConfigManager.DEFAULT_KNOWLEDGE_BASE_PATH);
            File knowledgeBaseDir = new File(knowledgeBasePath, knowledgeBase);
            
            // 更新数据库元数据
            SQLiteVectorDatabaseHandler vectorDb = null;
            try {
                vectorDb = new SQLiteVectorDatabaseHandler(knowledgeBaseDir, "unknown");
                if (vectorDb.loadDatabase()) {
                    // 获取选择的模型文件名
                    String selectedModelName = new File(selectedModel).getName();
                    
                    // 更新数据库元数据中的模型信息
                    if (vectorDb.updateEmbeddingModel(selectedModelName)) {
                        Log.d(TAG, "已更新数据库元数据中的模型信息: " + selectedModelName);
                        logToFile("已更新数据库元数据中的模型信息: " + selectedModelName);
                        
                        // 保存数据库
                        if (vectorDb.saveDatabase()) {
                            Log.d(TAG, "已保存数据库元数据");
                            logToFile("已保存数据库元数据");
                        } else {
                            Log.e(TAG, "保存数据库元数据失败");
                            logToFile("保存数据库元数据失败");
                        }
                    } else {
                        Log.e(TAG, "更新数据库元数据中的模型信息失败");
                        logToFile("更新数据库元数据中的模型信息失败");
                    }
                } else {
                    Log.e(TAG, "加载数据库失败");
                    logToFile("加载数据库失败");
                }
            } catch (Exception e) {
                Log.e(TAG, "更新数据库元数据失败", e);
                logToFile("更新数据库元数据失败: " + e.getMessage());
            } finally {
                if (vectorDb != null) {
                    try {
                        vectorDb.close();
                    } catch (Exception e) {
                        Log.e(TAG, "关闭数据库失败", e);
                    }
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "保存模型映射失败", e);
        }
    }
    
    // 获取所有可能的模型路径
    private List<String> getPossibleModelPaths() {
        List<String> possiblePaths = new ArrayList<>();
        
        // 获取配置中的嵌入模型路径
        String configPath = ConfigManager.getEmbeddingModelPath(requireContext());
        possiblePaths.add(configPath);
        
        // 添加可能的替代路径
        File externalStorageDir = android.os.Environment.getExternalStorageDirectory();
        File downloadDir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS);
        
        possiblePaths.add(new File(externalStorageDir, "starragdata/embeddings").getAbsolutePath());
        possiblePaths.add(new File(downloadDir, "starragdata/embeddings").getAbsolutePath());
        possiblePaths.add(new File(externalStorageDir, "Download/starragdata/embeddings").getAbsolutePath());
        possiblePaths.add("/storage/emulated/0/Download/starragdata/embeddings");
        possiblePaths.add("/sdcard/Download/starragdata/embeddings");
        
        return possiblePaths;
    }
    
    // 检查下拉框是否为空
    private boolean isSpinnerEmpty(Spinner spinner) {
        if (spinner == null) return true;
        if (spinner.getAdapter() == null) return true;
        SpinnerAdapter adapter = spinner.getAdapter();
        return adapter.getCount() == 0;
    }
    
    /**
     * 将选中的文本转为知识库笔记
     * @param text 要转为笔记的文本
     */
    private void transferToKnowledgeNote(String text) {
        if (text == null || text.trim().isEmpty()) {
            Toast.makeText(requireContext(), "没有选中文本或文本为空", Toast.LENGTH_SHORT).show();
            return;
        }
        
        try {
            // 保存要转换的文本到临时变量
            final String textToTransfer = text;
            
            // 获取MainActivity实例
            MainActivity activity = (MainActivity) requireActivity();
            
            // 导航到知识库笔记页面
            activity.navigateToKnowledgeNote();
            
            // 添加延迟，确保Fragment完全初始化
            new Handler(Looper.getMainLooper()).postDelayed(() -> {
                try {
                    // 尝试获取KnowledgeNoteFragment实例
                    Fragment fragment = null;
                    
                    // 尝试通过不同的标签获取Fragment
                    for (Fragment f : activity.getSupportFragmentManager().getFragments()) {
                        if (f instanceof KnowledgeNoteFragment) {
                            fragment = f;
                            break;
                        }
                        
                        // 检查子Fragment
                        if (f.getChildFragmentManager() != null) {
                            for (Fragment childFragment : f.getChildFragmentManager().getFragments()) {
                                if (childFragment instanceof KnowledgeNoteFragment) {
                                    fragment = childFragment;
                                    break;
                                }
                            }
                        }
                    }
                    
                    // 如果找到了KnowledgeNoteFragment
                    if (fragment instanceof KnowledgeNoteFragment) {
                        KnowledgeNoteFragment knowledgeNoteFragment = (KnowledgeNoteFragment) fragment;
                        
                        // 将文本插入到知识库笔记的内容编辑框中
                        knowledgeNoteFragment.insertTextToContentEditor(textToTransfer);
                        
                        // 显示提示信息
                        Toast.makeText(requireContext(), "已转为知识库笔记", Toast.LENGTH_SHORT).show();
                        Log.d(TAG, "已将文本转为知识库笔记，长度: " + textToTransfer.length());
                    } else {
                        // 第一次尝试失败，再次延迟重试
                        Log.d(TAG, "第一次尝试获取KnowledgeNoteFragment失败，将在500ms后重试");
                        
                        // 再次延迟500ms重试
                        new Handler(Looper.getMainLooper()).postDelayed(() -> {
                            try {
                                // 再次尝试获取Fragment
                                Fragment retryFragment = null;
                                for (Fragment f : activity.getSupportFragmentManager().getFragments()) {
                                    if (f instanceof KnowledgeNoteFragment) {
                                        retryFragment = f;
                                        break;
                                    }
                                    
                                    // 检查ViewPager中的Fragment
                                    if (f.getChildFragmentManager() != null) {
                                        for (Fragment childFragment : f.getChildFragmentManager().getFragments()) {
                                            if (childFragment instanceof KnowledgeNoteFragment) {
                                                retryFragment = childFragment;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (retryFragment instanceof KnowledgeNoteFragment) {
                                    KnowledgeNoteFragment knowledgeNoteFragment = (KnowledgeNoteFragment) retryFragment;
                                    knowledgeNoteFragment.insertTextToContentEditor(textToTransfer);
                                    Toast.makeText(requireContext(), "已转为知识库笔记", Toast.LENGTH_SHORT).show();
                                    Log.d(TAG, "重试成功：已将文本转为知识库笔记，长度: " + textToTransfer.length());
                                } else {
                                    Toast.makeText(requireContext(), "无法获取知识库笔记页面", Toast.LENGTH_SHORT).show();
                                    Log.e(TAG, "重试后仍无法获取KnowledgeNoteFragment实例");
                                }
                            } catch (Exception e) {
                                Toast.makeText(requireContext(), "重试转为笔记失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                                Log.e(TAG, "重试转为笔记失败", e);
                            }
                        }, 500); // 再延迟500ms
                    }
                } catch (Exception e) {
                    Toast.makeText(requireContext(), "转为笔记失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    Log.e(TAG, "转为笔记失败", e);
                }
            }, 300); // 初始延迟300ms
        } catch (Exception e) {
            Toast.makeText(requireContext(), "转为笔记失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            Log.e(TAG, "转为笔记失败", e);
        }
    }
    
    /**
     * 应用全局字体大小设置
     */
    private void applyGlobalTextSize() {
        if (textViewResponse != null) {
            float fontSize = ConfigManager.getGlobalTextSize(requireContext());
            textViewResponse.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSize);
            Log.d(TAG, "已应用全局字体大小: " + fontSize + "sp");
        }
    }
    
    @Override
    public void onResume() {
        super.onResume();
        // 在页面恢复时重新应用字体大小，以便在设置页面修改后能够立即生效
        applyGlobalTextSize();
    }
    
    // 设置自定义文本选择菜单
    private void setupCustomTextSelectionMenu() {
        if (textViewResponse != null) {
            textViewResponse.setCustomSelectionActionModeCallback(new ActionMode.Callback() {
                @Override
                public boolean onCreateActionMode(ActionMode mode, Menu menu) {
                    // 不干扰系统默认菜单创建
                    return true;
                }

                @Override
                public boolean onPrepareActionMode(ActionMode mode, Menu menu) {
                    // 先检查菜单中是否已有"转笔记"选项
                    boolean hasTransferOption = false;
                    for (int i = 0; i < menu.size(); i++) {
                        MenuItem item = menu.getItem(i);
                        if (item.getTitle().equals("转笔记")) {
                            hasTransferOption = true;
                            break;
                        }
                    }
                    
                    // 只有在没有"转笔记"选项时才添加
                    if (!hasTransferOption) {
                        menu.add(Menu.NONE, Menu.FIRST + 100, 5, "转笔记");
                    }
                    return true;
                }

                @Override
                public boolean onActionItemClicked(ActionMode mode, MenuItem item) {
                    if (item.getTitle().equals("转笔记")) {
                        // 获取选中的文本
                        String selectedText = "";
                        
                        // 获取选中的文本
                        int start = textViewResponse.getSelectionStart();
                        int end = textViewResponse.getSelectionEnd();
                        
                        if (start >= 0 && end >= 0 && start != end) {
                            // 有选中的文本
                            selectedText = textViewResponse.getText().toString().substring(start, end);
                        } else {
                            // 没有选中文本，使用全部内容
                            selectedText = textViewResponse.getText().toString();
                        }
                        
                        // 调用转笔记方法
                        transferToKnowledgeNote(selectedText);
                        
                        // 关闭选择模式
                        mode.finish();
                        return true;
                    }
                    return false;
                }

                @Override
                public void onDestroyActionMode(ActionMode mode) {
                    // 不需要特殊处理
                }
            });
        }
    }
}