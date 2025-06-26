package com.example.starlocalrag;

import android.app.AlertDialog;
import android.content.Context;


import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.text.method.ScrollingMovementMethod;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.view.MenuProvider;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Lifecycle;

import com.example.starlocalrag.ConfigManager;
import com.example.starlocalrag.LogManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ModelDownloadFragment extends Fragment {
    private static final String TAG = "ModelDownloadFragment";
    
    // Constants removed - now using string resources
    
    // UI组件
    private CheckBox checkBoxBgeM3;
    private CheckBox checkBoxBgeReranker;
    private CheckBox checkBoxQwen06B;
    private CheckBox checkBoxQwen17B;
    private TextView textViewProgress;
    private Button buttonDownload;
    
    // 下载相关
    private ExecutorService downloadExecutor;
    private Handler mainHandler;
    private boolean isDownloading = false;
    
    // 进度显示
    private StringBuilder progressText = new StringBuilder();
    private int lastReportedProgress = 0;
    
    // 电源管理
    private PowerManager.WakeLock wakeLock;
    
    // 模型配置
    private static final Map<String, ModelConfig> MODEL_CONFIGS = new HashMap<>();
    
    static {
        // 嵌入模型配置
        MODEL_CONFIGS.put("bge-m3", new ModelConfig(
            "bge-m3_dynamic_int8_onnx",
            ModelType.EMBEDDING,
            new String[]{
                "https://hf-mirror.com/er6y/bge-m3_dynamic_int8_onnx/resolve/main/tokenizer_config.json?download=true",
                "https://hf-mirror.com/er6y/bge-m3_dynamic_int8_onnx/resolve/main/tokenizer.json?download=true",
                "https://hf-mirror.com/er6y/bge-m3_dynamic_int8_onnx/resolve/main/special_tokens_map.json?download=true",
                "https://hf-mirror.com/er6y/bge-m3_dynamic_int8_onnx/resolve/main/model.onnx?download=true",
                "https://hf-mirror.com/er6y/bge-m3_dynamic_int8_onnx/resolve/main/conversion_info.json?download=true",
                "https://hf-mirror.com/er6y/bge-m3_dynamic_int8_onnx/resolve/main/config.json?download=true"
            },
            new String[]{
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "model.onnx",
                "conversion_info.json",
                "config.json"
            }
        ));
        
        // 重排模型配置
        MODEL_CONFIGS.put("bge-reranker", new ModelConfig(
            "bge-reranker-v2-m3_dynamic_int8_onnx",
            ModelType.RERANKER,
            new String[]{
                "https://hf-mirror.com/er6y/bge-reranker-v2-m3_dynamic_int8_onnx/resolve/main/config.json?download=true",
                "https://hf-mirror.com/er6y/bge-reranker-v2-m3_dynamic_int8_onnx/resolve/main/conversion_info.json?download=true",
                "https://hf-mirror.com/er6y/bge-reranker-v2-m3_dynamic_int8_onnx/resolve/main/model.onnx?download=true",
                "https://hf-mirror.com/er6y/bge-reranker-v2-m3_dynamic_int8_onnx/resolve/main/special_tokens_map.json?download=true",
                "https://hf-mirror.com/er6y/bge-reranker-v2-m3_dynamic_int8_onnx/resolve/main/tokenizer.json?download=true",
                "https://hf-mirror.com/er6y/bge-reranker-v2-m3_dynamic_int8_onnx/resolve/main/tokenizer_config.json?download=true"
            },
            new String[]{
                "config.json",
                "conversion_info.json",
                "model.onnx",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json"
            }
        ));
        
        // LLM模型配置
        MODEL_CONFIGS.put("qwen-0.6b", new ModelConfig(
            "Qwen3-0.6B-GGUF",
            ModelType.LLM,
            new String[]{
                "https://hf-mirror.com/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf?download=true",
                "https://hf-mirror.com/Qwen/Qwen3-0.6B-GGUF/resolve/main/params?download=true"
            },
            new String[]{
                "Qwen3-0.6B-Q8_0.gguf",
                "params"
            }
        ));
        
        MODEL_CONFIGS.put("qwen-1.7b", new ModelConfig(
            "Qwen3-1.7B-GGUF",
            ModelType.LLM,
            new String[]{
                "https://hf-mirror.com/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf?download=true",
                "https://hf-mirror.com/Qwen/Qwen3-1.7B-GGUF/resolve/main/params?download=true"
            },
            new String[]{
                "Qwen3-1.7B-Q8_0.gguf",
                "params"
            }
        ));
    }
    
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_model_download, container, false);
        
        initViews(view);
        setupListeners();
        
        downloadExecutor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());
        
        // 初始化进度文本
        progressText.setLength(0);
        progressText.append(textViewProgress.getText());
        
        return view;
    }
    
    private void initViews(View view) {
        checkBoxBgeM3 = view.findViewById(R.id.checkBoxBgeM3);
        checkBoxBgeReranker = view.findViewById(R.id.checkBoxBgeReranker);
        checkBoxQwen06B = view.findViewById(R.id.checkBoxQwen06B);
        checkBoxQwen17B = view.findViewById(R.id.checkBoxQwen17B);
        textViewProgress = view.findViewById(R.id.textViewProgress);
        buttonDownload = view.findViewById(R.id.buttonDownload);
        
        // 设置进度文本框支持文本选择和滚动
        textViewProgress.setTextIsSelectable(true);
        textViewProgress.setFocusable(true);
        textViewProgress.setFocusableInTouchMode(true);
    }
    
    private void setupListeners() {
        // 添加MenuProvider来处理菜单
        requireActivity().addMenuProvider(new MenuProvider() {
            @Override
            public void onCreateMenu(@NonNull Menu menu, @NonNull MenuInflater menuInflater) {
                menuInflater.inflate(R.menu.model_download_menu, menu);
            }
            
            @Override
            public boolean onMenuItemSelected(@NonNull MenuItem menuItem) {
                int id = menuItem.getItemId();
                
                if (id == R.id.action_close_download) {
                    // Close download page - Use Navigation component's back operation
                    requireActivity().getOnBackPressedDispatcher().onBackPressed();
                    return true;
                }
                
                return false;
            }
        }, getViewLifecycleOwner(), Lifecycle.State.RESUMED);
        
        buttonDownload.setOnClickListener(v -> {
            if (isDownloading) {
                stopDownload();
            } else {
                startDownload();
            }
        });
    }
    
    private void startDownload() {
        List<String> selectedModels = getSelectedModels();
        if (selectedModels.isEmpty()) {
            Toast.makeText(getContext(), "请至少选择一个模型", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // 直接开始下载
        proceedWithDownload(selectedModels);
    }
    
    private List<String> getSelectedModels() {
        List<String> selected = new ArrayList<>();
        
        if (checkBoxBgeM3.isChecked()) {
            selected.add("bge-m3");
        }
        if (checkBoxBgeReranker.isChecked()) {
            selected.add("bge-reranker");
        }
        if (checkBoxQwen06B.isChecked()) {
            selected.add("qwen-0.6b");
        }
        if (checkBoxQwen17B.isChecked()) {
            selected.add("qwen-1.7b");
        }
        
        return selected;
    }
    

    
    private void proceedWithDownload(List<String> selectedModels) {
        // 显示Wi-Fi下载提示
        showWifiDownloadDialog(selectedModels);
    }
    
    private void showWifiDownloadDialog(List<String> selectedModels) {
        new AlertDialog.Builder(getContext())
            .setTitle("下载确认")
                .setMessage("确定要下载选中的模型吗？")
                .setPositiveButton(getString(R.string.common_download), (dialog, which) -> checkDirectoryConflictsAndDownload(selectedModels))
            .setNegativeButton(getString(R.string.common_cancel), null)
            .show();
    }
    
    private void checkDirectoryConflictsAndDownload(List<String> selectedModels) {
        // 检查目录冲突
        List<String> conflictDirs = checkDirectoryConflicts(selectedModels);
        if (!conflictDirs.isEmpty()) {
            showOverwriteDialog(conflictDirs, selectedModels);
        } else {
            executeDownload(selectedModels);
        }
    }
    
    private List<String> checkDirectoryConflicts(List<String> selectedModels) {
        List<String> conflicts = new ArrayList<>();
        
        for (String modelKey : selectedModels) {
            ModelConfig config = MODEL_CONFIGS.get(modelKey);
            if (config != null) {
                File targetDir = getTargetDirectory(config);
                if (targetDir.exists() && targetDir.isDirectory()) {
                    conflicts.add(config.directoryName);
                }
            }
        }
        
        return conflicts;
    }
    
    private void showOverwriteDialog(List<String> conflictDirs, List<String> selectedModels) {
        String message = "以下目录已存在：" + "\n\n" + String.join("\n", conflictDirs);
        
        new AlertDialog.Builder(getContext())
            .setTitle("目录已存在")
                .setMessage(message)
                .setPositiveButton(getString(R.string.common_overwrite), (dialog, which) -> executeDownload(selectedModels))
                .setNegativeButton(getString(R.string.common_cancel), null)
            .show();
    }
    
    private void executeDownload(List<String> selectedModels) {
        isDownloading = true;
        buttonDownload.setText(R.string.button_interrupt);
        
        // 清空进度文本，开始新的下载
        progressText.setLength(0);
        
        // 获取电源锁
        acquireWakeLocks();
        
        appendProgress("下载选中的模型" + "\n");
        
        downloadExecutor.execute(() -> {
            boolean allSuccess = true;
            try {
                for (String modelKey : selectedModels) {
                    // 检查是否已被中断
                    if (!isDownloading) {
                        mainHandler.post(() -> appendProgress("\n" + "下载已被中断" + "\n"));
                        return;
                    }
                    boolean success = downloadModel(modelKey);
                    if (!success) {
                        allSuccess = false;
                    }
                }
                
                // 只有在所有模型都成功下载且未被中断的情况下才显示完成信息
                if (isDownloading && allSuccess) {
                    mainHandler.post(() -> {
                        appendProgress("\n" + "所有模型下载完成" + "\n");
                        finishDownload();
                    });
                } else {
                    mainHandler.post(() -> finishDownload());
                }
                
            } catch (Exception e) {
                LogManager.logE(TAG, "下载错误", e);
                mainHandler.post(() -> {
                    appendProgress("\n" + "下载错误" + ": " + e.getMessage() + "\n");
                    finishDownload();
                });
            }
        });
    }
    
    private boolean downloadModel(String modelKey) {
        ModelConfig config = MODEL_CONFIGS.get(modelKey);
        if (config == null) {
            mainHandler.post(() -> appendProgress("未知模型" + ": " + modelKey + "\n"));
            return false;
        }
        
        mainHandler.post(() -> appendProgress("\n" + "开始下载" + ": " + config.directoryName + "\n"));
        
        // 创建目标目录
        File targetDir = getTargetDirectory(config);
        if (!targetDir.exists()) {
            targetDir.mkdirs();
        }
        
        // 下载所有文件
        for (int i = 0; i < config.downloadUrls.length; i++) {
            // 检查是否已被中断
            if (!isDownloading) {
                mainHandler.post(() -> appendProgress("Download interrupted\n"));
                return false;
            }
            
            String url = config.downloadUrls[i];
            String filename = config.filenames[i];
            
            mainHandler.post(() -> appendProgress("正在下载文件" + ": " + filename + "\n"));
            
            boolean success = downloadFile(url, new File(targetDir, filename));
            if (!success && isDownloading) {
                // 只有在未被中断的情况下才尝试备用地址
                String backupUrl = url.replace("hf-mirror.com", "huggingface.co");
                mainHandler.post(() -> appendProgress("尝试备用地址" + "\n"));
                success = downloadFile(backupUrl, new File(targetDir, filename));
            }
            
            if (!success) {
                mainHandler.post(() -> appendProgress("下载失败" + ": " + filename + "\n"));
                return false;
            }
        }
        
        mainHandler.post(() -> appendProgress(config.directoryName + " " + "下载完成" + "\n"));
        return true;
    }
    
    private File getTargetDirectory(ModelConfig config) {
        String basePath;
        switch (config.type) {
            case EMBEDDING:
                basePath = ConfigManager.getEmbeddingModelPath(getContext());
                break;
            case RERANKER:
                basePath = ConfigManager.getRerankerModelPath(getContext());
                break;
            case LLM:
                basePath = ConfigManager.getModelPath(getContext());
                break;
            default:
                throw new IllegalArgumentException("未知模型" + "类型: " + config.type);
        }
        
        return new File(basePath, config.directoryName);
    }
    
    private boolean downloadFile(String urlString, File targetFile) {
        try {
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(30000);
            connection.setReadTimeout(30000);
            
            int responseCode = connection.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                LogManager.logE(TAG, "下载失败，HTTP响应码" + ": " + responseCode);
                return false;
            }
            
            long fileSize = connection.getContentLengthLong();
            
            // 初始化进度显示
            mainHandler.post(() -> appendProgress("进度" + ":"));
            
            try (InputStream inputStream = connection.getInputStream();
                 FileOutputStream outputStream = new FileOutputStream(targetFile)) {
                
                byte[] buffer = new byte[32768]; // 增大缓冲区到32KB提高性能
                long totalBytesRead = 0;
                int bytesRead;
                int lastReportedProgress = 0;
                
                LogManager.logI(TAG, "开始下载，文件大小: " + fileSize + " 字节");
                
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    // 检查是否已被中断
                    if (!isDownloading) {
                        LogManager.logI(TAG, "下载已被中断");
                        return false;
                    }
                    
                    outputStream.write(buffer, 0, bytesRead);
                    totalBytesRead += bytesRead;
                    
                    if (fileSize > 0) {
                        final int progress = (int) ((totalBytesRead * 100) / fileSize);
                        
                        // 确保每个百分点都显示一个点，防止跳跃
                        if (progress > lastReportedProgress) {
                            // 构建所有需要添加的点号和百分比
                            StringBuilder progressText = new StringBuilder();
                            while (progress > lastReportedProgress && lastReportedProgress < 100) {
                                lastReportedProgress++;
                                progressText.append(".");
                                
                                // 每10个点插入百分比数字，不换行
                                if (lastReportedProgress % 10 == 0) {
                                    progressText.append(lastReportedProgress).append("%");
                                }
                            }
                            
                            final String textToAdd = progressText.toString();
                            LogManager.logI(TAG, "进度更新: " + lastReportedProgress + "%, 添加内容: " + textToAdd);
                            
                            // 一次性添加所有内容，减少UI更新次数
                            mainHandler.post(() -> appendProgress(textToAdd));
                        }
                    }
                }
            }
            
            // 完成后显示结果
            if (fileSize > 0) {
                mainHandler.post(() -> appendProgress(" " + getString(R.string.log_100_percent) + "\n"));
            }
            
            return true;
            
        } catch (IOException e) {
            LogManager.logE(TAG, getString(R.string.log_download_file_failed) + ": " + urlString, e);
            return false;
        }
    }
    
    private void appendProgress(String text) {
        // 简单的文本追加
        progressText.append(text);
        
        // 直接设置文本内容
        textViewProgress.setText(progressText.toString());
        
        // 自动滚动到底部
        textViewProgress.post(() -> {
            if (textViewProgress.getLayout() != null) {
                int scrollAmount = textViewProgress.getLayout().getLineTop(textViewProgress.getLineCount()) - textViewProgress.getHeight();
                if (scrollAmount > 0) {
                    textViewProgress.scrollTo(0, scrollAmount);
                }
            }
        });
    }
    
    private void acquireWakeLocks() {
        try {
            PowerManager powerManager = (PowerManager) requireContext().getSystemService(Context.POWER_SERVICE);
            wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "StarLocalRAG:ModelDownload");
            wakeLock.acquire(30 * 60 * 1000L); // 30分钟超时
        } catch (Exception e) {
            LogManager.logE(TAG, getString(R.string.log_acquire_wakelock_failed), e);
        }
    }
    
    private void releaseWakeLocks() {
        try {
            if (wakeLock != null && wakeLock.isHeld()) {
                wakeLock.release();
                wakeLock = null;
            }
        } catch (Exception e) {
            LogManager.logE(TAG, getString(R.string.log_release_wakelock_failed), e);
        }
    }
    
    private void finishDownload() {
        isDownloading = false;
        buttonDownload.setText("Download Selected Models");
        releaseWakeLocks();
    }
    
    private void stopDownload() {
        if (isDownloading) {
            isDownloading = false;
            
            // 中断下载线程
            if (downloadExecutor != null) {
                downloadExecutor.shutdownNow();
                downloadExecutor = Executors.newSingleThreadExecutor();
            }
            
            mainHandler.post(() -> {
                appendProgress("\nDownload interrupted\n");
                buttonDownload.setText("Download Selected Models");
                releaseWakeLocks();
            });
        }
    }
    
    @Override
    public void onDestroy() {
        super.onDestroy();
        
        if (downloadExecutor != null) {
            downloadExecutor.shutdown();
        }
        
        releaseWakeLocks();
    }
    
    // 模型配置类
    private static class ModelConfig {
        final String directoryName;
        final ModelType type;
        final String[] downloadUrls;
        final String[] filenames;
        
        ModelConfig(String directoryName, ModelType type, String[] downloadUrls, String[] filenames) {
            this.directoryName = directoryName;
            this.type = type;
            this.downloadUrls = downloadUrls;
            this.filenames = filenames;
        }
    }
    
    // 模型类型枚举
    private enum ModelType {
        EMBEDDING, RERANKER, LLM
    }
}