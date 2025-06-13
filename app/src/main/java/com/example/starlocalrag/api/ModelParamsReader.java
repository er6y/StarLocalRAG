package com.example.starlocalrag.api;

import com.example.starlocalrag.LogManager;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * 模型参数读取工具类
 * 用于从模型目录下的配置文件中读取推理参数
 */
public class ModelParamsReader {
    private static final String TAG = "ModelParamsReader";
    
    /**
     * 从模型目录读取推理参数
     * @param modelDirPath 模型目录路径
     * @return 推理参数对象，如果读取失败返回null
     */
    public static LocalLlmHandler.InferenceParams readInferenceParams(String modelDirPath) {
        if (modelDirPath == null || modelDirPath.isEmpty()) {
            LogManager.logW(TAG, "模型目录路径为空");
            return null;
        }
        
        File modelDir = new File(modelDirPath);
        if (!modelDir.exists() || !modelDir.isDirectory()) {
            LogManager.logW(TAG, "模型目录不存在: " + modelDirPath);
            return null;
        }
        
        // 尝试读取 params 文件
        File paramsFile = new File(modelDir, "params");
        if (paramsFile.exists()) {
            LogManager.logI(TAG, "找到params文件: " + paramsFile.getAbsolutePath());
            return readFromParamsFile(paramsFile);
        }
        
        // 尝试读取 generation_config.json 文件
        File configFile = new File(modelDir, "generation_config.json");
        if (configFile.exists()) {
            LogManager.logI(TAG, "找到generation_config.json文件: " + configFile.getAbsolutePath());
            return readFromJsonFile(configFile);
        }
        
        LogManager.logI(TAG, "模型目录下未找到参数配置文件: " + modelDirPath);
        return null;
    }
    
    /**
     * 从params文件读取参数（支持JSON格式和简单键值对格式）
     */
    private static LocalLlmHandler.InferenceParams readFromParamsFile(File paramsFile) {
        try {
            LogManager.logI(TAG, "开始读取params文件: " + paramsFile.getAbsolutePath());
            String content = readFileContent(paramsFile);
            if (content == null || content.trim().isEmpty()) {
                LogManager.logW(TAG, "params文件内容为空");
                return null;
            }
            
            LogManager.logI(TAG, "params文件内容长度: " + content.length() + " 字符");
            
            // 检测文件格式：如果内容以{开头，尝试作为JSON解析
            String trimmedContent = content.trim();
            if (trimmedContent.startsWith("{")) {
                LogManager.logI(TAG, "检测到JSON格式，使用JSON解析器");
                return readFromJsonContent(content);
            } else {
                LogManager.logI(TAG, "检测到键值对格式，使用键值对解析器");
                return readFromKeyValueContent(content);
            }
            
        } catch (Exception e) {
            LogManager.logW(TAG, "读取params文件失败", e);
            return null;
        }
    }
    
    /**
     * 从JSON内容解析参数
     */
    private static LocalLlmHandler.InferenceParams readFromJsonContent(String content) {
        try {
            JSONObject json = new JSONObject(content);
            LocalLlmHandler.InferenceParams params = new LocalLlmHandler.InferenceParams();
            boolean hasParams = false;
            
            LogManager.logI(TAG, "开始解析JSON格式参数");
            
            // 解析temperature
            if (json.has("temperature")) {
                try {
                    params.setTemperature((float) json.getDouble("temperature"));
                    hasParams = true;
                    LogManager.logI(TAG, "✓ 解析temperature: " + json.getDouble("temperature"));
                } catch (Exception e) {
                    LogManager.logW(TAG, "解析temperature失败", e);
                }
            }
            
            // 解析top_p
            if (json.has("top_p")) {
                try {
                    params.setTopP((float) json.getDouble("top_p"));
                    hasParams = true;
                    LogManager.logI(TAG, "✓ 解析top_p: " + json.getDouble("top_p"));
                } catch (Exception e) {
                    LogManager.logW(TAG, "解析top_p失败", e);
                }
            }
            
            // 解析top_k
            if (json.has("top_k")) {
                try {
                    params.setTopK(json.getInt("top_k"));
                    hasParams = true;
                    LogManager.logI(TAG, "✓ 解析top_k: " + json.getInt("top_k"));
                } catch (Exception e) {
                    LogManager.logW(TAG, "解析top_k失败", e);
                }
            }
            
            // 解析repeat_penalty或repetition_penalty
            String[] repeatKeys = {"repeat_penalty", "repetition_penalty"};
            for (String key : repeatKeys) {
                if (json.has(key)) {
                    try {
                        params.setRepetitionPenalty((float) json.getDouble(key));
                        hasParams = true;
                        LogManager.logI(TAG, "✓ 解析" + key + ": " + json.getDouble(key));
                        break;
                    } catch (Exception e) {
                        LogManager.logW(TAG, "解析" + key + "失败", e);
                    }
                }
            }
            
            LogManager.logI(TAG, "JSON参数解析完成，hasParams: " + hasParams);
            return hasParams ? params : null;
            
        } catch (Exception e) {
            LogManager.logW(TAG, "JSON解析失败", e);
            return null;
        }
    }
    
    /**
     * 从键值对内容解析参数
     */
    private static LocalLlmHandler.InferenceParams readFromKeyValueContent(String content) {
        try {
            LocalLlmHandler.InferenceParams params = new LocalLlmHandler.InferenceParams();
            boolean hasParams = false;
            
            String[] lines = content.split("\n");
            LogManager.logI(TAG, "解析到 " + lines.length + " 行键值对内容");
            
            for (int i = 0; i < lines.length; i++) {
                String line = lines[i].trim();
                
                if (line.isEmpty() || line.startsWith("#")) {
                    continue; // 跳过空行和注释
                }
                
                String[] parts = line.split("=", 2);
                if (parts.length != 2) {
                    LogManager.logW(TAG, "行格式不正确，跳过: [" + line + "]");
                    continue;
                }
                
                String key = parts[0].trim();
                String value = parts[1].trim();
                LogManager.logI(TAG, "解析键值对: [" + key + "] = [" + value + "]");
                
                try {
                    switch (key.toLowerCase()) {
                        case "temperature":
                        case "temp":
                            params.setTemperature(Float.parseFloat(value));
                            hasParams = true;
                            LogManager.logI(TAG, "✓ 解析temperature: " + value);
                            break;
                        case "top_p":
                        case "topp":
                            params.setTopP(Float.parseFloat(value));
                            hasParams = true;
                            LogManager.logI(TAG, "✓ 解析top_p: " + value);
                            break;
                        case "top_k":
                        case "topk":
                            params.setTopK(Integer.parseInt(value));
                            hasParams = true;
                            LogManager.logI(TAG, "✓ 解析top_k: " + value);
                            break;
                        case "repeat_penalty":
                        case "repetition_penalty":
                            params.setRepetitionPenalty(Float.parseFloat(value));
                            hasParams = true;
                            LogManager.logI(TAG, "✓ 解析repeat_penalty: " + value);
                            break;
                        default:
                            LogManager.logI(TAG, "未识别的参数键: [" + key + "]，跳过");
                            break;
                    }
                } catch (NumberFormatException e) {
                    LogManager.logW(TAG, "解析参数失败 [" + key + "=" + value + "]", e);
                }
            }
            
            LogManager.logI(TAG, "键值对参数解析完成，hasParams: " + hasParams);
            return hasParams ? params : null;
            
        } catch (Exception e) {
            LogManager.logW(TAG, "键值对解析失败", e);
            return null;
        }
    }
    
    /**
     * 从JSON文件读取参数
     */
    private static LocalLlmHandler.InferenceParams readFromJsonFile(File jsonFile) {
        try {
            String content = readFileContent(jsonFile);
            if (content == null || content.trim().isEmpty()) {
                LogManager.logW(TAG, "JSON文件内容为空");
                return null;
            }
            
            JSONObject json = new JSONObject(content);
            LocalLlmHandler.InferenceParams params = new LocalLlmHandler.InferenceParams();
            boolean hasParams = false;
            
            // 解析temperature
            if (json.has("temperature")) {
                try {
                    params.setTemperature((float) json.getDouble("temperature"));
                    hasParams = true;
                    LogManager.logI(TAG, "✓ 解析temperature: " + json.getDouble("temperature"));
                } catch (Exception e) {
                    LogManager.logW(TAG, "解析temperature失败", e);
                }
            }
            
            // 解析top_p
            if (json.has("top_p")) {
                try {
                    params.setTopP((float) json.getDouble("top_p"));
                    hasParams = true;
                    LogManager.logI(TAG, "✓ 解析top_p: " + json.getDouble("top_p"));
                } catch (Exception e) {
                    LogManager.logW(TAG, "解析top_p失败", e);
                }
            }
            
            // 解析top_k
            if (json.has("top_k")) {
                try {
                    params.setTopK(json.getInt("top_k"));
                    hasParams = true;
                    LogManager.logI(TAG, "✓ 解析top_k: " + json.getInt("top_k"));
                } catch (Exception e) {
                    LogManager.logW(TAG, "解析top_k失败", e);
                }
            }
            
            // 解析repeat_penalty或repetition_penalty
            String[] repeatKeys = {"repeat_penalty", "repetition_penalty"};
            for (String key : repeatKeys) {
                if (json.has(key)) {
                    try {
                        params.setRepetitionPenalty((float) json.getDouble(key));
                        hasParams = true;
                        LogManager.logI(TAG, "✓ 解析" + key + ": " + json.getDouble(key));
                        break;
                    } catch (Exception e) {
                        LogManager.logW(TAG, "解析" + key + "失败", e);
                    }
                }
            }
            
            return hasParams ? params : null;
            
        } catch (Exception e) {
            LogManager.logW(TAG, "读取JSON文件失败", e);
            return null;
        }
    }
    
    /**
     * 读取文件内容
     */
    private static String readFileContent(File file) {
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] data = new byte[(int) file.length()];
            fis.read(data);
            return new String(data, StandardCharsets.UTF_8);
        } catch (IOException e) {
            LogManager.logW(TAG, "读取文件失败: " + file.getAbsolutePath(), e);
            return null;
        }
    }
}