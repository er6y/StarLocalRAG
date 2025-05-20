package com.example.starlocalrag;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.Set;

/**
 * 内部Trie节点结构
 */
class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    boolean isEnd = false;
    String token = null;
}


/**
 * BERT分词器实现，支持解析tokenizer.json文件
 */
public class BertTokenizer {
    private static final String TAG = "StarLocalRAG_Tokenizer";
    
    // 特殊token
    private String clsToken = "[CLS]";
    private String sepToken = "[SEP]";
    private String unkToken = "[UNK]";
    private String padToken = "[PAD]";
    private String maskToken = "[MASK]";
    
    // 特殊token ID - 更新为与PC端一致
    private int clsTokenId = 0; // <s>对应0 (之前是101)
    private int sepTokenId = 2; // </s>对应2 (之前是102)
    private int unkTokenId = 3; // <unk>对应3 (之前是100)
    private int padTokenId = 1; // <pad>对应1 (之前是0)
    private int maskTokenId = 250001; // <mask>对应250001 (之前是103)
    
    // 词汇表 (token -> id)
    private Map<String, Integer> vocab = new HashMap<>();
    
    // Trie前缀树根节点
    private TrieNode trieRoot = new TrieNode();
    
    // 反向词汇表 (id -> token)
    private Map<Integer, String> idToToken = new HashMap<>();
    
    // 最大序列长度
    private int maxLength = 512;
    
    // 是否区分大小写
    private boolean doLowerCase = false;
    
    // BPE合并规则
    private List<Map.Entry<String, String>> bpeMerges = new ArrayList<>();
    
    // 词汇表中是否使用了BPE编码
    private boolean useBpe = false;
    
    // 是否添加前缀空格
    private boolean addPrefixSpace = true;
    
    // 是否使用一致性分词策略
    private boolean useConsistentTokenization = false;
    
    // 是否启用调试模式
    private boolean debugMode = false;

    private boolean isQwen3Model = false;
    private static final String[] QWEN3_SPECIAL_TOKENS = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>"};
    
    private void detectModelType() {
        isQwen3Model = vocab.containsKey("<|im_start|>") || vocab.containsKey("<|im_end|>");
        Log.d(TAG, "模型类型检测: " + (isQwen3Model ? "Qwen3 模型" : "非 Qwen3 模型（可能是 BGE-M3）"));
    }

    
    /**
     * 从目录加载tokenizer
     * @param modelDir 模型目录
     * @return 是否加载成功
     */
    public boolean loadFromDirectory(File modelDir) {
        try {
            Log.d(TAG, "从目录加载tokenizer: " + modelDir.getAbsolutePath());
            
            // 加载tokenizer配置
            File tokenizerConfigFile = new File(modelDir, "tokenizer_config.json");
            if (tokenizerConfigFile.exists()) {
                loadTokenizerConfig(tokenizerConfigFile);
                Log.d(TAG, "成功加载tokenizer配置");
            } else {
                Log.d(TAG, "未找到tokenizer_config.json文件");
            }
            
            // 加载特殊token映射
            File specialTokensMapFile = new File(modelDir, "special_tokens_map.json");
            if (specialTokensMapFile.exists()) {
                loadSpecialTokensMap(specialTokensMapFile);
                Log.d(TAG, "成功加载特殊token映射");
            } else {
                Log.d(TAG, "未找到special_tokens_map.json文件");
            }
            
            // 加载词汇表
            File tokenizerFile = new File(modelDir, "tokenizer.json");
            if (tokenizerFile.exists()) {
                boolean result = loadTokenizerJson(tokenizerFile);
                if (result) {
                    // 构建Trie前缀树
                    buildTrieFromVocab();
                    
                    // 检测模型类型
                    detectModelType();
                    
                    Log.d(TAG, "词汇表加载完成，大小：" + vocab.size());
                    // 打印部分词汇表内容作为诊断
                    StringBuilder sampleVocab = new StringBuilder();
                    int count = 0;
                    for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
                        if (count < 20) {
                            sampleVocab.append(entry.getKey()).append("=").append(entry.getValue()).append(", ");
                        }
                        count++;
                    }
                    Log.d(TAG, "词汇表样本: " + sampleVocab.toString());
                }
                return result;
            } else {
                Log.e(TAG, "找不到tokenizer.json文件");
                return false;
            }
        } catch (Exception e) {
            Log.e(TAG, "加载tokenizer失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 加载tokenizer配置
     * @param configFile 配置文件
     */
    private void loadTokenizerConfig(File configFile) {
        try {
            Log.d(TAG, "加载tokenizer配置: " + configFile.getAbsolutePath());
            String content = readFileContent(configFile);
            JSONObject config = new JSONObject(content);
            
            // 读取配置参数
            if (config.has("do_lower_case")) {
                doLowerCase = config.getBoolean("do_lower_case");
                Log.d(TAG, "设置do_lower_case: " + doLowerCase);
            }
            
            if (config.has("model_max_length")) {
                maxLength = config.getInt("model_max_length");
                Log.d(TAG, "设置model_max_length: " + maxLength);
            }
        } catch (Exception e) {
            Log.e(TAG, "加载tokenizer配置失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 加载特殊token映射
     * @param mapFile 映射文件
     */
    private void loadSpecialTokensMap(File mapFile) {
        try {
            Log.d(TAG, "加载特殊token映射: " + mapFile.getAbsolutePath());
            String content = readFileContent(mapFile);
            JSONObject map = new JSONObject(content);
            
            // 读取特殊token
            if (map.has("cls_token")) {
                clsToken = map.getString("cls_token");
                Log.d(TAG, "设置cls_token: " + clsToken);
            }
            
            if (map.has("sep_token")) {
                sepToken = map.getString("sep_token");
                Log.d(TAG, "设置sep_token: " + sepToken);
            }
            
            if (map.has("unk_token")) {
                unkToken = map.getString("unk_token");
                Log.d(TAG, "设置unk_token: " + unkToken);
            }
            
            if (map.has("pad_token")) {
                padToken = map.getString("pad_token");
                Log.d(TAG, "设置pad_token: " + padToken);
            }
            
            if (map.has("mask_token")) {
                maskToken = map.getString("mask_token");
                Log.d(TAG, "设置mask_token: " + maskToken);
            }
        } catch (Exception e) {
            Log.e(TAG, "加载特殊token映射失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 加载tokenizer.json文件
     * @param tokenizerFile tokenizer文件
     * @return 是否加载成功
     */
    private boolean loadTokenizerJson(File tokenizerFile) {
        try {
            Log.d(TAG, "开始加载tokenizer.json: " + tokenizerFile.getAbsolutePath());
            String content = readFileContent(tokenizerFile);
            
            // 检查文件名和路径，预判模型类型
            String filePath = tokenizerFile.getAbsolutePath().toLowerCase();
            if (filePath.contains("bge-m3") || filePath.contains("bge_m3")) {
                Log.d(TAG, "检测到BGE-M3模型路径，优先使用特殊处理");
                return handleBgeM3Tokenizer(content);
            }
            
            // 打印文件内容的前200个字符，帮助调试
            Log.d(TAG, "文件内容前200个字符: " + content.substring(0, Math.min(200, content.length())));
            
            // 检查内容是否是BGE-M3模型的特殊格式
            if (content.contains("\"<s>\"") && content.contains("\"<pad>\"") && 
                content.contains("\"</s>\"") && content.contains("\"<unk>\"")) {
                Log.d(TAG, "文件内容包含BGE-M3特殊token，尝试特殊处理");
                return handleBgeM3Tokenizer(content);
            }
            
            // 检查文件内容是否为空
            if (content.isEmpty()) {
                Log.e(TAG, "tokenizer.json文件内容为空");
                return false;
            }
            
            // 尝试解析JSON对象
            try {
                JSONObject json = new JSONObject(content);
                Log.d(TAG, "成功解析为JSON对象");
                
                // 检查主要字段
                String[] fields = {"model", "added_tokens", "vocab"};
                for (String field : fields) {
                    if (json.has(field)) {
                        Log.d(TAG, "发现字段: " + field + "，类型: " + json.get(field).getClass().getSimpleName());
                    }
                }
                
                // 尝试从model.vocab字段解析词汇表
                if (json.has("model") && !json.isNull("model")) {
                    JSONObject model = json.getJSONObject("model");
                    if (model.has("vocab") && !model.isNull("vocab")) {
                        Log.d(TAG, "从model.vocab字段解析词汇表");
                        JSONObject vocabJson = model.getJSONObject("vocab");
                        
                        // 遍历词汇表
                        Iterator<String> keys = vocabJson.keys();
                        while (keys.hasNext()) {
                            String token = keys.next();
                            int id = vocabJson.getInt(token);
                            vocab.put(token, id);
                            idToToken.put(id, token);
                            
                            // 检查是否是特殊token
                            checkAndUpdateSpecialToken(token, id);
                        }
                        
                        Log.d(TAG, "成功从model.vocab解析词汇表，大小: " + vocab.size());
                        return !vocab.isEmpty();
                    }
                }
                
                // 尝试从added_tokens字段解析词汇表
                if (json.has("added_tokens") && !json.isNull("added_tokens")) {
                    Log.d(TAG, "从added_tokens字段解析词汇表");
                    JSONArray addedTokens = json.getJSONArray("added_tokens");
                    
                    for (int i = 0; i < addedTokens.length(); i++) {
                        JSONObject tokenObj = addedTokens.getJSONObject(i);
                        if (tokenObj.has("content") && tokenObj.has("id")) {
                            String token = tokenObj.getString("content");
                            int id = tokenObj.getInt("id");
                            vocab.put(token, id);
                            idToToken.put(id, token);
                            
                            // 检查是否是特殊token
                            checkAndUpdateSpecialToken(token, id);
                            
                            // 打印前几个token，帮助调试
                            if (i < 5) {
                                Log.d(TAG, "added_token #" + i + ": " + token + ", id: " + id);
                            }
                        }
                    }
                    
                    Log.d(TAG, "成功从added_tokens解析词汇表，大小: " + vocab.size());
                }
                
                // 尝试从顶层vocab字段解析词汇表
                if (json.has("vocab") && !json.isNull("vocab")) {
                    Log.d(TAG, "从顶层vocab字段解析词汇表");
                    Object vocabObj = json.get("vocab");
                    
                    if (vocabObj instanceof JSONObject) {
                        JSONObject vocabJson = (JSONObject) vocabObj;
                        
                        // 遍历词汇表
                        Iterator<String> keys = vocabJson.keys();
                        while (keys.hasNext()) {
                            String token = keys.next();
                            int id = vocabJson.getInt(token);
                            vocab.put(token, id);
                            idToToken.put(id, token);
                            
                            // 检查是否是特殊token
                            checkAndUpdateSpecialToken(token, id);
                        }
                        
                        Log.d(TAG, "成功从顶层vocab解析词汇表，大小: " + vocab.size());
                    } else if (vocabObj instanceof JSONArray) {
                        Log.d(TAG, "顶层vocab是JSONArray，尝试解析二维数组格式");
                        String vocabArrayStr = vocabObj.toString();
                        return parseVocabArrayFormat(vocabArrayStr);
                    } else {
                        Log.e(TAG, "顶层vocab既不是JSONObject也不是JSONArray，而是: " + vocabObj.getClass().getName());
                    }
                }
                
                // 尝试寻找"added_vocab_file"字段，可能包含词汇表路径
                if (json.has("added_vocab_file") && !json.isNull("added_vocab_file")) {
                    String vocabFilePath = json.getString("added_vocab_file");
                    Log.d(TAG, "发现added_vocab_file字段: " + vocabFilePath);
                    
                    // 尝试构造词汇表文件路径
                    File vocabFile = new File(tokenizerFile.getParentFile(), vocabFilePath);
                    if (vocabFile.exists() && vocabFile.isFile()) {
                        Log.d(TAG, "尝试从词汇表文件加载: " + vocabFile.getAbsolutePath());
                        String vocabContent = readFileContent(vocabFile);
                        return parseVocabArrayFormat(vocabContent);
                    }
                }
                
                // 如果以上方法都失败，尝试查找二维数组格式的词汇表
                if (vocab.isEmpty()) {
                    Log.d(TAG, "标准解析方法失败，尝试解析二维数组格式");
                    return parseVocabArrayFormat(content);
                }
                
                return !vocab.isEmpty();
            } catch (JSONException e) {
                // 打印异常详情，包括JSON的开头部分
                StringBuilder errorDetail = new StringBuilder();
                errorDetail.append("JSON解析失败: ").append(e.getMessage()).append("\n");
                errorDetail.append("JSON开头: ").append(content.substring(0, Math.min(100, content.length())));
                Log.e(TAG, errorDetail.toString());
                
                // 尝试查找Value字段，这可能是BGE-M3模型的词汇表格式
                int valueIndex = content.indexOf("Value");
                if (valueIndex != -1) {
                    Log.d(TAG, "找到Value字段，可能是BGE-M3模型的词汇表格式");
                    // 打印上下文
                    int contextStart = Math.max(0, valueIndex - 20);
                    int contextEnd = Math.min(content.length(), valueIndex + 100);
                    String contextStr = content.substring(contextStart, contextEnd);
                    Log.d(TAG, "Value字段上下文: " + contextStr);
                }
                
                // 尝试使用备用方法解析
                Log.d(TAG, "尝试解析二维数组格式");
                return parseVocabArrayFormat(content);
            }
        } catch (Exception e) {
            Log.e(TAG, "加载tokenizer.json失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 解析二维数组格式的词汇表
     * 格式为 [["token1", value1], ["token2", value2], ...]
     * @param content JSON内容
     * @return 是否解析成功
     */
    private boolean parseVocabArrayFormat(String content) {
        try {
            Log.d(TAG, "开始解析二维数组格式词汇表");
            
            // 检查内容是否为空
            if (content == null || content.isEmpty()) {
                Log.e(TAG, "内容为空");
                return false;
            }
            
            // 特殊处理bge-m3模型的tokenizer.json
            if (content.contains("[\"<s>\",0]") && content.contains("[\"<pad>\",0]") && 
                content.contains("[\"</s>\",0]") && content.contains("[\"<unk>\",0]")) {
                Log.d(TAG, "检测到bge-m3模型的特殊格式，尝试直接解析");
                
                // 尝试使用正则表达式直接提取所有token-score对
                return parseVocabWithScoresRegex(content);
            }
            
            // 检查内容是否直接是一个数组
            if (content.trim().startsWith("[") && content.trim().endsWith("]")) {
                // 尝试直接解析整个内容作为JSON数组
                try {
                    JSONArray vocabArray = new JSONArray(content);
                    Log.d(TAG, "成功解析整个内容作为JSON数组，大小: " + vocabArray.length());
                    
                    // 检查第一个元素是否也是数组
                    if (vocabArray.length() > 0) {
                        Object firstItem = vocabArray.get(0);
                        if (firstItem instanceof JSONArray) {
                            // 这是我们期望的二维数组格式
                            JSONArray firstArray = (JSONArray) firstItem;
                            if (firstArray.length() >= 2) {
                                // 检查是否包含分数
                                Object secondValue = firstArray.get(1);
                                if (secondValue instanceof Number) {
                                    double value = firstArray.getDouble(1);
                                    if (value <= 0) { // 负数分数是bge-m3的特征
                                        Log.d(TAG, "检测到词汇表使用分数而非ID，使用parseVocabWithScores处理");
                                        return parseVocabWithScores(vocabArray);
                                    }
                                }
                            }
                            return processVocabArray(vocabArray);
                        } else {
                            Log.e(TAG, "词汇表不是二维数组格式，第一个元素类型: " + firstItem.getClass().getName());
                        }
                    }
                } catch (JSONException e) {
                    Log.e(TAG, "解析整个内容作为JSON数组失败: " + e.getMessage());
                    
                    // 尝试使用正则表达式解析
                    return parseVocabWithScoresRegex(content);
                }
            }
            
            // 如果直接解析失败，尝试在内容中查找词汇表数组
            int startIndex = content.indexOf("[[");
            if (startIndex == -1) {
                // 尝试查找 "vocab": 后面的数组
                startIndex = content.indexOf("\"vocab\":");
                if (startIndex != -1) {
                    // 找到 "vocab": 后面的第一个 [
                    startIndex = content.indexOf("[", startIndex + 8);
                    if (startIndex != -1) {
                        Log.d(TAG, "在 \"vocab\": 后找到数组开始位置: " + startIndex);
                    }
                }
            }
            
            if (startIndex == -1) {
                Log.e(TAG, "找不到词汇表数组的开始位置");
                
                // 打印内容的前200个字符，帮助调试
                String preview = content.substring(0, Math.min(200, content.length()));
                Log.d(TAG, "内容前200个字符: " + preview);
                
                // 尝试使用正则表达式解析
                return parseVocabWithScoresRegex(content);
            }
            
            // 尝试使用正则表达式解析，不依赖于找到数组的结束位置
            return parseVocabWithScoresRegex(content);
        } catch (Exception e) {
            Log.e(TAG, "解析二维数组格式词汇表失败: " + e.getMessage(), e);
            
            // 最后尝试使用正则表达式解析
            try {
                return parseVocabWithScoresRegex(content);
            } catch (Exception ex) {
                Log.e(TAG, "使用正则表达式解析词汇表失败: " + ex.getMessage(), ex);
                return false;
            }
        }
    }
    
    /**
     * 使用正则表达式解析带有分数的词汇表
     * 不依赖于找到数组的开始和结束位置，直接提取所有token-score对
     * @param content 词汇表内容
     * @return 是否解析成功
     */
    private boolean parseVocabWithScoresRegex(String content) {
        try {
            Log.d(TAG, "开始使用正则表达式解析词汇表");
            
            // 创建一个列表来存储token和分数
            List<TokenScore> tokenScores = new ArrayList<>();
            
            // 检查内容是否包含可能的token-score对
            if (content.length() > 200) {
                // 打印内容的一部分，帮助调试
                String contentSample = content.substring(0, Math.min(500, content.length()));
                Log.d(TAG, "词汇表内容样本(前500字符): " + contentSample);
                
                // 打印出问题的字符或格式，帮助调试
                if (content.contains("[\"<s>\",0]")) {
                    int index = content.indexOf("[\"<s>\",0]");
                    int start = Math.max(0, index - 20);
                    int end = Math.min(content.length(), index + 40);
                    String contextStr = content.substring(start, end);
                    Log.d(TAG, "问题区域上下文: " + contextStr);
                }
            }
            
            // 使用正则表达式匹配所有形如 ["token",score] 的模式
            // 修改后的正则表达式，更灵活地匹配各种引号和格式
            Pattern pattern = Pattern.compile("\\[\\s*[\"']([^\"']+)[\"']\\s*,\\s*([-0-9.]+)\\s*\\]");
            Matcher matcher = pattern.matcher(content);
            
            int count = 0;
            while (matcher.find() && count < 100000) { // 设置上限，避免无限循环
                count++;
                String token = matcher.group(1);
                String scoreStr = matcher.group(2);
                
                try {
                    double score = Double.parseDouble(scoreStr);
                    tokenScores.add(new TokenScore(token, score));
                    
                    // 移除详细的token匹配日志
                    // Log.d(TAG, "正则匹配 #" + tokenScores.size() + ": token=" + token + ", score=" + score);
                } catch (NumberFormatException e) {
                    Log.e(TAG, "解析分数失败: " + scoreStr);
                }
            }
            
            Log.d(TAG, "正则表达式匹配到的token-score对数量: " + tokenScores.size());
            
            if (tokenScores.isEmpty()) {
                Log.e(TAG, "正则表达式没有匹配到任何token-score对");
                
                // 尝试更宽松的正则表达式，直接提取模式
                Pattern fallbackPattern = Pattern.compile("\\[(.*?),([-0-9.]+)\\]");
                Matcher fallbackMatcher = fallbackPattern.matcher(content);
                
                count = 0;
                while (fallbackMatcher.find() && count < 100) { // 只尝试匹配前100个
                    count++;
                    String tokenWithQuotes = fallbackMatcher.group(1);
                    String scoreStr = fallbackMatcher.group(2);
                    
                    // 打印匹配的原始格式，帮助调试
                    Log.d(TAG, "备用正则匹配 #" + count + ": 原始token=" + tokenWithQuotes + ", score=" + scoreStr);
                }
                
                if (count > 0) {
                    Log.d(TAG, "备用正则表达式找到了" + count + "个可能的匹配项，但格式可能不正确");
                }
                
                // 尝试手动解析JSON格式
                return parseJsonManually(content);
            }
            
            // 按分数的绝对值降序排序
            Collections.sort(tokenScores, (a, b) -> Double.compare(Math.abs(b.score), Math.abs(a.score)));
            
            // 分配ID，从0开始
            for (int i = 0; i < tokenScores.size(); i++) {
                TokenScore ts = tokenScores.get(i);
                vocab.put(ts.token, i);
                idToToken.put(i, ts.token);
                
                // 检查是否是特殊token
                checkAndUpdateSpecialToken(ts.token, i);
                
                // 移除详细的token映射日志
                // Log.d(TAG, "Token: '" + ts.token + "' -> ID: " + i);
            }
            
            Log.d(TAG, "使用正则表达式成功解析词汇表，大小: " + vocab.size());
            return !vocab.isEmpty();
        } catch (Exception e) {
            Log.e(TAG, "使用正则表达式解析词汇表失败: " + e.getMessage(), e);
            
            // 尝试手动解析JSON格式
            return parseJsonManually(content);
        }
    }
    
    /**
     * 手动解析JSON格式的词汇表
     * 处理特殊字符和转义问题
     * @param content 词汇表内容
     * @return 是否解析成功
     */
    private boolean parseJsonManually(String content) {
        try {
            Log.d(TAG, "开始手动解析JSON格式词汇表");
            
            // 定位数组开始位置
            int startIdx = content.indexOf("Value");
            if (startIdx == -1) {
                startIdx = content.indexOf("[");
            } else {
                startIdx = content.indexOf("[", startIdx);
            }
            
            if (startIdx == -1) {
                Log.e(TAG, "找不到JSON数组开始位置");
                return false;
            }
            
            // 尝试分析数据格式
            analyzeDataFormat(content, startIdx);
            
            // 尝试处理可能是二维数组的格式
            List<TokenScore> tokenScores = new ArrayList<>();
            StringBuilder currentToken = new StringBuilder();
            double currentScore = 0;
            boolean inToken = false;
            boolean expectingScore = false;
            boolean inQuotes = false;
            char lastChar = 0;
            
            for (int i = startIdx; i < Math.min(content.length(), startIdx + 10000); i++) {
                char c = content.charAt(i);
                
                if (c == '[' && !inQuotes) {
                    // 开始一个新的记录
                    inToken = false;
                    expectingScore = false;
                    currentToken.setLength(0);
                } else if (c == '\"' && (lastChar != '\\' || (lastChar == '\\' && content.charAt(i-2) == '\\'))) {
                    // 引号处理，注意转义
                    inQuotes = !inQuotes;
                    if (!inQuotes && !inToken) {
                        inToken = true;
                    }
                } else if (c == ',' && !inQuotes) {
                    // 分隔符
                    if (inToken) {
                        inToken = false;
                        expectingScore = true;
                    }
                } else if (c == ']' && !inQuotes) {
                    // 一个记录结束
                    if (expectingScore && currentToken.length() > 0) {
                        try {
                            TokenScore ts = new TokenScore(currentToken.toString(), currentScore);
                            tokenScores.add(ts);
                            
                            // 移除详细的token映射日志
                            // Log.d(TAG, "手动解析 #" + tokenScores.size() + ": token=" + ts.token + ", score=" + ts.score);
                        } catch (Exception e) {
                            Log.e(TAG, "处理token-score对时发生错误: " + e.getMessage());
                        }
                    }
                    inToken = false;
                    expectingScore = false;
                    currentToken.setLength(0);
                } else if (inToken && inQuotes) {
                    // 收集token文本
                    currentToken.append(c);
                } else if (expectingScore && Character.isDigit(c) || c == '-' || c == '.') {
                    // 解析score
                    try {
                        // 尝试从当前位置解析一个数字
                        int j = i;
                        while (j < content.length() && (Character.isDigit(content.charAt(j)) || content.charAt(j) == '-' || content.charAt(j) == '.')) {
                            j++;
                        }
                        String scoreStr = content.substring(i, j);
                        currentScore = Double.parseDouble(scoreStr);
                        i = j - 1; // 跳过已解析的数字部分
                    } catch (NumberFormatException e) {
                        Log.e(TAG, "解析分数失败");
                    }
                }
                
                lastChar = c;
            }
            
            Log.d(TAG, "手动解析出的token-score对数量: " + tokenScores.size());
            
            if (tokenScores.isEmpty()) {
                Log.e(TAG, "手动解析没有提取到任何token-score对");
                return false;
            }
            
            // 按分数的绝对值降序排序
            Collections.sort(tokenScores, (a, b) -> Double.compare(Math.abs(b.score), Math.abs(a.score)));
            
            // 分配ID，从0开始
            for (int i = 0; i < tokenScores.size(); i++) {
                TokenScore ts = tokenScores.get(i);
                vocab.put(ts.token, i);
                idToToken.put(i, ts.token);
                
                // 检查是否是特殊token
                checkAndUpdateSpecialToken(ts.token, i);
            }
            
            Log.d(TAG, "手动解析成功，词汇表大小: " + vocab.size());
            return !vocab.isEmpty();
        } catch (Exception e) {
            Log.e(TAG, "手动解析JSON格式词汇表失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 分析数据格式，打印出详细的格式信息，帮助调试
     * @param content 要分析的内容
     * @param startIdx 开始索引
     */
    private void analyzeDataFormat(String content, int startIdx) {
        try {
            Log.d(TAG, "开始分析数据格式，从索引 " + startIdx + " 开始");
            
            // 查找第一个完整的token-score对
            int firstPairStart = content.indexOf("[[", startIdx);
            if (firstPairStart != -1) {
                int firstPairEnd = content.indexOf("]]", firstPairStart);
                if (firstPairEnd != -1 && firstPairEnd - firstPairStart < 100) {
                    String firstPair = content.substring(firstPairStart, firstPairEnd + 2);
                    Log.d(TAG, "第一个完整的token对: " + firstPair);
                }
            }
            
            // 1. 分析前几个特殊token
            String[] specialTokens = {"<s>", "<pad>", "</s>", "<unk>"};
            for (String token : specialTokens) {
                // 查找token在内容中的位置和上下文
                String tokenWithQuotes = "\"" + token + "\"";
                int tokenIdx = content.indexOf(tokenWithQuotes, startIdx);
                if (tokenIdx != -1) {
                    int contextStart = Math.max(0, tokenIdx - 10);
                    int contextEnd = Math.min(content.length(), tokenIdx + tokenWithQuotes.length() + 10);
                    String context = content.substring(contextStart, contextEnd);
                    
                    // 打印token的详细信息，包括编码和十六进制表示
                    StringBuilder hexStr = new StringBuilder();
                    for (int i = 0; i < context.length(); i++) {
                        char c = context.charAt(i);
                        hexStr.append(String.format("\\u%04x ", (int)c));
                    }
                    
                    Log.d(TAG, "特殊token \"" + token + "\" 上下文: " + context);
                    Log.d(TAG, "特殊token \"" + token + "\" 十六进制表示: " + hexStr.toString());
                } else {
                    Log.d(TAG, "未找到特殊token: " + token);
                }
            }
            
            // 2. 查找问题组合
            String[] problematicPatterns = {
                "[\"<s>\",0]", 
                "[\"<pad>\",0]",
                "[\"</s>\",0]",
                "[\"<unk>\",0]"
            };
            
            for (String pattern : problematicPatterns) {
                // 字符级分析
                Log.d(TAG, "模式 \"" + pattern + "\" 的字符级分析:");
                StringBuilder charAnalysis = new StringBuilder();
                for (int i = 0; i < pattern.length(); i++) {
                    char c = pattern.charAt(i);
                    charAnalysis.append("位置 ").append(i).append(": '").append(c)
                               .append("' (ASCII: ").append((int)c).append(", Hex: ")
                               .append(String.format("\\u%04x", (int)c)).append(")\n");
                }
                Log.d(TAG, charAnalysis.toString());
                
                // 检查在内容中的出现情况
                int patternIdx = content.indexOf(pattern, startIdx);
                if (patternIdx != -1) {
                    int contextStart = Math.max(0, patternIdx - 15);
                    int contextEnd = Math.min(content.length(), patternIdx + pattern.length() + 15);
                    String context = content.substring(contextStart, contextEnd);
                    Log.d(TAG, "找到模式 \"" + pattern + "\" 在位置 " + patternIdx + ", 上下文: " + context);
                    
                    // 尝试匹配不同的正则表达式
                    String[] regexPatterns = {
                        "\\[\\s*\"([^\"]+)\"\\s*,\\s*([-0-9.]+)\\s*\\]",
                        "\\[\\s*'([^']+)'\\s*,\\s*([-0-9.]+)\\s*\\]",
                        "\\[([^,]+),([-0-9.]+)\\]"
                    };
                    
                    for (String regex : regexPatterns) {
                        try {
                            Pattern p = Pattern.compile(regex);
                            Matcher m = p.matcher(context);
                            if (m.find()) {
                                Log.d(TAG, "正则 \"" + regex + "\" 匹配成功: 组1=" + m.group(1) + ", 组2=" + m.group(2));
                            } else {
                                Log.d(TAG, "正则 \"" + regex + "\" 匹配失败");
                            }
                        } catch (Exception e) {
                            Log.e(TAG, "正则匹配过程中出错: " + e.getMessage());
                        }
                    }
                } else {
                    Log.d(TAG, "未找到模式: " + pattern);
                }
            }
            
            // 3. 查找特定的正则表达式匹配问题
            String simplePattern = "\\[(.*?),([-0-9.]+)\\]";
            Pattern p = Pattern.compile(simplePattern);
            Matcher m = p.matcher(content.substring(startIdx, Math.min(content.length(), startIdx + 500)));
            
            int count = 0;
            while (m.find() && count < 5) {
                count++;
                String token = m.group(1);
                String score = m.group(2);
                Log.d(TAG, "简单正则匹配 #" + count + ": token部分=" + token + ", score部分=" + score);
                
                // 详细分析token部分
                StringBuilder tokenAnalysis = new StringBuilder();
                for (int i = 0; i < token.length(); i++) {
                    char c = token.charAt(i);
                    tokenAnalysis.append("位置 ").append(i).append(": '").append(c)
                               .append("' (ASCII: ").append((int)c).append(")\n");
                }
                Log.d(TAG, "token部分字符分析:\n" + tokenAnalysis.toString());
            }
        } catch (Exception e) {
            Log.e(TAG, "分析数据格式时出错: " + e.getMessage(), e);
        }
    }
    
    /**
     * 处理词汇表数组
     * @param vocabArray 词汇表数组
     * @return 是否处理成功
     */
    private boolean processVocabArray(JSONArray vocabArray) {
        try {
            // 检查第一个元素，判断是否包含分数而不是ID
            if (vocabArray.length() > 0) {
                JSONArray firstItem = vocabArray.getJSONArray(0);
                if (firstItem.length() >= 2) {
                    // 检查第二个元素是否为负数或小数，如果是则表示这是分数而不是ID
                    Object secondElement = firstItem.get(1);
                    if (secondElement instanceof Double || 
                        (secondElement instanceof Integer && (Integer)secondElement < 0)) {
                        Log.d(TAG, "检测到词汇表使用分数而非ID，使用parseVocabWithScores方法");
                        return parseVocabWithScores(vocabArray);
                    }
                }
            }
            
            // 标准解析：每个元素是[token, id]
            for (int i = 0; i < vocabArray.length(); i++) {
                JSONArray item = vocabArray.getJSONArray(i);
                if (item.length() >= 2) {
                    String token = item.getString(0);
                    int id = item.getInt(1);
                    
                    vocab.put(token, id);
                    idToToken.put(id, token);
                    
                    // 检查是否是特殊token
                    checkAndUpdateSpecialToken(token, id);
                }
            }
            
            Log.d(TAG, "成功解析二维数组格式词汇表，大小: " + vocab.size());
            return !vocab.isEmpty();
        } catch (Exception e) {
            Log.e(TAG, "处理词汇表数组失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 逐行解析词汇表数组
     * @param arrayStr 词汇表数组字符串
     * @return 是否解析成功
     */
    private boolean parseArrayLineByLine(String arrayStr) {
        try {
            Log.d(TAG, "开始逐行解析词汇表数组");
            
            // 移除开头和结尾的方括号
            String content = arrayStr.trim();
            if (content.startsWith("[[")) {
                content = content.substring(2);
            }
            if (content.endsWith("]]")) {
                content = content.substring(0, content.length() - 2);
            }
            
            // 按行分割
            String[] lines = content.split("\\],\\s*\\[");
            Log.d(TAG, "分割后的行数: " + lines.length);
            
            // 检查第一行，判断是否包含分数而不是ID
            if (lines.length > 0) {
                String firstLine = lines[0];
                if (firstLine.contains(",")) {
                    String[] parts = firstLine.split(",", 2);
                    if (parts.length >= 2) {
                        String valueStr = parts[1].trim();
                        // 移除引号
                        if (valueStr.startsWith("\"") || valueStr.startsWith("'")) {
                            valueStr = valueStr.substring(1);
                        }
                        if (valueStr.endsWith("\"") || valueStr.endsWith("'")) {
                            valueStr = valueStr.substring(0, valueStr.length() - 1);
                        }
                        
                        try {
                            double value = Double.parseDouble(valueStr);
                            if (value < 0 || value != Math.floor(value)) {
                                Log.d(TAG, "检测到词汇表使用分数而非ID，尝试手动解析带分数的词汇表");
                                return parseVocabWithScoresManually(lines);
                            }
                        } catch (NumberFormatException e) {
                            // 忽略解析错误，继续标准解析
                        }
                    }
                }
            }
            
            // 标准解析：每行是[token, id]
            int validEntries = 0;
            for (String line : lines) {
                try {
                    // 清理行
                    line = line.trim();
                    if (line.startsWith("[")) {
                        line = line.substring(1);
                    }
                    if (line.endsWith("]")) {
                        line = line.substring(0, line.length() - 1);
                    }
                    
                    // 分割token和id
                    String[] parts = line.split(",", 2);
                    if (parts.length >= 2) {
                        // 解析token
                        String token = parts[0].trim();
                        if (token.startsWith("\"") || token.startsWith("'")) {
                            token = token.substring(1);
                        }
                        if (token.endsWith("\"") || token.endsWith("'")) {
                            token = token.substring(0, token.length() - 1);
                        }
                        
                        // 解析id
                        String idStr = parts[1].trim();
                        try {
                            int id = Integer.parseInt(idStr);
                            
                            vocab.put(token, id);
                            idToToken.put(id, token);
                            validEntries++;
                            
                            // 检查是否是特殊token
                            checkAndUpdateSpecialToken(token, id);
                        } catch (NumberFormatException e) {
                            Log.e(TAG, "解析ID失败: " + idStr);
                        }
                    }
                } catch (Exception e) {
                    Log.e(TAG, "解析行失败: " + line + ", 错误: " + e.getMessage());
                }
            }
            
            Log.d(TAG, "逐行解析完成，有效条目数: " + validEntries);
            return validEntries > 0;
        } catch (Exception e) {
            Log.e(TAG, "逐行解析词汇表数组失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 解析带有分数的词汇表
     * 格式为 [["token1", score1], ["token2", score2], ...]
     * 其中score可能是负数或小数
     * @param vocabArray 词汇表数组
     * @return 是否解析成功
     */
    private boolean parseVocabWithScores(JSONArray vocabArray) {
        try {
            Log.d(TAG, "开始解析带分数的词汇表，大小: " + vocabArray.length());
            
            // 创建一个列表来存储token和分数
            List<TokenScore> tokenScores = new ArrayList<>();
            
            // 收集所有token和分数
            for (int i = 0; i < vocabArray.length(); i++) {
                try {
                    JSONArray item = vocabArray.getJSONArray(i);
                    if (item.length() >= 2) {
                        String token = item.getString(0);
                        double score;
                        
                        // 分数可能是整数或小数
                        try {
                            score = item.getDouble(1);
                        } catch (JSONException e) {
                            // 如果无法解析为double，尝试解析为int
                            score = item.getInt(1);
                        }
                        
                        tokenScores.add(new TokenScore(token, score));
                    }
                } catch (Exception e) {
                    Log.e(TAG, "解析token-score对失败，索引: " + i + ", 错误: " + e.getMessage());
                }
            }
            
            Log.d(TAG, "收集到的token-score对数量: " + tokenScores.size());
            
            if (tokenScores.isEmpty()) {
                Log.e(TAG, "没有有效的token-score对");
                return false;
            }
            
            // 按分数的绝对值降序排序
            Collections.sort(tokenScores, (a, b) -> Double.compare(Math.abs(b.score), Math.abs(a.score)));
            
            // 分配ID，从0开始
            for (int i = 0; i < tokenScores.size(); i++) {
                TokenScore ts = tokenScores.get(i);
                vocab.put(ts.token, i);
                idToToken.put(i, ts.token);
                
                // 检查是否是特殊token
                checkAndUpdateSpecialToken(ts.token, i);
                
                // 移除详细的token映射日志
                // Log.d(TAG, "Token: '" + ts.token + "' -> ID: " + i);
            }
            
            Log.d(TAG, "成功解析带分数的词汇表，大小: " + vocab.size());
            return !vocab.isEmpty();
        } catch (Exception e) {
            Log.e(TAG, "解析带分数的词汇表失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 手动解析带有分数的词汇表
     * @param lines 分割后的行
     * @return 是否解析成功
     */
    private boolean parseVocabWithScoresManually(String[] lines) {
        try {
            Log.d(TAG, "开始手动解析带分数的词汇表，行数: " + lines.length);
            
            // 创建一个列表来存储token和分数
            List<TokenScore> tokenScores = new ArrayList<>();
            
            // 收集所有token和分数
            for (String line : lines) {
                try {
                    // 清理行
                    line = line.trim();
                    if (line.startsWith("[")) {
                        line = line.substring(1);
                    }
                    if (line.endsWith("]")) {
                        line = line.substring(0, line.length() - 1);
                    }
                    
                    // 分割token和分数
                    String[] parts = line.split(",", 2);
                    if (parts.length >= 2) {
                        // 解析token
                        String token = parts[0].trim();
                        if (token.startsWith("\"") || token.startsWith("'")) {
                            token = token.substring(1);
                        }
                        if (token.endsWith("\"") || token.endsWith("'")) {
                            token = token.substring(0, token.length() - 1);
                        }
                        
                        // 解析分数
                        String scoreStr = parts[1].trim();
                        try {
                            double score = Double.parseDouble(scoreStr);
                            tokenScores.add(new TokenScore(token, score));
                        } catch (NumberFormatException e) {
                            Log.e(TAG, "解析分数失败: " + scoreStr);
                        }
                    }
                } catch (Exception e) {
                    Log.e(TAG, "解析行失败: " + line + ", 错误: " + e.getMessage());
                }
            }
            
            Log.d(TAG, "收集到的token-score对数量: " + tokenScores.size());
            
            if (tokenScores.isEmpty()) {
                Log.e(TAG, "没有有效的token-score对");
                return false;
            }
            
            // 按分数的绝对值降序排序
            Collections.sort(tokenScores, (a, b) -> Double.compare(Math.abs(b.score), Math.abs(a.score)));
            
            // 分配ID，从0开始
            for (int i = 0; i < tokenScores.size(); i++) {
                TokenScore ts = tokenScores.get(i);
                vocab.put(ts.token, i);
                idToToken.put(i, ts.token);
                
                // 检查是否是特殊token
                checkAndUpdateSpecialToken(ts.token, i);
                
                // 移除详细的token映射日志
                // Log.d(TAG, "Token: '" + ts.token + "' -> ID: " + i);
            }
            
            Log.d(TAG, "手动解析带分数的词汇表成功，大小: " + vocab.size());
            return !vocab.isEmpty();
        } catch (Exception e) {
            Log.e(TAG, "手动解析带分数的词汇表失败: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Token和分数的辅助类
     */
    private static class TokenScore {
        String token;
        double score;
        
        TokenScore(String token, double score) {
            this.token = token;
            this.score = score;
        }
    }
    
    /**
     * 检查并更新特殊token的ID
     * @param token token字符串
     * @param id token的ID
     */
    private void checkAndUpdateSpecialToken(String token, int id) {
        if (token.equals(clsToken)) {
            clsTokenId = id;
            Log.d(TAG, "更新CLS token ID: " + id);
        } else if (token.equals(sepToken)) {
            sepTokenId = id;
            Log.d(TAG, "更新SEP token ID: " + id);
        } else if (token.equals(unkToken)) {
            unkTokenId = id;
            Log.d(TAG, "更新UNK token ID: " + id);
        } else if (token.equals(padToken)) {
            padTokenId = id;
            Log.d(TAG, "更新PAD token ID: " + id);
        } else if (token.equals(maskToken)) {
            maskTokenId = id;
            Log.d(TAG, "更新MASK token ID: " + id);
        }
    }

    /**
     * 基本分词方法
     * @param text 输入文本
     * @return 分词结果
     */
    private List<String> basicTokenize(String text) {
        List<String> tokens = new ArrayList<>();
        
        // BGE-M3使用的是XLM-RoBERTa tokenizer
        // 它会处理常见的中文词和分词
        
        // 尝试使用SentencePiece风格的分词
        // 为中文文本，我们尝试直接在词汇表中查找每个可能的子字符串
        StringBuilder currentToken = new StringBuilder();
        
        // 开始分词，先添加空格前缀
        if (text.charAt(0) != ' ') {
            currentToken.append('▁');
        }
        
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            currentToken.append(c);
            String token = currentToken.toString();
            
            // 检查词汇表中是否有这个token
            if (vocab.containsKey(token)) {
                tokens.add(token);
                currentToken = new StringBuilder();
                if (c != ' ') {
                    currentToken.append('▁');
                }
            } else {
                // 如果不是完整token，继续尝试
                // 如果已经积累了多个字符但仍未找到，使用单字符分词
                if (currentToken.length() > 3) { // 包括可能的▁前缀和至少2个字符
                    // 回退一个字符
                    currentToken.deleteCharAt(currentToken.length() - 1);
                    String prevToken = currentToken.toString();
                    if (vocab.containsKey(prevToken)) {
                        tokens.add(prevToken);
                    } else {
                        // 未能找到合适的token，使用单字符
                        if (i > 0) i--; // 回退一个字符重新处理
                        if (currentToken.length() > 0) {
                            for (int j = 0; j < currentToken.length(); j++) {
                                String charToken = String.valueOf(currentToken.charAt(j));
                                if (vocab.containsKey(charToken)) {
                                    tokens.add(charToken);
                                } else {
                                    tokens.add(unkToken);
                                }
                            }
                        }
                    }
                    currentToken = new StringBuilder("▁");
                }
            }
        }
        
        // 处理最后剩余的token
        if (currentToken.length() > 0) {
            String token = currentToken.toString();
            if (vocab.containsKey(token)) {
                tokens.add(token);
            } else {
                // 单字符分词
                for (int j = 0; j < currentToken.length(); j++) {
                    String charToken = String.valueOf(currentToken.charAt(j));
                    if (vocab.containsKey(charToken)) {
                        tokens.add(charToken);
                    } else {
                        tokens.add(unkToken);
                    }
                }
            }
        }
        
        return tokens;
    }
    
    /**
     * 对文本进行分词
     * @param text 输入文本
     * @return 分词结果（token ID列表）
     */
    public long[][] tokenize(String text) {
        List<Integer> tokenIds = new ArrayList<>();
        
        try {
            // 详细记录输入文本
            Log.d(TAG, "分词前提示词: '" + text + "', 长度: " + text.length());
            
            // 添加起始标记 (对应PC端的<s>)
            tokenIds.add(clsTokenId);
            Log.d(TAG, "添加起始标记: " + clsTokenId);
            
            // 打印词汇表样本
            if (vocab.size() > 0) {
                StringBuilder vocabSample = new StringBuilder();
                int count = 0;
                for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
                    if (count < 20) {
                        vocabSample.append("'").append(entry.getKey()).append("'=").append(entry.getValue()).append(", ");
                        count++;
                    } else {
                        break;
                    }
                }
                Log.d(TAG, "词汇表样本: " + vocabSample.toString());
            } else {
                Log.e(TAG, "词汇表为空！");
            }
            
            // 处理文本
            if (text != null && !text.isEmpty()) {
                // 如果需要，转换为小写
                if (doLowerCase) {
                    text = text.toLowerCase();
                    // Log.d(TAG, "转换为小写: " + text);
                }
                
                // 记录词汇表信息
                // Log.d(TAG, "词汇表大小: " + vocab.size());
                
                  // 根据模型类型选择分词策略
                if (isQwen3Model) {
                    Log.d(TAG, "使用 Qwen3 分词策略");
                    List<String> tokens = tokenizeQwen3(text);
                    
                    StringBuilder tokenDebug = new StringBuilder();
                    for (String token : tokens) {
                        tokenDebug.append("'").append(token).append("' ");
                    }
                    Log.d(TAG, "分词结果 (" + tokens.size() + " tokens): " + tokenDebug.toString());
                    
                    // 将token转换为ID
                    for (String token : tokens) {
                        if (vocab.containsKey(token)) {
                            int id = vocab.get(token);
                            tokenIds.add(id);
                            Log.d(TAG, "找到 token: '" + token + "', ID: " + id);
                        } else {
                            Log.d(TAG, "未知 token: '" + token + "', 使用 UNK ID: " + unkTokenId);
                            tokenIds.add(unkTokenId);
                        }
                    }
                } else if (useConsistentTokenization) {
                    // Log.d(TAG, "使用一致性分词策略");
                    
                    // 使用优化的最长匹配算法
                    List<String> tokens = longestMatchTokenize(text);
                    
                    // 移除分词结果日志
                    // StringBuilder tokenDebug = new StringBuilder();
                    // for (String token : tokens) {
                    //     tokenDebug.append("'").append(token).append("' ");
                    // }
                    // Log.d(TAG, "分词结果 (" + tokens.size() + " tokens): " + tokenDebug.toString());
                    
                    // 将token转换为ID
                    for (String token : tokens) {
                        if (vocab.containsKey(token)) {
                            int id = vocab.get(token);
                            tokenIds.add(id);
                        } else {
                            // 对于未知token，尝试查找前缀
                            if (token.startsWith("▁") && token.length() > 1) {
                                String withoutPrefix = token.substring(1);
                                if (vocab.containsKey(withoutPrefix)) {
                                    int id = vocab.get(withoutPrefix);
                                    tokenIds.add(id);
                                }
                            }
                            
                            // 如果是中文字符，尝试添加前缀再查找
                            if (token.length() == 1 && isCJKChar(token.charAt(0))) {
                                String withPrefix = "▁" + token;
                                if (vocab.containsKey(withPrefix)) {
                                    int id = vocab.get(withPrefix);
                                    tokenIds.add(id);
                                }
                            }
                            
                            // 未知token
                            // Log.d(TAG, "未知token: '" + token + "', 使用UNK ID: " + unkTokenId);
                            tokenIds.add(unkTokenId);
                        }
                    }
                }
                // 对于BGE-M3模型，使用特殊的分词方法
                else if (vocab.size() > 50000) { // BGE-M3词汇表大小约为250002
                    // Log.d(TAG, "检测到大型词汇表，使用BGE-M3分词方法");
                    
                    // 模拟SentencePiece分词
                    // 首先添加前缀空格，这是XLM-RoBERTa tokenizer的特性
                    String processedText = text;
                    if (!text.startsWith(" ")) {
                        processedText = " " + text;
                        // Log.d(TAG, "添加空格前缀: " + processedText);
                    }
                    
                    // 使用改进的分词算法
                    List<String> tokens = new ArrayList<>();
                    int position = 0;
                    
                    while (position < processedText.length()) {
                        // 尝试找到最长匹配的token
                        String longestToken = null;
                        int longestLength = 0;
                        
                        // 对于当前位置，尝试不同长度的子字符串
                        for (int endPos = position + 1; endPos <= processedText.length() && endPos <= position + 10; endPos++) {
                            String substring = processedText.substring(position, endPos);
                            
                            // 检查原始子字符串
                            if (vocab.containsKey(substring) && substring.length() > longestLength) {
                                longestToken = substring;
                                longestLength = substring.length();
                            }
                            
                            // 检查添加▁前缀的子字符串（对于非空格开头的token）
                            if (!substring.startsWith(" ") && !substring.startsWith("▁")) {
                                String withPrefix = "▁" + substring;
                                if (vocab.containsKey(withPrefix) && withPrefix.length() > longestLength) {
                                    longestToken = withPrefix;
                                    longestLength = withPrefix.length();
                                }
                            }
                        }
                        
                        // 如果找到匹配的token
                        if (longestToken != null) {
                            tokens.add(longestToken);
                            position += longestLength;
                            if (longestToken.startsWith("▁") && longestToken.length() > 1) {
                                position -= 1; // 调整位置，因为▁前缀不在原始文本中
                            }
                            // 移除token日志
                            // Log.d(TAG, "找到token: '" + longestToken + "', 长度: " + longestLength);
                        } else {
                            // 如果当前字符是CJK字符，单独处理
                            char currentChar = processedText.charAt(position);
                            if (isCJKChar(currentChar)) {
                                String cjkToken = String.valueOf(currentChar);
                                
                                // 尝试直接查找
                                if (vocab.containsKey(cjkToken)) {
                                    tokens.add(cjkToken);
                                    // 移除token日志
                                    // Log.d(TAG, "找到CJK token: '" + cjkToken + "'");
                                } else {
                                    // 尝试添加▁前缀
                                    String withPrefix = "▁" + cjkToken;
                                    if (vocab.containsKey(withPrefix)) {
                                        tokens.add(withPrefix);
                                        // 移除token日志
                                        // Log.d(TAG, "找到带前缀的CJK token: '" + withPrefix + "'");
                                    } else {
                                        // 未知token
                                        // Log.d(TAG, "未知CJK字符: '" + cjkToken + "', 使用UNK");
                                        tokens.add(unkToken);
                                    }
                                }
                            } else if (currentChar == ' ') {
                                // 空格处理
                                String spaceToken = "▁";
                                if (vocab.containsKey(spaceToken)) {
                                    tokens.add(spaceToken);
                                    // 移除token日志
                                    // Log.d(TAG, "找到空格token: '" + spaceToken + "'");
                                }
                            } else {
                                // 其他未知字符
                                String charToken = String.valueOf(currentChar);
                                // Log.d(TAG, "未知字符: '" + charToken + "', 使用UNK");
                                tokens.add(unkToken);
                            }
                            position++;
                        }
                    }
                    
                    // 移除分词结果日志
                    // StringBuilder tokenDebug = new StringBuilder();
                    // for (String token : tokens) {
                    //     tokenDebug.append("'").append(token).append("' ");
                    // }
                    // Log.d(TAG, "分词结果 (" + tokens.size() + " tokens): " + tokenDebug.toString());
                    
                    // 将token转换为ID
                    for (String token : tokens) {
                        if (vocab.containsKey(token)) {
                            int id = vocab.get(token);
                            tokenIds.add(id);
                        } else {
                            // 对于未知token，尝试查找前缀
                            if (token.startsWith("▁") && token.length() > 1) {
                                String withoutPrefix = token.substring(1);
                                if (vocab.containsKey(withoutPrefix)) {
                                    int id = vocab.get(withoutPrefix);
                                    tokenIds.add(id);
                                }
                            }
                            
                            // 如果是中文字符，尝试添加前缀再查找
                            if (token.length() == 1 && isCJKChar(token.charAt(0))) {
                                String withPrefix = "▁" + token;
                                if (vocab.containsKey(withPrefix)) {
                                    int id = vocab.get(withPrefix);
                                    tokenIds.add(id);
                                }
                            }
                            
                            // 未知token
                            // Log.d(TAG, "未知token: '" + token + "', 使用UNK ID: " + unkTokenId);
                            tokenIds.add(unkTokenId);
                        }
                    }
                } else {
                    // 使用基本分词
                    // Log.d(TAG, "使用基本分词方法");
                    List<String> tokens = basicTokenize(text);
                    for (String token : tokens) {
                        if (vocab.containsKey(token)) {
                            tokenIds.add(vocab.get(token));
                        } else {
                            // 未知token
                            tokenIds.add(unkTokenId);
                        }
                    }
                }
                
                // 移除分词结果日志
                // StringBuilder tokenIdDebug = new StringBuilder();
                // for (int i = 0; i < Math.min(tokenIds.size(), 30); i++) {
                //     int id = tokenIds.get(i);
                //     String token = idToToken.getOrDefault(id, "<unknown>");
                //     tokenIdDebug.append(id).append("(").append(token).append(") ");
                // }
                // Log.d(TAG, "分词结果 (" + tokenIds.size() + " tokens): " + tokenIdDebug.toString());
            }
            
            // 添加结束标记 (对应PC端的</s>)
            tokenIds.add(sepTokenId);
            // Log.d(TAG, "添加结束标记: " + sepTokenId);
            
            // 确保不超过最大长度
            if (tokenIds.size() > maxLength) {
                // Log.d(TAG, "截断序列长度从 " + tokenIds.size() + " 到 " + maxLength);
                tokenIds = tokenIds.subList(0, maxLength);
            }
        } catch (Exception e) {
            Log.e(TAG, "分词过程发生异常: " + e.getMessage(), e);
            // 返回最基本的token序列
            tokenIds.clear();
            tokenIds.add(clsTokenId);
            tokenIds.add(sepTokenId);
        }
        
        // 转换为long[][]格式
        int seqLength = tokenIds.size();
        long[] inputIds = new long[seqLength];
        long[] attentionMask = new long[seqLength];
        
        // 填充inputIds和attentionMask
        for (int i = 0; i < seqLength; i++) {
            inputIds[i] = tokenIds.get(i);
            attentionMask[i] = 1; // 所有token都是有效的，注意力掩码为1
        }
        
        // 返回格式为[inputIds, attentionMask]的二维数组
        return new long[][] { inputIds, attentionMask };
    }
    
    private List<String> tokenizeQwen3(String text) {
        List<String> tokens = new ArrayList<>();
        int position = 0;
        
        Log.d(TAG, "开始 Qwen3 分词处理");
        
        while (position < text.length()) {
            boolean matched = false;
            
            // 首先检查特殊 token
            for (String specialToken : QWEN3_SPECIAL_TOKENS) {
                if (text.startsWith(specialToken, position)) {
                    tokens.add(specialToken);
                    position += specialToken.length();
                    matched = true;
                    Log.d(TAG, "匹配到特殊 token: " + specialToken + " 在位置 " + (position - specialToken.length()));
                    break;
                }
            }
            
            if (matched) {
                continue;
            }
            
            // 尝试最长匹配
            String longestToken = null;
            int longestLength = 0;
            
            for (int endPos = position + 1; endPos <= text.length() && endPos <= position + 10; endPos++) {
                String substring = text.substring(position, endPos);
                if (vocab.containsKey(substring) && substring.length() > longestLength) {
                    longestToken = substring;
                    longestLength = substring.length();
                }
                
                // 尝试添加前缀
                if (!substring.startsWith("▁")) {
                    String withPrefix = "▁" + substring;
                    if (vocab.containsKey(withPrefix) && withPrefix.length() > longestLength) {
                        longestToken = withPrefix;
                        longestLength = withPrefix.length();
                    }
                }
            }
            
            if (longestToken != null) {
                tokens.add(longestToken);
                position += longestToken.length();
                if (longestToken.startsWith("▁") && longestToken.length() > 1) {
                    position -= 1; // 调整位置，因为▁前缀不在原始文本中
                }
                Log.d(TAG, "匹配到 token: " + longestToken + " 在位置 " + (position - longestToken.length()));
            } else {
                // 处理单个字符
                char currentChar = text.charAt(position);
                String charToken = String.valueOf(currentChar);
                
                if (isCJKChar(currentChar)) {
                    if (vocab.containsKey(charToken)) {
                        tokens.add(charToken);
                        Log.d(TAG, "匹配到 CJK 字符: " + charToken + " 在位置 " + position);
                    } else {
                        String withPrefix = "▁" + charToken;
                        if (vocab.containsKey(withPrefix)) {
                            tokens.add(withPrefix);
                            Log.d(TAG, "匹配到带前缀的 CJK 字符: " + withPrefix + " 在位置 " + position);
                        } else {
                            tokens.add(unkToken);
                            Log.d(TAG, "未知 CJK 字符: " + charToken + " 在位置 " + position + ", 使用 UNK");
                        }
                    }
                } else {
                    tokens.add(unkToken);
                    Log.d(TAG, "未知字符: " + charToken + " 在位置 " + position + ", 使用 UNK");
                }
                position++;
            }
        }
        
        return tokens;
    }


    /**
     * 判断字符是否为CJK字符（中日韩统一表意文字）
     * @param c 要检查的字符
     * @return 是否为CJK字符
     */
    private boolean isCJKChar(char c) {
        Character.UnicodeBlock block = Character.UnicodeBlock.of(c);
        return block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS
                || block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A
                || block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B
                || block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_C
                || block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_D
                || block == Character.UnicodeBlock.CJK_COMPATIBILITY_IDEOGRAPHS
                || block == Character.UnicodeBlock.CJK_COMPATIBILITY_IDEOGRAPHS_SUPPLEMENT
                || block == Character.UnicodeBlock.CJK_SYMBOLS_AND_PUNCTUATION
                || block == Character.UnicodeBlock.HALFWIDTH_AND_FULLWIDTH_FORMS;
    }
    
    /**
     * 将token ID转换回token
     * @param tokenId token ID
     * @return token字符串
     */
    public String convertIdToToken(int tokenId) {
        String token = idToToken.getOrDefault(tokenId, unkToken);
        // 移除详细的token映射日志
        // Log.d(TAG, "转换ID到Token: " + tokenId + " -> '" + token + "'");
        return token;
    }

    /**
     * 获取特定ID范围内的token映射
     * @param startId 起始ID（包含）
     * @param endId 结束ID（包含）
     * @return token到ID的映射
     */
    public Map<String, Integer> getTokensByIdRange(int startId, int endId) {
        Map<String, Integer> tokenMap = new HashMap<>();
        for (int id = startId; id <= endId; id++) {
            String token = idToToken.getOrDefault(id, null);
            if (token != null) {
                tokenMap.put(token, id);
            }
        }
        Log.d(TAG, "获取ID范围 " + startId + "-" + endId + " 的token映射，共 " + tokenMap.size() + " 个token");
        return tokenMap;
    }
    /**
     * 获取名称匹配特定模式的token映射
     * @param pattern 名称模式（正则表达式）
     * @return token到ID的映射
     */
    public Map<String, Integer> getTokensByNamePattern(String pattern) {
        Map<String, Integer> tokenMap = new HashMap<>();
        Pattern regex = Pattern.compile(pattern);
        
        for (Map.Entry<Integer, String> entry : idToToken.entrySet()) {
            String token = entry.getValue();
            if (token != null && regex.matcher(token).matches()) {
                tokenMap.put(token, entry.getKey());
            }
        }
        
    Log.d(TAG, "获取名称匹配模式 '" + pattern + "' 的token映射，共 " + tokenMap.size() + " 个token");
    return tokenMap;
}
    /**
     * 将token ID列表转换为文本
     * @param tokenIds token ID列表
     * @return 解码后的文本
     */
    public String decode(List<Integer> tokenIds) {
        StringBuilder text = new StringBuilder();
        // Log.d(TAG, "开始解码 " + tokenIds.size() + " 个token");
        
        for (int tokenId : tokenIds) {
            // 跳过特殊token
            if (tokenId == clsTokenId || tokenId == sepTokenId || tokenId == padTokenId) {
                // Log.d(TAG, "跳过特殊token ID: " + tokenId);
                continue;
            }
            
            String token = convertIdToToken(tokenId);
            
            // 移除BPE编码中的前缀和后缀
            if (token.startsWith("▁")) {
                token = " " + token.substring(1);
                // Log.d(TAG, "处理前缀: '" + token + "'");
            } else if (token.startsWith("Ġ")) {
                token = " " + token.substring(1);
                // Log.d(TAG, "处理前缀: '" + token + "'");
            }
            
            text.append(token);
        }
        
        String result = text.toString().trim();
        // Log.d(TAG, "解码结果: '" + result + "'");
        return result;
    }
    
    /**
     * 读取文件内容
     * @param file 文件对象
     * @return 文件内容字符串
     * @throws IOException 如果读取失败
     */
    private String readFileContent(File file) throws IOException {
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line).append("\n");
            }
        }
        return content.toString();
    }
    
    /**
     * 获取词汇表大小
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return vocab.size();
    }
    
    /**
     * 获取PAD token ID
     * @return PAD token ID
     */
    public int getPadTokenId() {
        return padTokenId;
    }
    
    /**
     * 获取CLS token ID
     * @return CLS token ID
     */
    public int getClsTokenId() {
        return clsTokenId;
    }
    
    /**
     * 获取SEP token ID
     * @return SEP token ID
     */
    public int getSepTokenId() {
        return sepTokenId;
    }
    
    /**
     * 获取UNK token ID
     * @return UNK token ID
     */
    public int getUnkTokenId() {
        return unkTokenId;
    }
    
    /**
     * 获取MASK token ID
     * @return MASK token ID
     */
    public int getMaskTokenId() {
        return maskTokenId;
    }
    
    /**
     * 获取最大序列长度
     * @return 最大序列长度
     */
    public int getMaxLength() {
        return maxLength;
    }
    
    /**
     * 处理BGE-M3模型的tokenizer.json
     * 这种模型使用特殊格式，包含added_tokens和词汇表
     * @param content tokenizer.json内容
     * @return 是否处理成功
     */
    private boolean handleBgeM3Tokenizer(String content) {
        try {
            // Log.d(TAG, "开始处理BGE-M3模型tokenizer");
            
            // 首先尝试提取特殊token
            // 检查是否能从added_tokens中提取特殊token
            Pattern addedTokenPattern = Pattern.compile("\"content\":\\s*\"([^\"]+)\"\\s*,\\s*\"id\":\\s*(\\d+)");
            Matcher addedTokenMatcher = addedTokenPattern.matcher(content);
            
            int specialTokensFound = 0;
            while (addedTokenMatcher.find() && specialTokensFound < 10) {
                String token = addedTokenMatcher.group(1);
                int id = Integer.parseInt(addedTokenMatcher.group(2));
                
                // 添加到词汇表
                vocab.put(token, id);
                idToToken.put(id, token);
                
                // 检查是否是特殊token
                checkAndUpdateSpecialToken(token, id);
                
                // Log.d(TAG, "从added_tokens提取特殊token: " + token + ", id: " + id);
                specialTokensFound++;
            }
            
            if (specialTokensFound > 0) {
                // Log.d(TAG, "成功从added_tokens提取了" + specialTokensFound + "个特殊token");
            }
            
            // 无论是否从added_tokens提取成功，都尝试使用正则表达式解析词汇表
            boolean success = parseVocabWithScoresRegex(content);
            
            // 如果已经提取了一些特殊token，则认为部分成功
            if (!success && specialTokensFound > 0) {
                // Log.d(TAG, "正则表达式解析词汇表失败，但已提取了特殊token，视为部分成功");
                success = true;
            }
            
            // 检查特殊token是否已经被正确设置
            String[] specialTokens = {"<s>", "<pad>", "</s>", "<unk>"};
            for (String token : specialTokens) {
                if (vocab.containsKey(token)) {
                    int id = vocab.get(token);
                    // Log.d(TAG, "特殊token已设置: " + token + " -> " + id);
                } else {
                    // Log.e(TAG, "特殊token未找到: " + token);
                }
            }
            
            return success;
        } catch (Exception e) {
            // Log.e(TAG, "处理BGE-M3模型tokenizer失败: " + e.getMessage(), e);
            // 即使处理失败，仍然尝试使用正则表达式进行解析
            return parseVocabWithScoresRegex(content);
        }
    }
    
    /**
     * 设置是否使用一致性分词策略
     * 一致性分词策略确保在知识库构建和查询过程中使用相同的分词逻辑
     * @param useConsistent 是否使用一致性分词
     */
    public void setUseConsistentTokenization(boolean useConsistent) {
        this.useConsistentTokenization = useConsistent;
        // Log.d(TAG, "设置一致性分词策略: " + useConsistent);
    }
    
    /**
     * 获取当前是否使用一致性分词策略
     * @return 是否使用一致性分词
     */
    public boolean isUseConsistentTokenization() {
        return this.useConsistentTokenization;
    }
    
    /**
     * 设置是否启用调试模式
     * 调试模式会输出更详细的日志信息
     * @param debug 是否启用调试
     */
    public void setDebugMode(boolean debug) {
        this.debugMode = debug;
        // Log.d(TAG, "设置调试模式: " + debug);
    }
    
    /**
     * 输出调试日志
     * 只有在调试模式启用时才会输出
     * @param message 日志信息
     */
    private void debugLog(String message) {
        if (debugMode) {
            Log.d(TAG, message);
        }
    }
    
    /**
     * 使用Trie前缀树和BPE优化的最长匹配分词算法
     * @param text 输入文本
     * @return 分词结果
     */
    private List<String> longestMatchTokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String processedText = text;
        if (addPrefixSpace && !text.startsWith(" ")) {
            processedText = " " + text;
        }
        int position = 0;
        while (position < processedText.length()) {
            TrieNode node = trieRoot;
            int longestLength = 0;
            String longestToken = null;
            int curr = position;
            while (curr < processedText.length()) {
                char c = processedText.charAt(curr);
                if (!node.children.containsKey(c)) break;
                node = node.children.get(c);
                if (node.isEnd) {
                    longestToken = node.token;
                    longestLength = curr - position + 1;
                }
                curr++;
            }
            if (longestToken != null) {
                tokens.add(longestToken);
                position += longestLength;
            } else {
                // fallback: 单字符分词或UNK
                char currentChar = processedText.charAt(position);
                String charToken = String.valueOf(currentChar);
                if (vocab.containsKey(charToken)) {
                    tokens.add(charToken);
                } else {
                    tokens.add(unkToken);
                }
                position++;
            }
        }
        return tokens;
    }

    /**
     * 构建Trie前缀树，提升token查找速度
     */
    private void buildTrieFromVocab() {
        trieRoot = new TrieNode();
        for (String token : vocab.keySet()) {
            TrieNode node = trieRoot;
            for (char c : token.toCharArray()) {
                node = node.children.computeIfAbsent(c, k -> new TrieNode());
            }
            node.isEnd = true;
            node.token = token;
        }
    }

}
