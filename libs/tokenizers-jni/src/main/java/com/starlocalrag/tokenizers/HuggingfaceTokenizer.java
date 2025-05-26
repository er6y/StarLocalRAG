package com.starlocalrag.tokenizers;

import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.io.Closeable;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Hugging Face分词器的Java包装类
 * 提供对Hugging Face Tokenizers库的访问
 * 同时实现Model接口以兼容现有代码
 */
public class HuggingfaceTokenizer implements Closeable, Model {
    private long nativePtr;
    private boolean isClosed = false;
    
    // 词汇表 (token -> id)
    private Map<String, Integer> vocab = new HashMap<>();
    
    // 反向词汇表 (id -> token)
    private Map<Integer, String> vocabReverse = new HashMap<>();
    
    // JNI方法
    private native long createTokenizer(String path, boolean consistentTokenization);
    private native String[] tokenizeToStrings(long tokenizerPtr, String text);
    private native int[] tokenizeToIds(long tokenizerPtr, String text);
    private native String applyChatTemplate(long tokenizerPtr, String text, String jsonConfig);
    
    // 使用已有的 JNI 方法
    private String getVocabFromJNI() {
        return TokenizerJNI.getTokenizerConfig(nativePtr);
    }
    
    private String decodeIdsToString(int[] ids) {
        // 将 int[] 转换为 JSON 字符串
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < ids.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(ids[i]);
        }
        sb.append("]");
        return TokenizerJNI.decode(nativePtr, sb.toString());
    }
    
    // 特殊token常量 - 这些常量可能在未来的方法中使用
    public static final String CLS_TOKEN = "[CLS]";
    public static final String SEP_TOKEN = "[SEP]";
    public static final String UNK_TOKEN = "[UNK]";
    public static final String PAD_TOKEN = "[PAD]";
    public static final String MASK_TOKEN = "[MASK]";
    
    // 特殊token变量 - 用于内部处理
    private String clsToken = CLS_TOKEN;
    private String sepToken = SEP_TOKEN;
    private String unkToken = UNK_TOKEN;
    private String padToken = PAD_TOKEN;
    private String maskToken = MASK_TOKEN;
    
    // 是否使用一致性分词策略
    private boolean useConsistentTokenization = false;
    
    // 是否启用调试模式
    private boolean debugMode = false;
    
    /**
     * 使用指定的模型类型创建分词器
     * @param modelType 模型类型，如"bpe"或"wordpiece"
     * @throws IllegalArgumentException 如果创建分词器失败
     */
    public HuggingfaceTokenizer(String modelType) {
        this.nativePtr = TokenizerJNI.createTokenizer(modelType);
        if (this.nativePtr == 0) {
            throw new IllegalArgumentException("创建分词器失败，模型类型: " + modelType);
        }
    }
    
    /**
     * 从文件加载分词器
     * @param path 分词器文件路径
     * @throws IllegalArgumentException 如果加载分词器失败
     */
    public HuggingfaceTokenizer(String path, boolean isFile) {
        if (isFile) {
            this.nativePtr = TokenizerJNI.loadTokenizerFromFile(path);
            if (this.nativePtr == 0) {
                throw new IllegalArgumentException("从文件加载分词器失败: " + path);
            }
            
            // 加载词汇表
            try {
                loadVocabFromFile(new File(path));
            } catch (Exception e) {
                System.err.println("加载词汇表时出错: " + e.getMessage());
                // 不抛出异常，因为词汇表加载失败不应影响分词器的基本功能
            }
        } else {
            this.nativePtr = TokenizerJNI.createTokenizer(path);
            if (this.nativePtr == 0) {
                throw new IllegalArgumentException("创建分词器失败，模型类型: " + path);
            }
        }
    }
    
    /**
     * 对文本进行分词
     * @param text 要分词的文本
     * @return 分词结果
     * @throws IllegalStateException 如果分词器已关闭
     */
    public TokenizerResult tokenize(String text) {
        checkClosed();
        String jsonResult = TokenizerJNI.tokenize(nativePtr, text);
        if (jsonResult == null) {
            throw new IllegalStateException("分词失败");
        }
        
        try {
            JSONObject json = new JSONObject(jsonResult);
            JSONArray idsArray = json.getJSONArray("ids");
            JSONArray tokensArray = json.getJSONArray("tokens");
            
            List<Integer> ids = new ArrayList<>();
            List<String> tokens = new ArrayList<>();
            
            for (int i = 0; i < idsArray.length(); i++) {
                ids.add(idsArray.getInt(i));
            }
            
            for (int i = 0; i < tokensArray.length(); i++) {
                tokens.add(tokensArray.getString(i));
            }
            
            return new TokenizerResult(ids, tokens);
        } catch (JSONException e) {
            throw new IllegalStateException("解析分词结果失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取分词器配置
     * @return 分词器配置的JSON对象
     * @throws IllegalStateException 如果分词器已关闭
     */
    public JSONObject getTokenizerConfig() {
        checkClosed();
        String jsonStr = TokenizerJNI.getTokenizerConfig(nativePtr);
        if (jsonStr == null) {
            throw new IllegalStateException("获取分词器配置失败");
        }
        
        try {
            return new JSONObject(jsonStr);
        } catch (JSONException e) {
            throw new IllegalStateException("解析分词器配置失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取所有特殊token
     * @return 特殊token的映射表，key为token名称，value为token内容
     * @throws IllegalStateException 如果分词器已关闭
     */
    public Map<String, String> getSpecialTokens() {
        checkClosed();
        Map<String, String> specialTokens = new HashMap<>();
        
        try {
            JSONObject config = getTokenizerConfig();
            
            // 尝试从配置中提取特殊token
            if (config.has("added_tokens")) {
                JSONArray addedTokens = config.getJSONArray("added_tokens");
                for (int i = 0; i < addedTokens.length(); i++) {
                    JSONObject tokenObj = addedTokens.getJSONObject(i);
                    if (tokenObj.has("content") && tokenObj.has("special") && tokenObj.getBoolean("special")) {
                        String tokenContent = tokenObj.getString("content");
                        // 尝试找出这个token的名称
                        String tokenName = "special_token_" + i;
                        
                        // 基于token内容猜测名称
                        if (tokenContent.contains("[CLS]") || tokenContent.equals("<s>")) {
                            tokenName = "cls_token";
                            clsToken = tokenContent;
                        } else if (tokenContent.contains("[SEP]") || tokenContent.equals("</s>")) {
                            tokenName = "sep_token";
                            sepToken = tokenContent;
                        } else if (tokenContent.contains("[UNK]") || tokenContent.equals("<unk>")) {
                            tokenName = "unk_token";
                            unkToken = tokenContent;
                        } else if (tokenContent.contains("[PAD]") || tokenContent.equals("<pad>")) {
                            tokenName = "pad_token";
                            padToken = tokenContent;
                        } else if (tokenContent.contains("[MASK]") || tokenContent.equals("<mask>")) {
                            tokenName = "mask_token";
                            maskToken = tokenContent;
                        }
                        
                        specialTokens.put(tokenName, tokenContent);
                    }
                }
            }
            
            // 如果没有找到特殊token，尝试从配置的其他部分查找
            if (specialTokens.isEmpty()) {
                // 检查常见的特殊token字段
                String[] tokenFields = {"cls_token", "sep_token", "unk_token", "pad_token", "mask_token"};
                for (String field : tokenFields) {
                    if (config.has(field)) {
                        String tokenContent = config.getString(field);
                        specialTokens.put(field, tokenContent);
                        
                        // 更新内部变量
                        if (field.equals("cls_token")) {
                            clsToken = tokenContent;
                        } else if (field.equals("sep_token")) {
                            sepToken = tokenContent;
                        } else if (field.equals("unk_token")) {
                            unkToken = tokenContent;
                        } else if (field.equals("pad_token")) {
                            padToken = tokenContent;
                        } else if (field.equals("mask_token")) {
                            maskToken = tokenContent;
                        }
                    }
                }
            }
            
            // 如果还是没有找到特殊token，使用默认值
            if (specialTokens.isEmpty()) {
                specialTokens.put("cls_token", clsToken);
                specialTokens.put("sep_token", sepToken);
                specialTokens.put("unk_token", unkToken);
                specialTokens.put("pad_token", padToken);
                specialTokens.put("mask_token", maskToken);
            }
            
            return specialTokens;
        } catch (Exception e) {
            // 如果出错，返回默认的特殊token
            specialTokens.put("cls_token", clsToken);
            specialTokens.put("sep_token", sepToken);
            specialTokens.put("unk_token", unkToken);
            specialTokens.put("pad_token", padToken);
            specialTokens.put("mask_token", maskToken);
            return specialTokens;
        }
    }
    
    /**
     * 获取特殊token的ID
     * @param tokenName token名称
     * @return token ID，如果不存在则返回-1
     * @throws IllegalStateException 如果分词器已关闭
     */
    public int getSpecialTokenId(String tokenName) {
        checkClosed();
        
        try {
            // 获取特殊token的内容
            Map<String, String> specialTokens = getSpecialTokens();
            String tokenContent = specialTokens.get(tokenName);
            
            if (tokenContent != null && vocab.containsKey(tokenContent)) {
                return vocab.get(tokenContent);
            }
            
            // 如果没有找到，尝试使用默认的特殊token
            if (tokenName.equals("cls_token") && vocab.containsKey(clsToken)) {
                return vocab.get(clsToken);
            } else if (tokenName.equals("sep_token") && vocab.containsKey(sepToken)) {
                return vocab.get(sepToken);
            } else if (tokenName.equals("unk_token") && vocab.containsKey(unkToken)) {
                return vocab.get(unkToken);
            } else if (tokenName.equals("pad_token") && vocab.containsKey(padToken)) {
                return vocab.get(padToken);
            } else if (tokenName.equals("mask_token") && vocab.containsKey(maskToken)) {
                return vocab.get(maskToken);
            }
            
            return -1; // 如果没有找到，返回-1
        } catch (Exception e) {
            return -1; // 如果出错，返回-1
        }
    }
    
    /**
     * 获取特殊token的内容
     * @param tokenName token名称
     * @return token内容，如果不存在则返回null
     * @throws IllegalStateException 如果分词器已关闭
     */
    public String getSpecialTokenContent(String tokenName) {
        checkClosed();
        Map<String, String> specialTokens = getSpecialTokens();
        return specialTokens.get(tokenName);
    }
    
    /**
     * 将token ID数组解码为文本
     * @param ids token ID数组
     * @return 解码后的文本
     * @throws IllegalStateException 如果分词器已关闭
     */
    public String decode(int[] ids) {
        checkClosed();
        
        if (ids == null || ids.length == 0) {
            return "";
        }
        
        String result = decodeIdsToString(ids);
        if (result == null) {
            throw new IllegalStateException("解码失败");
        }
        
        return result;
    }
    
    /**
     * 检查分词器是否已关闭
     * @throws IllegalStateException 如果分词器已关闭
     */
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("分词器已关闭");
        }
    }
    
    /**
     * 应用聊天模板
     * 这个方法与 Python 的 tokenizer.apply_chat_template 方法类似
     * @param messages 消息数组，每个消息应包含 role 和 content 字段
     * @param addGenerationPrompt 是否添加生成提示
     * @param enableThinking 是否启用思考模式
     * @param chatTemplate 自定义聊天模板，如果为null则使用模型默认模板
     * @param continueFinalMessage 是否继续最后一条消息，而不是开始新消息
     * @return 应用模板后的文本
     */
    public String applyChatTemplate(JSONArray messages, boolean addGenerationPrompt, boolean enableThinking,
                               String chatTemplate, boolean continueFinalMessage) {
        checkClosed();
        
        try {
            // 获取分词器配置
            JSONObject config = getTokenizerConfig();
            Map<String, String> specialTokens = getSpecialTokens();
            
            // 确定使用哪个模板
            String templateToUse = chatTemplate;
            if (templateToUse == null) {
                // 尝试从配置中获取聊天模板
                if (config.has("chat_template")) {
                    templateToUse = config.getString("chat_template");
                }
            }
            
            // 检查是否同时设置了 addGenerationPrompt 和 continueFinalMessage
            if (addGenerationPrompt && continueFinalMessage) {
                throw new IllegalArgumentException("不能同时设置 addGenerationPrompt 和 continueFinalMessage");
            }
            
            // 如果有模板，使用模板，否则使用默认模板
            if (templateToUse != null && !templateToUse.isEmpty()) {
                return applyTemplate(templateToUse, messages, specialTokens, addGenerationPrompt, enableThinking, continueFinalMessage);
            } else {
                return applyDefaultTemplate(messages, specialTokens, addGenerationPrompt, enableThinking, continueFinalMessage);
            }
        } catch (Exception e) {
            throw new RuntimeException("应用聊天模板失败: " + e.getMessage(), e);
        }
    }

    /**
     * 应用聊天模板（简化版本，兼容旧接口）
     * @param messages 消息数组，每个消息应包含 role 和 content 字段
     * @param addGenerationPrompt 是否添加生成提示
     * @param enableThinking 是否启用思考模式
     * @return 应用模板后的文本
     * @throws IllegalStateException 如果分词器已关闭
     */
    public String applyChatTemplate(JSONArray messages, boolean addGenerationPrompt, boolean enableThinking) {
        return applyChatTemplate(messages, addGenerationPrompt, enableThinking, null, false);
    }
    
    /**
     * 应用指定的模板
     * @param template 模板字符串
     * @param messages 消息数组
     * @param specialTokens 特殊标记映射
     * @param addGenerationPrompt 是否添加生成提示
     * @param enableThinking 是否启用思考模式
     * @param continueFinalMessage 是否继续最后一条消息
     * @return 应用模板后的文本
     */
    private String applyTemplate(String template, JSONArray messages, Map<String, String> specialTokens, 
                                boolean addGenerationPrompt, boolean enableThinking, boolean continueFinalMessage) {
        // 替换模板变量
        String result = template;
        
        // 替换 messages 变量
        result = replaceVariable(result, "messages", messages);
        
        // 替换 add_generation_prompt 变量
        result = replaceVariable(result, "add_generation_prompt", addGenerationPrompt);
        
        // 替换 enable_thinking 变量
        result = replaceVariable(result, "enable_thinking", enableThinking);
        
        // 替换 continue_final_message 变量
        result = replaceVariable(result, "continue_final_message", continueFinalMessage);
        
        // 替换特殊标记
        for (Map.Entry<String, String> entry : specialTokens.entrySet()) {
            String tokenName = entry.getKey();
            String tokenValue = entry.getValue();
            result = result.replace("<|" + tokenName + "|>", tokenValue);
        }
        
        // 处理其他变量表达式
        // 移除所有未替换的变量表达式
        result = result.replaceAll("\\{\\{[^\\}]*\\}\\}", "");
        
        // 如果启用思考模式，确保添加思考标记
        if (enableThinking && !result.contains("<think>")) {
            String thinkTag = specialTokens.getOrDefault("think_tag", "<think>");
            String imStart = specialTokens.getOrDefault("im_start", "");
            String assistantRole = specialTokens.getOrDefault("assistant", "assistant");
            
            // 尝试在助手角色后添加思考标记
            if (imStart != null && !imStart.isEmpty()) {
                String assistantStart = imStart + assistantRole;
                if (result.contains(assistantStart)) {
                    int pos = result.indexOf(assistantStart) + assistantStart.length();
                    result = result.substring(0, pos) + "\n" + thinkTag + result.substring(pos);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 应用默认模板
     * @param messages 消息数组
     * @param specialTokens 特殊标记映射
     * @param addGenerationPrompt 是否添加生成提示
     * @param enableThinking 是否启用思考模式
     * @param continueFinalMessage 是否继续最后一条消息
     * @return 应用模板后的文本
     */
    private String applyDefaultTemplate(JSONArray messages, Map<String, String> specialTokens,
                                     boolean addGenerationPrompt, boolean enableThinking, boolean continueFinalMessage) {
        StringBuilder result = new StringBuilder();
        String imStart = specialTokens.getOrDefault("im_start", "<|im_start|>");
        String imEnd = specialTokens.getOrDefault("im_end", "<|im_end|>");
        String thinkTag = specialTokens.getOrDefault("think_tag", "<think>");
        
        // 处理消息数组
        for (int i = 0; i < messages.length(); i++) {
            JSONObject message = messages.getJSONObject(i);
            String role = message.getString("role");
            String content = message.getString("content");
            
            // 根据角色添加相应的标记
            result.append(imStart).append(role).append("\n");
            
            // 如果是助手角色且启用思考模式，添加思考标记
            if ("assistant".equals(role) && enableThinking) {
                result.append(thinkTag).append("\n");
            }
            
            result.append(content).append("\n");
            result.append(imEnd).append("\n");
        }
        
        // 如果需要添加生成提示
        if (addGenerationPrompt) {
            result.append(imStart).append("assistant\n");
            if (enableThinking) {
                result.append(thinkTag).append("\n");
            }
        }
        
        // 如果需要继续最后一条消息
        if (continueFinalMessage && messages.length() > 0) {
            // 获取最后一条消息
            JSONObject lastMessage = messages.getJSONObject(messages.length() - 1);
            String role = lastMessage.getString("role");
            
            // 只有当最后一条消息是助手消息时才继续
            if ("assistant".equals(role)) {
                // 移除最后一个 im_end 标记
                int lastImEndIndex = result.lastIndexOf(imEnd);
                if (lastImEndIndex >= 0) {
                    result.delete(lastImEndIndex, result.length());
                }
            }
        }
        
        return result.toString();
    }
    
    /**
     * 替换模板中的变量
     * @param template 模板字符串
     * @param variable 变量名
     * @param value 变量值
     * @return 替换后的字符串
     */
    private String replaceVariable(String template, String variable, Object value) {
        // 将值转换为字符串
        String valueStr;
        if (value instanceof JSONArray || value instanceof JSONObject) {
            valueStr = value.toString();
        } else {
            valueStr = String.valueOf(value);
        }
        
        // 支持多种变量格式: {{ var }}, {{var}}, {{ var}}, {{var }}
        String result = template;
        result = result.replace("{{ " + variable + " }}", valueStr);
        result = result.replace("{{" + variable + "}}", valueStr);
        result = result.replace("{{ " + variable + "}}", valueStr);
        result = result.replace("{{" + variable + " }}", valueStr);
        return result;
    }
    
    /**
     * 获取当前时间格式化字符串
     * @param format 格式化字符串
     * @return 格式化后的时间字符串
     */
    private String getCurrentTimeFormatted(String format) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
        return now.format(formatter);
    }
    
    /**
     * 从文件加载词汇表
     * @param file 词汇表文件
     * @throws IOException 如果文件读取失败
     * @throws JSONException 如果JSON解析失败
     */
    private void loadVocabFromFile(File file) throws IOException, JSONException {
        // 尝试从同目录下的vocab.json文件加载词汇表
        File vocabFile = new File(file.getParentFile(), "vocab.json");
        if (!vocabFile.exists()) {
            // 尝试从tokenizer.json文件中提取词汇表
            vocabFile = new File(file.getParentFile(), "tokenizer.json");
        }
        
        if (vocabFile.exists()) {
            StringBuilder content = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new FileReader(vocabFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    content.append(line);
                }
            }
            
            JSONObject json = new JSONObject(content.toString());
            
            // 尝试从不同的路径获取词汇表
            if (json.has("model") && json.getJSONObject("model").has("vocab")) {
                JSONObject vocabJson = json.getJSONObject("model").getJSONObject("vocab");
                loadVocabFromMap(vocabJson);
            } else if (json.has("vocab")) {
                JSONObject vocabJson = json.getJSONObject("vocab");
                loadVocabFromMap(vocabJson);
            }
            
            // 检查特殊token
            if (json.has("added_tokens")) {
                JSONArray addedTokens = json.getJSONArray("added_tokens");
                for (int i = 0; i < addedTokens.length(); i++) {
                    JSONObject tokenObj = addedTokens.getJSONObject(i);
                    if (tokenObj.has("content") && tokenObj.has("id")) {
                        String tokenContent = tokenObj.getString("content");
                        int tokenId = tokenObj.getInt("id");
                        vocab.put(tokenContent, tokenId);
                        vocabReverse.put(tokenId, tokenContent);
                        
                        // 检查是否是特殊token
                        if (tokenObj.has("special") && tokenObj.getBoolean("special")) {
                            if (tokenContent.contains("[CLS]") || tokenContent.equals("<s>")) {
                                clsToken = tokenContent;
                            } else if (tokenContent.contains("[SEP]") || tokenContent.equals("</s>")) {
                                sepToken = tokenContent;
                            } else if (tokenContent.contains("[UNK]") || tokenContent.equals("<unk>")) {
                                unkToken = tokenContent;
                            } else if (tokenContent.contains("[PAD]") || tokenContent.equals("<pad>")) {
                                padToken = tokenContent;
                            } else if (tokenContent.contains("[MASK]") || tokenContent.equals("<mask>")) {
                                maskToken = tokenContent;
                            }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * 对文本进行分词，返回long[][]格式的结果
     * 这个方法是为了兼容HuggingfaceTokenizer的接口
     * @param text 输入文本
     * @return 分词结果（token ID矩阵）
     */
    public long[][] tokenizeToLongArray(String text) {
        checkClosed();
        
        try {
            // 使用现有的tokenize方法
            TokenizerResult result = tokenize(text);
            List<Integer> ids = result.getIds();
            
            // 将List<Integer>转换为long[][]格式
            long[][] longResult = new long[1][ids.size()];
            for (int i = 0; i < ids.size(); i++) {
                longResult[0][i] = ids.get(i);
            }
            
            if (debugMode) {
                System.out.println("分词结果: " + ids.size() + " tokens");
            }
            
            return longResult;
        } catch (Exception e) {
            System.err.println("分词失败: " + e.getMessage());
            return new long[1][0]; // 返回空结果
        }
    }
    
    /**
     * 关闭分词器并释放资源
     */
    @Override
    public void close() {
        if (!isClosed) {
            TokenizerJNI.freeTokenizer(nativePtr);
            isClosed = true;
        }
    }
    
    // ===== 实现Model接口的方法 =====
    
    /**
     * 对文本列表进行分词（实现Model接口）
     * @param tokens 要分词的文本列表
     * @return 分词结果的token ID列表
     */
    @Override
    public List<Integer> tokenize(List<String> tokens) {
        StringBuilder text = new StringBuilder();
        for (String token : tokens) {
            if (text.length() > 0) {
                text.append(" ");
            }
            text.append(token);
        }
        
        try {
            TokenizerResult result = tokenize(text.toString());
            return result.getIds();
        } catch (Exception e) {
            // 如果分词失败，返回未知标记
            List<Integer> fallback = new ArrayList<>();
            int unkTokenId = getSpecialTokenId("unk_token");
            if (unkTokenId != -1) {
                fallback.add(unkTokenId);
            }
            return fallback;
        }
    }
    
    /**
     * 将token转换为ID（实现Model接口）
     * @param token 要转换的token
     * @return token的ID，如果不存在则返回未知标记ID
     */
    @Override
    public int tokenToId(String token) {
        return vocab.getOrDefault(token, getSpecialTokenId("unk_token"));
    }
    
    /**
     * 将ID转换为token（实现Model接口）
     * @param id 要转换的ID
     * @return 对应的token，如果不存在则返回空字符串
     */
    @Override
    public String idToToken(int id) {
        return vocabReverse.getOrDefault(id, "");
    }
    
    /**
     * 获取词汇表（实现Model接口）
     * @return 词汇表映射（token -> id）
     */
    @Override
    public Map<String, Integer> getVocab() {
        return vocab;
    }
    
    /**
     * 获取词汇表大小（实现Model接口）
     * @return 词汇表大小
     */
    @Override
    public int getVocabSize() {
        return vocab.size();
    }
    
    /**
     * 加载配置（实现TokenizerComponent接口）
     * @param config 配置信息
     * @throws JSONException 如果JSON解析失败
     */
    @Override
    public void loadConfig(JSONObject config) throws JSONException {
        // 重新加载配置
        if (config != null && config.has("vocab")) {
            vocab.clear();
            vocabReverse.clear();
            
            JSONObject vocabJson = config.getJSONObject("vocab");
            for (String key : vocabJson.keySet()) {
                int id = vocabJson.getInt(key);
                vocab.put(key, id);
                vocabReverse.put(id, key);
            }
        }
        
        // 更新特殊标记
        if (config != null) {
            if (config.has("cls_token")) {
                clsToken = config.getString("cls_token");
            }
            if (config.has("sep_token")) {
                sepToken = config.getString("sep_token");
            }
            if (config.has("unk_token")) {
                unkToken = config.getString("unk_token");
            }
            if (config.has("pad_token")) {
                padToken = config.getString("pad_token");
            }
            if (config.has("mask_token")) {
                maskToken = config.getString("mask_token");
            }
        }
    }
    
    /**
     * 从映射加载词汇表（实现Model接口）
     * @param vocabJson 词汇表JSON对象
     * @throws JSONException 如果JSON解析失败
     */
    @Override
    public void loadVocabFromMap(JSONObject vocabJson) throws JSONException {
        vocab.clear();
        vocabReverse.clear();
        
        for (String key : vocabJson.keySet()) {
            int id = vocabJson.getInt(key);
            vocab.put(key, id);
            vocabReverse.put(id, key);
        }
        
        if (debugMode) {
            System.out.println("从JSONObject加载词汇表，大小: " + vocab.size());
        }
    }
    
    /**
     * 从数组加载词汇表（实现Model接口）
     * @param vocabArray 词汇表数组
     * @throws JSONException 如果JSON解析失败
     */
    @Override
    public void loadVocabFromArray(JSONArray vocabArray) throws JSONException {
        vocab.clear();
        vocabReverse.clear();
        for (int i = 0; i < vocabArray.length(); i++) {
            String token = vocabArray.getString(i);
            vocab.put(token, i);
            vocabReverse.put(i, token);
        }
        if (debugMode) {
            System.out.println("从JSONArray加载词汇表，大小: " + vocab.size());
        }
    }
    
    /**
     * 设置未知标记ID（实现Model接口）
     * @param unkId 未知标记ID
     */
    @Override
    public void setUnkTokenId(int unkId) {
        // 在已有的词汇表中查找对应的token
        String token = vocabReverse.get(unkId);
        if (token != null) {
            unkToken = token;
        }
    }
    
    /**
     * 检查是否存在指定的token（实现Model接口）
     * @param token 要检查的token
     * @return 是否存在
     */
    @Override
    public boolean hasToken(String token) {
        return vocab.containsKey(token);
    }
    
    /**
     * 添加新的token（实现Model接口）
     * @param token 要添加的token
     * @param id token的ID
     */
    @Override
    public void addToken(String token, int id) {
        vocab.put(token, id);
        vocabReverse.put(id, token);
    }
    
    /**
     * 分词结果类
     */
    public static class TokenizerResult {
        private final List<Integer> ids;
        private final List<String> tokens;
        
        public TokenizerResult(List<Integer> ids, List<String> tokens) {
            this.ids = ids;
            this.tokens = tokens;
        }
        
        /**
         * 获取token IDs
         * @return token ID列表
         */
        public List<Integer> getIds() {
            return ids;
        }
        
        /**
         * 获取tokens
         * @return token字符串列表
         */
        public List<String> getTokens() {
            return tokens;
        }
    }
}
