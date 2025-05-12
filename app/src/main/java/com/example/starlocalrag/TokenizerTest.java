package com.example.starlocalrag;

import android.util.Log;
import java.io.File;

/**
 * 测试BertTokenizer的加载和分词功能
 */
public class TokenizerTest {
    private static final String TAG = "StarLocalRAG_TokenizerTest";

    /**
     * 测试tokenizer加载和分词
     * @param modelDir 模型目录
     * @return 测试结果
     */
    public static boolean testTokenizer(File modelDir) {
        try {
            Log.d(TAG, "开始测试tokenizer加载和分词功能");
            
            // 创建tokenizer实例
            BertTokenizer tokenizer = new BertTokenizer();
            
            // 加载tokenizer
            boolean loadResult = tokenizer.loadFromDirectory(modelDir);
            if (!loadResult) {
                Log.e(TAG, "加载tokenizer失败");
                return false;
            }
            
            Log.d(TAG, "成功加载tokenizer，词汇表大小: " + tokenizer.getVocabSize());
            Log.d(TAG, "特殊token ID: CLS=" + tokenizer.getClsTokenId() + 
                  ", SEP=" + tokenizer.getSepTokenId() + 
                  ", PAD=" + tokenizer.getPadTokenId() + 
                  ", UNK=" + tokenizer.getUnkTokenId());
            
            // 测试分词功能
            String testText = "Hello world, this is a test.";
            long[][] tokenized = tokenizer.tokenize(testText);
            
            Log.d(TAG, "分词结果长度: " + tokenized[0].length);
            
            // 打印前10个token ID
            StringBuilder tokenIdsStr = new StringBuilder("Token IDs: ");
            for (int i = 0; i < Math.min(10, tokenized[0].length); i++) {
                tokenIdsStr.append(tokenized[0][i]).append(" ");
            }
            Log.d(TAG, tokenIdsStr.toString());
            
            // 打印前10个attention mask
            StringBuilder attentionStr = new StringBuilder("Attention Mask: ");
            for (int i = 0; i < Math.min(10, tokenized[1].length); i++) {
                attentionStr.append(tokenized[1][i]).append(" ");
            }
            Log.d(TAG, attentionStr.toString());
            
            Log.d(TAG, "Tokenizer测试完成，结果正常");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "测试tokenizer时发生异常: " + e.getMessage(), e);
            return false;
        }
    }
}
