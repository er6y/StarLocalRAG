import os
import sys
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class ONNXEmbeddingModel:
    def __init__(self, model_dir):
        """
        初始化ONNX嵌入模型
        
        Args:
            model_dir: 包含ONNX模型文件的目录
        """
        self.model_dir = model_dir
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # 加载transformer模型
        transformer_path = os.path.join(model_dir, "transformer_model.onnx")
        self.transformer_session = ort.InferenceSession(transformer_path)
        
        # 加载pooling模型
        pooling_path = os.path.join(model_dir, "pooling_model.onnx")
        self.pooling_session = ort.InferenceSession(pooling_path)
        
        print(f"模型已加载: {model_dir}")
    
    def encode(self, texts, batch_size=32, normalize_embeddings=True):
        """
        编码文本为嵌入向量
        
        Args:
            texts: 要编码的文本或文本列表
            batch_size: 批处理大小
            normalize_embeddings: 是否对嵌入进行归一化
            
        Returns:
            numpy数组，形状为[len(texts), embedding_dim]
        """
        # 确保texts是列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 对文本进行编码
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="np")
            
            # 运行transformer模型
            transformer_outputs = self.transformer_session.run(
                ["output"], 
                {name: inputs[name] for name in inputs}
            )
            
            # 运行pooling模型
            pooling_outputs = self.pooling_session.run(
                ["sentence_embedding"], 
                {
                    "token_embeddings": transformer_outputs[0],
                    "attention_mask": inputs["attention_mask"]
                }
            )
            
            # 获取句子嵌入
            embeddings = pooling_outputs[0]
            
            # 对于BGE-M3模型，我们需要取第一个token的嵌入作为句子嵌入
            # 检查嵌入的维度
            if len(embeddings.shape) == 3:
                # 使用平均池化将token嵌入转换为句子嵌入
                embeddings = np.mean(embeddings, axis=1)
            
            # 归一化嵌入
            if normalize_embeddings:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

def test_onnx_model(model_dir):
    """测试ONNX嵌入模型"""
    try:
        # 加载ONNX模型
        model = ONNXEmbeddingModel(model_dir)
        
        # 测试文本
        test_texts = [
            "这是一个测试句子，用于验证ONNX模型",
            "另一个测试句子，看看模型是否正常工作",
            "第三个测试句子，确保批处理功能正常"
        ]
        
        # 编码文本
        embeddings = model.encode(test_texts)
        
        # 打印结果
        print(f"嵌入维度: {embeddings.shape}")
        print(f"第一个嵌入的前10个值: {embeddings[0][:10]}")
        
        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        print(f"相似度矩阵:\n{sim_matrix}")
        
        print("ONNX模型测试成功!")
        return True
    except Exception as e:
        print(f"ONNX模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_onnx_embedding.py <ONNX模型目录>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    success = test_onnx_model(model_dir)
    sys.exit(0 if success else 1)
