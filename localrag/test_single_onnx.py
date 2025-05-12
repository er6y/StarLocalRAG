import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def test_onnx_model(original_model_path, onnx_model_path):
    """比较原始模型和ONNX模型的嵌入向量"""
    test_sentences = [
        "这是一个测试句子，用于比较嵌入模型",
        "人工智能正在改变我们的生活方式",
        "深度学习模型需要大量的数据进行训练",
        "自然语言处理是人工智能的一个重要分支",
        "向量数据库可以高效地存储和检索嵌入向量"
    ]
    
    print(f"[INFO] 加载原始模型: {original_model_path}")
    original_model = SentenceTransformer(original_model_path)
    
    print(f"[INFO] 加载ONNX模型...")
    sys.path.insert(0, onnx_model_path)
    try:
        from inference import ONNXEmbeddingModel
        onnx_model = ONNXEmbeddingModel(onnx_model_path)
    except Exception as e:
        print(f"[ERROR] 加载ONNX模型失败: {str(e)}")
        return
    
    print("[INFO] 生成原始模型嵌入")
    original_embeddings = original_model.encode(test_sentences, normalize_embeddings=True)
    
    print("[INFO] 生成ONNX模型嵌入")
    try:
        onnx_embeddings = onnx_model.encode(test_sentences)
    except Exception as e:
        print(f"[ERROR] ONNX模型推理失败: {str(e)}")
        return
    
    # 计算余弦相似度
    similarities = []
    for i in range(len(test_sentences)):
        sim = cosine_similarity([original_embeddings[i]], [onnx_embeddings[i]])[0][0]
        similarities.append(sim)
        print(f"句子 {i+1} 相似度: {sim:.6f}")
    
    # 计算平均相似度
    avg_similarity = np.mean(similarities)
    print(f"[INFO] 平均相似度: {avg_similarity:.6f}")
    
    # 计算L2距离
    l2_distances = []
    for i in range(len(test_sentences)):
        dist = np.linalg.norm(original_embeddings[i] - onnx_embeddings[i])
        l2_distances.append(dist)
        print(f"句子 {i+1} L2距离: {dist:.6f}")
    
    # 计算平均L2距离
    avg_distance = np.mean(l2_distances)
    print(f"[INFO] 平均L2距离: {avg_distance:.6f}")
    
    # 评估转换质量
    if avg_similarity > 0.99:
        print("[INFO] 模型转换质量: 优秀 (相似度 > 0.99)")
    elif avg_similarity > 0.95:
        print("[INFO] 模型转换质量: 良好 (相似度 > 0.95)")
    elif avg_similarity > 0.9:
        print("[INFO] 模型转换质量: 一般 (相似度 > 0.9)")
    else:
        print("[INFO] 模型转换质量: 较差 (相似度 < 0.9)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python test_single_onnx.py <原始模型路径> <ONNX模型路径>")
        sys.exit(1)
    
    original_model_path = sys.argv[1]
    onnx_model_path = sys.argv[2]
    
    test_onnx_model(original_model_path, onnx_model_path)
