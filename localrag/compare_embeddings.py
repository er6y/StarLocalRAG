import os
import sys
import numpy as np
import shutil
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_onnx_model(model_dir):
    """加载ONNX模型"""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        # 检查必要的文件
        transformer_path = os.path.join(model_dir, "transformer_model.onnx")
        pooling_path = os.path.join(model_dir, "pooling_model.onnx")
        
        if not os.path.exists(transformer_path) or not os.path.exists(pooling_path):
            print(f"[ERROR] 找不到必要的ONNX模型文件: {transformer_path} 或 {pooling_path}")
            return None
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # 加载transformer模型
        transformer_session = ort.InferenceSession(transformer_path)
        
        # 加载pooling模型
        pooling_session = ort.InferenceSession(pooling_path)
        
        print(f"[INFO] ONNX模型加载成功: {model_dir}")
        return {"tokenizer": tokenizer, "transformer": transformer_session, "pooling": pooling_session}
    except Exception as e:
        print(f"[ERROR] 加载ONNX模型失败: {str(e)}")
        return None

def encode_with_onnx(model, texts):
    """使用ONNX模型编码文本"""
    tokenizer = model["tokenizer"]
    transformer_session = model["transformer"]
    pooling_session = model["pooling"]
    
    # 确保texts是列表
    if isinstance(texts, str):
        texts = [texts]
    
    # 对文本进行编码
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    
    # 获取输入名称
    transformer_input_names = [input.name for input in transformer_session.get_inputs()]
    
    # 准备transformer输入
    transformer_inputs = {}
    for name in transformer_input_names:
        if name in inputs:
            transformer_inputs[name] = inputs[name]
    
    # 运行transformer推理
    transformer_outputs = transformer_session.run(
        ["output"], 
        transformer_inputs
    )
    
    # 准备pooling输入
    pooling_inputs = {
        "token_embeddings": transformer_outputs[0],
        "attention_mask": inputs["attention_mask"]
    }
    
    # 运行pooling推理
    pooling_outputs = pooling_session.run(
        ["sentence_embedding"], 
        pooling_inputs
    )
    
    # 获取句子嵌入
    embeddings = pooling_outputs[0]
    
    # 检查嵌入的维度
    if len(embeddings.shape) == 3:
        # 对于BGE-M3模型，使用平均池化将token嵌入转换为句子嵌入
        embeddings = np.mean(embeddings, axis=1)
    
    # 归一化嵌入
    for i in range(len(embeddings)):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] = embeddings[i] / norm
    
    return embeddings

def clean_onnx_model_dir(model_dir):
    """清理ONNX模型目录，只保留必要的文件"""
    # 必要的文件列表
    essential_files = [
        "transformer_model.onnx",
        "pooling_model.onnx",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "config.json",
        "config_sentence_transformers.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "README.md",
        "inference.py"
    ]
    
    # 获取目录中的所有文件
    all_files = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), model_dir)
            all_files.append(rel_path)
    
    # 找出不必要的文件
    unnecessary_files = []
    for file in all_files:
        base_name = os.path.basename(file)
        if base_name not in essential_files and not any(base_name.endswith(ext) for ext in [".md", ".txt"]):
            unnecessary_files.append(file)
    
    # 删除不必要的文件
    if unnecessary_files:
        print(f"[INFO] 删除以下不必要的文件:")
        for file in unnecessary_files:
            file_path = os.path.join(model_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"  - {file}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"  - {file}/")
            except Exception as e:
                print(f"  [ERROR] 无法删除 {file}: {str(e)}")
    else:
        print("[INFO] 没有找到不必要的文件")

def compare_embeddings(original_model_path, onnx_model_path, test_sentences=None):
    """比较原始模型和ONNX模型的嵌入向量"""
    if test_sentences is None:
        test_sentences = [
            "这是一个测试句子，用于比较嵌入模型",
            "人工智能正在改变我们的生活方式",
            "深度学习模型需要大量的数据进行训练",
            "自然语言处理是人工智能的一个重要分支",
            "向量数据库可以高效地存储和检索嵌入向量"
        ]
    
    print(f"[INFO] 加载原始模型: {original_model_path}")
    original_model = SentenceTransformer(original_model_path)
    
    print(f"[INFO] 加载ONNX模型: {onnx_model_path}")
    onnx_model = load_onnx_model(onnx_model_path)
    
    if onnx_model is None:
        print("[ERROR] ONNX模型加载失败，无法进行比较")
        return
    
    print("[INFO] 生成原始模型嵌入")
    original_embeddings = original_model.encode(test_sentences, normalize_embeddings=True)
    
    print("[INFO] 生成ONNX模型嵌入")
    onnx_embeddings = encode_with_onnx(onnx_model, test_sentences)
    
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
    
    # 计算相对误差
    if avg_similarity > 0.99:
        print("[INFO] 模型转换质量: 优秀 (相似度 > 0.99)")
    elif avg_similarity > 0.95:
        print("[INFO] 模型转换质量: 良好 (相似度 > 0.95)")
    elif avg_similarity > 0.9:
        print("[INFO] 模型转换质量: 一般 (相似度 > 0.9)")
    else:
        print("[INFO] 模型转换质量: 较差 (相似度 < 0.9)")
    
    return {
        "similarities": similarities,
        "avg_similarity": avg_similarity,
        "l2_distances": l2_distances,
        "avg_distance": avg_distance
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python compare_embeddings.py <原始模型路径> <ONNX模型路径> [--clean]")
        sys.exit(1)
    
    original_model_path = sys.argv[1]
    onnx_model_path = sys.argv[2]
    clean_model = "--clean" in sys.argv
    
    # 比较嵌入向量
    results = compare_embeddings(original_model_path, onnx_model_path)
    
    # 如果指定了清理选项，则清理ONNX模型目录
    if clean_model and results and results["avg_similarity"] > 0.9:
        print("\n[INFO] 开始清理ONNX模型目录...")
        clean_onnx_model_dir(onnx_model_path)
        print("[INFO] ONNX模型目录清理完成")
