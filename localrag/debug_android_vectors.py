"""
专门用于调试Android端向量计算和相似度问题的脚本

该脚本模拟Android端的嵌入向量计算和相似度计算过程，
打印详细的中间结果，以便找出相似度计算偏低的原因。
"""
import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
import sqlite3
from transformers import AutoTokenizer
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

def load_vector_from_file(file_path):
    """从文件加载向量"""
    with open(file_path, 'r') as f:
        vector_data = json.load(f)
    return np.array(vector_data)

def save_vector_to_file(vector, file_path):
    """保存向量到文件"""
    with open(file_path, 'w') as f:
        json.dump(vector.tolist(), f)

def cosine_similarity_debug(vec1, vec2, name1="向量1", name2="向量2"):
    """
    计算两个向量的余弦相似度，并打印详细的计算过程
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        name1: 第一个向量的名称
        name2: 第二个向量的名称
        
    Returns:
        余弦相似度值
    """
    # 确保输入是numpy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    print(f"[DEBUG] {name1}形状: {vec1.shape}")
    print(f"[DEBUG] {name2}形状: {vec2.shape}")
    print(f"[DEBUG] {name1}类型: {vec1.dtype}")
    print(f"[DEBUG] {name2}类型: {vec2.dtype}")
    
    # 打印向量的前几个值用于对比
    print(f"[DEBUG] {name1}前10个值: {vec1[:10]}")
    print(f"[DEBUG] {name2}前10个值: {vec2[:10]}")
    
    # 检查向量是否已经归一化
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    print(f"[DEBUG] {name1}范数: {norm1:.6f}")
    print(f"[DEBUG] {name2}范数: {norm2:.6f}")
    
    # 如果向量已经归一化，范数应该接近1
    is_vec1_normalized = abs(norm1 - 1.0) < 0.001
    is_vec2_normalized = abs(norm2 - 1.0) < 0.001
    
    print(f"[DEBUG] {name1}是否已归一化: {is_vec1_normalized}")
    print(f"[DEBUG] {name2}是否已归一化: {is_vec2_normalized}")
    
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    print(f"[DEBUG] 点积: {dot_product:.6f}")
    
    # 计算点积的详细过程
    print(f"[DEBUG] 点积计算详细过程:")
    sum_product = 0
    for i in range(min(5, len(vec1))):  # 只打印前5个元素的计算过程
        product = vec1[i] * vec2[i]
        sum_product += product
        print(f"[DEBUG]   {name1}[{i}] * {name2}[{i}] = {vec1[i]:.6f} * {vec2[i]:.6f} = {product:.6f}")
    print(f"[DEBUG]   ... (只显示前5个元素的计算)")
    
    # 避免除以零
    if norm1 == 0 or norm2 == 0:
        print("[DEBUG] 警告: 向量范数为零，返回相似度0")
        return 0
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    print(f"[DEBUG] 余弦相似度计算: {dot_product:.6f} / ({norm1:.6f} * {norm2:.6f}) = {similarity:.6f}")
    
    # 如果两个向量都已经归一化，那么点积就等于余弦相似度
    if is_vec1_normalized and is_vec2_normalized:
        print(f"[DEBUG] 两个向量都已归一化，点积等于余弦相似度")
        # 验证一下计算结果
        direct_similarity = dot_product
        print(f"[DEBUG] 直接计算的相似度: {direct_similarity:.6f}")
        print(f"[DEBUG] 差异: {abs(similarity - direct_similarity):.10f}")
    
    return similarity

def compare_android_pc_vectors(android_vector_file, pc_vector_file):
    """
    比较Android端和PC端的向量
    
    Args:
        android_vector_file: Android端向量文件路径
        pc_vector_file: PC端向量文件路径
    """
    print(f"\n{'='*80}")
    print(f"比较Android端和PC端的向量")
    print(f"Android端向量文件: {android_vector_file}")
    print(f"PC端向量文件: {pc_vector_file}")
    print(f"{'='*80}\n")
    
    # 加载向量
    android_vector = load_vector_from_file(android_vector_file)
    pc_vector = load_vector_from_file(pc_vector_file)
    
    # 打印向量信息
    print(f"[DEBUG] Android端向量形状: {android_vector.shape}")
    print(f"[DEBUG] PC端向量形状: {pc_vector.shape}")
    print(f"[DEBUG] Android端向量类型: {android_vector.dtype}")
    print(f"[DEBUG] PC端向量类型: {pc_vector.dtype}")
    
    # 打印向量的前几个值用于对比
    print(f"[DEBUG] Android端向量前10个值: {android_vector[:10]}")
    print(f"[DEBUG] PC端向量前10个值: {pc_vector[:10]}")
    
    # 检查向量是否已经归一化
    android_norm = np.linalg.norm(android_vector)
    pc_norm = np.linalg.norm(pc_vector)
    
    print(f"[DEBUG] Android端向量范数: {android_norm:.6f}")
    print(f"[DEBUG] PC端向量范数: {pc_norm:.6f}")
    
    # 如果向量已经归一化，范数应该接近1
    is_android_normalized = abs(android_norm - 1.0) < 0.001
    is_pc_normalized = abs(pc_norm - 1.0) < 0.001
    
    print(f"[DEBUG] Android端向量是否已归一化: {is_android_normalized}")
    print(f"[DEBUG] PC端向量是否已归一化: {is_pc_normalized}")
    
    # 如果维度不同，无法比较
    if android_vector.shape != pc_vector.shape:
        print(f"[ERROR] 向量维度不同，无法比较")
        return
    
    # 计算向量差异
    diff = android_vector - pc_vector
    abs_diff = np.abs(diff)
    
    print(f"[DEBUG] 向量差异统计:")
    print(f"[DEBUG] - 平均差异: {np.mean(abs_diff):.6f}")
    print(f"[DEBUG] - 最大差异: {np.max(abs_diff):.6f}")
    print(f"[DEBUG] - 最小差异: {np.min(abs_diff):.6f}")
    print(f"[DEBUG] - 标准差: {np.std(abs_diff):.6f}")
    
    # 打印差异最大的前5个元素
    diff_indices = np.argsort(-abs_diff)[:5]
    print(f"[DEBUG] 差异最大的5个元素:")
    for i, idx in enumerate(diff_indices):
        print(f"[DEBUG] {i+1}. 索引 {idx}: Android={android_vector[idx]:.6f}, PC={pc_vector[idx]:.6f}, 差异={abs_diff[idx]:.6f}")
    
    # 计算余弦相似度
    print(f"\n[DEBUG] 计算Android端和PC端向量的余弦相似度:")
    similarity = cosine_similarity_debug(android_vector, pc_vector, "Android端向量", "PC端向量")
    
    # 计算L2距离
    l2_distance = np.linalg.norm(diff)
    print(f"[DEBUG] L2距离: {l2_distance:.6f}")
    
    # 计算相对误差
    relative_error = l2_distance / np.linalg.norm(pc_vector) if np.linalg.norm(pc_vector) > 0 else 0
    print(f"[DEBUG] 相对误差: {relative_error:.6f}")
    
    return similarity, l2_distance, relative_error

def compare_android_db_similarity(android_vector_file, db_path, collection_name="default", k=4):
    """
    使用Android端的向量与数据库中的向量计算相似度
    
    Args:
        android_vector_file: Android端向量文件路径
        db_path: SQLite数据库路径
        collection_name: 集合名称
        k: 返回的结果数量
    """
    print(f"\n{'='*80}")
    print(f"使用Android端向量计算相似度")
    print(f"Android端向量文件: {android_vector_file}")
    print(f"数据库路径: {db_path}")
    print(f"集合名称: {collection_name}")
    print(f"{'='*80}\n")
    
    # 加载Android端向量
    android_vector = load_vector_from_file(android_vector_file)
    print(f"[DEBUG] Android端向量形状: {android_vector.shape}")
    
    # 连接数据库
    print(f"[DEBUG] 连接数据库: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取所有文档向量
    print(f"[DEBUG] 查询集合 '{collection_name}' 中的文档...")
    cursor.execute(
        "SELECT id, content, metadata, embedding FROM documents WHERE collection = ?",
        (collection_name,)
    )
    
    # 定义向量转换函数
    def blob_to_vector(blob):
        """将二进制blob转换为向量"""
        return np.frombuffer(blob, dtype=np.float32)
    
    # 计算相似度并收集结果
    results = []
    rows = cursor.fetchall()
    print(f"[DEBUG] 找到 {len(rows)} 个文档")
    
    for doc_id, content, metadata_json, embedding_blob in rows:
        # 转换向量
        doc_embedding = blob_to_vector(embedding_blob)
        
        # 计算余弦相似度
        if doc_id <= 5:  # 只打印前5个文档的详细信息
            print(f"\n[DEBUG] 文档ID: {doc_id}")
            print(f"[DEBUG] 文档向量形状: {doc_embedding.shape}")
            print(f"[DEBUG] 计算余弦相似度...")
            similarity = cosine_similarity_debug(android_vector, doc_embedding, "Android端向量", "文档向量")
        else:
            # 对于其他文档，直接计算相似度
            dot_product = np.dot(android_vector, doc_embedding)
            norm_android = np.linalg.norm(android_vector)
            norm_doc = np.linalg.norm(doc_embedding)
            similarity = dot_product / (norm_android * norm_doc) if norm_android > 0 and norm_doc > 0 else 0
        
        # 解析元数据
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        results.append((similarity, doc_id, content, metadata))
    
    conn.close()
    
    # 排序并返回前k个结果
    print(f"\n[DEBUG] 对结果进行排序...")
    results.sort(reverse=True)  # 余弦相似度越高越好
    
    top_k_results = results[:k]
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"相似度搜索结果 (前 {k} 个):")
    print(f"{'='*80}")
    
    for i, (similarity, doc_id, content, metadata) in enumerate(top_k_results):
        print(f"\n结果 {i+1}:")
        print(f"相似度: {similarity:.6f}")
        print(f"文档ID: {doc_id}")
        print(f"元数据: {metadata}")
    
    return top_k_results

def generate_pc_vector(model_dir, query, output_file=None):
    """
    使用PC端模型生成查询向量
    
    Args:
        model_dir: ONNX模型目录
        query: 查询文本
        output_file: 输出文件路径，如果为None则不保存
    
    Returns:
        查询向量
    """
    print(f"\n{'='*80}")
    print(f"使用PC端模型生成查询向量")
    print(f"模型目录: {model_dir}")
    print(f"查询文本: {query}")
    print(f"{'='*80}\n")
    
    # 加载tokenizer
    print(f"[DEBUG] 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # 添加更详细的tokenizer信息
    print(f"[DEBUG] Tokenizer配置: {tokenizer.vocab_size}")
    print(f"[DEBUG] Tokenizer特殊tokens: {tokenizer.all_special_tokens}")
    print(f"[DEBUG] Tokenizer特殊token IDs: {tokenizer.all_special_ids}")
    print(f"[DEBUG] Tokenizer类型: {type(tokenizer).__name__}")
    
    # 检查模型文件
    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    print(f"[DEBUG] 加载ONNX模型: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # 获取模型输入输出名称
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    print(f"[DEBUG] 模型输入名称: {input_names}")
    print(f"[DEBUG] 模型输出名称: {output_names}")
    
    # 对文本进行编码 - 更详细的分词信息
    print(f"[DEBUG] Tokenizer处理文本...")
    tokens = tokenizer(query, padding=True, truncation=True, return_tensors="np")
    print(f"[DEBUG] 分词结果详细信息:")
    print(f"[DEBUG] Token序列: {tokens['input_ids'][0]}")
    print(f"[DEBUG] Token字符串: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
    print(f"[DEBUG] 注意力掩码: {tokens['attention_mask'][0]}")
    
    # 打印分词器的词汇表大小和一些关键词的token ID
    if len(query) > 0:
        print(f"[DEBUG] 查询文本的第一个字符: '{query[0]}'")
        try:
            first_char_token = tokenizer.encode(query[0], add_special_tokens=False)[0]
            print(f"[DEBUG] 第一个字符的token ID: {first_char_token}")
        except Exception as e:
            print(f"[DEBUG] 获取第一个字符token ID失败: {str(e)}")
    
    # 准备输入
    onnx_inputs = {}
    for name in input_names:
        if name in tokens:
            onnx_inputs[name] = tokens[name]
    
    # 运行推理
    print(f"[DEBUG] 运行ONNX模型...")
    start_time = time.time()
    outputs = session.run(None, onnx_inputs)
    inference_time = time.time() - start_time
    print(f"[DEBUG] ONNX推理完成，耗时: {inference_time:.4f}秒")
    
    # 获取嵌入向量
    embedding = outputs[0]
    print(f"[DEBUG] 原始嵌入向量形状: {embedding.shape}")
    print(f"[DEBUG] 原始嵌入向量类型: {embedding.dtype}")
    
    # 打印原始嵌入向量的一些值
    if len(embedding.shape) > 1 and embedding.shape[0] > 0:
        print(f"[DEBUG] 原始嵌入向量前10个值: {embedding[0][:10]}")
    
    # 如果是批处理输出，取第一个
    if len(embedding.shape) > 1:
        embedding = embedding[0]
    
    # 检查嵌入的维度
    if len(embedding.shape) == 2:
        # 对于BGE-M3模型，使用平均池化将token嵌入转换为句子嵌入
        print(f"[DEBUG] 检测到2D嵌入输出，应用平均池化: {embedding.shape}")
        embedding = np.mean(embedding, axis=0)
        print(f"[DEBUG] 平均池化后形状: {embedding.shape}")
    
    # 打印未归一化的向量信息
    print(f"[DEBUG] 未归一化向量统计信息:")
    print(f"[DEBUG] - 平均值: {np.mean(embedding):.6f}")
    print(f"[DEBUG] - 标准差: {np.std(embedding):.6f}")
    print(f"[DEBUG] - 最小值: {np.min(embedding):.6f}")
    print(f"[DEBUG] - 最大值: {np.max(embedding):.6f}")
    print(f"[DEBUG] - 前10个值: {embedding[:10]}")
    
    # 归一化
    print(f"[DEBUG] 对嵌入向量进行归一化...")
    norm = np.linalg.norm(embedding)
    print(f"[DEBUG] 归一化前范数: {norm:.6f}")
    
    if norm > 0:
        embedding = embedding / norm
    
    # 验证归一化结果
    new_norm = np.linalg.norm(embedding)
    print(f"[DEBUG] 归一化后范数: {new_norm:.6f}")
    
    # 打印嵌入向量的统计信息
    print(f"[DEBUG] 归一化后向量统计信息:")
    print(f"[DEBUG] - 平均值: {np.mean(embedding):.6f}")
    print(f"[DEBUG] - 标准差: {np.std(embedding):.6f}")
    print(f"[DEBUG] - 最小值: {np.min(embedding):.6f}")
    print(f"[DEBUG] - 最大值: {np.max(embedding):.6f}")
    print(f"[DEBUG] - 前10个值: {embedding[:10]}")
    
    # 保存向量到文件
    if output_file:
        print(f"[DEBUG] 保存向量到文件: {output_file}")
        save_vector_to_file(embedding, output_file)
    
    return embedding

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="调试Android端向量计算和相似度问题")
    parser.add_argument("--android-vector", "-a", help="Android端向量文件路径")
    parser.add_argument("--pc-vector", "-p", help="PC端向量文件路径")
    parser.add_argument("--model", "-m", help="ONNX模型目录路径")
    parser.add_argument("--query", "-q", help="查询文本")
    parser.add_argument("--db", "-d", help="SQLite数据库路径")
    parser.add_argument("--collection", "-c", default="default", help="集合名称")
    parser.add_argument("--output", "-o", help="输出向量文件路径")
    
    args = parser.parse_args()
    
    # 生成PC端向量
    if args.model and args.query:
        pc_vector = generate_pc_vector(args.model, args.query, args.output)
    
    # 比较Android端和PC端的向量
    if args.android_vector and args.pc_vector:
        compare_android_pc_vectors(args.android_vector, args.pc_vector)
    
    # 使用Android端向量计算相似度
    if args.android_vector and args.db:
        compare_android_db_similarity(args.android_vector, args.db, args.collection)

if __name__ == "__main__":
    main()
