"""
调试Android端相似度计算问题的脚本

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

# 设置日志详细程度
VERBOSE = True

class ONNXEmbeddingModel:
    """ONNX嵌入模型类，用于生成文本的嵌入向量"""
    
    def __init__(self, model_dir):
        """
        初始化ONNX嵌入模型
        
        Args:
            model_dir: 包含ONNX模型文件的目录
        """
        print(f"[DEBUG] 初始化ONNX嵌入模型: {model_dir}")
        self.model_dir = model_dir
        
        # 检查模型文件
        model_files = os.listdir(model_dir)
        print(f"[DEBUG] 模型目录文件: {', '.join(model_files)}")
        
        # 加载tokenizer
        print(f"[DEBUG] 加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"[DEBUG] Tokenizer加载完成，词汇表大小: {len(self.tokenizer.vocab)}")
        
        # 检查是否使用单一模型或分离模型
        model_path = os.path.join(model_dir, "model.onnx")
        transformer_path = os.path.join(model_dir, "transformer_model.onnx")
        pooling_path = os.path.join(model_dir, "pooling_model.onnx")
        
        if os.path.exists(model_path):
            # 使用单一模型
            print(f"[DEBUG] 使用单一ONNX模型: {model_path}")
            self.session = ort.InferenceSession(model_path)
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"[DEBUG] 模型输入名称: {self.input_names}")
            print(f"[DEBUG] 模型输出名称: {self.output_names}")
            self.use_separate_models = False
        elif os.path.exists(transformer_path) and os.path.exists(pooling_path):
            # 使用分离的transformer和pooling模型
            print(f"[DEBUG] 使用分离的ONNX模型:")
            print(f"[DEBUG] - Transformer模型: {transformer_path}")
            print(f"[DEBUG] - Pooling模型: {pooling_path}")
            
            self.transformer_session = ort.InferenceSession(transformer_path)
            self.pooling_session = ort.InferenceSession(pooling_path)
            
            self.transformer_input_names = [input.name for input in self.transformer_session.get_inputs()]
            self.transformer_output_names = [output.name for output in self.transformer_session.get_outputs()]
            
            self.pooling_input_names = [input.name for input in self.pooling_session.get_inputs()]
            self.pooling_output_names = [output.name for output in self.pooling_session.get_outputs()]
            
            print(f"[DEBUG] Transformer模型输入名称: {self.transformer_input_names}")
            print(f"[DEBUG] Transformer模型输出名称: {self.transformer_output_names}")
            print(f"[DEBUG] Pooling模型输入名称: {self.pooling_input_names}")
            print(f"[DEBUG] Pooling模型输出名称: {self.pooling_output_names}")
            
            self.use_separate_models = True
        else:
            raise ValueError(f"在{model_dir}中找不到有效的ONNX模型文件")
    
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
        
        print(f"[DEBUG] 开始编码 {len(texts)} 个文本，批处理大小: {batch_size}")
        
        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"[DEBUG] 处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}, 文本数量: {len(batch_texts)}")
            
            # 对文本进行编码
            print(f"[DEBUG] Tokenizer处理文本...")
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="np")
            
            if VERBOSE:
                for j, text in enumerate(batch_texts):
                    print(f"[DEBUG] 文本 {j+1}: {text}")
                    print(f"[DEBUG] - Token IDs: {inputs['input_ids'][j]}")
                    print(f"[DEBUG] - 注意力掩码: {inputs['attention_mask'][j]}")
            
            if self.use_separate_models:
                # 使用分离的transformer和pooling模型
                print(f"[DEBUG] 运行Transformer模型...")
                
                # 准备transformer输入
                transformer_inputs = {}
                for name in self.transformer_input_names:
                    if name in inputs:
                        transformer_inputs[name] = inputs[name]
                
                # 运行transformer推理
                start_time = time.time()
                transformer_outputs = self.transformer_session.run(
                    self.transformer_output_names, 
                    transformer_inputs
                )
                transformer_time = time.time() - start_time
                print(f"[DEBUG] Transformer推理完成，耗时: {transformer_time:.4f}秒")
                
                if VERBOSE:
                    print(f"[DEBUG] Transformer输出形状: {transformer_outputs[0].shape}")
                
                # 准备pooling输入
                pooling_inputs = {
                    "token_embeddings": transformer_outputs[0],
                    "attention_mask": inputs["attention_mask"]
                }
                
                # 运行pooling推理
                print(f"[DEBUG] 运行Pooling模型...")
                start_time = time.time()
                pooling_outputs = self.pooling_session.run(
                    self.pooling_output_names, 
                    pooling_inputs
                )
                pooling_time = time.time() - start_time
                print(f"[DEBUG] Pooling推理完成，耗时: {pooling_time:.4f}秒")
                
                # 获取句子嵌入
                embeddings = pooling_outputs[0]
                
                if VERBOSE:
                    print(f"[DEBUG] Pooling输出形状: {embeddings.shape}")
            else:
                # 使用单一的model.onnx
                print(f"[DEBUG] 运行单一ONNX模型...")
                
                # 准备输入
                onnx_inputs = {}
                for name in self.input_names:
                    if name in inputs:
                        onnx_inputs[name] = inputs[name]
                
                # 运行推理
                start_time = time.time()
                outputs = self.session.run(None, onnx_inputs)
                inference_time = time.time() - start_time
                print(f"[DEBUG] ONNX推理完成，耗时: {inference_time:.4f}秒")
                
                # 获取嵌入向量
                embeddings = outputs[0]
                
                if VERBOSE:
                    print(f"[DEBUG] ONNX输出形状: {embeddings.shape}")
            
            # 检查嵌入的维度
            if len(embeddings.shape) == 3:
                # 对于BGE-M3模型，使用平均池化将token嵌入转换为句子嵌入
                print(f"[DEBUG] 检测到3D嵌入输出，应用平均池化: {embeddings.shape}")
                embeddings = np.mean(embeddings, axis=1)
                print(f"[DEBUG] 平均池化后形状: {embeddings.shape}")
            
            # 归一化嵌入
            if normalize_embeddings:
                print(f"[DEBUG] 对嵌入向量进行归一化...")
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                if VERBOSE:
                    for j in range(len(batch_texts)):
                        print(f"[DEBUG] 文本 {j+1} 嵌入范数: {norms[j][0]:.6f}")
                
                embeddings = embeddings / norms
                
                # 验证归一化结果
                new_norms = np.linalg.norm(embeddings, axis=1)
                if VERBOSE:
                    for j in range(len(batch_texts)):
                        print(f"[DEBUG] 文本 {j+1} 归一化后范数: {new_norms[j]:.6f}")
            
            all_embeddings.append(embeddings)
        
        # 合并所有批次的嵌入
        result = np.vstack(all_embeddings)
        print(f"[DEBUG] 嵌入生成完成，最终形状: {result.shape}")
        
        if VERBOSE:
            # 打印嵌入向量的统计信息
            print(f"[DEBUG] 嵌入向量统计信息:")
            print(f"[DEBUG] - 平均值: {np.mean(result):.6f}")
            print(f"[DEBUG] - 标准差: {np.std(result):.6f}")
            print(f"[DEBUG] - 最小值: {np.min(result):.6f}")
            print(f"[DEBUG] - 最大值: {np.max(result):.6f}")
        
        return result

def cosine_similarity_debug(vec1, vec2):
    """
    计算两个向量的余弦相似度，并打印详细的计算过程
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        余弦相似度值
    """
    # 确保输入是numpy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    print(f"[DEBUG] 点积: {dot_product:.6f}")
    
    # 计算范数
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    print(f"[DEBUG] 向量1范数: {norm_vec1:.6f}")
    print(f"[DEBUG] 向量2范数: {norm_vec2:.6f}")
    
    # 避免除以零
    if norm_vec1 == 0 or norm_vec2 == 0:
        print("[DEBUG] 警告: 向量范数为零，返回相似度0")
        return 0
    
    # 计算余弦相似度
    similarity = dot_product / (norm_vec1 * norm_vec2)
    print(f"[DEBUG] 余弦相似度: {similarity:.6f}")
    
    return similarity

def test_similarity_with_database(model_dir, db_path, query, collection_name="default", k=4):
    """
    使用给定的模型和数据库，测试查询相似度计算
    
    Args:
        model_dir: ONNX模型目录
        db_path: SQLite数据库路径
        query: 查询文本
        collection_name: 集合名称
        k: 返回的结果数量
    """
    print(f"\n{'='*80}")
    print(f"开始相似度测试")
    print(f"模型目录: {model_dir}")
    print(f"数据库路径: {db_path}")
    print(f"查询文本: {query}")
    print(f"集合名称: {collection_name}")
    print(f"{'='*80}\n")
    
    # 加载模型
    model = ONNXEmbeddingModel(model_dir)
    
    # 生成查询向量
    print(f"\n[DEBUG] 生成查询向量...")
    query_embedding = model.encode(query)
    print(f"[DEBUG] 查询向量形状: {query_embedding.shape}")
    
    # 连接数据库
    print(f"\n[DEBUG] 连接数据库: {db_path}")
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
        print(f"\n[DEBUG] 文档ID: {doc_id}")
        print(f"[DEBUG] 文档向量形状: {doc_embedding.shape}")
        
        # 计算余弦相似度
        print(f"[DEBUG] 计算余弦相似度...")
        similarity = cosine_similarity_debug(query_embedding[0], doc_embedding)
        
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

def test_similarity_between_texts(model_dir, texts):
    """
    测试文本之间的相似度
    
    Args:
        model_dir: ONNX模型目录
        texts: 要比较的文本列表
    """
    print(f"\n{'='*80}")
    print(f"测试文本之间的相似度")
    print(f"模型目录: {model_dir}")
    print(f"{'='*80}\n")
    
    # 加载模型
    model = ONNXEmbeddingModel(model_dir)
    
    # 生成所有文本的嵌入向量
    print(f"\n[DEBUG] 生成所有文本的嵌入向量...")
    embeddings = model.encode(texts)
    print(f"[DEBUG] 嵌入向量形状: {embeddings.shape}")
    
    # 计算相似度矩阵
    print(f"\n[DEBUG] 计算相似度矩阵...")
    similarity_matrix = np.zeros((len(texts), len(texts)))
    
    for i in range(len(texts)):
        for j in range(len(texts)):
            print(f"\n[DEBUG] 计算文本 {i+1} 和文本 {j+1} 之间的相似度:")
            similarity_matrix[i, j] = cosine_similarity_debug(embeddings[i], embeddings[j])
    
    # 打印相似度矩阵
    print(f"\n{'='*80}")
    print(f"相似度矩阵:")
    print(f"{'='*80}")
    
    # 打印表头
    header = "      "
    for i in range(len(texts)):
        header += f"文本{i+1:2d}     "
    print(header)
    
    # 打印矩阵内容
    for i in range(len(texts)):
        row = f"文本{i+1:2d} "
        for j in range(len(texts)):
            row += f"{similarity_matrix[i, j]:10.6f} "
        print(row)
    
    return similarity_matrix

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="调试Android端相似度计算问题")
    parser.add_argument("--model", "-m", required=True, help="ONNX模型目录路径")
    parser.add_argument("--db", "-d", help="SQLite数据库路径")
    parser.add_argument("--query", "-q", help="查询文本")
    parser.add_argument("--collection", "-c", default="default", help="集合名称")
    parser.add_argument("--texts", "-t", nargs="+", help="要比较的文本列表")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    global VERBOSE
    VERBOSE = args.verbose
    
    if args.db and args.query:
        # 测试数据库相似度查询
        test_similarity_with_database(args.model, args.db, args.query, args.collection)
    elif args.texts:
        # 测试文本之间的相似度
        test_similarity_between_texts(args.model, args.texts)
    else:
        print("错误: 必须提供数据库路径和查询文本，或者文本列表")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
