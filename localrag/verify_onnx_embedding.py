"""
ONNX词嵌入模型验证工具 - 用于验证转换后的ONNX词嵌入模型是否可用
"""
import os
import sys
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import argparse
import json

def verify_onnx_embedding(model_path, verbose=True):
    """验证ONNX词嵌入模型的有效性"""
    results = {
        "model_name": os.path.basename(model_path),
        "model_type": "ONNX词嵌入模型",
        "status": "未知",
        "errors": [],
        "details": {},
        "embedding_dim": None,
        "test_results": []
    }
    
    print(f"开始验证ONNX词嵌入模型: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        error = f"错误: 模型路径不存在: {model_path}"
        print(error)
        results["status"] = "失败"
        results["errors"].append(error)
        return results
    
    # 检查模型文件
    model_files = os.listdir(model_path)
    print(f"模型目录内容 ({len(model_files)}个文件): {', '.join(model_files)}")
    results["details"]["files"] = model_files
    
    # 检查必要文件
    onnx_model_path = os.path.join(model_path, "model.onnx")
    if not os.path.exists(onnx_model_path):
        error = f"错误: ONNX模型文件不存在: {onnx_model_path}"
        print(error)
        results["status"] = "失败"
        results["errors"].append(error)
        return results
    
    # 检查模型大小
    model_size = os.path.getsize(onnx_model_path)
    print(f"ONNX模型文件大小: {model_size} 字节 ({model_size / 1024 / 1024:.2f} MB)")
    results["details"]["model_size"] = {
        "bytes": model_size,
        "mb": round(model_size / 1024 / 1024, 2)
    }
    
    if model_size < 1000:
        warning = "警告: ONNX模型文件过小，可能存在问题"
        print(warning)
        results["errors"].append(warning)
    
    # 加载分词器
    try:
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab_size = len(tokenizer)
        print(f"分词器加载成功，词汇表大小: {vocab_size}")
        results["details"]["tokenizer"] = {
            "vocab_size": vocab_size
        }
    except Exception as e:
        error = f"错误: 加载分词器失败: {str(e)}"
        print(error)
        results["status"] = "失败"
        results["errors"].append(error)
        return results
    
    # 加载ONNX模型
    try:
        print("加载ONNX模型...")
        start_time = time.time()
        session = ort.InferenceSession(onnx_model_path)
        load_time = time.time() - start_time
        print(f"ONNX模型加载成功，耗时: {load_time:.2f}秒")
        results["details"]["load_time"] = round(load_time, 2)
        
        # 获取模型输入输出信息
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        print(f"模型输入名称: {input_names}")
        print(f"模型输出名称: {output_names}")
        results["details"]["input_names"] = input_names
        results["details"]["output_names"] = output_names
        
        # 准备测试样本
        test_texts = [
            "这是一个测试句子，用于验证词嵌入模型",
            "人工智能正在改变世界",
            "词嵌入是自然语言处理的基础技术"
        ]
        results["test_results"] = []
        
        for i, test_text in enumerate(test_texts):
            print(f"\n测试样本 {i+1}: '{test_text}'")
            test_result = {
                "text": test_text,
                "success": False,
                "embedding_shape": None,
                "embedding_stats": None,
                "inference_time": None
            }
            
            try:
                # 准备输入数据
                inputs = tokenizer(test_text, return_tensors="np")
                input_ids = inputs.input_ids.astype(np.int64)  # 确保是int64类型
                attention_mask = inputs.attention_mask.astype(np.int64) if hasattr(inputs, "attention_mask") else None
                
                print(f"输入形状: {input_ids.shape}")
                print(f"输入token IDs: {input_ids[0]}")
                print(f"输入数据类型: {input_ids.dtype}")  # 打印数据类型以便确认
                
                # 执行推理
                print("执行ONNX模型推理...")
                start_time = time.time()
                
                # 准备输入字典
                ort_inputs = {input_names[0]: input_ids}
                if len(input_names) > 1 and attention_mask is not None:
                    ort_inputs[input_names[1]] = attention_mask
                
                # 运行推理
                ort_outputs = session.run(output_names, ort_inputs)
                inference_time = time.time() - start_time
                test_result["inference_time"] = round(inference_time, 4)
                
                print(f"推理完成，耗时: {inference_time:.4f}秒")
                
                # 检查输出
                embedding = ort_outputs[0]
                print(f"输出形状: {embedding.shape}")
                test_result["embedding_shape"] = list(embedding.shape)
                
                # 保存嵌入维度
                if results["embedding_dim"] is None and len(embedding.shape) >= 2:
                    if len(embedding.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                        results["embedding_dim"] = embedding.shape[2]
                    elif len(embedding.shape) == 2:  # [batch_size, hidden_dim]
                        results["embedding_dim"] = embedding.shape[1]
                
                if verbose:
                    # 计算嵌入向量的统计信息
                    if len(embedding.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                        # 对序列长度维度取平均，得到每个样本的嵌入向量
                        avg_embedding = np.mean(embedding, axis=1)
                        sample_embedding = avg_embedding[0]
                    elif len(embedding.shape) == 2:  # [batch_size, hidden_dim]
                        sample_embedding = embedding[0]
                    else:
                        sample_embedding = embedding.flatten()
                    
                    # 计算统计信息
                    stats = {
                        "min": float(np.min(sample_embedding)),
                        "max": float(np.max(sample_embedding)),
                        "mean": float(np.mean(sample_embedding)),
                        "std": float(np.std(sample_embedding)),
                        "norm": float(np.linalg.norm(sample_embedding))
                    }
                    test_result["embedding_stats"] = stats
                    
                    print(f"嵌入向量统计信息:")
                    print(f"  最小值: {stats['min']:.4f}")
                    print(f"  最大值: {stats['max']:.4f}")
                    print(f"  均值: {stats['mean']:.4f}")
                    print(f"  标准差: {stats['std']:.4f}")
                    print(f"  L2范数: {stats['norm']:.4f}")
                
                # 验证输出是否合理
                if np.isnan(embedding).any():
                    error = "警告: 输出包含NaN值"
                    print(error)
                    test_result["error"] = error
                    continue
                
                if np.isinf(embedding).any():
                    error = "警告: 输出包含无穷大值"
                    print(error)
                    test_result["error"] = error
                    continue
                
                test_result["success"] = True
                
            except Exception as e:
                error = f"错误: 测试样本 {i+1} 推理失败: {str(e)}"
                print(error)
                test_result["error"] = str(e)
            
            results["test_results"].append(test_result)
        
        # 检查测试结果
        success_count = sum(1 for result in results["test_results"] if result["success"])
        if success_count == len(test_texts):
            print("\n验证成功: ONNX词嵌入模型可以正常加载和执行推理")
            results["status"] = "成功"
        elif success_count > 0:
            print(f"\n部分验证成功: {success_count}/{len(test_texts)} 个测试样本通过")
            results["status"] = "部分成功"
        else:
            print("\n验证失败: 所有测试样本都失败")
            results["status"] = "失败"
        
        return results
        
    except Exception as e:
        error = f"错误: ONNX模型验证失败: {str(e)}"
        print(error)
        import traceback
        print(traceback.format_exc())
        results["status"] = "失败"
        results["errors"].append(error)
        return results

def main():
    parser = argparse.ArgumentParser(description="ONNX词嵌入模型验证工具")
    parser.add_argument("model_path", help="ONNX模型路径")
    parser.add_argument("--output", "-o", help="结果输出JSON文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    results = verify_onnx_embedding(args.model_path, args.verbose)
    
    # 输出结果摘要
    print("\n=== 验证结果摘要 ===")
    print(f"模型: {results['model_name']}")
    print(f"状态: {results['status']}")
    if results["embedding_dim"]:
        print(f"嵌入维度: {results['embedding_dim']}")
    if results["errors"]:
        print(f"错误数量: {len(results['errors'])}")
        for i, error in enumerate(results["errors"]):
            print(f"  错误 {i+1}: {error}")
    
    # 保存结果到JSON文件
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
