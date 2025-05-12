"""
ONNX模型验证工具 - 用于验证转换后的ONNX模型是否可用
"""
import os
import sys
import time
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer

def verify_onnx_model(model_path, verbose=True):
    """验证ONNX模型的有效性"""
    print(f"开始验证ONNX模型: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查模型文件
    model_files = os.listdir(model_path)
    print(f"模型目录内容 ({len(model_files)}个文件): {', '.join(model_files)}")
    
    # 检查必要文件
    onnx_model_path = os.path.join(model_path, "model.onnx")
    if not os.path.exists(onnx_model_path):
        print(f"错误: ONNX模型文件不存在: {onnx_model_path}")
        return False
    
    # 检查模型大小
    model_size = os.path.getsize(onnx_model_path)
    print(f"ONNX模型文件大小: {model_size} 字节 ({model_size / 1024:.2f} KB)")
    
    if model_size < 1000:
        print(f"警告: ONNX模型文件过小，可能只包含嵌入层或存在问题")
    
    # 加载分词器
    try:
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"分词器加载成功，词汇表大小: {len(tokenizer)}")
    except Exception as e:
        print(f"错误: 加载分词器失败: {str(e)}")
        return False
    
    # 加载ONNX模型
    try:
        print("加载ONNX模型...")
        start_time = time.time()
        session = ort.InferenceSession(onnx_model_path)
        load_time = time.time() - start_time
        print(f"ONNX模型加载成功，耗时: {load_time:.2f}秒")
        
        # 获取模型输入输出信息
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        print(f"模型输入名称: {input_names}")
        print(f"模型输出名称: {output_names}")
        
        # 准备输入数据
        test_text = "你好，我是一个语言模型。"
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs.input_ids.numpy()
        
        print(f"测试输入: '{test_text}'")
        print(f"输入形状: {input_ids.shape}")
        print(f"输入token IDs: {input_ids[0]}")
        
        # 执行推理
        print("执行ONNX模型推理...")
        start_time = time.time()
        
        # 准备输入字典
        ort_inputs = {input_names[0]: input_ids}
        
        # 运行推理
        ort_outputs = session.run(output_names, ort_inputs)
        inference_time = time.time() - start_time
        
        print(f"推理完成，耗时: {inference_time:.2f}秒")
        
        # 检查输出
        output = ort_outputs[0]
        print(f"输出形状: {output.shape}")
        
        if verbose:
            # 显示部分输出
            if len(output.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                print(f"输出示例 (前5个值): {output[0, 0, :5]}")
            elif len(output.shape) == 2:  # [batch_size, hidden_dim]
                print(f"输出示例 (前5个值): {output[0, :5]}")
            else:
                print(f"输出示例 (前5个值): {output.flatten()[:5]}")
        
        # 验证输出是否合理
        if np.isnan(output).any():
            print("警告: 输出包含NaN值")
            return False
        
        if np.isinf(output).any():
            print("警告: 输出包含无穷大值")
            return False
        
        print("验证成功: ONNX模型可以正常加载和执行推理")
        return True
        
    except Exception as e:
        print(f"错误: ONNX模型验证失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python verify_onnx_model.py <ONNX模型路径>")
        return
    
    model_path = sys.argv[1]
    verify_onnx_model(model_path)

if __name__ == "__main__":
    main()
