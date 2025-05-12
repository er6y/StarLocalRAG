"""
Gemma模型加载测试脚本 - 用于诊断模型加载问题
"""
import os
import sys
import torch
import json
import time
import traceback
from pathlib import Path

def test_gemma_model_loading(model_path):
    """测试Gemma模型加载"""
    print(f"开始测试Gemma模型加载: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查模型文件
    model_files = os.listdir(model_path)
    print(f"模型目录内容 ({len(model_files)}个文件): {', '.join(model_files)}")
    
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 尝试导入必要的库
    try:
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        print("成功导入transformers库")
    except ImportError as e:
        print(f"错误: 无法导入transformers库: {str(e)}")
        return False
    
    # 尝试加载tokenizer
    try:
        print("尝试加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"成功加载tokenizer, 词汇表大小: {len(tokenizer)}")
    except Exception as e:
        print(f"错误: 加载tokenizer失败: {str(e)}")
        print(traceback.format_exc())
        return False
    
    # 尝试加载模型配置
    try:
        print("尝试加载模型配置...")
        config = AutoConfig.from_pretrained(model_path)
        print(f"成功加载模型配置, 模型类型: {config.model_type}")
        print(f"模型配置详情: {config}")
    except Exception as e:
        print(f"错误: 加载模型配置失败: {str(e)}")
        print(traceback.format_exc())
        # 尝试继续，因为有些模型可能没有标准配置文件
    
    # 尝试加载模型
    print("\n===== 尝试方法1: 使用AutoModelForCausalLM =====")
    try:
        print("尝试使用AutoModelForCausalLM加载模型...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        elapsed = time.time() - start_time
        print(f"成功加载模型! 耗时: {elapsed:.2f}秒")
        print(f"模型类型: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"错误: 使用AutoModelForCausalLM加载模型失败: {str(e)}")
        print(traceback.format_exc())
    
    print("\n===== 尝试方法2: 使用特定的GemmaForCausalLM类 =====")
    try:
        from transformers import GemmaForCausalLM, GemmaConfig
        print("尝试使用GemmaForCausalLM加载模型...")
        start_time = time.time()
        model = GemmaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        elapsed = time.time() - start_time
        print(f"成功加载模型! 耗时: {elapsed:.2f}秒")
        print(f"模型类型: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"错误: 使用GemmaForCausalLM加载模型失败: {str(e)}")
        print(traceback.format_exc())
    
    print("\n===== 尝试方法3: 使用CPU加载 =====")
    try:
        print("尝试使用CPU加载模型...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # 使用float32以避免可能的精度问题
            trust_remote_code=True,
            device_map=None,  # 不使用device_map
            low_cpu_mem_usage=True
        )
        elapsed = time.time() - start_time
        print(f"成功加载模型! 耗时: {elapsed:.2f}秒")
        print(f"模型类型: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"错误: 使用CPU加载模型失败: {str(e)}")
        print(traceback.format_exc())
    
    print("\n所有加载方法都失败了。")
    return False

def main():
    if len(sys.argv) < 2:
        print("用法: python gemma_model_test.py <模型路径>")
        return
    
    model_path = sys.argv[1]
    log_path = os.path.join(os.path.dirname(model_path), "gemma_model_test_log.txt")
    
    # 将输出重定向到文件
    original_stdout = sys.stdout
    with open(log_path, 'w', encoding='utf-8') as log_file:
        sys.stdout = log_file
        success = test_gemma_model_loading(model_path)
        sys.stdout = original_stdout
    
    if success:
        print(f"模型加载测试成功! 详细日志已保存到: {log_path}")
    else:
        print(f"模型加载测试失败。详细日志已保存到: {log_path}")
        
        # 读取日志文件并显示最后几行
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print("\n最后的错误信息:")
            for line in lines[-20:]:  # 显示最后20行
                print(line.strip())

if __name__ == "__main__":
    main()
