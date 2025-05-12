import os
import json
import shutil
from pathlib import Path

def create_gemma_config(model_path):
    """为Gemma模型创建必要的配置文件"""
    print(f"正在为模型创建配置文件: {model_path}")
    
    # 检查模型目录是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查模型文件
    model_files = os.listdir(model_path)
    print(f"模型目录内容: {', '.join(model_files)}")
    
    # 检查是否有model.safetensors文件
    has_safetensors = any(f == "model.safetensors" for f in model_files)
    
    if not has_safetensors:
        print("错误: 未找到model.safetensors文件，这可能不是标准的Gemma模型")
        return False
    
    # 创建config.json - 根据诊断结果修正参数
    config = {
        "architectures": ["GemmaForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "hidden_act": "gelu",
        "hidden_size": 1152,  # 修正为1152，而不是3072
        "initializer_range": 0.02,
        "intermediate_size": 9216,  # 调整为hidden_size的8倍
        "max_position_embeddings": 8192,
        "model_type": "gemma",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 16,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.36.2",
        "use_cache": True,
        "vocab_size": 262144  # 修正为262144，而不是262145
    }
    
    # 写入config.json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"已创建配置文件: {config_path}")
    
    # 创建tokenizer_config.json
    tokenizer_config = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "model_max_length": 8192,
        "tokenizer_class": "GemmaTokenizer",
        "unk_token": "<unk>"
    }
    
    # 写入tokenizer_config.json
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"已创建分词器配置文件: {tokenizer_config_path}")
    
    # 创建generation_config.json
    generation_config = {
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "transformers_version": "4.36.2"
    }
    
    # 写入generation_config.json
    generation_config_path = os.path.join(model_path, "generation_config.json")
    with open(generation_config_path, 'w', encoding='utf-8') as f:
        json.dump(generation_config, f, indent=2)
    
    print(f"已创建生成配置文件: {generation_config_path}")
    
    return True

if __name__ == "__main__":
    # 使用示例
    model_path = r"d:\work\localrag\models\gemma-3-1b-it"
    create_gemma_config(model_path)
    print("配置文件创建完成，请重新尝试转换模型")
