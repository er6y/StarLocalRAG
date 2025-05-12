import os
import sys
import shutil

def clean_onnx_model_dir(model_dir):
    """清理ONNX模型目录，只保留必要的文件"""
    # 必要的文件列表
    essential_files = [
        "model.onnx",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "config.json",
        "config_sentence_transformers.json",
        "sentencepiece.bpe.model",
        "README.md",
        "inference.py"
    ]
    
    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(model_dir), "temp_onnx_model")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 复制必要的文件到临时目录
    print(f"[INFO] 复制必要的文件到临时目录...")
    for file in essential_files:
        src_path = os.path.join(model_dir, file)
        if os.path.exists(src_path):
            dst_path = os.path.join(temp_dir, file)
            shutil.copy2(src_path, dst_path)
            print(f"  - 已复制: {file}")
    
    # 删除原目录
    print(f"[INFO] 删除原目录: {model_dir}")
    shutil.rmtree(model_dir)
    
    # 重命名临时目录为原目录名
    print(f"[INFO] 重命名临时目录为: {model_dir}")
    shutil.move(temp_dir, model_dir)
    
    print("[INFO] ONNX模型目录清理完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python clean_onnx_model_simple.py <ONNX模型目录>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    if not os.path.exists(model_dir):
        print(f"[ERROR] 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    clean_onnx_model_dir(model_dir)
