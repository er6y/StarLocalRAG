import os
import sys
import shutil

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
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, model_dir)
            all_files.append(rel_path)
    
    # 检查必要文件是否存在
    missing_files = []
    for file in essential_files:
        if file not in all_files and not os.path.exists(os.path.join(model_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"[WARNING] 以下必要文件不存在:")
        for file in missing_files:
            print(f"  - {file}")
    
    # 找出不必要的文件
    unnecessary_files = []
    for file in all_files:
        if file not in essential_files and not any(file.endswith(ext) for ext in [".md", ".txt"]):
            unnecessary_files.append(file)
    
    # 删除不必要的文件
    if unnecessary_files:
        print(f"[INFO] 删除以下不必要的文件:")
        deleted_count = 0
        for file in unnecessary_files:
            file_path = os.path.join(model_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    if deleted_count <= 10:
                        print(f"  - {file}")
                    elif deleted_count == 11:
                        print(f"  - ... 以及其他 {len(unnecessary_files) - 10} 个文件")
            except Exception as e:
                print(f"  [ERROR] 无法删除 {file}: {str(e)}")
        
        print(f"[INFO] 共删除 {deleted_count} 个不必要的文件")
    else:
        print("[INFO] 没有找到不必要的文件")
    
    # 计算清理前后的大小
    total_size_before = sum(os.path.getsize(os.path.join(model_dir, f)) for f in all_files if os.path.isfile(os.path.join(model_dir, f)))
    remaining_files = [f for f in all_files if f in essential_files or any(f.endswith(ext) for ext in [".md", ".txt"])]
    total_size_after = sum(os.path.getsize(os.path.join(model_dir, f)) for f in remaining_files if os.path.isfile(os.path.join(model_dir, f)))
    
    print(f"[INFO] 清理前大小: {total_size_before / (1024 * 1024):.2f} MB")
    print(f"[INFO] 清理后大小: {total_size_after / (1024 * 1024):.2f} MB")
    print(f"[INFO] 节省空间: {(total_size_before - total_size_after) / (1024 * 1024):.2f} MB ({(1 - total_size_after / total_size_before) * 100:.2f}%)")
    
    print("[INFO] ONNX模型目录清理完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python clean_onnx_model.py <ONNX模型目录>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    if not os.path.exists(model_dir):
        print(f"[ERROR] 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    clean_onnx_model_dir(model_dir)
