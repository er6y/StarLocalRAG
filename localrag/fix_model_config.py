"""
修复模型配置文件中的token ID不一致问题
"""
import os
import sys
import json
import shutil

def fix_model_config(model_path):
    """修复模型配置文件中的token ID不一致问题"""
    print(f"开始修复模型配置: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查配置文件
    config_path = os.path.join(model_path, "config.json")
    gen_config_path = os.path.join(model_path, "generation_config.json")
    
    if not os.path.exists(config_path):
        print(f"错误: 未找到config.json文件")
        return False
    
    if not os.path.exists(gen_config_path):
        print(f"错误: 未找到generation_config.json文件")
        return False
    
    # 读取config.json
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"成功读取config.json")
    except Exception as e:
        print(f"错误: 读取config.json失败: {str(e)}")
        return False
    
    # 读取generation_config.json
    try:
        with open(gen_config_path, 'r', encoding='utf-8') as f:
            gen_config_data = json.load(f)
        print(f"成功读取generation_config.json")
    except Exception as e:
        print(f"错误: 读取generation_config.json失败: {str(e)}")
        return False
    
    # 获取token IDs
    config_bos_id = config_data.get('bos_token_id')
    config_eos_id = config_data.get('eos_token_id')
    gen_bos_id = gen_config_data.get('bos_token_id')
    gen_eos_id = gen_config_data.get('eos_token_id')
    
    print(f"config.json中的token IDs: bos={config_bos_id}, eos={config_eos_id}")
    print(f"generation_config.json中的token IDs: bos={gen_bos_id}, eos={gen_eos_id}")
    
    # 检查是否需要修复
    needs_fix = False
    if config_bos_id is not None and gen_bos_id is not None and config_bos_id != gen_bos_id:
        print(f"需要修复bos_token_id: {gen_bos_id} -> {config_bos_id}")
        gen_config_data['bos_token_id'] = config_bos_id
        needs_fix = True
    
    if config_eos_id is not None and gen_eos_id is not None and config_eos_id != gen_eos_id:
        print(f"需要修复eos_token_id: {gen_eos_id} -> {config_eos_id}")
        gen_config_data['eos_token_id'] = config_eos_id
        needs_fix = True
    
    if not needs_fix:
        print("配置文件一致，无需修复")
        return True
    
    # 备份原始文件
    backup_path = gen_config_path + ".backup"
    shutil.copy2(gen_config_path, backup_path)
    print(f"已备份原始文件到: {backup_path}")
    
    # 写入修复后的配置
    try:
        with open(gen_config_path, 'w', encoding='utf-8') as f:
            json.dump(gen_config_data, f, ensure_ascii=False, indent=2)
        print(f"成功写入修复后的generation_config.json")
        return True
    except Exception as e:
        print(f"错误: 写入修复后的配置失败: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python fix_model_config.py <模型路径>")
        return
    
    model_path = sys.argv[1]
    success = fix_model_config(model_path)
    
    if success:
        print("修复完成！请重新尝试加载模型。")
    else:
        print("修复失败，请检查错误信息。")

if __name__ == "__main__":
    main()
