"""
模型诊断工具 - 用于检查模型文件和结构，帮助诊断加载问题
"""
import os
import sys
import json
import traceback
import codecs

def check_model_directory(model_path, log_file=None):
    """检查模型目录的完整性"""
    def log(message):
        print(message)
        if log_file:
            log_file.write(message + "\n")
    
    log(f"开始诊断模型: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        log(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查模型文件
    model_files = os.listdir(model_path)
    log(f"模型目录内容 ({len(model_files)}个文件): {', '.join(model_files)}")
    
    # 必要的文件检查
    required_files = ["config.json", "tokenizer_config.json", "generation_config.json"]
    for req_file in required_files:
        if req_file in model_files:
            file_path = os.path.join(model_path, req_file)
            file_size = os.path.getsize(file_path)
            log(f"找到必要文件: {req_file}, 大小: {file_size} 字节")
            
            # 检查JSON文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                log(f"{req_file} 内容解析成功，包含 {len(data)} 个键")
                
                # 对特定文件进行更详细的检查
                if req_file == "config.json":
                    arch = data.get('architectures', ['未知'])
                    model_type = data.get('model_type', '未知')
                    log(f"模型架构: {arch}, 模型类型: {model_type}")
                
                if req_file == "generation_config.json":
                    bos_token_id = data.get('bos_token_id', '未知')
                    eos_token_id = data.get('eos_token_id', '未知')
                    log(f"生成配置: bos_token_id={bos_token_id}, eos_token_id={eos_token_id}")
                    
                    # 检查生成配置中的token ID是否一致
                    if 'config.json' in model_files:
                        config_path = os.path.join(model_path, 'config.json')
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                            config_bos = config_data.get('bos_token_id')
                            config_eos = config_data.get('eos_token_id')
                            
                            if config_bos is not None and bos_token_id != config_bos:
                                log(f"警告: bos_token_id不一致 - config.json: {config_bos}, generation_config.json: {bos_token_id}")
                            
                            if config_eos is not None and eos_token_id != config_eos:
                                log(f"警告: eos_token_id不一致 - config.json: {config_eos}, generation_config.json: {eos_token_id}")
                        except Exception as e:
                            log(f"比较token ID时出错: {str(e)}")
            except Exception as e:
                log(f"错误: 解析 {req_file} 失败: {str(e)}")
                log(traceback.format_exc())
        else:
            log(f"警告: 未找到必要文件 {req_file}")
    
    # 检查模型权重文件
    weight_files = [f for f in model_files if f.endswith('.bin') or f.endswith('.safetensors')]
    if weight_files:
        log(f"找到 {len(weight_files)} 个权重文件:")
        for wf in weight_files:
            wf_path = os.path.join(model_path, wf)
            wf_size = os.path.getsize(wf_path)
            log(f"  - {wf}: {wf_size} 字节 ({wf_size / 1024 / 1024:.2f} MB)")
    else:
        log("警告: 未找到任何权重文件 (.bin 或 .safetensors)")
    
    # 特别检查safetensors索引文件
    index_files = [f for f in model_files if f.endswith('.safetensors.index.json')]
    if index_files:
        log(f"找到 {len(index_files)} 个safetensors索引文件:")
        for idx_file in index_files:
            idx_path = os.path.join(model_path, idx_file)
            idx_size = os.path.getsize(idx_path)
            log(f"  - {idx_file}: {idx_size} 字节")
            
            # 检查索引文件内容 - 尝试多种编码
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            index_content = None
            
            for encoding in encodings:
                try:
                    with open(idx_path, 'r', encoding=encoding) as f:
                        index_content = f.read()
                    log(f"使用 {encoding} 编码成功读取索引文件")
                    break
                except UnicodeDecodeError:
                    continue
            
            if index_content is None:
                # 如果所有编码都失败，尝试二进制读取
                try:
                    with open(idx_path, 'rb') as f:
                        binary_content = f.read()
                    # 检查BOM标记
                    if binary_content.startswith(b'\xef\xbb\xbf'):
                        log("检测到UTF-8 BOM标记")
                        binary_content = binary_content[3:]
                    index_content = binary_content.decode('utf-8', errors='replace')
                    log("使用二进制模式读取并转换为UTF-8")
                except Exception as e:
                    log(f"所有编码读取尝试都失败: {str(e)}")
            
            if index_content:
                log(f"索引文件内容长度: {len(index_content)} 字符")
                
                # 尝试解析JSON
                try:
                    # 尝试修复常见的JSON问题
                    # 1. 移除可能的BOM标记
                    if index_content.startswith('\ufeff'):
                        index_content = index_content[1:]
                        log("移除了BOM标记")
                    
                    # 2. 检查并修复不正确的引号
                    if "'" in index_content:
                        index_content = index_content.replace("'", "\"")
                        log("将单引号替换为双引号")
                    
                    index_data = json.loads(index_content)
                    log(f"索引文件JSON解析成功，包含 {len(index_data)} 个键")
                    
                    # 保存解析后的JSON到文件以便检查
                    parsed_json_path = os.path.join(os.path.dirname(model_path), "parsed_index.json")
                    with open(parsed_json_path, 'w', encoding='utf-8') as f:
                        json.dump(index_data, f, ensure_ascii=False, indent=2)
                    log(f"已保存解析后的JSON到: {parsed_json_path}")
                    
                    # 检查weight_map键
                    if 'weight_map' in index_data:
                        weight_map = index_data['weight_map']
                        log(f"weight_map包含 {len(weight_map)} 个条目")
                        
                        # 检查权重文件是否存在
                        missing_weights = []
                        for weight_name, file_name in weight_map.items():
                            weight_file = os.path.join(model_path, file_name)
                            if not os.path.exists(weight_file):
                                missing_weights.append((weight_name, file_name))
                        
                        if missing_weights:
                            log(f"警告: 有 {len(missing_weights)} 个权重文件缺失:")
                            for weight_name, file_name in missing_weights[:10]:  # 只显示前10个
                                log(f"  - 权重 {weight_name} -> 文件 {file_name} 不存在")
                            if len(missing_weights) > 10:
                                log(f"  ... 以及 {len(missing_weights) - 10} 个其他缺失文件")
                        else:
                            log("所有权重文件都存在")
                    else:
                        log("警告: 索引文件中没有weight_map键")
                    
                    # 检查metadata键
                    if 'metadata' in index_data:
                        metadata = index_data['metadata']
                        log(f"metadata: {json.dumps(metadata, ensure_ascii=False)}")
                except json.JSONDecodeError as je:
                    log(f"错误: 索引文件JSON解析失败: {str(je)}")
                    # 显示文件前100个字符和最后100个字符作为诊断信息
                    log(f"文件内容前100个字符: {index_content[:100]}")
                    log(f"文件内容最后100个字符: {index_content[-100:] if len(index_content) > 100 else index_content}")
                    
                    # 尝试修复并保存索引文件
                    try:
                        log("尝试修复索引文件...")
                        fixed_path = os.path.join(os.path.dirname(model_path), "fixed_" + idx_file)
                        with open(fixed_path, 'w', encoding='utf-8') as f:
                            # 尝试创建一个基本的索引文件结构
                            weight_files_dict = {}
                            for wf in weight_files:
                                if wf.endswith('.safetensors'):
                                    # 为每个权重文件创建一个占位符条目
                                    weight_files_dict[f"placeholder_{wf}"] = wf
                            
                            fixed_data = {
                                "weight_map": weight_files_dict,
                                "metadata": {"total_size": sum(os.path.getsize(os.path.join(model_path, wf)) for wf in weight_files)}
                            }
                            json.dump(fixed_data, f, ensure_ascii=False, indent=2)
                        log(f"已创建修复后的索引文件: {fixed_path}")
                    except Exception as fix_e:
                        log(f"修复尝试失败: {str(fix_e)}")
            else:
                log("无法读取索引文件内容")
    else:
        log("未找到safetensors索引文件")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("用法: python model_diagnostic.py <模型路径>")
        return
    
    model_path = sys.argv[1]
    log_path = os.path.join(os.path.dirname(model_path), "model_diagnostic_log.txt")
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        check_model_directory(model_path, log_file)
    
    print(f"诊断完成，日志已保存到: {log_path}")

if __name__ == "__main__":
    main()
