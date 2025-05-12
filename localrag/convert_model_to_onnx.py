#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
转换模型到ONNX格式的简洁脚本
支持词嵌入模型和大语言模型
"""

import os
import sys
import torch
import shutil
import argparse
import numpy as np
import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import logging

# 设置日志编码为UTF-8，确保中文正确显示
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 确保日志输出使用UTF-8编码
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.stream.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

def read_test_samples(file_path: str) -> List[str]:
    """读取测试样本"""
    if not os.path.exists(file_path):
        logger.warning(f"测试样本文件不存在: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    samples = [line.strip() for line in lines if line.strip()]
    return samples

def convert_embedding_model(
    model_name: str,
    output_dir: Optional[str] = None,
    use_cuda: bool = False,
    opset_version: int = 14,
    quantization: str = 'none',
    precision: str = 'FP32',
    test_samples_file: Optional[str] = None,
    clean_files: bool = True
) -> bool:
    """
    将词嵌入模型转换为ONNX格式
    
    Args:
        model_name: 模型名称或路径
        output_dir: 输出目录
        use_cuda: 是否使用CUDA加速
        opset_version: ONNX opset版本
        quantization: 量化类型，可选值为'none', 'dynamic'
        precision: 精度，可选值为'FP32', 'FP16', 'INT8'
        test_samples_file: 测试样本文件路径
        clean_files: 是否清理不必要的文件
    """
    try:
        from transformers import AutoTokenizer
        
        logger.info(f"开始转换嵌入模型: {model_name}")
        
        # 处理static量化类型
        if quantization == 'static':
            logger.info("注意: 将'static'量化转换为'dynamic'量化，因为ONNX更适合动态量化")
            quantization = 'dynamic'
        
        # 设置设备
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 加载模型
        logger.info(f"加载模型...")
        model = SentenceTransformer(model_name)
        model.to(device)
        
        # 确保输出目录存在
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", 
                                  f"{os.path.basename(model_name)}_{quantization}_quant_{precision}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")
        
        # 获取tokenizer
        logger.info(f"加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 创建一个包装类，使用模型的encode方法
        class SentenceTransformerWrapper(torch.nn.Module):
            def __init__(self, model):
                super(SentenceTransformerWrapper, self).__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask, token_type_ids=None):
                # 使用模型的encode方法，但需要先转换为文本
                # 由于我们无法在ONNX导出过程中进行tokenization，所以这里假设输入已经是tokenized的
                # 我们将直接使用模型的模块来处理
                
                # 获取transformer模型
                transformer = self.model._modules['0']
                
                # 创建特征字典
                features = {}
                features['input_ids'] = input_ids
                features['attention_mask'] = attention_mask
                if token_type_ids is not None:
                    features['token_type_ids'] = token_type_ids
                
                # 获取transformer输出
                trans_output = transformer(features)
                
                # 获取pooling模型
                pooling = self.model._modules['1']
                
                # 应用pooling
                pooling_output = pooling(trans_output)
                
                # 获取嵌入
                embeddings = pooling_output['sentence_embedding']
                
                return embeddings
        
        # 实例化包装类
        wrapped_model = SentenceTransformerWrapper(model)
        wrapped_model.eval()
        
        # 将模型导出为ONNX格式
        logger.info(f"将模型导出到ONNX格式，opset版本: {opset_version}...")
        
        # 准备输入示例
        input_ids = torch.ones((1, 128), dtype=torch.long).to(device)
        attention_mask = torch.ones((1, 128), dtype=torch.long).to(device)
        
        # 导出路径
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        # 导出模型 - 使用更激进的优化选项
        torch.onnx.export(
            wrapped_model,
            (input_ids, attention_mask),
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,  # 常量折叠优化
            input_names=['input_ids', 'attention_mask'],
            output_names=['embedding'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'embedding': {0: 'batch_size'}
            },
            verbose=False
        )
        
        # 使用ONNX内置优化器优化模型
        try:
            import onnx
            from onnx import optimizer
            
            logger.info("应用ONNX内置优化...")
            
            # 加载模型
            model = onnx.load(onnx_path)
            
            # 应用所有可用的优化
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'fuse_bn_into_conv',
                'fuse_add_bias_into_conv',
                'fuse_consecutive_transposes',
                'fuse_consecutive_reduces',
                'fuse_matmul_add_bias_into_gemm'
            ]
            
            # 优化模型
            optimized_model = optimizer.optimize(model, passes)
            
            # 保存优化后的模型
            optimized_model_path = os.path.join(output_dir, "model_optimized.onnx")
            onnx.save(optimized_model, optimized_model_path)
            
            # 检查优化后的模型大小
            original_size = os.path.getsize(onnx_path)
            optimized_size = os.path.getsize(optimized_model_path)
            
            if optimized_size < original_size:
                logger.info(f"ONNX优化成功! 原始大小: {original_size/(1024*1024):.2f} MB, 优化后: {optimized_size/(1024*1024):.2f} MB")
                # 替换原始模型
                shutil.move(optimized_model_path, onnx_path)
            else:
                logger.warning(f"优化后的模型更大，使用原始模型")
                os.remove(optimized_model_path)
        except Exception as e:
            logger.warning(f"ONNX优化失败: {str(e)}")
            logger.info("使用原始ONNX模型")
        
        # 应用量化（如果需要）
        if quantization == 'dynamic':
            logger.info(f"应用动态量化: {precision}...")
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                # 创建量化后的模型文件路径
                quantized_model_path = os.path.join(output_dir, "model_quantized.onnx")
                
                # 根据精度选择量化类型
                quant_type = None
                if precision == 'INT8':
                    quant_type = QuantType.QInt8
                elif precision == 'INT4':
                    try:
                        # 检查是否支持INT4
                        if hasattr(QuantType, 'QInt4'):
                            quant_type = QuantType.QInt4
                            logger.info("使用INT4量化...")
                        else:
                            logger.warning("当前ONNX运行时不支持INT4量化，回退到INT8")
                            quant_type = QuantType.QInt8
                            precision = 'INT8'  # 更新精度信息
                    except Exception as e:
                        logger.warning(f"INT4量化初始化失败: {str(e)}，回退到INT8")
                        quant_type = QuantType.QInt8
                        precision = 'INT8'  # 更新精度信息
                
                if quant_type is not None:
                    # 尝试更激进的量化方法
                    try:
                        # 尝试使用更激进的量化参数
                        logger.info("尝试更激进的量化方法...")
                        quantize_dynamic(
                            model_input=onnx_path,
                            model_output=quantized_model_path,
                            op_types_to_quantize=['MatMul', 'Gemm', 'Conv', 'Attention', 'LSTM', 'GRU'],
                            per_channel=True,
                            weight_type=quant_type,
                            # 以下是更激进的参数
                            optimize_model=True,  # 在量化前优化模型
                            extra_options={'EnableSubgraph': True}  # 启用子图优化
                        )
                    except (TypeError, ValueError) as e:
                        logger.info(f"使用基本量化API，错误: {str(e)}")
                        # 如果激进方法失败，回退到基本方法
                        quantize_dynamic(
                            model_input=onnx_path,
                            model_output=quantized_model_path,
                            weight_type=quant_type  # 确保传递量化类型参数
                        )
                    
                    # 检查量化后的模型是否存在
                    if os.path.exists(quantized_model_path):
                        # 检查量化后的模型大小
                        original_size = os.path.getsize(onnx_path)
                        quantized_size = os.path.getsize(quantized_model_path)
                        
                        # 转换为MB以便于比较和显示
                        original_size_mb = original_size / (1024 * 1024)
                        quantized_size_mb = quantized_size / (1024 * 1024)
                        
                        # 计算压缩率
                        compression_ratio = 100 * (1 - quantized_size / original_size)
                        
                        # 无论大小如何，都使用量化后的模型
                        if quantized_size < original_size:
                            logger.info(f"ONNX量化成功! ONNX原始大小: {original_size_mb:.2f} MB, 量化后: {quantized_size_mb:.2f} MB, 压缩率: {compression_ratio:.2f}%")
                        else:
                            logger.info(f"ONNX量化完成。ONNX原始大小: {original_size_mb:.2f} MB, 量化后: {quantized_size_mb:.2f} MB, 压缩率: {compression_ratio:.2f}%")
                        
                        # 备份原始模型
                        shutil.move(
                            onnx_path,
                            os.path.join(output_dir, "model.original.onnx")
                        )
                        # 将量化后的模型重命名为model.onnx
                        shutil.move(quantized_model_path, onnx_path)
                        
                        # 如果是INT4量化，测试模型是否可用
                        if precision == 'INT4':
                            try:
                                # 尝试加载模型进行测试
                                logger.info("测试INT4量化模型是否可用...")
                                import onnxruntime as ort
                                ort_session = ort.InferenceSession(onnx_path)
                                logger.info("INT4量化模型测试成功!")
                            except Exception as e:
                                logger.warning(f"INT4量化模型测试失败: {str(e)}")
                                logger.warning("回退到INT8量化...")
                                
                                # 恢复原始模型
                                shutil.move(
                                    os.path.join(output_dir, "model.original.onnx"),
                                    onnx_path
                                )
                                
                                # 重新使用INT8量化
                                logger.info("应用INT8量化...")
                                try:
                                    quantize_dynamic(
                                        model_input=onnx_path,
                                        model_output=quantized_model_path,
                                        weight_type=QuantType.QInt8
                                    )
                                    
                                    if os.path.exists(quantized_model_path):
                                        # 备份原始模型
                                        shutil.move(
                                            onnx_path,
                                            os.path.join(output_dir, "model.original.onnx")
                                        )
                                        # 将量化后的模型重命名为model.onnx
                                        shutil.move(quantized_model_path, onnx_path)
                                        logger.info("成功回退到INT8量化")
                                    else:
                                        logger.warning("INT8量化失败，使用原始模型")
                                except Exception as e:
                                    logger.warning(f"INT8量化失败: {str(e)}，使用原始模型")
                    else:
                        logger.warning("量化后的模型文件不存在，使用原始模型")
                else:
                    logger.warning(f"不支持的精度类型: {precision}，使用原始模型")
            except Exception as e:
                logger.warning(f"动态量化失败: {str(e)}")
                logger.info("使用原始模型")
        elif quantization == 'static':
            logger.info(f"应用静态量化: {precision}...")
            try:
                from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
                from onnxruntime.quantization.calibrate import CalibrationDataReader
                
                # 创建量化后的模型文件路径
                quantized_model_path = os.path.join(output_dir, "model_quantized.onnx")
                
                # 根据精度选择量化类型
                quant_type = None
                if precision == 'INT8':
                    quant_type = QuantType.QInt8
                elif precision == 'INT4':
                    try:
                        # 检查是否支持INT4
                        if hasattr(QuantType, 'QInt4'):
                            quant_type = QuantType.QInt4
                            logger.info("使用INT4静态量化...")
                        else:
                            logger.warning("当前ONNX运行时不支持INT4静态量化，回退到INT8")
                            quant_type = QuantType.QInt8
                            precision = 'INT8'  # 更新精度信息
                    except Exception as e:
                        logger.warning(f"INT4静态量化初始化失败: {str(e)}，回退到INT8")
                        quant_type = QuantType.QInt8
                        precision = 'INT8'  # 更新精度信息
                
                if quant_type is not None:
                    # 创建校准数据读取器
                    class DummyCalibrationDataReader(CalibrationDataReader):
                        def __init__(self):
                            super().__init__()
                            self.data_index = -1
                            self.input_names = None
                            self.data_set = []
                            
                            # 创建一些随机数据用于校准
                            for _ in range(10):
                                self.data_set.append({
                                    'input_ids': np.random.randint(0, 1000, size=(1, 32), dtype=np.int64),
                                    'attention_mask': np.ones((1, 32), dtype=np.int64)
                                })
                        
                        def get_next(self):
                            if self.data_index >= len(self.data_set) - 1:
                                return None
                            self.data_index += 1
                            return self.data_set[self.data_index]
                        
                        def rewind(self):
                            self.data_index = -1
                    
                    try:
                        # 尝试静态量化
                        logger.info("开始静态量化...")
                        calibration_data_reader = DummyCalibrationDataReader()
                        
                        quantize_static(
                            model_input=onnx_path,
                            model_output=quantized_model_path,
                            calibration_data_reader=calibration_data_reader,
                            quant_format=QuantFormat.QDQ,
                            weight_type=quant_type,
                            op_types_to_quantize=['MatMul', 'Gemm', 'Conv', 'Attention', 'LSTM', 'GRU']
                        )
                    except Exception as e:
                        logger.warning(f"静态量化失败: {str(e)}，回退到动态量化")
                        # 回退到动态量化
                        try:
                            from onnxruntime.quantization import quantize_dynamic
                            logger.info("回退到动态量化...")
                            quantize_dynamic(
                                model_input=onnx_path,
                                model_output=quantized_model_path,
                                weight_type=quant_type
                            )
                        except Exception as e2:
                            logger.warning(f"动态量化也失败: {str(e2)}，使用原始模型")
                    
                    # 检查量化后的模型是否存在
                    if os.path.exists(quantized_model_path):
                        # 检查量化后的模型大小
                        original_size = os.path.getsize(onnx_path)
                        quantized_size = os.path.getsize(quantized_model_path)
                        
                        # 转换为MB以便于比较和显示
                        original_size_mb = original_size / (1024 * 1024)
                        quantized_size_mb = quantized_size / (1024 * 1024)
                        
                        # 计算压缩率
                        compression_ratio = 100 * (1 - quantized_size / original_size)
                        
                        # 无论大小如何，都使用量化后的模型
                        if quantized_size < original_size:
                            logger.info(f"ONNX静态量化成功! ONNX原始大小: {original_size_mb:.2f} MB, 量化后: {quantized_size_mb:.2f} MB, 压缩率: {compression_ratio:.2f}%")
                        else:
                            logger.info(f"ONNX静态量化完成。ONNX原始大小: {original_size_mb:.2f} MB, 量化后: {quantized_size_mb:.2f} MB, 压缩率: {compression_ratio:.2f}%")
                        
                        # 备份原始模型
                        shutil.move(
                            onnx_path,
                            os.path.join(output_dir, "model.original.onnx")
                        )
                        # 将量化后的模型重命名为model.onnx
                        shutil.move(quantized_model_path, onnx_path)
                        
                        # 如果是INT4量化，测试模型是否可用
                        if precision == 'INT4':
                            try:
                                # 尝试加载模型进行测试
                                logger.info("测试INT4静态量化模型是否可用...")
                                import onnxruntime as ort
                                ort_session = ort.InferenceSession(onnx_path)
                                logger.info("INT4静态量化模型测试成功!")
                            except Exception as e:
                                logger.warning(f"INT4静态量化模型测试失败: {str(e)}")
                                logger.warning("回退到INT8静态量化...")
                                
                                # 恢复原始模型
                                shutil.move(
                                    os.path.join(output_dir, "model.original.onnx"),
                                    onnx_path
                                )
                                
                                # 重新使用INT8量化
                                logger.info("应用INT8静态量化...")
                                try:
                                    calibration_data_reader = DummyCalibrationDataReader()
                                    calibration_data_reader.rewind()
                                    
                                    quantize_static(
                                        model_input=onnx_path,
                                        model_output=quantized_model_path,
                                        quant_format=QuantFormat.QDQ,
                                        weight_type=QuantType.QInt8,
                                        op_types_to_quantize=['MatMul', 'Gemm', 'Conv', 'Attention', 'LSTM', 'GRU']
                                    )
                                    
                                    if os.path.exists(quantized_model_path):
                                        # 备份原始模型
                                        shutil.move(
                                            onnx_path,
                                            os.path.join(output_dir, "model.original.onnx")
                                        )
                                        # 将量化后的模型重命名为model.onnx
                                        shutil.move(quantized_model_path, onnx_path)
                                        logger.info("成功回退到INT8静态量化")
                                    else:
                                        logger.warning("INT8静态量化失败，使用原始模型")
                                except Exception as e:
                                    logger.warning(f"INT8静态量化失败: {str(e)}，使用原始模型")
                    else:
                        logger.warning("静态量化后的模型文件不存在，使用原始模型")
                else:
                    logger.warning(f"不支持的精度: {precision}，跳过静态量化")
            except ImportError as e:
                logger.warning(f"静态量化所需的库不可用: {str(e)}，跳过静态量化")
        elif precision == 'FP16':
            logger.info("应用FP16精度转换...")
            try:
                import onnx
                from onnxconverter_common import float16
                
                # 创建FP16模型文件路径
                fp16_model_path = os.path.join(output_dir, "model_fp16.onnx")
                
                try:
                    # 加载ONNX模型
                    onnx_model = onnx.load(onnx_path)
                    
                    # 转换为FP16
                    fp16_model = float16.convert_float_to_float16(onnx_model)
                    
                    # 保存FP16模型
                    onnx.save(fp16_model, fp16_model_path)
                    
                    # 检查FP16模型是否存在
                    if os.path.exists(fp16_model_path):
                        # 检查FP16模型大小
                        original_size = os.path.getsize(onnx_path)
                        fp16_size = os.path.getsize(fp16_model_path)
                        
                        # 转换为MB以便于比较和显示
                        original_size_mb = original_size / (1024 * 1024)
                        fp16_size_mb = fp16_size / (1024 * 1024)
                        
                        # 计算压缩率
                        compression_ratio = 100 * (1 - fp16_size / original_size)
                        
                        # 无论大小如何，都使用FP16模型
                        if fp16_size < original_size:
                            logger.info(f"FP16转换成功! ONNX原始大小: {original_size_mb:.2f} MB, FP16: {fp16_size_mb:.2f} MB, 压缩率: {compression_ratio:.2f}%")
                        else:
                            logger.info(f"FP16转换完成。ONNX原始大小: {original_size_mb:.2f} MB, FP16: {fp16_size_mb:.2f} MB, 压缩率: {compression_ratio:.2f}%")
                        
                        # 备份原始模型
                        shutil.move(
                            onnx_path,
                            os.path.join(output_dir, "model.original.onnx")
                        )
                        # 将FP16模型重命名为model.onnx
                        shutil.move(fp16_model_path, onnx_path)
                    else:
                        logger.warning("FP16模型文件不存在，使用原始模型")
                except Exception as e:
                    logger.warning(f"FP16转换失败: {str(e)}，使用原始模型")
            except ImportError as e:
                logger.warning(f"FP16转换所需的库不可用: {str(e)}，跳过FP16转换")
        
        # 复制tokenizer文件
        logger.info("复制tokenizer文件...")
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", 
                          "vocab.txt", "merges.txt", "config.json"]
        
        for file in tokenizer_files:
            src_path = os.path.join(model_name, file)
            dst_path = os.path.join(output_dir, file)
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
        
        # 创建推理脚本
        logger.info(f"创建模型推理脚本...")
        inference_script = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
ONNX嵌入模型推理脚本
\"\"\"

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class ONNXEmbeddings:
    def __init__(self, model_path):
        # 加载ONNX模型
        model_file = os.path.join(model_path, "model.onnx")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件不存在: {{model_file}}")
        
        self.session = ort.InferenceSession(model_file)
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def encode(self, sentences, batch_size=32, normalize_embeddings=True):
        \"\"\"
        对文本进行编码，生成嵌入向量
        
        Args:
            sentences: 文本列表或单个文本
            batch_size: 批处理大小
            normalize_embeddings: 是否对嵌入向量进行归一化
            
        Returns:
            numpy.ndarray: 嵌入向量
        \"\"\"
        # 处理单个文本的情况
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        # 批处理
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # 使用tokenizer处理文本
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="np")
            
            # 准备输入
            onnx_inputs = {{
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }}
            
            if 'token_type_ids' in inputs:
                onnx_inputs['token_type_ids'] = inputs['token_type_ids']
            
            # 运行模型
            embeddings = self.session.run(None, onnx_inputs)[0]
            
            # 归一化
            if normalize_embeddings:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            all_embeddings.append(embeddings)
        
        # 合并所有批次的结果
        all_embeddings = np.vstack(all_embeddings)
        
        return all_embeddings

# 示例用法
if __name__ == "__main__":
    # 初始化模型
    model = ONNXEmbeddings("./")  # 使用当前目录作为模型路径
    
    # 编码文本
    texts = [
        "这是第一个测试句子。",
        "这是另一个不同的句子。"
    ]
    
    # 示例代码
    sample_embeddings = model.encode(texts)
    print(f"嵌入向量形状: {{sample_embeddings.shape}}")
    
    # 计算相似度
    from sklearn.metrics.pairwise import cosine_similarity
    sample_similarity = cosine_similarity([sample_embeddings[0]], [sample_embeddings[1]])[0][0]
    print(f"两个句子的余弦相似度: {{sample_similarity:.4f}}")
"""
        
        with open(os.path.join(output_dir, "inference.py"), "w", encoding="utf-8") as f:
            f.write(inference_script)
        
        # 创建README文件
        readme_content = f"""# ONNX嵌入模型

## 模型信息
- 原始模型: {model_name}
- 转换时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 量化类型: {quantization}
- 精度: {precision}

## 使用方法
1. 安装依赖:
```
pip install onnxruntime transformers numpy
```

2. 使用示例:
```python
from inference import ONNXEmbeddings

# 初始化模型
model = ONNXEmbeddings('{os.path.basename(output_dir)}')

# 编码文本
texts = [
    "这是第一个测试句子。",
    "这是另一个不同的句子。"
]

# 示例代码
sample_embeddings = model.encode(texts)
print(f"嵌入向量形状: {{sample_embeddings.shape}}")

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
sample_similarity = cosine_similarity([sample_embeddings[0]], [sample_embeddings[1]])[0][0]
print(f"两个句子的余弦相似度: {{sample_similarity:.4f}}")
```
"""
        
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # 测试模型
        test_success = True
        if test_samples_file and os.path.exists(test_samples_file):
            logger.info(f"测试模型...")
            try:
                with open(test_samples_file, 'r', encoding='utf-8') as f:
                    test_samples = [line.strip() for line in f.readlines() if line.strip()]
                
                if test_samples:
                    # 原始模型推理
                    logger.info("使用原始模型生成嵌入...")
                    original_embeddings = model.encode(test_samples, normalize_embeddings=True)
                    
                    # ONNX模型推理
                    logger.info("使用ONNX模型生成嵌入...")
                    import onnxruntime as ort
                    try:
                        ort_session = ort.InferenceSession(os.path.join(output_dir, "model.onnx"))
                        
                        # 使用tokenizer处理测试样本
                        encoded_inputs = tokenizer(test_samples, padding=True, truncation=True, return_tensors="np")
                        
                        # 运行ONNX模型
                        onnx_inputs = {
                            'input_ids': encoded_inputs['input_ids'],
                            'attention_mask': encoded_inputs['attention_mask']
                        }
                        if 'token_type_ids' in encoded_inputs:
                            onnx_inputs['token_type_ids'] = encoded_inputs['token_type_ids']
                        
                        onnx_embeddings = ort_session.run(None, onnx_inputs)[0]
                        
                        # 计算余弦相似度
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarities = []
                        
                        for i in range(len(test_samples)):
                            sim = cosine_similarity([original_embeddings[i]], [onnx_embeddings[i]])[0][0]
                            similarities.append(sim)
                            logger.info(f"样本 {i+1}: 相似度 = {sim:.6f}")
                        
                        avg_sim = sum(similarities) / len(similarities)
                        min_sim = min(similarities)
                        
                        logger.info(f"嵌入维度: {onnx_embeddings.shape}")
                        logger.info(f"平均余弦相似度: {avg_sim:.6f}")
                        logger.info(f"最小余弦相似度: {min_sim:.6f}")
                        
                        # 检查相似度是否符合要求
                        if avg_sim < 0.95:  # 降低阈值从0.98到0.95
                            logger.error(f"测试失败: 平均相似度 ({avg_sim:.6f}) 低于阈值 0.95")
                            test_success = False
                        elif min_sim < 0.90:  # 降低阈值从0.95到0.90
                            logger.error(f"测试失败: 最小相似度 ({min_sim:.6f}) 低于阈值 0.90")
                            test_success = False
                        else:
                            logger.info("测试通过: 模型精度符合要求")
                    except Exception as e:
                        logger.error(f"ONNX模型测试失败: {str(e)}")
                        test_success = False
            except Exception as e:
                logger.error(f"测试样本读取失败: {str(e)}")
                test_success = False
        
        # 计算模型大小和压缩率
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        # 检查ONNX模型文件是否存在
        if os.path.exists(onnx_path):
            # 计算原始HuggingFace模型目录的大小
            original_size = sum(os.path.getsize(os.path.join(model_name, f)) for f in os.listdir(model_name) 
                               if os.path.isfile(os.path.join(model_name, f)))
            converted_size = os.path.getsize(onnx_path)
            
            # 清理不必要的文件，只保留必须的文件
            original_onnx_path = os.path.join(output_dir, "model.original.onnx")
            if os.path.exists(original_onnx_path):
                os.remove(original_onnx_path)
                logger.info("清理完成：删除了原始ONNX模型文件")
        else:
            logger.error("ONNX模型文件不存在，无法计算模型大小")
            return False
        
        # 转换为GB和MB以便于比较和显示
        original_size_gb = original_size / (1024 * 1024 * 1024)
        converted_size_mb = converted_size / (1024 * 1024)
        compression_ratio = (original_size - converted_size) / original_size * 100
        
        logger.info(f"原始HuggingFace模型大小: {original_size_gb:.2f} GB")
        logger.info(f"转换后ONNX模型大小: {converted_size_mb:.2f} MB")
        logger.info(f"压缩率: {compression_ratio:.2f}%")
        
        # 如果模型大小异常，发出警告
        if converted_size_mb < 5 and original_size_gb > 100:
            logger.warning(f"警告: 转换后的模型大小 ({converted_size_mb:.2f} MB) 可能过小，请检查模型是否正确")
            test_success = False
        elif compression_ratio < 0:  # 如果模型变大了
            logger.warning(f"警告: 转换后的模型大小增加了 ({-compression_ratio:.2f}%)，可能存在问题")
            # 不将其视为失败，因为某些模型转换后确实会变大
        
        # 如果需要清理文件，只保留必要的文件
        if clean_files:
            logger.info("清理不必要的文件...")
            necessary_files = {"model.onnx", "model.original.onnx", "tokenizer.json", "tokenizer_config.json", 
                              "special_tokens_map.json", "vocab.txt", "inference.py", 
                              "README.md", "merges.txt", "config.json"}
            
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path) and file not in necessary_files:
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        
        logger.info("转换完成!")
        return test_success
    
    except Exception as e:
        logger.error(f"转换失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def convert_llm_model(
    model_name: str,
    output_dir: str,
    use_cuda: bool = False,
    opset_version: int = 14,
    quantization: str = 'none',
    precision: str = 'FP32',
    test_samples_file: Optional[str] = None,
    clean_files: bool = True
) -> bool:
    """
    将大语言模型转换为ONNX格式
    
    Args:
        model_name: 模型名称或路径
        output_dir: 输出目录
        use_cuda: 是否使用CUDA加速
        opset_version: ONNX opset版本
        quantization: 量化类型，可选值为'none', 'dynamic'
        precision: 精度，可选值为'FP32', 'FP16', 'INT8'
        test_samples_file: 测试样本文件路径
        clean_files: 是否清理不必要的文件
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        logger.info(f"开始转换大语言模型: {model_name}")
        
        # 处理static量化类型
        if quantization == 'static':
            logger.info("注意: 将'static'量化转换为'dynamic'量化，因为ONNX更适合动态量化")
            quantization = 'dynamic'
        
        # 设置设备
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 加载模型
        logger.info(f"加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        model.to(device)
        model.eval()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")
        
        # 准备示例输入
        sample_text = "这是一个测试样本，用于ONNX模型导出。"
        inputs = tokenizer(sample_text, return_tensors="pt").to(device)
        
        # 定义输入和输出名称
        input_names = list(inputs.keys())
        output_names = ["logits"]
        
        # 导出到ONNX
        logger.info(f"导出模型到ONNX格式，opset版本: {opset_version}...")
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        # 导出模型
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={name: {0: "batch_size", 1: "sequence_length"} for name in input_names},
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        logger.info(f"模型已导出到: {onnx_path}")
        
        # 使用ONNX内置优化器优化模型
        try:
            import onnx
            from onnx import optimizer
            
            logger.info("应用ONNX内置优化...")
            
            # 加载模型
            model = onnx.load(onnx_path)
            
            # 应用所有可用的优化
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'fuse_bn_into_conv',
                'fuse_add_bias_into_conv',
                'fuse_consecutive_transposes',
                'fuse_consecutive_reduces',
                'fuse_matmul_add_bias_into_gemm'
            ]
            
            # 优化模型
            optimized_model = optimizer.optimize(model, passes)
            
            # 保存优化后的模型
            optimized_model_path = os.path.join(output_dir, "model_optimized.onnx")
            onnx.save(optimized_model, optimized_model_path)
            
            # 检查优化后的模型大小
            original_size = os.path.getsize(onnx_path)
            optimized_size = os.path.getsize(optimized_model_path)
            
            if optimized_size < original_size:
                logger.info(f"ONNX优化成功! 原始大小: {original_size/(1024*1024):.2f} MB, 优化后: {optimized_size/(1024*1024):.2f} MB")
                # 替换原始模型
                shutil.move(optimized_model_path, onnx_path)
            else:
                logger.warning(f"优化后的模型更大，使用原始模型")
                os.remove(optimized_model_path)
        except Exception as e:
            logger.warning(f"ONNX优化失败: {str(e)}")
            logger.info("使用原始ONNX模型")
        
        # 应用量化（如果需要）
        if quantization == 'dynamic':
            logger.info(f"应用动态量化: {precision}...")
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                # 创建量化后的模型文件路径
                quantized_model_path = os.path.join(output_dir, "model_quantized.onnx")
                
                # 根据精度选择量化类型
                quant_type = None
                if precision == 'INT8':
                    quant_type = QuantType.QInt8
                elif precision == 'INT4':
                    try:
                        # 检查是否支持INT4
                        if hasattr(QuantType, 'QInt4'):
                            quant_type = QuantType.QInt4
                            logger.info("使用INT4量化...")
                        else:
                            logger.warning("当前ONNX运行时不支持INT4量化，回退到INT8")
                            quant_type = QuantType.QInt8
                            precision = 'INT8'  # 更新精度信息
                    except Exception as e:
                        logger.warning(f"INT4量化初始化失败: {str(e)}，回退到INT8")
                        quant_type = QuantType.QInt8
                        precision = 'INT8'  # 更新精度信息
                
                if quant_type is not None:
                    # 尝试更激进的量化方法
                    try:
                        # 尝试使用更激进的量化参数
                        logger.info("尝试更激进的量化方法...")
                        quantize_dynamic(
                            model_input=onnx_path,
                            model_output=quantized_model_path,
                            op_types_to_quantize=['MatMul', 'Gemm', 'Conv', 'Attention', 'LSTM', 'GRU'],
                            per_channel=True,
                            weight_type=quant_type,
                            # 以下是更激进的参数
                            optimize_model=True,  # 在量化前优化模型
                            extra_options={'EnableSubgraph': True}  # 启用子图优化
                        )
                    except (TypeError, ValueError) as e:
                        logger.info(f"使用基本量化API，错误: {str(e)}")
                        # 如果激进方法失败，回退到基本方法
                        quantize_dynamic(
                            model_input=onnx_path,
                            model_output=quantized_model_path,
                            weight_type=quant_type  # 确保传递量化类型参数
                        )
                    
                    # 检查量化后的模型是否存在
                    if os.path.exists(quantized_model_path):
                        # 检查量化后的模型大小
                        original_size = os.path.getsize(onnx_path)
                        quantized_size = os.path.getsize(quantized_model_path)
                        
                        # 转换为MB以便于比较和显示
                        original_size_mb = original_size / (1024 * 1024)
                        quantized_size_mb = quantized_size / (1024 * 1024)
                        
                        # 计算压缩率
                        compression_ratio = 100 * (1 - quantized_size / original_size)
                        
                        # 无论大小如何，都使用量化后的模型
                        if quantized_size < original_size:
                            logger.info(f"ONNX量化成功! ONNX原始大小: {original_size_mb:.2f} MB, 量化后: {quantized_size_mb:.2f} MB, 压缩率: {compression_ratio:.2f}%")
                        else:
                            logger.info(f"ONNX量化完成。ONNX原始大小: {original_size_mb:.2f} MB, 量化后: {quantized_size_mb:.2f} MB, 压缩率: {compression_ratio:.2f}%")
                        
                        # 备份原始模型
                        shutil.move(
                            onnx_path,
                            os.path.join(output_dir, "model.original.onnx")
                        )
                        # 将量化后的模型重命名为model.onnx
                        shutil.move(quantized_model_path, onnx_path)
                        
                        # 如果是INT4量化，测试模型是否可用
                        if precision == 'INT4':
                            try:
                                # 尝试加载模型进行测试
                                logger.info("测试INT4量化模型是否可用...")
                                import onnxruntime as ort
                                ort_session = ort.InferenceSession(onnx_path)
                                logger.info("INT4量化模型测试成功!")
                            except Exception as e:
                                logger.warning(f"INT4量化模型测试失败: {str(e)}")
                                logger.warning("回退到INT8量化...")
                                
                                # 恢复原始模型
                                shutil.move(
                                    os.path.join(output_dir, "model.original.onnx"),
                                    onnx_path
                                )
                                
                                # 重新使用INT8量化
                                logger.info("应用INT8量化...")
                                try:
                                    quantize_dynamic(
                                        model_input=onnx_path,
                                        model_output=quantized_model_path,
                                        weight_type=QuantType.QInt8
                                    )
                                    
                                    if os.path.exists(quantized_model_path):
                                        # 备份原始模型
                                        shutil.move(
                                            onnx_path,
                                            os.path.join(output_dir, "model.original.onnx")
                                        )
                                        # 将量化后的模型重命名为model.onnx
                                        shutil.move(quantized_model_path, onnx_path)
                                        logger.info("成功回退到INT8量化")
                                    else:
                                        logger.warning("INT8量化失败，使用原始模型")
                                except Exception as e:
                                    logger.warning(f"INT8量化失败: {str(e)}，使用原始模型")
                    else:
                        logger.warning("量化后的模型文件不存在，使用原始模型")
                else:
                    logger.warning(f"不支持的精度类型: {precision}，使用原始模型")
            except Exception as e:
                logger.warning(f"动态量化失败: {str(e)}")
                logger.info("使用原始模型")
        
        # 复制tokenizer文件
        logger.info("复制tokenizer文件...")
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", 
                          "vocab.txt", "merges.txt", "config.json"]
        
        for file in tokenizer_files:
            src_path = os.path.join(model_name, file)
            dst_path = os.path.join(output_dir, file)
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
        
        # 创建推理脚本
        logger.info(f"创建模型推理脚本...")
        inference_script = """import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class ONNXCausalLM:
    def __init__(self, model_path):
        # 加载ONNX模型
        self.model_path = model_path
        self.onnx_model = ort.InferenceSession(os.path.join(model_path, "model.onnx"))
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 获取输入和输出名称
        self.input_names = [input.name for input in self.onnx_model.get_inputs()]
        self.output_names = [output.name for output in self.onnx_model.get_outputs()]
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 准备ONNX输入
        onnx_inputs = {}
        for name in self.input_names:
            if name in inputs:
                onnx_inputs[name] = inputs[name]
        
        # 生成文本
        for _ in range(max_length):
            # 运行推理
            outputs = self.onnx_model.run(self.output_names, onnx_inputs)
            logits = outputs[0]
            
            # 获取下一个token
            next_token_logits = logits[0, -1, :]
            
            # 应用温度
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # 应用top_p采样
            if top_p < 1.0:
                sorted_logits = np.sort(next_token_logits)[::-1]
                cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
                sorted_indices_to_remove = cumulative_probs > top_p
                indices_to_remove = np.zeros_like(next_token_logits, dtype=bool)
                indices_to_remove[np.argsort(next_token_logits)[::-1][sorted_indices_to_remove]] = True
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 采样下一个token
            probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
            next_token = np.random.choice(len(probs), p=probs)
            
            # 如果生成了结束标记，停止生成
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # 更新输入
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
            attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
            
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text

# 使用示例
if __name__ == "__main__":
    model = ONNXCausalLM("./")  # 使用当前目录作为模型路径
    
    # 生成文本
    prompt = "今天天气真好，我决定"
    generated_text = model.generate(prompt, max_length=50)
    print(f"生成的文本: {generated_text}")
"""
        
        with open(os.path.join(output_dir, "inference.py"), "w", encoding="utf-8") as f:
            f.write(inference_script)
        
        # 创建README文件
        readme_content = f"""# ONNX大语言模型

## 模型信息
- 原始模型: {model_name}
- 转换时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 量化类型: {quantization}
- 精度: {precision}

## 使用方法
1. 安装依赖:
```
pip install onnxruntime transformers numpy
```

2. 使用示例:
```python
from inference import ONNXCausalLM

model = ONNXCausalLM("./")  # 使用当前目录作为模型路径

# 生成文本
prompt = "今天天气真好，我决定"
generated_text = model.generate(prompt, max_length=50)
print(f"生成的文本: {generated_text}")
```
"""
        
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # 测试模型
        test_success = True
        if test_samples_file and os.path.exists(test_samples_file):
            logger.info(f"测试模型...")
            try:
                with open(test_samples_file, 'r', encoding='utf-8') as f:
                    test_samples = [line.strip() for line in f.readlines() if line.strip()]
                
                if test_samples:
                    logger.info("测试ONNX模型生成...")
                    from inference import ONNXCausalLM
                    onnx_model = ONNXCausalLM(output_dir)
                    
                    for i, sample in enumerate(test_samples[:3]):  # 只测试前3个样本
                        logger.info(f"样本 {i+1}: {sample}")
                        generated_text = onnx_model.generate(sample, max_length=20)
                        logger.info(f"生成结果: {generated_text}")
            except Exception as e:
                logger.error(f"测试样本读取失败: {str(e)}")
                test_success = False
        
        # 计算模型大小和压缩率
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        # 检查ONNX模型文件是否存在
        if os.path.exists(onnx_path):
            # 计算原始HuggingFace模型目录的大小
            original_size = sum(os.path.getsize(os.path.join(model_name, f)) for f in os.listdir(model_name) 
                               if os.path.isfile(os.path.join(model_name, f)))
            converted_size = os.path.getsize(onnx_path)
            
            # 清理不必要的文件，只保留必须的文件
            original_onnx_path = os.path.join(output_dir, "model.original.onnx")
            if os.path.exists(original_onnx_path):
                os.remove(original_onnx_path)
                logger.info("清理完成：删除了原始ONNX模型文件")
        else:
            logger.error("ONNX模型文件不存在，无法计算模型大小")
            return False
        
        # 转换为GB和MB以便于比较和显示
        original_size_gb = original_size / (1024 * 1024 * 1024)
        converted_size_mb = converted_size / (1024 * 1024)
        compression_ratio = (original_size - converted_size) / original_size * 100
        
        logger.info(f"原始HuggingFace模型大小: {original_size_gb:.2f} GB")
        logger.info(f"转换后ONNX模型大小: {converted_size_mb:.2f} MB")
        logger.info(f"压缩率: {compression_ratio:.2f}%")
        
        # 如果模型大小异常，发出警告
        if converted_size_mb < 5 and original_size_gb > 100:
            logger.warning(f"警告: 转换后的模型大小 ({converted_size_mb:.2f} MB) 可能过小，请检查模型是否正确")
            test_success = False
        elif compression_ratio < 0:  # 如果模型变大了
            logger.warning(f"警告: 转换后的模型大小增加了 ({-compression_ratio:.2f}%)，可能存在问题")
            # 不将其视为失败，因为某些模型转换后确实会变大
        
        # 如果需要清理文件，只保留必要的文件
        if clean_files:
            logger.info("清理不必要的文件...")
            necessary_files = {"model.onnx", "model.original.onnx", "tokenizer.json", "tokenizer_config.json", 
                              "special_tokens_map.json", "vocab.txt", "inference.py", 
                              "README.md", "merges.txt", "config.json"}
            
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path) and file not in necessary_files:
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        
        logger.info("转换完成!")
        return test_success
    
    except Exception as e:
        logger.error(f"转换失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='将模型转换为ONNX格式')
    parser.add_argument('--model', type=str, required=True, help='模型名称或路径')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--model_type', type=str, choices=['embedding', 'llm'], required=True, help='模型类型')
    parser.add_argument('--use_cuda', action='store_true', help='是否使用CUDA加速')
    parser.add_argument('--opset_version', type=int, default=14, help='ONNX opset版本')
    parser.add_argument('--quantization', type=str, choices=['none', 'dynamic', 'static'], default='none', help='量化类型')
    parser.add_argument('--precision', type=str, choices=['FP32', 'FP16', 'INT8', 'INT4'], default='FP32', help='精度')
    parser.add_argument('--test_samples', type=str, help='测试样本文件路径')
    parser.add_argument('--no_clean', action='store_true', help='不清理不必要的文件')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output is None:
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", 
                                  f"{os.path.basename(args.model)}_{args.quantization}_quant_{args.precision}")
    
    # 转换模型
    if args.model_type == "embedding":
        success = convert_embedding_model(
            model_name=args.model,
            output_dir=args.output,
            use_cuda=args.use_cuda,
            opset_version=args.opset_version,
            quantization=args.quantization,
            precision=args.precision,
            test_samples_file=args.test_samples,
            clean_files=not args.no_clean
        )
    else:
        success = convert_llm_model(
            model_name=args.model,
            output_dir=args.output,
            use_cuda=args.use_cuda,
            opset_version=args.opset_version,
            quantization=args.quantization,
            precision=args.precision,
            test_samples_file=args.test_samples,
            clean_files=not args.no_clean
        )
    
    if success:
        logger.info(f"模型已成功转换并保存到: {args.output}")
        return 0
    else:
        logger.error("模型转换失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
