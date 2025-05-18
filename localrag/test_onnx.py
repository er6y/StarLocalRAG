#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试原始模型和ONNX模型生成能力的脚本
"""

import os
import sys
import time
import json
import logging
import re
import argparse
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnxruntime as ort

# 添加softmax函数用于ONNX模型生成
def softmax(x):
    """计算softmax值"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_model_generation(model_path, prompt, max_new_tokens=200, enable_thinking=True, num_threads=4):
    """
    测试模型生成能力
    
    Args:
        model_path: 模型路径
        prompt: 提示词
        max_new_tokens: 最大生成token数
        enable_thinking: 是否启用思考模式
        num_threads: ONNX推理线程数
        
    Returns:
        生成的文本
    """
    try:
        logger.info(f"开始测试模型生成能力: {model_path}")
        logger.info(f"提示词: {prompt}")
        logger.info(f"线程数: {num_threads}, 最大token数: {max_new_tokens}, 思考模式: {enable_thinking}")
        
        # 检查是否是ONNX模型
        is_onnx = model_path.lower().endswith(".onnx") or "onnx" in os.path.basename(model_path).lower()
        
        # 检查是否是Qwen3模型
        is_qwen3 = "qwen3" in model_path.lower()
        
        if is_onnx:
            # ONNX模型生成
            logger.info("使用ONNX模型生成文本")
            
            # 如果model_path是目录，则查找model.onnx文件
            if os.path.isdir(model_path):
                model_file = os.path.join(model_path, "model.onnx")
                if not os.path.exists(model_file):
                    # 尝试查找其他onnx文件
                    onnx_files = [f for f in os.listdir(model_path) if f.endswith(".onnx")]
                    if onnx_files:
                        model_file = os.path.join(model_path, onnx_files[0])
                    else:
                        raise FileNotFoundError(f"在{model_path}中找不到ONNX模型文件")
            else:
                model_file = model_path
                
            # 加载tokenizer
            tokenizer_path = os.path.dirname(model_file)
            logger.info(f"加载tokenizer: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            
            # 创建ONNX运行时会话，使用优化选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = num_threads  # 使用传入的线程数
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 顺序执行模式
            sess_options.enable_profiling = False  # 禁用分析以提高性能
            
            # 检测可用的执行提供程序
            providers = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            logger.info(f"使用执行提供程序: {providers}，线程数: {num_threads}")
            session = ort.InferenceSession(model_file, sess_options, providers=providers)
            
            # 获取模型输入和输出名称
            input_names = [input.name for input in session.get_inputs()]
            output_names = [output.name for output in session.get_outputs()]
            
            logger.info(f"模型输入: {input_names}")
            logger.info(f"模型输出: {output_names}")
            
            # 使用与官方推荐相同的方式处理输入
            logger.info("尝试使用ONNX模型生成文本")
            
            # 准备消息格式
            if enable_thinking:
                logger.info("启用思考模式")
            else:
                logger.info("禁用思考模式")
                
            # 如果禁用思考模式，添加/no_think指令
            if not enable_thinking:
                if not prompt.endswith("/no_think"):
                    prompt = prompt + " /no_think"
                    logger.info(f"添加/no_think指令到提示词: {prompt}")
            
            messages = [{"role": "user", "content": prompt}]
            
            # 尝试使用apply_chat_template
            try:
                # 尝试使用apply_chat_template并指定思考模式
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking  # 支持思考模式
                )
                logger.info(f"使用chat_template格式化输入，思考模式: {enable_thinking}")
            except Exception as e:
                # 如果失败，尝试不指定思考模式
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    logger.info("使用默认chat_template格式化输入")
                except Exception as e2:
                    logger.warning(f"使用chat_template失败: {str(e2)}，回退到直接使用tokenizer")
                    text = prompt
            
            # 编码输入
            inputs = tokenizer(text, return_tensors="np")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            logger.info(f"输入文本: {text[:100]}...")
            
            # 生成的token IDs
            all_token_ids = input_ids[0].tolist()
            
            # 生成循环
            start_time = time.time()
            logger.info(f"设置生成的最大token数量为: {max_new_tokens}")
            
            # 使用流式模式输出，显示生成过程
            logger.info("开始生成文本，流式输出...")
            
            try:
                # 生成文本
                for i in range(max_new_tokens):
                    # 准备模型输入
                    model_inputs = {}
                    if "input_ids" in input_names:
                        model_inputs["input_ids"] = input_ids
                    if "attention_mask" in input_names:
                        model_inputs["attention_mask"] = attention_mask
                    
                    # 运行推理
                    try:
                        outputs = session.run(output_names, model_inputs)
                    except Exception as e:
                        logger.error(f"推理时出错: {str(e)}")
                        break
                        
                    # 每20个token显示一次当前生成进度
                    if i % 20 == 0 and i > 0:
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0:
                            tokens_per_second = i / elapsed_time
                            # 打印当前生成的内容和速度
                            current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                            logger.info(f"当前生成速度: {tokens_per_second:.2f} tokens/秒, 已生成{i}个token")
                    
                    # 获取logits
                    logits = outputs[0]
                    
                    # 获取最后一个位置的logits
                    next_token_logits = logits[0, -1, :]
                    
                    # 使用官方推荐的采样参数
                    # 思考模式: Temperature=0.6, TopP=0.95, TopK=20
                    # 非思考模式: Temperature=0.7, TopP=0.8, TopK=20
                    
                    # 设置采样参数
                    if enable_thinking:
                        temperature = 0.6
                        top_p = 0.95
                        top_k = 20
                    else:
                        temperature = 0.7
                        top_p = 0.8
                        top_k = 20
                    
                    # 应用温度采样
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # 应用top-k采样
                    if top_k > 0:
                        indices_to_remove = np.argpartition(next_token_logits, -top_k)[:len(next_token_logits) - top_k]
                        next_token_logits[indices_to_remove] = -float("inf")
                    
                    # 应用top-p采样
                    if top_p < 1.0:
                        sorted_indices = np.argsort(next_token_logits)[::-1]
                        sorted_logits = next_token_logits[sorted_indices]
                        cumulative_probs = np.cumsum(softmax(sorted_logits))
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                        sorted_indices_to_remove[0] = False
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float("inf")
                    
                    # 计算softmax并采样
                    probs = softmax(next_token_logits)
                    next_token_id = np.random.choice(len(probs), p=probs)
                    
                    # 添加到生成的ID列表
                    all_token_ids.append(int(next_token_id))
                    
                    # 检查是否生成了EOS
                    if next_token_id == tokenizer.eos_token_id:
                        logger.info(f"生成了EOS标记，结束生成")
                        break
                    
                    # 更新输入，包含所有已生成的tokens
                    input_ids = np.array([all_token_ids], dtype=np.int64)
                    attention_mask = np.ones((1, len(all_token_ids)), dtype=np.int64)
                    
                    # 每10个token输出一次当前生成的文本
                    if i % 10 == 0 and i > 0:
                        current_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
                        logger.info(f"当前生成 ({i} tokens): {current_text[-50:]}...")
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 解码生成的文本
                generated_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
                
                # 如果启用了思考模式，尝试解析思考内容
                if enable_thinking:
                    try:
                        # 尝试分离思考内容和最终回答
                        import re
                        think_pattern = r"<think>(.*?)</think>"
                        match = re.search(think_pattern, generated_text, re.DOTALL)
                        
                        if match:
                            thinking_content = match.group(1).strip()
                            # 提取最终回答 - 在</think>之后的所有内容
                            final_answer_pattern = r"</think>\s*(.*)"
                            final_match = re.search(final_answer_pattern, generated_text, re.DOTALL)
                            
                            if final_match:
                                content = final_match.group(1).strip()
                            else:
                                content = "无法提取最终回答"
                                
                            logger.info(f"思考内容: {thinking_content[:100]}...")
                            logger.info(f"最终回答: {content[:100]}...")
                            generated_text = f"思考内容:\n{thinking_content}\n\n最终回答:\n{content}"
                    except Exception as e:
                        logger.warning(f"解析思考内容时出错: {str(e)}")
                
                # 计算生成时间
                logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
                
                # 计算生成速度
                tokens_generated = len(all_token_ids) - len(inputs["input_ids"][0])
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                logger.info(f"生成了 {tokens_generated} 个token，速度为 {tokens_per_second:.2f} tokens/秒")
                
                result = f"输入提示词: {prompt}\n\n生成的文本:\n{generated_text}"
                logger.info(result)
                return result
                
            except Exception as e:
                logger.error(f"ONNX自回归生成时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 如果生成失败，回退到单步推理
                logger.info("回退到单步推理模式")
                
                # 使用原始输入进行单步推理
                model_inputs = {}
                if "input_ids" in input_names:
                    model_inputs["input_ids"] = inputs["input_ids"]
                if "attention_mask" in input_names:
                    model_inputs["attention_mask"] = inputs["attention_mask"]
                
                # 运行推理
                start_time = time.time()
                outputs = session.run(output_names, model_inputs)
                end_time = time.time()
                inference_time = end_time - start_time
                
                # 获取logits并找到最可能的下一个token
                logits = outputs[0]
                next_token_logits = logits[0, -1, :]
                next_token_id = np.argmax(next_token_logits)
                
                # 将token ID转换为文本
                next_token_text = tokenizer.decode([next_token_id])
                
                # 计算tokens per second
                num_tokens = inputs["input_ids"].shape[1]
                tokens_per_second = num_tokens / inference_time if inference_time > 0 else 0
                logger.info(f"推理时间: {inference_time:.4f}秒")
                logger.info(f"Tokens/秒: {tokens_per_second:.2f}")
                
                result = f"输入提示词: {prompt}\n\n模型可以正常工作\n预测的下一个token: {next_token_text}"
                logger.info(result)
                return result
                
        else:
            # 原始模型生成
            logger.info("使用原始模型生成文本")
            
            # 加载模型和tokenizer
            logger.info("加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",  # 使用CPU以确保兼容性
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=False
            )
            
            # 加载tokenizer
            logger.info(f"加载tokenizer: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # 将模型设置为评估模式
            model.eval()
            
            # 处理Qwen3模型的特殊情况
            if is_qwen3:
                logger.info("检测到Qwen3模型，使用特殊处理")
                try:
                    # 先尝试修改transformers库中的sdpa_attention.py文件
                    import importlib.util
                    import sys
                    from types import ModuleType
                    
                    # 找到sdpa_attention模块的路径
                    sdpa_module_name = "transformers.integrations.sdpa_attention"
                    if sdpa_module_name in sys.modules:
                        # 如果模块已加载，获取其路径
                        sdpa_module = sys.modules[sdpa_module_name]
                        sdpa_file_path = sdpa_module.__file__
                        logger.info(f"找到SDPA模块路径: {sdpa_file_path}")
                        
                        # 创建一个修改过的sdpa_attention_forward函数
                        def patched_sdpa_attention_forward(self, *args, **kwargs):
                            # 获取原始函数的参数
                            query = kwargs.get("query", args[0] if len(args) > 0 else None)
                            key = kwargs.get("key", args[1] if len(args) > 1 else None)
                            value = kwargs.get("value", args[2] if len(args) > 2 else None)
                            causal_mask = kwargs.get("causal_mask", args[3] if len(args) > 3 else None)
                            
                            # 修复问题行，不使用.item()
                            is_causal = (query.shape[2] > 1) and causal_mask is None
                            
                            # 其余代码与原函数相同
                            attn_output = torch.nn.functional.scaled_dot_product_attention(
                                query, key, value, attn_mask=causal_mask, is_causal=is_causal
                            )
                            return attn_output, None
                        
                        # 替换原始函数
                        from transformers.integrations.sdpa_attention import sdpa_attention_forward
                        original_func = sdpa_attention_forward
                        sys.modules[sdpa_module_name].sdpa_attention_forward = patched_sdpa_attention_forward
                        logger.info("成功替换SDPA函数用于生成")
                        
                        # 如果禁用思考模式，添加/no_think指令
                        if not enable_thinking:
                            if not prompt.endswith("/no_think"):
                                prompt = prompt + " /no_think"
                                logger.info(f"添加/no_think指令到提示词: {prompt}")
                        
                        # 准备消息格式
                        messages = [{"role": "user", "content": prompt}]
                        
                        # 尝试使用apply_chat_template
                        try:
                            # 尝试使用apply_chat_template并指定思考模式
                            text = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=enable_thinking  # 支持思考模式
                            )
                            logger.info(f"使用chat_template格式化输入，思考模式: {enable_thinking}")
                        except Exception as e:
                            # 如果失败，尝试不指定思考模式
                            try:
                                text = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True
                                )
                                logger.info("使用默认chat_template格式化输入")
                            except Exception as e2:
                                logger.warning(f"使用chat_template失败: {str(e2)}，回退到直接使用tokenizer")
                                text = prompt
                        
                        # 编码输入
                        inputs = tokenizer(text, return_tensors="pt")
                        
                        # 生成文本
                        logger.info("开始生成文本...")
                        start_time = time.time()
                        with torch.no_grad():
                            # 使用官方推荐的采样参数
                            if enable_thinking:
                                temperature = 0.6
                                top_p = 0.95
                                top_k = 20
                            else:
                                temperature = 0.7
                                top_p = 0.8
                                top_k = 20
                                
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_new_tokens=max_new_tokens,  # 使用max_new_tokens而非max_length
                                num_return_sequences=1,
                                do_sample=True,
                                top_p=top_p,
                                top_k=top_k,
                                temperature=temperature,
                                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                                repetition_penalty=1.1  # 添加重复惩罚来避免生成重复内容
                            )
                        end_time = time.time()
                        
                        # 恢复原始函数
                        sys.modules[sdpa_module_name].sdpa_attention_forward = original_func
                        
                        # 解码生成的文本
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # 计算生成时间
                        generation_time = end_time - start_time
                        logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
                        
                        result = f"输入提示词: {prompt}\n\n生成的文本:\n{generated_text}"
                        logger.info(result)
                        return result
                    else:
                        # 如果找不到模块，尝试另一种方法
                        logger.warning(f"找不到SDPA模块，尝试其他方法")
                        raise Exception("找不到SDPA模块")
                except Exception as e:
                    logger.warning(f"Qwen3模型特殊处理失败: {str(e)}")
                    # 如果特殊处理失败，则使用标准方法
            
            # 对于非Qwen3模型或Qwen3特殊处理失败时，使用标准生成方法
            try:
                # 如果禁用思考模式，添加/no_think指令
                if not enable_thinking:
                    if not prompt.endswith("/no_think"):
                        prompt = prompt + " /no_think"
                        logger.info(f"添加/no_think指令到提示词: {prompt}")
                
                # 准备消息格式
                messages = [{"role": "user", "content": prompt}]
                
                # 尝试使用apply_chat_template
                try:
                    # 尝试使用apply_chat_template并指定思考模式
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking  # 支持思考模式
                    )
                    logger.info(f"使用chat_template格式化输入，思考模式: {enable_thinking}")
                except Exception as e:
                    # 如果失败，尝试不指定思考模式
                    try:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        logger.info("使用默认chat_template格式化输入")
                    except Exception as e2:
                        logger.warning(f"使用chat_template失败: {str(e2)}，回退到直接使用tokenizer")
                        text = prompt
                
                # 编码输入
                inputs = tokenizer(text, return_tensors="pt")
                
                # 生成文本
                logger.info("开始生成文本...")
                start_time = time.time()
                with torch.no_grad():
                    # 使用官方推荐的采样参数
                    if enable_thinking:
                        temperature = 0.6
                        top_p = 0.95
                        top_k = 20
                    else:
                        temperature = 0.7
                        top_p = 0.8
                        top_k = 20
                        
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_new_tokens,  # 使用max_new_tokens而非max_length
                        num_return_sequences=1,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                        repetition_penalty=1.1  # 添加重复惩罚
                    )
                end_time = time.time()
                
                # 解码生成的文本
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 计算生成时间
                generation_time = end_time - start_time
                logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
                
                result = f"输入提示词: {prompt}\n\n生成的文本:\n{generated_text}"
                logger.info(result)
                return result
                
            except Exception as e:
                logger.error(f"生成文本时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return f"生成失败: {str(e)}"
    
    except Exception as e:
        logger.error(f"测试模型生成能力时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"生成失败: {str(e)}"

def main():
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="测试原始模型和ONNX模型的生成能力")
    parser.add_argument("--original", "-o", required=True, help="原始模型路径")
    parser.add_argument("--onnx", "-x", required=True, help="ONNX模型路径")
    parser.add_argument("--max-new-tokens", "-m", type=int, default=200, help="生成的最大token数量")
    parser.add_argument("--no-thinking", "-n", action="store_true", help="禁用思考模式")
    parser.add_argument("--threads", "-j", type=int, default=4, help="ONNX推理线程数")
    parser.add_argument("--prompt", "-p", type=str, default="用中文介绍下你自己，并夸赞下自己，不少于100个字。", help="测试提示词")
    args = parser.parse_args()
    
    # 先测试原始模型
    logger.info("\n" + "="*50)
    logger.info("测试原始模型的生成能力...")
    original_result = test_model_generation(args.original, args.prompt, args.max_new_tokens, not args.no_thinking, args.threads)
    
    # 再测试ONNX模型
    logger.info("\n" + "="*50)
    logger.info("测试ONNX模型的生成能力...")
    onnx_result = test_model_generation(args.onnx, args.prompt, args.max_new_tokens, not args.no_thinking, args.threads)

if __name__ == "__main__":
    main()
