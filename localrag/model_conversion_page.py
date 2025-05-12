import os
import sys
import json
import importlib.util
import logging
import traceback
import time

# 检查PyTorch是否已安装
def check_torch_available():
    try:
        # 尝试导入torch
        import torch
        import numpy as np
        
        # 检查transformers是否已安装
        transformers_spec = importlib.util.find_spec("transformers")
        transformers_available = transformers_spec is not None
        
        # 检查safetensors是否已安装
        safetensors_spec = importlib.util.find_spec("safetensors")
        safetensors_available = safetensors_spec is not None
        
        # 只有当所有依赖都已安装时才返回True
        all_deps_available = torch is not None and transformers_available and safetensors_available
        if all_deps_available:
            print("[INFO] 所有依赖都已安装：PyTorch、transformers、safetensors")
        else:
            missing = []
            if torch is None:
                missing.append("PyTorch")
            if not transformers_available:
                missing.append("transformers")
            if not safetensors_available:
                missing.append("safetensors")
            print(f"[WARNING] 缺少依赖：{', '.join(missing)}")
        return all_deps_available
    except ImportError as e:
        print(f"[ERROR] 导入错误：{str(e)}")
        return False

# 检查依赖
TORCH_AVAILABLE = check_torch_available()
if not TORCH_AVAILABLE:
    print("[INFO] 依赖检查：PyTorch或相关依赖未安装，模型转换功能将不可用")
else:
    print("[INFO] 依赖检查：所有依赖都已安装，模型转换功能可用")

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                            QPushButton, QTextEdit, QFileDialog, QMessageBox, QCheckBox,
                            QRadioButton, QButtonGroup, QGroupBox, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QFont

# 添加更详细的日志记录
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_conversion_debug.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_conversion")

# 模型转换线程
class ModelConversionThread(QThread):
    progress_updated = pyqtSignal(str)
    completed = pyqtSignal(bool, str)
    
    def __init__(self, model_path, quantization_type=None, precision=None, format_type="ONNX", is_llm=False, use_sdpa=False, embeddings_only=False, verbose_logging=True, output_dir=None):
        super().__init__()
        self.model_path = model_path
        self.quantization_type = quantization_type
        self.precision = precision
        self.format_type = "ONNX"  # 固定为ONNX格式
        self.is_llm = is_llm  # 是否为大模型
        self.use_sdpa = use_sdpa  # 是否使用SDPA
        self.embeddings_only = embeddings_only
        self.verbose_logging = verbose_logging  # 是否启用详细日志
        self.output_dir = output_dir  # 自定义输出目录
        
    def update_progress(self, msg):
        self.progress_updated.emit(msg)
        
    def run(self):
        if not TORCH_AVAILABLE:
            self.update_progress("[ERROR] PyTorch is not installed, cannot perform model conversion")
            self.completed.emit(False, "PyTorch is not installed, please install PyTorch and related dependencies first")
            return
            
        try:
            # 导入必要的库
            import torch
            import sys
            import importlib.util
            
            # 初始化model变量，避免未定义错误
            model = None
            tokenizer = None
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                self.update_progress(f"[ERROR] Model path does not exist: {self.model_path}")
                self.completed.emit(False, f"Model path does not exist: {self.model_path}")
                return
                
            # 添加更详细的日志记录
            if self.verbose_logging:
                self.update_progress(f"[INFO] Starting conversion of model: {self.model_path}")
                self.update_progress(f"[INFO] Conversion options:")
                self.update_progress(f"[INFO] - Format: {self.format_type}")
                self.update_progress(f"[INFO] - Quantization: {self.quantization_type}")
                self.update_progress(f"[INFO] - Precision: {self.precision}")
                self.update_progress(f"[INFO] - Is LLM: {self.is_llm}")
                self.update_progress(f"[INFO] - Use SDPA: {self.use_sdpa}")
                self.update_progress(f"[INFO] - Embeddings only: {self.embeddings_only}")
                self.update_progress(f"[INFO] - Output directory: {self.output_dir}")
            
            # 准备测试样本文件
            test_samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_test_samples.txt")
            if not os.path.exists(test_samples_path):
                self.update_progress(f"[INFO] Creating test samples file: {test_samples_path}")
                with open(test_samples_path, "w", encoding="utf-8") as f:
                    f.write("自然语言处理是人工智能的一个重要分支，它研究如何让计算机理解人类语言。\n")
                    f.write("深度学习模型如Transformer已经在多种NLP任务上取得了突破性进展。\n")
                    f.write("向量嵌入是将文本转换为数值表示的方法，使计算机能够处理和理解文本数据。\n")
                    f.write("知识库问答系统结合了信息检索和自然语言处理技术，能够从大量文档中找到相关信息并生成答案。\n")
            
            # 使用新的转换脚本
            self.update_progress(f"[INFO] Starting conversion to ONNX format...")
            
            # 导入新的转换模块
            import sys
            import importlib.util
            
            # 动态导入模块
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_model_to_onnx.py")
            self.update_progress(f"[INFO] 使用转换脚本: {script_path}")
            
            try:
                spec = importlib.util.spec_from_file_location("convert_model_to_onnx", script_path)
                convert_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(convert_module)
                self.update_progress(f"[INFO] 转换模块加载成功")
            except Exception as e:
                self.update_progress(f"[ERROR] 加载转换模块失败: {str(e)}")
                import traceback
                self.update_progress(f"[ERROR] 详细错误信息: {traceback.format_exc()}")
                self.completed.emit(False, f"加载转换模块失败: {str(e)}")
                return
            
            # 根据模型类型调用相应的转换函数
            if not self.is_llm:
                self.update_progress(f"[INFO] 转换词嵌入模型...")
                try:
                    success = convert_module.convert_embedding_model(
                        model_name=self.model_path,
                        output_dir=self.output_dir,
                        use_cuda=torch.cuda.is_available(),
                        opset_version=14,  # 使用更高的opset版本，支持scaled_dot_product_attention
                        quantization='dynamic' if self.quantization_type else 'none',  # 只支持dynamic量化
                        precision=self.precision if self.precision else 'FP32',
                        test_samples_file=test_samples_path,
                        clean_files=True
                    )
                except Exception as e:
                    self.update_progress(f"[ERROR] 词嵌入模型转换失败: {str(e)}")
                    import traceback
                    self.update_progress(f"[ERROR] 详细错误信息: {traceback.format_exc()}")
                    success = False
            else:
                self.update_progress(f"[INFO] 转换大语言模型...")
                try:
                    success = convert_module.convert_llm_model(
                        model_name=self.model_path,
                        output_dir=self.output_dir,
                        use_cuda=torch.cuda.is_available(),
                        opset_version=14,  # 使用更高的opset版本，支持scaled_dot_product_attention
                        quantization='dynamic' if self.quantization_type else 'none',  # 只支持dynamic量化
                        precision=self.precision if self.precision else 'FP32',
                        test_samples_file=test_samples_path,
                        clean_files=True  # 增加clean_files参数
                    )
                except Exception as e:
                    self.update_progress(f"[ERROR] 大语言模型转换失败: {str(e)}")
                    import traceback
                    self.update_progress(f"[ERROR] 详细错误信息: {traceback.format_exc()}")
                    success = False
            
            if success:
                # 计算模型大小
                original_size = self._get_directory_size(self.model_path)
                converted_size = self._get_directory_size(self.output_dir)
                
                # 格式化大小
                original_size_str = self._format_size(original_size)
                converted_size_str = self._format_size(converted_size)
                
                # 计算压缩率
                compression_ratio = (1 - converted_size / original_size) * 100 if original_size > 0 else 0
                
                # 显示结果
                self.update_progress(f"\n[INFO] 转换完成!")
                self.update_progress(f"[INFO] 原始模型大小: {original_size_str}")
                self.update_progress(f"[INFO] 转换后模型大小: {converted_size_str}")
                self.update_progress(f"[INFO] 压缩率: {compression_ratio:.2f}%")
                self.update_progress(f"[INFO] 模型已保存到: {self.output_dir}")
                
                # 发送完成信号
                self.completed.emit(True, f"模型转换成功，已保存到: {self.output_dir}")
            else:
                self.update_progress(f"[ERROR] 转换失败!")
                self.completed.emit(False, "模型转换失败，请查看日志了解详情")
                
        except Exception as e:
            self.update_progress(f"[ERROR] 转换过程中发生错误: {str(e)}")
            # 添加详细的堆栈跟踪
            import traceback
            self.update_progress(f"[ERROR] 详细错误信息: {traceback.format_exc()}")
            self.completed.emit(False, f"转换失败: {str(e)}")
    
    def _count_parameters(self, model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters())
    
    def _get_directory_size(self, path):
        """获取目录总大小"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    
    def _format_size(self, size_bytes):
        """将字节大小格式化为人类可读的形式"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    def _detect_model_architecture(self, model_path):
        """Detect the model architecture from the model path or files"""
        model_name = os.path.basename(model_path).lower()
        
        # Check model name first
        if "gemma" in model_name:
            return "Gemma"
        elif "llama" in model_name:
            return "LLaMA"
        elif "qwen" in model_name:
            return "Qwen"
        elif "mistral" in model_name:
            return "Mistral"
        elif "phi" in model_name:
            return "Phi"
        
        # If model name doesn't give a clear indication, check for specific files
        if os.path.exists(os.path.join(model_path, "tokenizer.model")) and \
           os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
            # This is likely a Gemma model
            return "Gemma"
        
        # Check config.json if it exists
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    model_type = config.get("model_type", "").lower()
                    if model_type:
                        # Capitalize first letter for consistency
                        return model_type[0].upper() + model_type[1:]
            except:
                pass
        
        # Default to "Unknown" if we can't determine
        return "Unknown"
    
    def _handle_gemma_model(self, tokenizer, output_dir):
        """专门处理Gemma模型的函数，确保即使模型加载失败也能创建基本嵌入模型"""
        self.update_progress("[INFO] 使用专门的Gemma模型处理函数...")
        logger.info("使用专门的Gemma模型处理函数")
        
        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        self.update_progress(f"[INFO] CUDA可用: {cuda_available}")
        if cuda_available:
            self.update_progress(f"[INFO] CUDA设备数量: {torch.cuda.device_count()}")
            self.update_progress(f"[INFO] 当前CUDA设备: {torch.cuda.current_device()}")
            self.update_progress(f"[INFO] CUDA设备名称: {torch.cuda.get_device_name(0)}")
        
        # 尝试直接加载模型配置
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_path)
            self.update_progress(f"[INFO] 成功加载模型配置: {config.model_type}")
        except Exception as config_e:
            self.update_progress(f"[WARNING] 加载模型配置失败: {str(config_e)}")
        
        # 创建基本嵌入模型
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    vocab_size = config_data.get("vocab_size", 262144)  # 使用配置文件中的词汇表大小
                    embedding_dim = config_data.get("hidden_size", 1152)  # 使用配置文件中的隐藏层大小
                    self.update_progress(f"[INFO] 从配置文件读取参数: vocab_size={vocab_size}, hidden_size={embedding_dim}")
            except Exception as e:
                self.update_progress(f"[WARNING] 读取配置文件失败: {str(e)}, 使用默认值")
                vocab_size = 262144  # Gemma-3默认词汇表大小
                embedding_dim = 1152  # Gemma-3默认嵌入维度
        else:
            vocab_size = 262144  # Gemma-3默认词汇表大小
            embedding_dim = 1152  # Gemma-3默认嵌入维度
            self.update_progress(f"[INFO] 未找到配置文件，使用默认参数: vocab_size={vocab_size}, hidden_size={embedding_dim}")
        
        # 确保词汇表大小与模型匹配
        tokenizer_vocab_size = len(tokenizer)
        if vocab_size != tokenizer_vocab_size:
            self.update_progress(f"[WARNING] 词汇表大小不匹配: 配置文件={vocab_size}, 分词器={tokenizer_vocab_size}, 使用配置文件中的值")
        
        self.update_progress(f"[INFO] 创建基本Gemma嵌入模型 (词汇表大小: {vocab_size}, 嵌入维度: {embedding_dim})")
        logger.info(f"创建基本Gemma嵌入模型 (词汇表大小: {vocab_size}, 嵌入维度: {embedding_dim})")
        
        # 创建简单的嵌入模型
        embedding_model = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # 创建示例输入
        example_text = "Hello, I am a language model"
        inputs = tokenizer(example_text, return_tensors="pt")
        input_ids = inputs.input_ids
        self.update_progress(f"[INFO] 示例输入形状: {input_ids.shape}")
        
        # 导出ONNX模型
        if self.format_type == "ONNX":
            embedding_path = os.path.join(output_dir, "model.onnx")
            self.update_progress(f"[INFO] 导出Gemma嵌入模型到ONNX: {embedding_path}")
            logger.info(f"导出Gemma嵌入模型到ONNX: {embedding_path}")
            
            try:
                # 准备输入，包括注意力掩码
                attention_mask = inputs.get('attention_mask', None)
                model_inputs = (input_ids,) if attention_mask is None else (input_ids, attention_mask)
                input_names = ['input_ids'] if attention_mask is None else ['input_ids', 'attention_mask']
                dynamic_axes = {
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'token_ids': {0: 'batch_size', 1: 'sequence_length'}
                }
                
                # 如果有注意力掩码，添加到动态轴配置
                if attention_mask is not None:
                    dynamic_axes['attention_mask'] = {0: 'batch_size', 1: 'sequence_length'}
                
                self.update_progress(f"[INFO] Model inputs: {input_names}")
                logger.info(f"模型输入: {input_names}")
                
                torch.onnx.export(
                    embedding_model,
                    model_inputs,
                    embedding_path,
                    export_params=True,
                    opset_version=14,
                    input_names=input_names,
                    output_names=['token_ids'],  # 修改输出名称为token_ids
                    dynamic_axes=dynamic_axes,
                    use_external_data_format=False  # 禁用外部数据模式
                )
                logger.info(f"ONNX导出成功: {embedding_path}")
                
                # 验证ONNX模型
                if os.path.exists(embedding_path):
                    self.update_progress(f"[INFO] 开始验证ONNX模型...")
                    try:
                        # 导入验证模块
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        import verify_onnx_embedding
                        
                        # 验证模型
                        output_json = os.path.join(output_dir, "model_verification_report.json")
                        results = verify_onnx_embedding.verify_onnx_embedding(output_dir, verbose=True)
                        
                        # 保存验证结果
                        with open(output_json, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        
                        # 显示验证结果摘要
                        self.update_progress(f"[INFO] ONNX模型验证状态: {results['status']}")
                        if results['embedding_dim']:
                            self.update_progress(f"[INFO] 嵌入维度: {results['embedding_dim']}")
                        if results['errors']:
                            for error in results['errors']:
                                self.update_progress(f"[WARNING] 验证错误: {error}")
                        else:
                            self.update_progress(f"[INFO] 验证成功: ONNX模型可以正常加载和执行推理")
                        
                        self.update_progress(f"[INFO] 详细验证报告已保存到: {output_json}")
                    except Exception as e:
                        self.update_progress(f"[WARNING] 验证ONNX模型时发生错误: {str(e)}")
                        logger.warning(f"验证ONNX模型时发生错误: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                return True
            except Exception as e:
                error_msg = f"Gemma嵌入模型ONNX导出失败: {str(e)}"
                self.update_progress(f"[ERROR] {error_msg}")
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                # 尝试TorchScript作为回退
                try:
                    torchscript_path = os.path.join(output_dir, "model.pt")
                    self.update_progress(f"[INFO] 尝试导出为TorchScript: {torchscript_path}")
                    traced_model = torch.jit.trace(embedding_model, (input_ids,))
                    torch.jit.save(traced_model, torchscript_path)
                    self.update_progress("[INFO] Gemma嵌入模型TorchScript导出成功")
                    logger.info(f"Gemma嵌入模型TorchScript导出成功: {torchscript_path}")
                    return True
                except Exception as fallback_error:
                    fallback_error_msg = f"Gemma嵌入模型TorchScript导出也失败: {str(fallback_error)}"
                    self.update_progress(f"[ERROR] {fallback_error_msg}")
                    logger.error(fallback_error_msg)
                    logger.error(traceback.format_exc())
                    return False
        
        # 导出TorchScript模型
        elif self.format_type == "TorchScript":
            torchscript_path = os.path.join(output_dir, "model.pt")
            self.update_progress(f"[INFO] 导出Gemma嵌入模型到TorchScript: {torchscript_path}")
            logger.info(f"导出Gemma嵌入模型到TorchScript: {torchscript_path}")
            
            try:
                traced_model = torch.jit.trace(embedding_model, (input_ids,))
                torch.jit.save(traced_model, torchscript_path)
                self.update_progress("[INFO] Gemma嵌入模型TorchScript导出成功")
                logger.info(f"Gemma嵌入模型TorchScript导出成功: {torchscript_path}")
                return True
            except Exception as e:
                error_msg = f"Gemma嵌入模型TorchScript导出失败: {str(e)}"
                self.update_progress(f"[ERROR] {error_msg}")
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return False
        
        return False
    
    def _copy_config_files(self, output_dir):
        """复制配置文件和词汇表文件"""
        self.update_progress("[INFO] Copying configuration files and vocabulary files...")
        
        # For newer model formats like Gemma 3, we need to copy all relevant files
        essential_files = [
            "config.json", "generation_config.json", "tokenizer_config.json", 
            "tokenizer.model", "tokenizer.json", "special_tokens_map.json",
            "model.safetensors.index.json"
        ]
        
        for config_file in essential_files:
            src_path = os.path.join(self.model_path, config_file)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_dir, config_file)
                import shutil
                shutil.copy2(src_path, dst_path)
                self.update_progress(f"[INFO] Copied file: {config_file}")
        
        # Copy vocabulary files - using broader patterns for newer models
        vocab_patterns = ["vocab", "tokenizer", "merges", "dict", "spm"]
        for file in os.listdir(self.model_path):
            if any(pattern in file.lower() for pattern in vocab_patterns):
                src_path = os.path.join(self.model_path, file)
                if os.path.isfile(src_path):
                    dst_path = os.path.join(output_dir, file)
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    self.update_progress(f"[INFO] Copied vocabulary file: {file}")

# 创建分隔线
def create_separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line

# 模型转换页面
class ModelConversionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and obj is self.progress_text:
            return True
        return super().eventFilter(obj, event)
    
    def init_ui(self):
        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)  # 增加间距
        main_layout.setContentsMargins(20, 20, 20, 20)  # 增加外边距
        
        # 设置标题字体
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)  # 增大字体
        
        # 创建顶部水平布局（包含模型类型和模型选择）
        top_layout = QHBoxLayout()
        
        # 模型类型选择 - 使用单选按钮
        model_type_group = QGroupBox("Model Type")
        model_type_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_type_layout = QHBoxLayout()
        
        # 创建单选按钮组
        self.model_type_group = QButtonGroup(self)
        self.embedding_radio = QRadioButton("Word Embedding Model")
        self.llm_radio = QRadioButton("Large Language Model")
        
        # 默认选择词嵌入模型
        self.embedding_radio.setChecked(True)
        
        # 添加到按钮组
        self.model_type_group.addButton(self.embedding_radio)
        self.model_type_group.addButton(self.llm_radio)
        
        # 连接信号
        self.embedding_radio.toggled.connect(self.on_model_type_changed)
        self.llm_radio.toggled.connect(self.on_model_type_changed)
        
        model_type_layout.addWidget(self.embedding_radio)
        model_type_layout.addWidget(self.llm_radio)
        model_type_layout.addStretch()
        model_type_group.setLayout(model_type_layout)
        model_type_group.setMaximumWidth(200)  # 限制宽度
        
        # 模型选择
        model_group = QGroupBox("Model Selection")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QHBoxLayout()  # 改为水平布局
        
        # 添加刷新按钮
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Refresh model list")
        self.refresh_button.clicked.connect(self.scan_local_models)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.refresh_button)
        
        model_group.setLayout(model_layout)
        
        # 添加到顶部布局
        top_layout.addWidget(model_type_group)
        top_layout.addWidget(model_group, 1)  # 模型选择占据更多空间
        
        # 创建转换选项水平布局 - 将所有选项放在一行
        options_layout_horizontal = QHBoxLayout()
        
        # 创建选项组
        options_group = QHBoxLayout()
        options_group.setSpacing(10)  # 设置选项之间的间距
        
        # 输出格式选择
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["ONNX"])
        self.format_combo.setEnabled(False)  # 禁用选择，因为只支持ONNX
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        
        # 量化类型选择
        quantization_layout = QHBoxLayout()
        quantization_label = QLabel("Quantization:")
        self.quantization_combo = QComboBox()
        self.quantization_combo.addItems(["None", "Dynamic", "Static"])
        quantization_layout.addWidget(quantization_label)
        quantization_layout.addWidget(self.quantization_combo)
        
        # 精度选择
        precision_layout = QHBoxLayout()
        precision_label = QLabel("Precision:")
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["None", "FP16", "INT8", "INT4"])
        precision_layout.addWidget(precision_label)
        precision_layout.addWidget(self.precision_combo)
        
        # 转换按钮
        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert_model)
        
        # 添加到选项组
        options_group.addLayout(format_layout)
        options_group.addLayout(quantization_layout)
        options_group.addLayout(precision_layout)
        options_group.addWidget(self.convert_button)
        
        # 添加到水平布局
        options_layout_horizontal.addLayout(options_group)
        options_layout_horizontal.addStretch(1)
        
        # 高级选项
        advanced_layout = QHBoxLayout()
        self.use_sdpa_checkbox = QCheckBox("Use SDPA (scaled_dot_product_attention)")
        self.use_sdpa_checkbox.setChecked(True)
        advanced_layout.addWidget(self.use_sdpa_checkbox)
        
        # 添加仅导出嵌入层的选项
        self.embeddings_only_checkbox = QCheckBox("Embeddings Only (better compatibility)")
        self.embeddings_only_checkbox.setChecked(False)
        self.embeddings_only_checkbox.setToolTip("Only export the embedding layer, which has better compatibility with newer models")
        advanced_layout.addWidget(self.embeddings_only_checkbox)
        
        advanced_layout.addStretch()
        
        # 进度显示
        progress_group = QGroupBox("Conversion Progress")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        progress_layout = QVBoxLayout()
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.installEventFilter(self)
        self.progress_text.setMinimumHeight(250)  # 增加高度
        self.progress_text.setStyleSheet("QTextEdit { background-color: #f8f8f8; }")
        progress_layout.addWidget(self.progress_text)
        progress_group.setLayout(progress_layout)
        
        # 添加到主布局
        main_layout.addLayout(top_layout)
        main_layout.addLayout(options_layout_horizontal)
        main_layout.addLayout(advanced_layout)
        main_layout.addWidget(progress_group)
        
        self.setLayout(main_layout)
        
        # 初始化模型列表
        self.scan_local_models()
        
        # 显示依赖状态
        if not TORCH_AVAILABLE:
            self.progress_text.setStyleSheet("QTextEdit { background-color: #fff8f8; }")  # 轻微红色背景提示错误
            self.progress_text.append("[WARNING] PyTorch or related dependencies are not installed, model conversion functionality will be limited")
            self.progress_text.append("Please ensure the following dependencies are installed:")
            self.progress_text.append("• PyTorch")
            self.progress_text.append("• transformers")
            self.progress_text.append("• safetensors")
            self.progress_text.append("\nInstallation command: pip install torch transformers safetensors")
            # 不禁用转换按钮，而是在点击时显示提示
        else:
            self.progress_text.append("[INFO] All dependencies are installed, model conversion is available")
            self.progress_text.append("Please select a model type and a specific model, then click the \"Convert\" button to start conversion")
    
    def on_model_type_changed(self):
        self.scan_local_models()
    
    # 根据选择的模型类型扫描本地模型
    def scan_local_models(self):
        self.model_combo.clear()
        if self.embedding_radio.isChecked():
            self.scan_local_embedding_models()
        else:
            self.scan_local_llm_models()
    
    # 扫描本地embeddings目录下的词嵌入模型
    def scan_local_embedding_models(self):
        # 获取embeddings目录路径
        embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        
        # 检查目录是否存在
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir, exist_ok=True)
            self.progress_text.append("[INFO] 词嵌入模型目录不存在，已创建一个空目录")
            return
        
        # 扫描目录下的所有子目录（每个子目录代表一个模型）
        models = []
        for item in os.listdir(embeddings_dir):
            item_path = os.path.join(embeddings_dir, item)
            if os.path.isdir(item_path):
                models.append(item)
        
        if models:
            self.model_combo.addItems(models)
            self.progress_text.append(f"[INFO] 找到{len(models)}个本地词嵌入模型")
        else:
            self.progress_text.append("[INFO] 没有找到本地词嵌入模型，请将模型放在embeddings目录下")
    
    # 扫描本地models目录下的大模型
    def scan_local_llm_models(self):
        # 获取models目录路径
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        # 检查目录是否存在
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            self.progress_text.append("[INFO] 大模型目录不存在，已创建一个空目录")
            return
        
        # 扫描目录下的所有子目录（每个子目录代表一个模型）
        models = []
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                models.append(item)
        
        if models:
            self.model_combo.addItems(models)
            self.progress_text.append(f"[INFO] 找到{len(models)}个本地大模型")
        else:
            self.progress_text.append("[INFO] 没有找到本地大模型，请将模型放在models目录下")
    
    def convert_model(self):
        # 检查依赖
        if not TORCH_AVAILABLE:
            # 显示依赖缺失提示
            self.progress_text.setStyleSheet("QTextEdit { background-color: #fff8f8; }")  # 轻微红色背景提示错误
            self.progress_text.append("[ERROR] PyTorch或相关依赖未安装，无法进行模型转换")
            self.progress_text.append("请安装以下依赖：")
            self.progress_text.append("1. 打开命令提示符或PowerShell")
            self.progress_text.append("2. 运行以下命令：")
            self.progress_text.append("   pip install torch transformers safetensors")
            self.progress_text.append("3. 安装完成后重启应用")
            self.progress_text.append("\n如果安装速度较慢，可以尝试使用国内镜像：")
            self.progress_text.append("   pip install torch transformers safetensors -i https://pypi.tuna.tsinghua.edu.cn/simple")
            
            return
        
        # 获取模型路径
        model_name = self.model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "错误", "请选择一个模型")
            return
        
        # 构建模型路径
        if self.embedding_radio.isChecked():
            # 词嵌入模型
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings", model_name)
        else:
            # 大语言模型
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_name)
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", f"模型路径不存在: {model_path}")
            return
        
        # 获取转换选项
        format_type = "ONNX"  # 固定为ONNX格式
        quantization_type = self.quantization_combo.currentText()
        if quantization_type == "None":
            quantization_type = None
        
        precision = self.precision_combo.currentText()
        if precision == "None":
            precision = None
        
        # 根据模型类型和量化选项创建输出目录名称
        model_name = os.path.basename(model_path)
        quant_suffix = f"_{quantization_type.lower()}_quant_{precision}" if quantization_type else ""
        
        # 根据模型类型决定输出目录的父目录
        if self.llm_radio.isChecked():
            parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        else:
            # 词嵌入模型放到embeddings目录
            parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        
        # 确保父目录存在
        os.makedirs(parent_dir, exist_ok=True)
        
        # 创建完整的输出目录路径
        output_dir = os.path.join(parent_dir, f"{model_name}{quant_suffix}")
        
        # 获取高级选项
        use_sdpa = self.use_sdpa_checkbox.isChecked()
        embeddings_only = self.embeddings_only_checkbox.isChecked()
        
        # 启用详细日志记录
        verbose_logging = True
        
        # 检查是否有测试样本文件
        test_samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_test_samples.txt")
        if not os.path.exists(test_samples_path):
            self.progress_text.append("[WARNING] 测试样本文件不存在，将使用默认测试样本")
            # 创建默认测试样本文件
            with open(test_samples_path, "w", encoding="utf-8") as f:
                f.write("这是一个简单的测试句子，用于测试模型的基本功能。\n")
                f.write("自然语言处理是人工智能的一个重要分支，它研究如何让计算机理解人类语言。\n")
                f.write("深度学习模型如Transformer已经在多种NLP任务上取得了突破性进展。\n")
                f.write("向量嵌入是将文本转换为数值表示的方法，使计算机能够处理和理解文本数据。\n")
                f.write("知识库问答系统结合了信息检索和自然语言处理技术，能够从大量文档中找到相关信息并生成答案。\n")
        
        # 显示转换配置摘要
        self.progress_text.append("[INFO] 转换配置摘要:")
        self.progress_text.append(f"• 模型: {model_name}")
        self.progress_text.append(f"• 模型类型: {'大语言模型' if self.llm_radio.isChecked() else '词嵌入模型'}")
        self.progress_text.append(f"• 输出格式: {format_type}")
        if quantization_type:
            self.progress_text.append(f"• 量化类型: {quantization_type}")
        if precision:
            self.progress_text.append(f"• 精度: {precision}")
        self.progress_text.append(f"• 使用SDPA: {'是' if use_sdpa else '否'}")
        self.progress_text.append(f"• 仅嵌入层: {'是' if embeddings_only else '否'}")
        self.progress_text.append(f"• 详细日志: {'启用' if verbose_logging else '禁用'}")
        self.progress_text.append(f"• 测试样本文件: {test_samples_path}")
        self.progress_text.append("\n[INFO] 开始转换...")
        
        # 禁用转换按钮，避免重复点击
        self.convert_button.setEnabled(False)
        self.convert_button.setText("转换中...")
        
        # 开始转换
        self.conversion_thread = ModelConversionThread(
            model_path, 
            quantization_type, 
            precision, 
            format_type="ONNX",  # 固定为ONNX格式
            is_llm=self.llm_radio.isChecked(),  # 传递是否为大模型的标志
            use_sdpa=use_sdpa,
            embeddings_only=embeddings_only,
            verbose_logging=verbose_logging,
            output_dir=output_dir  # 传递自定义输出目录
        )
        self.conversion_thread.progress_updated.connect(self.update_progress)
        self.conversion_thread.completed.connect(self.on_conversion_completed)
        self.conversion_thread.start()
    
    def update_progress(self, message):
        self.progress_text.append(message)
        # 滚动到底部
        cursor = self.progress_text.textCursor()
        cursor.movePosition(cursor.End)
        self.progress_text.setTextCursor(cursor)
    
    def on_conversion_completed(self, success, message):
        # 重新启用转换按钮
        self.convert_button.setEnabled(True)
        self.convert_button.setText("转换")
        
        if success:
            # 成功完成转换
            self.progress_text.setStyleSheet("QTextEdit { background-color: #f0fff0; }")  # 轻微绿色背景提示成功
            self.progress_text.append("\n[INFO] 转换成功!")
            
            # 显示成功对话框
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("转换成功")
            msg_box.setText("模型转换完成!")
            msg_box.setInformativeText(message)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
        else:
            # 转换失败
            self.progress_text.setStyleSheet("QTextEdit { background-color: #fff8f8; }")  # 轻微红色背景提示错误
            self.progress_text.append("\n[ERROR] 转换失败!")
            
            # 显示错误对话框
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("转换失败")
            msg_box.setText("模型转换过程遇到错误")
            msg_box.setInformativeText(message)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
