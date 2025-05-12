import sys
import os
import json
import datetime
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, 
                             QFileDialog, QListWidget, QTextEdit, QMenu, QAction, 
                             QMessageBox, QCheckBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QEvent, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
import requests
import numpy as np

# 配置文件路径
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".localrag")
API_KEYS_FILE = os.path.join(CONFIG_DIR, "api_keys.json")
PRESETS_FILE = os.path.join(CONFIG_DIR, "presets.json")
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_bases")
TOKENIZER_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_config.json")

# 确保目录存在
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

# API提供商列表
API_PROVIDERS = [
    "本地",
    "ollama (本地)",
    "DeepSeek",
    "千问",
    "月之暗面",
    "豆包",
    "自定义"
]

# API提供商默认地址
DEFAULT_API_URLS = {
    "本地": "",  # 本地模型不需要API地址
    "ollama (本地)": "http://localhost:11434",
    "DeepSeek": "https://api.deepseek.com",
    "千问": "https://dashscope.aliyuncs.com",
    "月之暗面": "https://api.moonshot.cn",
    "豆包": "https://api.doubao.com"
}

# 导入必要的库
from langchain.embeddings.base import Embeddings
from sqlite_vectorstore import SQLiteVectorStore

# 加载API密钥
def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载API密钥文件出错: {e}")
    return {}

# 保存API密钥
def save_api_keys(api_keys):
    try:
        with open(API_KEYS_FILE, 'w', encoding='utf-8') as f:
            json.dump(api_keys, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存API密钥文件出错: {e}")

# 加载提示词预设
def load_presets():
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载提示词预设文件出错: {e}")
    return []

# 保存提示词预设
def save_presets(presets):
    try:
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存提示词预设文件出错: {e}")

# 加载分词器配置
def load_tokenizer_config():
    if os.path.exists(TOKENIZER_CONFIG_FILE):
        try:
            with open(TOKENIZER_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载分词器配置文件出错: {e}")
    return {"use_android_tokenizer": False}

# 保存分词器配置
def save_tokenizer_config(config):
    try:
        with open(TOKENIZER_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存分词器配置文件出错: {e}")

# 获取知识库列表
def get_knowledge_bases():
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        return []
    return [d for d in os.listdir(KNOWLEDGE_BASE_DIR) 
            if os.path.isdir(os.path.join(KNOWLEDGE_BASE_DIR, d))]

# 获取本地模型列表
def get_local_models():
    """扫描models目录下的本地模型"""
    models = []
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # 确保目录存在
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        return models
    
    # 扫描目录下的所有子目录（每个子目录代表一个模型）
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # 检查是否是ONNX模型（包含model.onnx文件）
            if os.path.exists(os.path.join(item_path, "model.onnx")):
                models.append(f"{item} (ONNX)")
            # 检查是否是分离的ONNX模型（包含transformer_model.onnx和pooling_model.onnx文件）
            elif os.path.exists(os.path.join(item_path, "transformer_model.onnx")) and os.path.exists(os.path.join(item_path, "pooling_model.onnx")):
                models.append(f"{item} (ONNX)")
            # 检查是否是HuggingFace模型（包含config.json文件）
            elif os.path.exists(os.path.join(item_path, "config.json")):
                models.append(f"{item} (HuggingFace)")
            else:
                models.append(item)  # 未知类型的模型目录
    
    return models

# 模型获取线程
class ModelFetchThread(QThread):
    models_fetched = pyqtSignal(list, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, api_url, api_key, provider):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.provider = provider
        
    def run(self):
        try:
            models = self.fetch_models()
            self.models_fetched.emit(models, self.provider)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def fetch_models(self):
        # 根据不同的API提供商实现不同的模型获取逻辑
        if self.provider == "本地":
            # 获取本地模型列表
            return get_local_models()
        elif self.provider == "ollama (本地)" or "ollama" in self.api_url.lower():
            try:
                response = requests.get(f"{self.api_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [model['name'] for model in data.get('models', [])]
                else:
                    raise Exception(f"获取模型列表失败: {response.status_code}")
            except requests.RequestException as e:
                raise Exception(f"连接Ollama服务失败: {e}")
        
        # 其他API提供商的模型获取逻辑
        # 这里需要根据各个API提供商的实际接口进行实现
        elif self.provider == "DeepSeek":
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.api_url}/v1/models", headers=headers)
            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            else:
                raise Exception(f"获取模型列表失败: {response.status_code}")
        
        elif self.provider == "月之暗面":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.get(f"{self.api_url}/v1/models", headers=headers)
            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            else:
                raise Exception(f"获取月之暗面模型列表失败: {response.status_code}")
        
        # 其他API提供商...
        elif self.provider == "千问":
            # 千问API的模型获取逻辑
            return ["qwen-turbo", "qwen-plus", "qwen-max"]
            
        elif self.provider == "豆包":
            # 豆包API的模型获取逻辑
            return ["dbgpt-8k", "dbgpt-16k"]
            
        else:
            # 自定义API或其他未实现的提供商
            return ["default-model-1", "default-model-2"]

# 自定义Embeddings类，使用sentence-transformers
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model, progress_callback=None):
        self.model = model
        self.progress_callback = progress_callback
    
    def embed_documents(self, texts):
        if self.progress_callback:
            self.progress_callback(f"开始向量化 {len(texts)} 个文本块")
        
        # 分批处理，每处理一定数量的文本块就更新进度
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch).tolist()
            all_embeddings.extend(batch_embeddings)
            
            if self.progress_callback and i > 0:
                self.progress_callback(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 个文本块")
        
        return all_embeddings
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()
        
    # 确保对象可以被序列化
    def __getstate__(self):
        return {"model": self.model}
        
    def __setstate__(self, state):
        self.model = state["model"]
        self.progress_callback = None

# ONNX嵌入模型类，支持ONNX格式的模型
class ONNXEmbeddings(Embeddings):
    def __init__(self, model_path, progress_callback=None):
        self.model_path = model_path
        self.progress_callback = progress_callback
        
        try:
            # 加载分词器配置
            tokenizer_config = load_tokenizer_config()
            self.use_android_tokenizer = tokenizer_config.get("use_android_tokenizer", False)
            
            if self.progress_callback:
                try:
                    self.progress_callback(f"分词器配置: {'Android分词器' if self.use_android_tokenizer else 'HuggingFace分词器'}")
                except:
                    print(f"分词器配置: {'Android分词器' if self.use_android_tokenizer else 'HuggingFace分词器'}")
            
            # 检查是否存在transformer_model.onnx和pooling_model.onnx
            transformer_path = os.path.join(model_path, "transformer_model.onnx")
            pooling_path = os.path.join(model_path, "pooling_model.onnx")
            model_path_onnx = os.path.join(model_path, "model.onnx")
            
            if os.path.exists(model_path_onnx):
                # 使用单一的model.onnx
                import onnxruntime as ort
                from transformers import AutoTokenizer
                self.session = ort.InferenceSession(model_path_onnx)
                self.input_names = [input.name for input in self.session.get_inputs()]
                self.output_names = [output.name for output in self.session.get_outputs()]
                
                # 根据配置加载不同的分词器
                if self.use_android_tokenizer:
                    from android_tokenizer import BertTokenizer as AndroidTokenizer
                    vocab_file = os.path.join(model_path, "tokenizer.json")
                    self.tokenizer = AndroidTokenizer(vocab_file)
                    if self.progress_callback:
                        try:
                            self.progress_callback(f"已加载Android分词器，词汇表大小: {len(self.tokenizer.vocab)}")
                        except:
                            print(f"已加载Android分词器，词汇表大小: {len(self.tokenizer.vocab)}")
                else:
                    # 使用HuggingFace分词器
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    if self.progress_callback:
                        try:
                            self.progress_callback(f"已加载HuggingFace分词器，词汇表大小: {self.tokenizer.vocab_size}")
                        except:
                            print(f"已加载HuggingFace分词器，词汇表大小: {self.tokenizer.vocab_size}")
                
                self.use_separate_models = False
            elif os.path.exists(transformer_path) and os.path.exists(pooling_path):
                # 使用分离的transformer和pooling模型
                import onnxruntime as ort
                from transformers import AutoTokenizer
                
                self.transformer_session = ort.InferenceSession(transformer_path)
                self.pooling_session = ort.InferenceSession(pooling_path)
                
                # 根据配置加载不同的分词器
                if self.use_android_tokenizer:
                    from android_tokenizer import BertTokenizer as AndroidTokenizer
                    vocab_file = os.path.join(model_path, "tokenizer.json")
                    self.tokenizer = AndroidTokenizer(vocab_file)
                    if self.progress_callback:
                        try:
                            self.progress_callback(f"已加载Android分词器，词汇表大小: {len(self.tokenizer.vocab)}")
                        except:
                            print(f"已加载Android分词器，词汇表大小: {len(self.tokenizer.vocab)}")
                else:
                    # 使用HuggingFace分词器
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    if self.progress_callback:
                        try:
                            self.progress_callback(f"已加载HuggingFace分词器，词汇表大小: {self.tokenizer.vocab_size}")
                        except:
                            print(f"已加载HuggingFace分词器，词汇表大小: {self.tokenizer.vocab_size}")
                
                self.transformer_input_names = [input.name for input in self.transformer_session.get_inputs()]
                self.transformer_output_names = [output.name for output in self.transformer_session.get_outputs()]
                
                self.pooling_input_names = [input.name for input in self.pooling_session.get_inputs()]
                self.pooling_output_names = [output.name for output in self.pooling_session.get_outputs()]
                
                self.use_separate_models = True
            else:
                # 找不到ONNX模型文件
                raise ValueError(f"在{model_path}中找不到有效的ONNX模型文件")
        except Exception as e:
            print(f"ONNX模型初始化出错: {e}")
            raise
    
    def embed_documents(self, texts):
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if self.progress_callback and i % 10 == 0:
                try:
                    self.progress_callback(f"正在处理文档 {i+1}/{total}")
                except Exception as e:
                    # 如果回调函数调用失败，忽略错误并继续
                    print(f"进度回调函数调用失败: {e}")
            
            # 获取单个文档的嵌入
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text):
        try:
            # 使用tokenizer处理文本
            if self.use_android_tokenizer:
                # Android分词器的输出格式不同，需要特殊处理
                inputs = self.tokenizer(text)  # 这里会返回正确格式的输入
                
                # 详细的分词调试信息
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                token_details = " ".join([f"{t}" for t in tokens])
                
                if self.progress_callback:
                    try:
                        self.progress_callback(f"Android分词器处理文本: {len(inputs['input_ids'][0])}个token")
                        self.progress_callback(f"分词结果: {token_details}")
                    except:
                        print(f"Android分词器处理文本: {len(inputs['input_ids'][0])}个token")
                        print(f"分词结果: {token_details}")
            else:
                # 使用HuggingFace分词器处理文本
                inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
                
                # 详细的分词调试信息
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                token_details = " ".join([f"{t}" for t in tokens])
                
                if self.progress_callback:
                    try:
                        self.progress_callback(f"HuggingFace分词器处理文本: {len(inputs['input_ids'][0])}个token")
                        self.progress_callback(f"分词结果: {token_details}")
                    except:
                        print(f"HuggingFace分词器处理文本: {len(inputs['input_ids'][0])}个token")
                        print(f"分词结果: {token_details}")
            
            if self.use_separate_models:
                # 使用分离的transformer和pooling模型
                
                # 准备transformer输入
                transformer_inputs = {}
                for name in self.transformer_input_names:
                    if name in inputs:
                        transformer_inputs[name] = inputs[name]
                
                # 运行transformer推理
                try:
                    transformer_outputs = self.transformer_session.run(
                        self.transformer_output_names, 
                        transformer_inputs
                    )
                except Exception as e:
                    if self.progress_callback:
                        try:
                            self.progress_callback(f"Transformer推理失败: {str(e)}")
                        except:
                            print(f"Transformer推理失败: {str(e)}")
                    raise
                
                # 准备pooling输入
                pooling_inputs = {
                    "token_embeddings": transformer_outputs[0],
                    "attention_mask": inputs["attention_mask"]
                }
                
                # 运行pooling推理
                try:
                    pooling_outputs = self.pooling_session.run(
                        self.pooling_output_names, 
                        pooling_inputs
                    )
                except Exception as e:
                    if self.progress_callback:
                        try:
                            self.progress_callback(f"Pooling推理失败: {str(e)}")
                        except:
                            print(f"Pooling推理失败: {str(e)}")
                    raise
                
                # 获取嵌入向量
                embedding = pooling_outputs[0]
                
                # 检查嵌入的维度
                if len(embedding.shape) == 3:
                    # 对于BGE-M3模型，使用平均池化将token嵌入转换为句子嵌入
                    if self.progress_callback:
                        try:
                            self.progress_callback(f"检测到3D嵌入输出，应用平均池化: {embedding.shape}")
                        except:
                            print(f"检测到3D嵌入输出，应用平均池化: {embedding.shape}")
                    embedding = np.mean(embedding, axis=1)
                
                # 如果是批处理输出，取第一个
                if len(embedding.shape) > 1:
                    embedding = embedding[0]
            else:
                # 使用单一的model.onnx
                
                # 准备输入
                onnx_inputs = {}
                for name in self.input_names:
                    if name in inputs:
                        onnx_inputs[name] = inputs[name]
                
                # 运行推理
                try:
                    outputs = self.session.run(None, onnx_inputs)
                except Exception as e:
                    if self.progress_callback:
                        try:
                            self.progress_callback(f"ONNX推理失败: {str(e)}")
                        except:
                            print(f"ONNX推理失败: {str(e)}")
                    raise
                
                # 获取嵌入向量
                embedding = outputs[0]
                
                # 如果是批处理输出，取第一个
                if len(embedding.shape) > 1:
                    embedding = embedding[0]
            
            # 归一化
            import numpy as np
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.tolist()
        except Exception as e:
            if self.progress_callback:
                try:
                    self.progress_callback(f"嵌入查询失败: {str(e)}")
                except:
                    print(f"嵌入查询失败: {str(e)}")
            raise
    
    # 添加序列化支持
    def __getstate__(self):
        # 只保存模型路径，不保存模型对象
        return {"model_path": self.model_path}
        
    def __setstate__(self, state):
        # 重新初始化模型
        self.__init__(state["model_path"])

# 知识库构建线程
class KnowledgeBaseBuilderThread(QThread):
    progress_updated = pyqtSignal(str)
    completed = pyqtSignal(bool, str)
    
    def __init__(self, kb_name, files, embedding_model, mode="overwrite"):
        super().__init__()
        self.kb_name = kb_name
        self.files = files
        self.embedding_model = embedding_model
        self.mode = mode  # 添加模式参数：overwrite（覆盖）或append（追加）
    
    def update_progress(self, msg):
        """用于向量化过程中的进度回调，可以被序列化"""
        self.progress_updated.emit(msg)
        
    def run(self):
        try:
            # 导入必要的库
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.document_loaders import (
                PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader,
                UnstructuredExcelLoader, TextLoader, UnstructuredHTMLLoader, JSONLoader
            )
            # 尝试导入处理旧版Word文档的库
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "jq", "python-docx", "pywin32"])
                self.progress_updated.emit("已安装必要的依赖包")
            except Exception as e:
                self.progress_updated.emit(f"安装依赖包时出错: {str(e)}，但将继续处理")
            
            # 导入处理旧版Word文档的库
            try:
                import win32com.client
                import pythoncom
                self.has_win32com = True
                self.progress_updated.emit("成功加载COM组件，可以处理旧版Word文档")
            except ImportError:
                self.has_win32com = False
                self.progress_updated.emit("未能加载COM组件，旧版Word文档(.doc)可能无法正确处理")
            from sentence_transformers import SentenceTransformer
            import pickle
            import shutil
            
            # 创建知识库目录
            kb_dir = os.path.join(KNOWLEDGE_BASE_DIR, self.kb_name)
            # 根据模式处理目录
            if self.mode == "overwrite":
                # 覆盖模式：如果目录已存在，先清空
                if os.path.exists(kb_dir):
                    shutil.rmtree(kb_dir)
                os.makedirs(kb_dir, exist_ok=True)
                self.progress_updated.emit("覆盖模式：将替换现有知识库")
            else:  # append模式
                # 追加模式：保留目录，但需要加载现有向量存储
                os.makedirs(kb_dir, exist_ok=True)
                self.progress_updated.emit("追加模式：将向现有知识库添加内容")
            
            self.progress_updated.emit("开始处理文件...")
            
            # 存储所有文档的文本内容
            all_docs = []
            
            # 处理每个文件
            for file_path in self.files:
                file_name = os.path.basename(file_path)
                self.progress_updated.emit(f"处理文件: {file_name}")
                
                # 根据文件类型选择合适的加载器
                try:
                    if file_path.lower().endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif file_path.lower().endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                    elif file_path.lower().endswith('.doc'):
                        # 处理旧版Word文档
                        if self.has_win32com:
                            # 使用COM组件处理旧版Word文档
                            try:
                                pythoncom.CoInitialize()  # 初始化COM环境
                                word_app = win32com.client.Dispatch("Word.Application")
                                word_app.Visible = False
                                doc = word_app.Documents.Open(file_path)
                                text = doc.Content.Text
                                doc.Close()
                                word_app.Quit()
                                
                                # 创建临时文本文件
                                temp_txt_path = file_path + ".txt"
                                with open(temp_txt_path, 'w', encoding='utf-8') as f:
                                    f.write(text)
                                
                                # 使用文本加载器加载临时文件
                                loader = TextLoader(temp_txt_path)
                                docs = loader.load()
                                
                                # 删除临时文件
                                try:
                                    os.remove(temp_txt_path)
                                except:
                                    pass
                                
                                self.progress_updated.emit(f"成功加载文档: {file_name}，共{len(docs)}页/段")
                                all_docs.extend(docs)
                                continue
                            except Exception as e:
                                self.progress_updated.emit(f"使用COM组件处理Word文档失败: {str(e)}，尝试使用备选方法")
                                # 如果COM处理失败，尝试使用Docx2txtLoader
                                try:
                                    loader = Docx2txtLoader(file_path)
                                except:
                                    self.progress_updated.emit(f"无法处理旧版Word文档: {file_name}，跳过")
                                    continue
                        else:
                            # 尝试使用Docx2txtLoader，可能会失败
                            try:
                                loader = Docx2txtLoader(file_path)
                            except:
                                self.progress_updated.emit(f"无法处理旧版Word文档: {file_name}，跳过")
                                continue
                    elif file_path.lower().endswith(('.pptx', '.ppt')):
                        loader = UnstructuredPowerPointLoader(file_path)
                    elif file_path.lower().endswith(('.xlsx', '.xls')):
                        loader = UnstructuredExcelLoader(file_path)
                    elif file_path.lower().endswith(('.txt', '.md', '.html', '.htm')):
                        loader = TextLoader(file_path)
                    elif file_path.lower().endswith('.json'):
                        # 先尝试安装jq
                        try:
                            import jq
                        except ImportError:
                            try:
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "jq"])
                                self.progress_updated.emit("已安装jq包")
                                import jq
                            except Exception as e:
                                self.progress_updated.emit(f"安装jq包失败: {str(e)}，将使用备选方法处理JSON文件")
                                # 使用备选方法处理JSON文件
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        json_data = json.load(f)
                                    # 将JSON数据转换为文本
                                    text = json.dumps(json_data, ensure_ascii=False, indent=2)
                                    # 创建临时文本文件
                                    temp_txt_path = file_path + ".txt"
                                    with open(temp_txt_path, 'w', encoding='utf-8') as f:
                                        f.write(text)
                                    # 使用文本加载器加载临时文件
                                    loader = TextLoader(temp_txt_path)
                                    docs = loader.load()
                                    # 删除临时文件
                                    try:
                                        os.remove(temp_txt_path)
                                    except:
                                        pass
                                    self.progress_updated.emit(f"成功加载JSON文档: {file_name}，共{len(docs)}页/段")
                                    all_docs.extend(docs)
                                    continue
                                except Exception as e:
                                    self.progress_updated.emit(f"处理JSON文件失败: {str(e)}，跳过")
                                    continue
                        # 使用JSONLoader加载
                        loader = JSONLoader(file_path, jq_schema='.', text_content=False)
                    else:
                        self.progress_updated.emit(f"不支持的文件类型: {file_name}，跳过")
                        continue
                    
                    # 加载文档
                    try:
                        docs = loader.load()
                        self.progress_updated.emit(f"成功加载文档: {file_name}，共{len(docs)}页/段")
                        all_docs.extend(docs)
                    except Exception as doc_error:
                        # 处理特定错误："There is no item named 'word/document.xml' in the archive"
                        if "word/document.xml" in str(doc_error) and file_path.lower().endswith('.docx'):
                            self.progress_updated.emit(f"文档XML结构错误，尝试备选方法处理: {file_name}")
                            try:
                                # 尝试使用文本方式读取
                                import zipfile
                                text_content = ""
                                with zipfile.ZipFile(file_path) as z:
                                    # 尝试从document.xml或其他可能的位置提取文本
                                    for item in z.namelist():
                                        if item.endswith('.xml') and not item.startswith('_'):
                                            try:
                                                content = z.read(item).decode('utf-8')
                                                # 简单去除XML标签
                                                import re
                                                text_content += re.sub('<[^<]+>', ' ', content) + "\n"
                                            except:
                                                pass
                                
                                if text_content:
                                    # 创建临时文本文件
                                    temp_txt_path = file_path + ".txt"
                                    with open(temp_txt_path, 'w', encoding='utf-8') as f:
                                        f.write(text_content)
                                    # 使用文本加载器
                                    temp_loader = TextLoader(temp_txt_path)
                                    docs = temp_loader.load()
                                    # 删除临时文件
                                    try:
                                        os.remove(temp_txt_path)
                                    except:
                                        pass
                                    self.progress_updated.emit(f"使用备选方法成功加载文档: {file_name}，共{len(docs)}页/段")
                                    all_docs.extend(docs)
                                else:
                                    raise Exception("无法提取文本内容")
                            except Exception as backup_error:
                                self.progress_updated.emit(f"备选方法处理失败: {str(backup_error)}")
                        else:
                            self.progress_updated.emit(f"处理文件 {file_name} 时出错: {str(doc_error)}")
                except Exception as e:
                    self.progress_updated.emit(f"处理文件 {file_name} 时出错: {str(e)}")
            
            if not all_docs:
                raise Exception("没有成功加载任何文档")
            
            # 文本分割
            self.progress_updated.emit("正在进行文本分割...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(all_docs)
            
            # 过滤掉过小的文本块，防止向量偏差
            MIN_CHUNK_SIZE = 200  # 最小文本块大小，与Android版本保持一致
            filtered_chunks = []
            for chunk in chunks:
                if len(chunk.page_content) >= MIN_CHUNK_SIZE:
                    filtered_chunks.append(chunk)
                else:
                    self.progress_updated.emit(f"过滤掉过小的文本块，长度: {len(chunk.page_content)}")
            
            self.progress_updated.emit(f"文本分割完成，共生成{len(chunks)}个文本块，过滤后剩余{len(filtered_chunks)}个")
            chunks = filtered_chunks
            
            # 使用词嵌入模型生成向量
            self.progress_updated.emit("正在生成向量嵌入...")
            
            # 使用本地词嵌入模型进行向量化
            self.progress_updated.emit(f"使用本地词嵌入模型 {self.embedding_model} 进行向量化")
            
            try:
                # 导入必要的库
                from sentence_transformers import SentenceTransformer
                from langchain.embeddings.base import Embeddings
                
                # 获取embeddings目录路径
                embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
                
                # 检查模型路径是否存在
                model_path = os.path.join(embeddings_dir, self.embedding_model)
                if not os.path.exists(model_path):
                    raise Exception(f"模型路径不存在: {model_path}")
                
                # 加载本地模型
                self.progress_updated.emit(f"正在加载本地模型: {self.embedding_model}")
                
                # 加载分词器配置，确保RAG查询使用与知识库构建相同的分词器
                tokenizer_config = load_tokenizer_config()
                use_android_tokenizer = tokenizer_config.get("use_android_tokenizer", False)
                self.progress_updated.emit(f"分词器配置: {'Android分词器' if use_android_tokenizer else 'HuggingFace分词器'}")
                
                # 检查是否是ONNX模型
                if "_onnx" in self.embedding_model.lower() or self.embedding_model.lower().endswith("onnx") or "quant" in self.embedding_model.lower():
                    self.progress_updated.emit("检测到ONNX格式模型，使用ONNX运行时加载")
                    try:
                        # 导入ONNX运行时
                        import onnxruntime as ort
                        import numpy as np
                        
                        # 使用ONNXEmbeddings类加载模型
                        embeddings = ONNXEmbeddings(model_path, progress_callback=self.update_progress)
                        self.progress_updated.emit("ONNX模型加载成功")
                    except Exception as onnx_error:
                        self.progress_updated.emit(f"ONNXEmbeddings初始化失败: {str(onnx_error)}")
                        raise onnx_error
                else:
                    # 使用标准SentenceTransformer加载
                    self.progress_updated.emit(f"使用标准SentenceTransformer加载模型: {model_path}")
                    try:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer(model_path)
                        embeddings = SentenceTransformerEmbeddings(model, progress_callback=self.update_progress)
                    except Exception as e:
                        self.progress_updated.emit(f"SentenceTransformer加载失败: {str(e)}")
                        raise e
                # 创建或更新向量存储
                self.progress_updated.emit("正在创建向量存储...")
                
                # 使用SQLite作为向量存储
                db_path = os.path.join(kb_dir, "vectorstore.db")
                
                if self.mode == "append" and os.path.exists(db_path):
                    # 追加模式：加载现有向量存储
                    try:
                        self.progress_updated.emit("正在加载现有向量存储...")
                        
                        # 加载现有SQLite向量存储
                        metadata_path = os.path.join(kb_dir, "metadata.json")
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        
                        embedding_model_name = metadata.get("embedding_model", "all-MiniLM-L6-v2")
                        embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
                        
                        # 检查是否使用自定义嵌入模型
                        if self.embedding_model and self.embedding_model != "默认 (使用知识库原模型)":
                            # 提取模型名称（去掉类型标识）
                            custom_model_name = self.embedding_model
                            if " (ONNX)" in custom_model_name:
                                custom_model_name = custom_model_name.replace(" (ONNX)", "")
                            elif " (HuggingFace)" in custom_model_name:
                                custom_model_name = custom_model_name.replace(" (HuggingFace)", "")
                            
                            self.progress_updated.emit(f"使用自定义嵌入模型: {custom_model_name}，覆盖知识库原模型: {embedding_model_name}")
                            embedding_model_name = custom_model_name
                        else:
                            self.progress_updated.emit(f"使用知识库原嵌入模型: {embedding_model_name}")
                        
                        model_path = os.path.join(embeddings_dir, embedding_model_name)
                        
                        # 加载嵌入模型
                        if "_onnx" in embedding_model_name.lower() or embedding_model_name.lower().endswith("onnx") or "quant" in embedding_model_name.lower():
                            self.progress_updated.emit("检测到ONNX格式模型，使用ONNX运行时加载")
                            try:
                                # 导入ONNX运行时
                                import onnxruntime as ort
                                import numpy as np
                                
                                # 使用ONNXEmbeddings类加载模型
                                embeddings = ONNXEmbeddings(model_path, progress_callback=self.update_progress)
                                self.progress_updated.emit("ONNX模型加载成功")
                            except Exception as onnx_error:
                                self.progress_updated.emit(f"ONNXEmbeddings初始化失败: {str(onnx_error)}")
                                raise onnx_error
                        else:
                            # 使用标准SentenceTransformer加载
                            self.progress_updated.emit(f"使用标准SentenceTransformer加载模型: {model_path}")
                            try:
                                from sentence_transformers import SentenceTransformer
                                model = SentenceTransformer(model_path)
                                embeddings = SentenceTransformerEmbeddings(model, progress_callback=self.update_progress)
                            except Exception as e:
                                self.progress_updated.emit(f"SentenceTransformer加载失败: {str(e)}")
                                raise e
                        # 加载向量存储
                        vectorstore = SQLiteVectorStore(
                            embedding_function=embeddings,
                            db_path=db_path,
                            collection_name=self.kb_name
                        )
                        self.progress_updated.emit("向量存储加载成功")
                        
                        # 将新文档添加到现有向量存储
                        self.progress_updated.emit(f"正在向现有向量存储添加{len(chunks)}个文本块...")
                        vectorstore.add_documents(chunks)
                        self.progress_updated.emit("成功添加新文档到现有向量存储")
                    except Exception as e:
                        self.progress_updated.emit(f"加载现有向量存储失败: {str(e)}，将创建新的向量存储")
                        vectorstore = SQLiteVectorStore.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            db_path=db_path,
                            collection_name=self.kb_name
                        )
                else:
                    # 覆盖模式或追加模式但没有现有向量存储：创建新的向量存储
                    vectorstore = SQLiteVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        db_path=db_path,
                        collection_name=self.kb_name
                    )
                
                # 保存文档元数据
                metadata_path = os.path.join(kb_dir, "metadata.json")
                metadata = {
                    "file_count": len(self.files),
                    "chunk_count": len(chunks),
                    "files": [os.path.basename(f) for f in self.files],
                    "embedding_model": self.embedding_model,
                    "model_type": "local",
                    "created_at": str(datetime.datetime.now()),
                    "vector_store_type": "sqlite"  # 添加向量存储类型信息
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                self.progress_updated.emit("知识库构建完成!")
                self.completed.emit(True, "知识库构建成功!")
            except Exception as e:
                self.progress_updated.emit(f"创建向量存储失败: {str(e)}")
                raise Exception(f"创建向量存储失败: {str(e)}")
        except Exception as e:
            self.progress_updated.emit(f"错误: {str(e)}")
            self.completed.emit(False, f"知识库构建失败: {str(e)}")

# RAG查询线程
class RAGQueryThread(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    completed = pyqtSignal()
    progress_updated = pyqtSignal(str)  # 添加进度更新信号
    
    # 添加类变量来跟踪加载的本地模型
    loaded_local_model = None
    loaded_tokenizer = None
    last_activity_time = 0
    model_unload_timer = None
    
    def __init__(self, query, preset, kb_name, llm_api_url, llm_api_key, llm_model, provider="ollama (本地)", custom_embedding_model=None):
        super().__init__()
        self.query = query
        self.preset = preset
        self.kb_name = kb_name
        self.llm_api_url = llm_api_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.provider = provider
        self.custom_embedding_model = custom_embedding_model
        self.is_stopped = False
        
        # 添加进度回调函数
        self.update_progress = self.debug_log
        
        # 更新最后活动时间
        RAGQueryThread.last_activity_time = time.time()
        
        # 如果定时器不存在，创建一个定时器用于自动卸载模型
        if RAGQueryThread.model_unload_timer is None and provider == "本地":
            RAGQueryThread.model_unload_timer = QTimer()
            RAGQueryThread.model_unload_timer.timeout.connect(RAGQueryThread.check_model_unload)
            RAGQueryThread.model_unload_timer.start(60000)  # 每分钟检查一次
    
    def debug_log(self, message):
        """输出调试信息到控制台并通过信号发送到UI"""
        print(f"[DEBUG] {message}")
        self.response_received.emit(f"[调试] {message}\n")
        
    def run(self):
        try:
            # 导入必要的库
            import pickle
            import requests
            from langchain.prompts import PromptTemplate
            # 移除FAISS导入
            # from langchain_community.vectorstores import FAISS
            from sqlite_vectorstore import SQLiteVectorStore
            
            # 检查是否使用知识库
            if self.kb_name is None:
                self.debug_log("未选择知识库，将直接使用大模型回答")
                self.response_received.emit("未使用知识库，直接使用大模型回答...\n")
                context = None
            else:
                # 从知识库中检索相关内容
                kb_dir = os.path.join(KNOWLEDGE_BASE_DIR, self.kb_name)
                self.response_received.emit("正在从知识库检索相关内容...\n")
                self.debug_log(f"知识库目录: {kb_dir}")
                
                # 检查知识库是否存在
                vectorstore_path = os.path.join(kb_dir, "vectorstore.db")
                self.debug_log(f"向量存储路径: {vectorstore_path}")
                if not os.path.exists(vectorstore_path):
                    raise Exception(f"知识库 {self.kb_name} 不存在或未正确构建")
                
                # 加载向量存储
                self.debug_log("开始加载向量存储...")
                try:
                    # 加载SQLite向量存储
                    # 首先获取嵌入模型
                    metadata_path = os.path.join(kb_dir, "metadata.json")
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    embedding_model_name = metadata.get("embedding_model", "all-MiniLM-L6-v2")
                    embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
                    
                    # 检查是否使用自定义嵌入模型
                    if self.custom_embedding_model and self.custom_embedding_model != "默认 (使用知识库原模型)":
                        # 提取模型名称（去掉类型标识）
                        custom_model_name = self.custom_embedding_model
                        if " (ONNX)" in custom_model_name:
                            custom_model_name = custom_model_name.replace(" (ONNX)", "")
                        elif " (HuggingFace)" in custom_model_name:
                            custom_model_name = custom_model_name.replace(" (HuggingFace)", "")
                        
                        self.debug_log(f"使用自定义嵌入模型: {custom_model_name}，覆盖知识库原模型: {embedding_model_name}")
                        embedding_model_name = custom_model_name
                    else:
                        self.debug_log(f"使用知识库原嵌入模型: {embedding_model_name}")
                    
                    model_path = os.path.join(embeddings_dir, embedding_model_name)
                    
                    # 加载嵌入模型
                    if "_onnx" in embedding_model_name.lower() or embedding_model_name.lower().endswith("onnx") or "quant" in embedding_model_name.lower():
                        self.debug_log("检测到ONNX格式模型，使用ONNX运行时加载")
                        try:
                            # 导入ONNX运行时
                            import onnxruntime as ort
                            import numpy as np
                            
                            # 检查模型文件是否存在
                            model_onnx_path = os.path.join(model_path, "model.onnx")
                            transformer_path = os.path.join(model_path, "transformer_model.onnx")
                            pooling_path = os.path.join(model_path, "pooling_model.onnx")
                            
                            # 检查是否是量化模型
                            is_quantized = "quant" in embedding_model_name.lower() or "int8" in embedding_model_name.lower() or "int4" in embedding_model_name.lower()
                            if is_quantized:
                                self.debug_log(f"检测到量化模型: {embedding_model_name}")
                                # 检查目录中的所有文件，帮助诊断问题
                                self.debug_log(f"目录内容: {os.listdir(model_path)}")
                            
                            if not (os.path.exists(model_onnx_path) or (os.path.exists(transformer_path) and os.path.exists(pooling_path))):
                                self.debug_log(f"找不到ONNX模型文件，模型路径: {model_path}")
                                self.debug_log(f"model.onnx存在: {os.path.exists(model_onnx_path)}")
                                self.debug_log(f"transformer_model.onnx存在: {os.path.exists(transformer_path)}")
                                self.debug_log(f"pooling_model.onnx存在: {os.path.exists(pooling_path)}")
                                raise FileNotFoundError(f"在{model_path}中找不到有效的ONNX模型文件")
                            
                            # 使用ONNXEmbeddings类加载模型
                            self.debug_log(f"初始化ONNX模型: {model_path}")
                            try:
                                # 加载分词器配置，确保RAG查询使用与知识库构建相同的分词器
                                tokenizer_config = load_tokenizer_config()
                                use_android_tokenizer = tokenizer_config.get("use_android_tokenizer", False)
                                self.debug_log(f"分词器配置: {'Android分词器' if use_android_tokenizer else 'HuggingFace分词器'}")
                                
                                # 确保ONNXEmbeddings类使用正确的分词器
                                # 注意：这里不需要额外设置，因为ONNXEmbeddings类的__init__方法会自动加载分词器配置
                                embeddings = ONNXEmbeddings(model_path, progress_callback=self.debug_log)
                                self.debug_log("ONNX模型加载成功")
                            except Exception as onnx_error:
                                self.debug_log(f"ONNXEmbeddings初始化失败: {str(onnx_error)}")
                                raise onnx_error
                        except Exception as onnx_error:
                            self.debug_log(f"加载ONNX模型失败: {str(onnx_error)}，尝试使用标准方法")
                            from sentence_transformers import SentenceTransformer
                            try:
                                self.debug_log(f"尝试使用SentenceTransformer加载模型: {model_path}")
                                model = SentenceTransformer(model_path)
                                embeddings = SentenceTransformerEmbeddings(model, progress_callback=self.debug_log)
                                self.debug_log("使用SentenceTransformer加载成功")
                            except Exception as st_error:
                                self.debug_log(f"使用SentenceTransformer加载也失败: {str(st_error)}")
                                raise Exception(f"无法加载模型: {str(onnx_error)}, {str(st_error)}")
                    else:
                        # 使用标准SentenceTransformer加载
                        self.debug_log(f"使用标准SentenceTransformer加载模型: {model_path}")
                        try:
                            from sentence_transformers import SentenceTransformer
                            model = SentenceTransformer(model_path)
                            embeddings = SentenceTransformerEmbeddings(model, progress_callback=self.debug_log)
                            self.debug_log("标准模型加载成功")
                        except Exception as e:
                            self.debug_log(f"SentenceTransformer加载失败: {str(e)}")
                            raise e
                    # 加载向量存储
                    vectorstore = SQLiteVectorStore(
                        embedding_function=embeddings,
                        db_path=vectorstore_path,
                        collection_name=self.kb_name
                    )
                    self.debug_log("向量存储加载成功")
                    
                    # 检索相关文档
                    self.debug_log(f"开始检索与问题相关的文档: {self.query}")
                    docs = vectorstore.similarity_search(self.query, k=5)
                    self.debug_log(f"检索到 {len(docs)} 个相关文档")
                    
                    # 显示每个文档的相似度值（调试用）
                    self.debug_log("===== 相似度详细信息 =====")
                    for i, doc in enumerate(docs):
                        similarity = doc.metadata.get("similarity_score", "未知")
                        source = doc.metadata.get("source", "未知")
                        self.debug_log(f"文档 {i+1} - 相似度: {similarity:.6f} - 来源: {source}")
                        # 显示文档内容的前100个字符
                        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        self.debug_log(f"内容预览: {preview}")
                    self.debug_log("===== 相似度详细信息结束 =====")
                    
                    # 构建上下文
                    context = "\n\n".join([doc.page_content for doc in docs])
                    self.debug_log(f"构建的上下文长度: {len(context)} 字符")
                except Exception as e:
                    self.debug_log(f"加载向量存储失败: {str(e)}")
                    raise Exception(f"加载知识库失败: {str(e)}")
            
            # 构建提示词
            self.debug_log("开始构建提示词")
            if self.kb_name is None:
                # 不使用知识库时的提示词模板
                if self.preset and self.preset.strip():
                    # 使用用户提供的提示词模板，但不包含知识库相关内容
                    prompt_template = PromptTemplate.from_template(self.preset + """

用户问题：{query}

请根据你的知识回答用户的问题。
""")
                    self.debug_log("使用用户自定义提示词模板（无知识库）")
                else:
                    # 使用默认提示词模板（无知识库）
                    prompt_template = PromptTemplate.from_template("""你是一个有用的AI助手。

用户问题：{query}

请根据你的知识回答用户的问题。
""")
                    self.debug_log("使用默认提示词模板（无知识库）")
                
                # 构建不包含context的提示词
                prompt = prompt_template.format(query=self.query)
            else:
                # 使用知识库时的提示词模板
                if self.preset and self.preset.strip():
                    # 使用用户提供的提示词模板
                    prompt_template = PromptTemplate.from_template(self.preset + """

系统已从知识库中检索到以下相关内容：
{context}

用户问题：{query}

请根据知识库中的相关内容回答用户的问题。如果知识库中没有相关信息，请明确指出并尝试根据你的知识提供一个合理的回答。
""")
                    self.debug_log("使用用户自定义提示词模板")
                else:
                    # 使用默认提示词模板
                    prompt_template = PromptTemplate.from_template("""你是一个有用的AI助手。

系统已从知识库中检索到以下相关内容：
{context}

请根据以上信息回答用户的问题。如果知识库中没有相关信息，请明确指出并尝试根据你的知识提供一个合理的回答。

问题：{query}

回答：""")
                    self.debug_log("使用默认提示词模板")
                
                # 构建包含context的提示词
                prompt = prompt_template.format(context=context, query=self.query)
            
            # 调用大模型API
            self.response_received.emit("正在生成回答...\n")
            
            # 根据不同的API提供商调用不同的API
            self.debug_log(f"使用API提供商: {self.provider}, API地址: {self.llm_api_url}, 模型: {self.llm_model}")
            
            # 处理本地模型
            if self.provider == "本地":
                self.debug_log("使用本地模型进行推理")
                self.handle_local_model(prompt)
            # 处理Ollama API
            elif self.provider == "ollama (本地)" or "ollama" in self.llm_api_url.lower():
                # 使用Ollama API
                try:
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": True
                    }
                    
                    self.debug_log(f"准备调用Ollama API: {self.llm_api_url}/api/generate")
                    self.debug_log(f"请求头: {headers}")
                    self.debug_log(f"请求数据: model={self.llm_model}, prompt长度={len(prompt)}字符, stream=True")
                    
                    try:
                        response = requests.post(f"{self.llm_api_url}/api/generate", json=data, headers=headers, stream=True)
                        self.debug_log(f"API响应状态码: {response.status_code}")
                        response.raise_for_status()
                        
                        self.debug_log("开始处理流式响应...")
                        line_count = 0
                        for line in response.iter_lines():
                            if self.is_stopped:
                                self.debug_log("用户停止了生成")
                                break
                            if line:
                                try:
                                    chunk_data = json.loads(line)
                                    if "response" in chunk_data:
                                        self.response_received.emit(chunk_data["response"])
                                        line_count += 1
                                        if line_count % 10 == 0:
                                            self.debug_log(f"已处理 {line_count} 行响应")
                                except json.JSONDecodeError as e:
                                    self.debug_log(f"JSON解析错误: {str(e)}, 行内容: {line[:100]}...")
                                    continue
                        self.debug_log(f"流式响应处理完成，共处理 {line_count} 行")
                    except requests.exceptions.RequestException as e:
                        self.debug_log(f"请求异常: {str(e)}")
                        raise
                except Exception as e:
                    self.debug_log(f"调用Ollama API过程中发生异常: {str(e)}")
                    raise Exception(f"调用Ollama API失败: {str(e)}")
            elif self.provider == "DeepSeek":
                # DeepSeek API
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.llm_api_key}"
                    }
                    data = {
                        "model": self.llm_model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": True
                    }
                    
                    with requests.post(f"{self.llm_api_url}/v1/chat/completions", json=data, headers=headers, stream=True) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if self.is_stopped:
                                break
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    line = line[6:]
                                    if line.strip() == '[DONE]':
                                        break
                                    try:
                                        chunk_data = json.loads(line)
                                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                            delta = chunk_data['choices'][0].get('delta', {})
                                            if 'content' in delta and delta['content']:
                                                self.response_received.emit(delta['content'])
                                    except json.JSONDecodeError:
                                        continue
                except Exception as e:
                    raise Exception(f"调用DeepSeek API失败: {str(e)}")
            elif self.provider == "千问":
                # 千问 API (DashScope)
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.llm_api_key}"
                    }
                    data = {
                        "model": self.llm_model,
                        "input": {
                            "messages": [
                                {"role": "user", "content": prompt}
                            ]
                        },
                        "parameters": {
                            "result_format": "message",
                            "incremental_output": True
                        }
                    }
                    
                    with requests.post(f"{self.llm_api_url}/v1/services/aigc/text-generation/generation", 
                                     json=data, headers=headers, stream=True) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if self.is_stopped:
                                break
                            if line:
                                try:
                                    chunk_data = json.loads(line)
                                    if 'output' in chunk_data and 'text' in chunk_data['output']:
                                        self.response_received.emit(chunk_data['output']['text'])
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    raise Exception(f"调用千问API失败: {str(e)}")
            elif self.provider == "月之暗面":
                # 月之暗面 API
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.llm_api_key}"
                    }
                    data = {
                        "model": self.llm_model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": True
                    }
                    
                    with requests.post(f"{self.llm_api_url}/v1/chat/completions", 
                                     json=data, headers=headers, stream=True) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if self.is_stopped:
                                break
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    line = line[6:]
                                    if line.strip() == '[DONE]':
                                        break
                                    try:
                                        chunk_data = json.loads(line)
                                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                            delta = chunk_data['choices'][0].get('delta', {})
                                            if 'content' in delta and delta['content']:
                                                self.response_received.emit(delta['content'])
                                    except json.JSONDecodeError:
                                        continue
                except Exception as e:
                    raise Exception(f"调用月之暗面API失败: {str(e)}")
            elif self.provider == "豆包":
                # 豆包 API
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.llm_api_key}"
                    }
                    data = {
                        "model": self.llm_model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": True
                    }
                    
                    with requests.post(f"{self.llm_api_url}/v1/chat/completions", 
                                     json=data, headers=headers, stream=True) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if self.is_stopped:
                                break
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    line = line[6:]
                                    if line.strip() == '[DONE]':
                                        break
                                    try:
                                        chunk_data = json.loads(line)
                                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                            delta = chunk_data['choices'][0].get('delta', {})
                                            if 'content' in delta and delta['content']:
                                                self.response_received.emit(delta['content'])
                                    except json.JSONDecodeError:
                                        continue
                except Exception as e:
                    raise Exception(f"调用豆包API失败: {str(e)}")
            elif self.provider == "自定义":
                # 自定义API - 假设使用OpenAI兼容格式
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.llm_api_key}"
                    }
                    data = {
                        "model": self.llm_model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": True
                    }
                    
                    with requests.post(f"{self.llm_api_url}/v1/chat/completions", 
                                     json=data, headers=headers, stream=True) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if self.is_stopped:
                                break
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    line = line[6:]
                                    if line.strip() == '[DONE]':
                                        break
                                    try:
                                        chunk_data = json.loads(line)
                                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                            delta = chunk_data['choices'][0].get('delta', {})
                                            if 'content' in delta and delta['content']:
                                                self.response_received.emit(delta['content'])
                                    except json.JSONDecodeError:
                                        continue
                except Exception as e:
                    raise Exception(f"调用自定义API失败: {str(e)}")
            else:
                # 未知API提供商，返回错误
                raise Exception(f"不支持的API提供商: {self.provider}")
            
            if not self.is_stopped:
                self.debug_log("生成过程正常完成")
                self.response_received.emit("\n\n回答完成。")
            else:
                self.debug_log("生成过程被用户中断")
                self.response_received.emit("\n\n生成已停止。")
                
        except Exception as e:
            self.debug_log(f"发生异常: {str(e)}")
            self.error_occurred.emit(f"错误: {str(e)}")
        finally:
            self.debug_log("查询线程执行完毕，发送completed信号")
            self.completed.emit()
    
    def stop(self):
        print("[DEBUG] 用户请求停止生成")
        self.is_stopped = True
    
    def handle_local_model(self, prompt):
        """处理本地模型推理"""
        # 解析模型名称和类型
        model_info = self.llm_model.split(" (")
        if len(model_info) > 1:
            model_name = model_info[0]
            model_type = model_info[1].rstrip(")")  # 移除右括号
        else:
            model_name = self.llm_model
            model_type = "HuggingFace"  # 默认为HuggingFace类型
        
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_name)
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            raise Exception(f"模型路径不存在: {model_path}")
        
        # 更新最后活动时间
        RAGQueryThread.last_activity_time = time.time()
        
        # 检查是否已经加载了相同的模型
        if (RAGQueryThread.loaded_local_model is not None and 
            RAGQueryThread.loaded_tokenizer is not None and 
            hasattr(RAGQueryThread, 'loaded_model_name') and 
            self.llm_model == RAGQueryThread.loaded_model_name):
            self.debug_log(f"使用已加载的模型: {self.llm_model}")
            model = RAGQueryThread.loaded_local_model
            tokenizer = RAGQueryThread.loaded_tokenizer
        else:
            # 如果已经加载了其他模型，先卸载它
            if (RAGQueryThread.loaded_local_model is not None and 
                hasattr(RAGQueryThread, 'loaded_model_name') and 
                self.llm_model != RAGQueryThread.loaded_model_name):
                self.debug_log(f"切换模型: 从 {RAGQueryThread.loaded_model_name} 到 {self.llm_model}")
                RAGQueryThread.unload_model()
                # 强制进行垃圾回收，释放内存
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.debug_log("已卸载之前的模型并清理内存")
            
            # 根据模型类型加载不同格式的模型
            self.debug_log(f"正在加载本地模型: {model_name}, 类型: {model_type}")
            
            try:
                if model_type == "ONNX":
                    # 加载ONNX模型
                    import onnxruntime as ort
                    from transformers import AutoTokenizer
                    
                    # 加载分词器
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                    # 检查是否存在model.onnx文件
                    onnx_model_path = os.path.join(model_path, "model.onnx")
                    if not os.path.exists(onnx_model_path):
                        raise Exception(f"ONNX模型文件不存在: {onnx_model_path}")
                    
                    # 加载ONNX模型
                    model = ort.InferenceSession(onnx_model_path)
                    self.debug_log(f"ONNX模型加载成功: {onnx_model_path}")
                    
                elif model_type == "TorchScript":
                    # 不再支持TorchScript模型
                    raise Exception("不再支持TorchScript格式，请将模型转换为ONNX格式")
                else:  # HuggingFace或其他类型
                    # 加载HuggingFace模型
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    # 加载分词器和模型
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    try:
                        # 尝试使用device_map自动分配
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                    except Exception as e:
                        if "requires `accelerate`" in str(e):
                            self.debug_log("未安装accelerate库，使用替代方法加载模型")
                            # 使用不依赖accelerate的方式加载模型
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                low_cpu_mem_usage=True
                            )
                        else:
                            # 其他错误，直接抛出
                            raise
                
                # 保存加载的模型以供后续使用
                RAGQueryThread.loaded_local_model = model
                RAGQueryThread.loaded_tokenizer = tokenizer
                RAGQueryThread.loaded_model_name = self.llm_model
                self.debug_log(f"模型加载成功")
                
            except Exception as e:
                self.debug_log(f"加载模型失败: {str(e)}")
                raise Exception(f"加载模型失败: {str(e)}")
        
        try:
            # 处理提示词
            self.debug_log("开始处理提示词...")
            
            # 根据模型类型进行不同的推理
            if model_type == "ONNX":
                # ONNX模型推理
                import numpy as np
                
                # 检查提示词长度
                self.debug_log("检查提示词长度...")
                encoded_input = tokenizer.encode(prompt)
                self.debug_log(f"提示词长度: {len(encoded_input)} tokens")
                
                # 如果提示词太长，进行截断
                max_input_length = 2048  # ONNX模型通常有输入长度限制
                if len(encoded_input) > max_input_length:
                    self.debug_log(f"提示词过长，将截断到{max_input_length}个tokens")
                    encoded_input = encoded_input[:max_input_length]
                    prompt = tokenizer.decode(encoded_input)
                
                inputs = tokenizer(prompt, return_tensors="np")
                
                # 获取模型输入和输出名称
                input_names = [input.name for input in model.get_inputs()]
                output_names = [output.name for output in model.get_outputs()]
                
                self.debug_log(f"模型输入名称: {input_names}")
                self.debug_log(f"模型输出名称: {output_names}")
                
                # 准备输入
                onnx_inputs = {}
                for name in input_names:
                    if name in inputs:
                        onnx_inputs[name] = inputs[name]
                
                # 运行推理
                self.debug_log("开始ONNX模型推理...")
                outputs = model.run(output_names, onnx_inputs)
                
                # 处理输出
                try:
                    if len(outputs) > 0:
                        # 获取输出张量
                        output_tensor = outputs[0]
                        
                        # 检查输出形状
                        self.debug_log(f"输出张量形状: {output_tensor.shape}")
                        
                        # 如果是token IDs，解码为文本
                        if len(output_tensor.shape) > 1 and output_tensor.shape[1] > 1:
                            # 可能是token IDs
                            if isinstance(output_tensor, np.ndarray):
                                # 转换为整数类型
                                output_ids = output_tensor.astype(np.int64)
                                # 解码
                                answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                            else:
                                answer = str(output_tensor)
                        else:
                            # 可能是嵌入向量或其他格式
                            answer = "模型输出不是标准的token IDs格式，无法直接解码为文本。请检查模型类型是否正确。"
                    else:
                        answer = "模型未返回任何输出"
                except Exception as e:
                    self.debug_log(f"处理ONNX模型输出时出错: {str(e)}")
                    answer = f"处理模型输出时出错: {str(e)}"
            
            elif model_type == "TorchScript":
                # 不再支持TorchScript模型
                raise Exception("不再支持TorchScript格式，请将模型转换为ONNX格式")
            else:  # HuggingFace或其他类型
                # HuggingFace模型推理
                import torch
                
                # 检查提示词长度
                self.debug_log("检查提示词长度...")
                encoded_input = tokenizer.encode(prompt)
                self.debug_log(f"提示词长度: {len(encoded_input)} tokens")
                
                inputs = tokenizer(prompt, return_tensors="pt")
                input_length = inputs["input_ids"].shape[1]
                self.debug_log(f"输入长度: {input_length} tokens")
                
                # 检查输入长度是否超过模型最大长度
                if input_length > 2048:
                    self.debug_log(f"输入长度({input_length})超过2048，将使用max_new_tokens参数")
                    max_new_tokens = min(1024, 4096 - input_length)  # 确保总长度不超过4096
                    self.debug_log(f"设置max_new_tokens={max_new_tokens}")
                    
                    with torch.no_grad():
                        try:
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_new_tokens=max_new_tokens,  # 使用max_new_tokens而不是max_length
                                temperature=0.7,
                                top_p=0.9,
                                do_sample=True
                            )
                        except Exception as e:
                            self.debug_log(f"生成失败，尝试截断输入: {str(e)}")
                            # 如果生成失败，尝试截断输入
                            max_input_length = 2048
                            self.debug_log(f"截断输入到{max_input_length}个tokens")
                            encoded_input = encoded_input[:max_input_length]
                            truncated_prompt = tokenizer.decode(encoded_input)
                            inputs = tokenizer(truncated_prompt, return_tensors="pt")
                            
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_length=max_input_length + 1024,
                                temperature=0.7,
                                top_p=0.9,
                                do_sample=True
                            )
                else:
                    # 输入长度在正常范围内
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_length=2048,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True
                        )
                
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            self.debug_log(f"回答长度: {len(answer)} 字符")
            
            # 发送回答
            self.response_received.emit(answer)
            
        except Exception as e:
            self.debug_log(f"模型推理失败: {str(e)}")
            import traceback
            self.debug_log(traceback.format_exc())
            raise Exception(f"模型推理失败: {str(e)}")
    
    @staticmethod
    def check_model_unload():
        """检查是否需要卸载模型"""
        # 注释掉自动卸载逻辑，保持模型在内存中
        # 只有在程序关闭或用户切换模型时才会卸载
        pass
        # 原来的自动卸载逻辑（已注释）
        # if RAGQueryThread.loaded_local_model is not None and time.time() - RAGQueryThread.last_activity_time > 300:
        #     # 如果最后一次活动时间超过5分钟，则卸载模型
        #     RAGQueryThread.unload_model()
    
    @staticmethod
    def unload_model():
        """卸载模型"""
        if RAGQueryThread.loaded_local_model is not None:
            print("[DEBUG] 自动卸载模型")
            
            # 获取模型名称用于日志
            model_name = "未知模型"
            if hasattr(RAGQueryThread, 'loaded_model_name'):
                model_name = RAGQueryThread.loaded_model_name
            
            # 释放模型资源
            try:
                # 清除引用
                RAGQueryThread.loaded_local_model = None
                RAGQueryThread.loaded_tokenizer = None
                
                # 强制进行垃圾回收
                import gc
                gc.collect()
                
                # 如果有PyTorch，尝试清空CUDA缓存
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, AttributeError):
                    # 如果没有PyTorch或者CUDA不可用，忽略错误
                    pass
                
                print(f"[DEBUG] 成功卸载模型: {model_name}")
            except Exception as e:
                print(f"[DEBUG] 卸载模型时出错: {str(e)}")

# 构建知识库页面
class KnowledgeBasePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            if obj == self.embedding_model_combo:
                self.scan_local_embedding_models()
            elif obj == self.kb_name_combo:
                self.refresh_kb_list()
            elif obj == self.kb_combo:
                self.refresh_kb_list()
        return super().eventFilter(obj, event)
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 文件浏览部分
        file_browse_layout = QHBoxLayout()
        self.browse_button = QPushButton("浏览文件")
        self.browse_button.clicked.connect(self.browse_files)
        file_browse_layout.addWidget(self.browse_button, 1)
        
        # 添加清空文件按钮
        self.clear_files_button = QPushButton("清空文件")
        self.clear_files_button.clicked.connect(self.clear_files)
        file_browse_layout.addWidget(self.clear_files_button, 1)
        
        layout.addLayout(file_browse_layout)
        
        # 文件列表
        self.file_list = QListWidget()
        layout.addWidget(QLabel("已选择的文件:"))
        layout.addWidget(self.file_list)
        
        # 进度显示
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        layout.addWidget(QLabel("构建进度:"))
        layout.addWidget(self.progress_text)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("词嵌入模型:"))
        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.setEditable(False)
        # 添加下拉菜单点击事件，自动扫描本地模型
        self.embedding_model_combo.installEventFilter(self)
        model_layout.addWidget(self.embedding_model_combo)
        layout.addLayout(model_layout)
        
        # 自动扫描本地模型
        self.scan_local_embedding_models()
        
        # 知识库名称
        kb_name_layout = QHBoxLayout()
        kb_name_layout.addWidget(QLabel("知识库名称:"))
        self.kb_name_combo = QComboBox()
        self.kb_name_combo.setEditable(True)
        # 添加下拉菜单点击事件，自动刷新知识库列表
        self.kb_name_combo.installEventFilter(self)
        kb_name_layout.addWidget(self.kb_name_combo)
        layout.addLayout(kb_name_layout)
        
        # 刷新知识库列表
        self.refresh_kb_list()
        
        # 构建按钮
        self.build_button = QPushButton("构建知识库")
        self.build_button.setMaximumWidth(120)
        self.build_button.clicked.connect(self.build_knowledge_base)
        layout.addWidget(self.build_button)
        
        self.setLayout(layout)
    
    def scan_local_embedding_models(self):
        """扫描本地embeddings目录下的词嵌入模型"""
        self.embedding_model_combo.clear()
        
        # 添加"默认"选项，使用知识库原有的嵌入模型
        self.embedding_model_combo.addItem("默认 (使用知识库原模型)")
        
        # 获取embeddings目录路径
        embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        
        # 检查目录是否存在
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir, exist_ok=True)
            self.progress_text.append("未找到embeddings目录，已创建空目录")
            return
        
        # 扫描目录下的所有子目录（每个子目录代表一个模型）
        models = []
        for item in os.listdir(embeddings_dir):
            item_path = os.path.join(embeddings_dir, item)
            if os.path.isdir(item_path):
                # 检查是否是ONNX模型（包含model.onnx文件）
                if os.path.exists(os.path.join(item_path, "model.onnx")):
                    models.append(f"{item} (ONNX)")
                # 检查是否是分离的ONNX模型（包含transformer_model.onnx和pooling_model.onnx文件）
                elif os.path.exists(os.path.join(item_path, "transformer_model.onnx")) and os.path.exists(os.path.join(item_path, "pooling_model.onnx")):
                    models.append(f"{item} (ONNX)")
                # 检查是否是HuggingFace模型（包含config.json文件）
                elif os.path.exists(os.path.join(item_path, "config.json")):
                    models.append(f"{item} (HuggingFace)")
                else:
                    models.append(item)  # 未知类型的模型目录
        
        if models:
            self.embedding_model_combo.addItems(models)
            self.progress_text.append(f"找到{len(models)}个本地词嵌入模型")
        else:
            self.progress_text.append("未找到本地词嵌入模型，请将模型放置在embeddings目录下")
    
    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择文件",
            "",
            "所有文件 (*.*);;PDF文件 (*.pdf);;Word文件 (*.docx *.doc);;Excel文件 (*.xlsx *.xls);;PowerPoint文件 (*.pptx *.ppt);;文本文件 (*.txt);;Markdown文件 (*.md);;HTML文件 (*.html *.htm);;JSON文件 (*.json)"
        )
        
        if files:
            # 追加到文件列表，不清空之前选择的文件
            for file_path in files:
                # 检查文件是否已经在列表中，避免重复添加
                existing_items = [self.file_list.item(i).text() for i in range(self.file_list.count())]
                if file_path not in existing_items:
                    self.file_list.addItem(file_path)
    
    def clear_files(self):
        """清空文件列表"""
        self.file_list.clear()
    
    def fetch_embedding_models(self):
        provider = self.embedding_provider_combo.currentText()
        api_url = self.embedding_api_url_edit.text().strip()
        api_key = self.embedding_api_key_edit.text().strip()
        
        if not api_url:
            QMessageBox.warning(self, "错误", "请输入API地址")
            return
        
        # 保存API密钥
        if api_key:
            api_keys = load_api_keys()
            api_keys[provider] = api_key
            save_api_keys(api_keys)
        
        # 获取模型列表
        self.progress_text.append(f"正在获取{provider}的模型列表...")
        self.fetch_models_thread = ModelFetchThread(api_url, api_key, provider)
        self.fetch_models_thread.models_fetched.connect(self.update_embedding_models)
        self.fetch_models_thread.error_occurred.connect(self.on_fetch_error)
        self.fetch_models_thread.start()
    
    def update_embedding_models(self, models, provider):
        self.embedding_model_combo.clear()
        self.embedding_model_combo.addItems(models)
        self.progress_text.append(f"成功获取{provider}的模型列表")
    
    def on_fetch_error(self, error_msg):
        self.progress_text.append(f"获取模型列表失败: {error_msg}")
        QMessageBox.warning(self, "错误", f"获取模型列表失败: {error_msg}")
    
    def refresh_kb_list(self):
        self.kb_name_combo.clear()
        knowledge_bases = get_knowledge_bases()
        if knowledge_bases:
            self.kb_name_combo.addItems(knowledge_bases)
        else:
            self.progress_text.append("未找到知识库")
    
    def build_knowledge_base(self):
        kb_name = self.kb_name_combo.currentText().strip()
        if not kb_name:
            QMessageBox.warning(self, "错误", "请输入知识库名称")
            return
        
        # 如果当前正在构建，则中断构建
        if hasattr(self, 'kb_builder_thread') and self.kb_builder_thread.isRunning():
            reply = QMessageBox.question(self, "确认中断", 
                                      "确定要中断当前知识库构建过程吗？",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.kb_builder_thread.terminate()
                self.kb_builder_thread.wait()
                self.progress_text.append("知识库构建已中断")
                self.build_button.setText("构建知识库")
                self.build_button.setEnabled(True)
            return
        
        # 检查知识库是否已存在
        kb_dir = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
        kb_mode = "overwrite"  # 默认为覆盖模式
        if os.path.exists(kb_dir):
            # 创建自定义对话框，提供三个选项：覆盖、追加、取消
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("知识库已存在")
            msg_box.setText(f"知识库 '{kb_name}' 已存在，请选择操作：")
            
            overwrite_button = msg_box.addButton("覆盖", QMessageBox.ActionRole)
            append_button = msg_box.addButton("追加", QMessageBox.ActionRole)
            cancel_button = msg_box.addButton("取消", QMessageBox.RejectRole)
            
            msg_box.exec_()
            
            clicked_button = msg_box.clickedButton()
            if clicked_button == overwrite_button:
                kb_mode = "overwrite"  # 覆盖模式
            elif clicked_button == append_button:
                kb_mode = "append"  # 追加模式
            else:  # 取消
                return
        
        # 检查文件列表
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            QMessageBox.warning(self, "错误", "请选择至少一个文件")
            return
        
        # 获取选择的本地词嵌入模型
        model = self.embedding_model_combo.currentText().strip()
        
        if not model:
            QMessageBox.warning(self, "错误", "请选择词嵌入模型")
            return
        
        # 清空进度显示
        self.progress_text.clear()
        
        # 开始构建知识库
        self.progress_text.append(f"开始构建知识库: {kb_name}")
        
        self.kb_builder_thread = KnowledgeBaseBuilderThread(
            kb_name, files, model, kb_mode
        )
        self.kb_builder_thread.progress_updated.connect(self.update_progress)
        self.kb_builder_thread.completed.connect(self.on_build_completed)
        self.kb_builder_thread.start()
        
        # 将构建按钮改为中断构建
        self.build_button.setText("中断构建")
    
    def update_progress(self, message):
        self.progress_text.append(message)
    
    def on_build_completed(self, success, message):
        # 恢复构建按钮文本
        self.build_button.setText("构建知识库")
        self.build_button.setEnabled(True)
        if success:
            self.progress_text.append(message)
            QMessageBox.information(self, "成功", message)
        else:
            self.progress_text.append(f"错误: {message}")
            QMessageBox.warning(self, "错误", message)

# RAG问答页面
class RAGQueryPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_saved_api_keys()
        self.load_saved_presets()
        self.query_thread = None
        self.model_fetch_thread = None
        
        # 扫描嵌入模型
        self.scan_local_embedding_models()
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            if obj == self.embedding_model_combo:
                self.scan_local_embedding_models()
            elif obj == self.kb_name_combo:
                self.refresh_kb_list()
            elif obj == self.kb_combo:
                self.refresh_kb_list()
            elif obj == self.llm_model_combo:
                self.fetch_llm_models()
        return super().eventFilter(obj, event)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 大模型API设置
        api_group_layout = QVBoxLayout()
        
        # API提供商选择
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("大模型API提供商:"))
        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(API_PROVIDERS)
        self.llm_provider_combo.currentTextChanged.connect(self.on_llm_provider_changed)
        provider_layout.addWidget(self.llm_provider_combo)
        api_group_layout.addLayout(provider_layout)
        
        # API地址
        api_url_layout = QHBoxLayout()
        api_url_layout.addWidget(QLabel("API地址:"))
        self.llm_api_url_edit = QLineEdit()
        api_url_layout.addWidget(self.llm_api_url_edit)
        api_group_layout.addLayout(api_url_layout)
        
        # API密钥
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API密钥:"))
        self.llm_api_key_edit = QLineEdit()
        self.llm_api_key_edit.setEchoMode(QLineEdit.Password)
        api_key_layout.addWidget(self.llm_api_key_edit)
        api_group_layout.addLayout(api_key_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("大模型:"))
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        # 添加下拉菜单点击事件，自动获取模型列表
        self.llm_model_combo.installEventFilter(self)
        model_layout.addWidget(self.llm_model_combo)
        
        # 添加刷新按钮
        self.refresh_models_button = QPushButton("刷新")
        self.refresh_models_button.clicked.connect(self.fetch_llm_models)
        model_layout.addWidget(self.refresh_models_button)
        
        api_group_layout.addLayout(model_layout)
        
        layout.addLayout(api_group_layout)
        
        # 知识库选择
        kb_layout = QHBoxLayout()
        kb_layout.addWidget(QLabel("知识库:"))
        self.kb_combo = QComboBox()
        # 添加下拉菜单点击事件，自动刷新知识库列表
        self.kb_combo.installEventFilter(self)
        kb_layout.addWidget(self.kb_combo, 1)  # 设置为1，减少宽度
        
        # 添加嵌入模型选择
        kb_layout.addWidget(QLabel("嵌入模型:"))
        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.installEventFilter(self)
        kb_layout.addWidget(self.embedding_model_combo, 1)  # 设置为1，减少宽度
        
        layout.addLayout(kb_layout)
        
        # 对话区域
        splitter = QSplitter(Qt.Vertical)
        
        # 对话返回框
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setContextMenuPolicy(Qt.CustomContextMenu)
        self.response_text.customContextMenuRequested.connect(self.show_response_context_menu)
        # 设置复制粘贴快捷键
        setup_text_edit_shortcuts(self.response_text, read_only=True)
        splitter.addWidget(self.response_text)
        
        # 提示词预设和输入区域
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        
        # 系统提示词
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("系统提示词:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        # 添加编辑完成信号，自动保存提示词
        self.preset_combo.editTextChanged.connect(self.auto_save_preset)
        preset_layout.addWidget(self.preset_combo)
        input_layout.addLayout(preset_layout)
        
        # 提示词输入框
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("在此输入您的问题...")
        self.query_edit.setContextMenuPolicy(Qt.CustomContextMenu)
        self.query_edit.customContextMenuRequested.connect(self.show_query_context_menu)
        self.query_edit.installEventFilter(self)  # 安装事件过滤器用于捕获Ctrl+Enter
        # 设置复制粘贴快捷键
        setup_text_edit_shortcuts(self.query_edit, read_only=False)
        input_layout.addWidget(self.query_edit)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        self.send_button = QPushButton("发送 ▶")
        self.send_button.clicked.connect(self.send_query)
        button_layout.addWidget(self.send_button)
        
        self.new_chat_button = QPushButton("新对话")
        self.new_chat_button.clicked.connect(self.new_chat)
        button_layout.addWidget(self.new_chat_button)
        input_layout.addLayout(button_layout)
        
        splitter.addWidget(input_widget)
        
        # 设置初始大小比例
        splitter.setSizes([400, 200])
        layout.addWidget(splitter)
        
        self.setLayout(layout)
        
        # 初始化API地址
        self.on_llm_provider_changed(self.llm_provider_combo.currentText())
        
        # 刷新知识库列表
        self.refresh_kb_list()
    
    def eventFilter(self, obj, event):
        if obj is self.query_edit and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
                self.send_query()
                return True
        elif event.type() == QEvent.MouseButtonPress:
            if obj == self.llm_model_combo:
                self.fetch_llm_models()
            elif obj == self.kb_combo:
                self.refresh_kb_list()
        return super().eventFilter(obj, event)
    
    def load_saved_api_keys(self):
        api_keys = load_api_keys()
        provider = self.llm_provider_combo.currentText()
        if provider in api_keys:
            self.llm_api_key_edit.setText(api_keys[provider])
    
    def load_saved_presets(self):
        presets = load_presets()
        self.preset_combo.clear()
        self.preset_combo.addItems(presets)
    
    def on_llm_provider_changed(self, provider):
        # 更新API地址
        if provider in DEFAULT_API_URLS:
            self.llm_api_url_edit.setText(DEFAULT_API_URLS[provider])
        else:
            self.llm_api_url_edit.clear()
        
        # 根据提供商类型启用或禁用API相关控件
        is_local = (provider == "本地")
        self.llm_api_url_edit.setEnabled(not is_local)
        self.llm_api_key_edit.setEnabled(not is_local)
        
        # 加载保存的API密钥
        api_keys = load_api_keys()
        if provider in api_keys:
            self.llm_api_key_edit.setText(api_keys[provider])
        else:
            self.llm_api_key_edit.clear()
        
        # 清空模型列表
        self.llm_model_combo.clear()
        
        # 如果是本地模型，立即加载本地模型列表
        if provider == "本地":
            local_models = get_local_models()
            if local_models:
                self.llm_model_combo.addItems(local_models)
                self.response_text.append(f"找到{len(local_models)}个本地模型")
            else:
                self.response_text.append("未找到本地模型，请将模型放置在models目录下")
                self.response_text.append("或使用\"转换模型\"功能将HuggingFace模型转换为ONNX或TorchScript格式")
    
    def refresh_kb_list(self):
        self.kb_combo.clear()
        # 添加"无"选项，允许用户不使用知识库直接查询
        self.kb_combo.addItem("无")
        knowledge_bases = get_knowledge_bases()
        if knowledge_bases:
            self.kb_combo.addItems(knowledge_bases)
        else:
            self.response_text.append("未找到知识库，请先在'构建知识库'页面创建知识库。")
    
    def fetch_llm_models(self):
        provider = self.llm_provider_combo.currentText()
        api_url = self.llm_api_url_edit.text().strip()
        api_key = self.llm_api_key_edit.text().strip()
        
        # 如果是本地模型，直接获取本地模型列表
        if provider == "本地":
            local_models = get_local_models()
            self.llm_model_combo.clear()
            if local_models:
                self.llm_model_combo.addItems(local_models)
                self.response_text.append(f"找到{len(local_models)}个本地模型")
            else:
                self.response_text.append("未找到本地模型，请将模型放置在models目录下")
                self.response_text.append("或使用\"转换模型\"功能将HuggingFace模型转换为ONNX或TorchScript格式")
            return
        
        # 对于非本地模型，检查API地址
        if not api_url:
            self.response_text.append("错误: 请输入API地址")
            return
        
        # 保存API密钥
        if api_key:
            api_keys = load_api_keys()
            api_keys[provider] = api_key
            save_api_keys(api_keys)
        
        # 获取模型列表
        self.response_text.append(f"正在获取{provider}的模型列表...")
        self.model_fetch_thread = ModelFetchThread(api_url, api_key, provider)
        self.model_fetch_thread.models_fetched.connect(self.update_llm_models)
        self.model_fetch_thread.error_occurred.connect(self.on_fetch_error)
        self.model_fetch_thread.start()
    
    def update_llm_models(self, models, provider):
        self.llm_model_combo.clear()
        self.llm_model_combo.addItems(models)
        self.response_text.append(f"成功获取{provider}的模型列表")
    
    def on_fetch_error(self, error_msg):
        self.response_text.append(f"获取模型列表失败: {error_msg}")
        QMessageBox.warning(self, "错误", f"获取模型列表失败: {error_msg}")
    
    def show_response_context_menu(self, position):
        menu = QMenu()
        copy_action = QAction("复制", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(lambda: self.response_text.copy())
        select_all_action = QAction("全选", self)
        select_all_action.triggered.connect(self.response_text.selectAll)
        menu.addAction(copy_action)
        menu.addAction(select_all_action)
        menu.exec_(self.response_text.mapToGlobal(position))
    
    def show_query_context_menu(self, position):
        menu = QMenu()
        copy_action = QAction("复制", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(lambda: self.query_edit.copy())
        paste_action = QAction("粘贴", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(lambda: self.query_edit.paste())
        cut_action = QAction("剪切", self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(lambda: self.query_edit.cut())
        select_all_action = QAction("全选", self)
        select_all_action.triggered.connect(self.query_edit.selectAll)
        menu.addAction(copy_action)
        menu.addAction(paste_action)
        menu.addAction(cut_action)
        menu.addAction(select_all_action)
        menu.exec_(self.query_edit.mapToGlobal(position))
    
    def auto_save_preset(self):
        preset = self.preset_combo.currentText().strip()
        if not preset:
            return
        
        presets = load_presets()
        if preset not in presets:
            presets.append(preset)
            save_presets(presets)
            # 不需要清空和重新添加，因为这会触发editTextChanged信号
            # 只需要确保当前编辑的文本已经在列表中
            index = self.preset_combo.findText(preset)
            if index < 0:  # 如果不在下拉列表中，添加它
                self.preset_combo.addItem(preset)
    
    def send_query(self):
        # 如果已经有查询在进行中，则停止它
        if self.query_thread and self.query_thread.isRunning():
            self.query_thread.stop()
            self.send_button.setText("发送 ▶")
            return
        
        # 获取查询内容
        query = self.query_edit.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "错误", "请输入问题")
            return
        
        # 获取知识库
        if self.kb_combo.count() == 0:
            QMessageBox.warning(self, "错误", "没有可用的知识库，请先创建知识库")
            return
        
        kb_name = self.kb_combo.currentText()
        # 如果选择了"无"，则不使用知识库
        if kb_name == "无":
            kb_name = None
        
        # 获取API信息
        provider = self.llm_provider_combo.currentText()
        api_url = self.llm_api_url_edit.text().strip()
        api_key = self.llm_api_key_edit.text().strip()
        model = self.llm_model_combo.currentText().strip()
        
        # 检查API地址（本地模型除外）
        if provider != "本地" and not api_url:
            self.response_text.append("错误: 请输入API地址")
            return
        
        if not model:
            QMessageBox.warning(self, "错误", "请选择或输入模型名称")
            return
        
        # 保存API密钥
        if api_key:
            api_keys = load_api_keys()
            api_keys[provider] = api_key
            save_api_keys(api_keys)
        
        # 获取提示词预设
        preset = self.preset_combo.currentText()
        
        # 获取自定义嵌入模型
        custom_embedding_model = self.embedding_model_combo.currentText().strip()
        
        # 显示用户问题
        self.response_text.append(f"\n用户: {query}\n")
        self.response_text.append("AI: ")
        
        # 更改按钮状态
        self.send_button.setText("停止 ■")
        
        # 开始查询
        self.query_thread = RAGQueryThread(
            query, preset, kb_name, api_url, api_key, model, provider, custom_embedding_model
        )
        self.query_thread.response_received.connect(self.update_response)
        self.query_thread.error_occurred.connect(self.on_query_error)
        self.query_thread.completed.connect(self.on_query_completed)
        self.query_thread.start()
        
        # 清空问题输入框
        self.query_edit.clear()
    
    def update_response(self, text):
        self.response_text.insertPlainText(text)
        # 滚动到底部
        cursor = self.response_text.textCursor()
        cursor.movePosition(cursor.End)
        self.response_text.setTextCursor(cursor)
    
    def on_query_error(self, error_msg):
        self.response_text.append(f"\n错误: {error_msg}")
        self.send_button.setText("发送 ▶")
    
    def on_query_completed(self):
        self.send_button.setText("发送 ▶")
    
    def new_chat(self):
        self.response_text.clear()
        self.query_edit.clear()

    def scan_local_embedding_models(self):
        """扫描本地embeddings目录下的词嵌入模型"""
        self.embedding_model_combo.clear()
        
        # 添加"默认"选项，使用知识库原有的嵌入模型
        self.embedding_model_combo.addItem("默认 (使用知识库原模型)")
        
        # 获取embeddings目录路径
        embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        
        # 检查目录是否存在
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir, exist_ok=True)
            self.response_text.append("未找到embeddings目录，已创建空目录")
            return
        
        # 扫描目录下的所有子目录（每个子目录代表一个模型）
        models = []
        for item in os.listdir(embeddings_dir):
            item_path = os.path.join(embeddings_dir, item)
            if os.path.isdir(item_path):
                # 检查是否是ONNX模型（包含model.onnx文件）
                if os.path.exists(os.path.join(item_path, "model.onnx")):
                    models.append(f"{item} (ONNX)")
                # 检查是否是分离的ONNX模型（包含transformer_model.onnx和pooling_model.onnx文件）
                elif os.path.exists(os.path.join(item_path, "transformer_model.onnx")) and os.path.exists(os.path.join(item_path, "pooling_model.onnx")):
                    models.append(f"{item} (ONNX)")
                # 检查是否是HuggingFace模型（包含config.json文件）
                elif os.path.exists(os.path.join(item_path, "config.json")):
                    models.append(f"{item} (HuggingFace)")
                else:
                    models.append(item)  # 未知类型的模型目录
        
        if models:
            self.embedding_model_combo.addItems(models)
            self.response_text.append(f"找到{len(models)}个本地词嵌入模型")
        else:
            self.response_text.append("未找到本地词嵌入模型，请将模型放置在embeddings目录下")

# 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("LocalRAG - 本地知识库问答系统")
        self.setGeometry(100, 100, 1000, 800)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        
        # 创建全局设置区域
        global_settings_layout = QHBoxLayout()
        
        # 添加分词器设置勾选框（全局设置）
        self.android_tokenizer_checkbox = QCheckBox("使用Android分词器")
        
        # 加载当前分词器设置
        tokenizer_config = load_tokenizer_config()
        use_android_tokenizer = tokenizer_config.get("use_android_tokenizer", False)
        
        self.android_tokenizer_checkbox.setChecked(use_android_tokenizer)
        
        # 连接信号
        self.android_tokenizer_checkbox.stateChanged.connect(self.on_tokenizer_changed)
        
        # 添加说明标签
        tokenizer_label = QLabel("分词器设置:")
        global_settings_layout.addWidget(tokenizer_label)
        global_settings_layout.addWidget(self.android_tokenizer_checkbox)
        global_settings_layout.addStretch(1)  # 添加弹性空间
        
        # 将全局设置添加到主布局
        main_layout.addLayout(global_settings_layout)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 添加RAG问答页面（放在第一位）
        self.rag_page = RAGQueryPage()
        self.tab_widget.addTab(self.rag_page, "RAG问答")
        
        # 添加构建知识库页面
        self.kb_page = KnowledgeBasePage()
        self.tab_widget.addTab(self.kb_page, "构建知识库")
        
        # 添加知识库笔记页面
        from knowledge_note_page import KnowledgeNotePage
        self.note_page = KnowledgeNotePage()
        self.tab_widget.addTab(self.note_page, "知识库笔记")
        
        # 添加模型转换页面
        from model_conversion_hub import ModelConversionHub
        self.model_conversion_hub = ModelConversionHub()
        self.tab_widget.addTab(self.model_conversion_hub, "模型转换")
        
        # 将标签页添加到主布局
        main_layout.addWidget(self.tab_widget)
        
        # 创建中央部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def on_tokenizer_changed(self, state):
        """处理分词器选择变化"""
        use_android = (state == Qt.Checked)
        
        # 更新配置
        config = load_tokenizer_config()
        config["use_android_tokenizer"] = use_android
        save_tokenizer_config(config)
        
        # 显示提示信息
        message = ""
        if use_android:
            message = "已切换到Android分词器，将使用与Android端一致的分词方式生成向量。\n注意：使用此设置生成的知识库向量将与PC端默认方式不同，但与Android端一致。"
            print("已切换到Android分词器，将使用与Android端一致的分词方式生成向量。")
            print("注意：使用此设置生成的知识库向量将与PC端默认方式不同，但与Android端一致。")
        else:
            message = "已切换到HuggingFace分词器（默认），将使用标准分词方式生成向量。"
            print("已切换到HuggingFace分词器（默认），将使用标准分词方式生成向量。")
        
        # 通知各个页面分词器设置已更改
        # 在知识库页面显示提示
        if hasattr(self, 'kb_page') and hasattr(self.kb_page, 'progress_text'):
            self.kb_page.progress_text.append(message)
        
        # 在RAG查询页面显示提示
        if hasattr(self, 'rag_page') and hasattr(self.rag_page, 'response_text'):
            self.rag_page.response_text.append(message)
        
        # 显示对话框提示用户
        QMessageBox.information(self, "分词器设置已更改", 
                               f"分词器设置已更改为: {'Android分词器' if use_android else 'HuggingFace分词器'}\n\n{message}")
    
    def notify_pages(self, message):
        # 在知识库页面显示提示
        if hasattr(self, 'kb_page') and hasattr(self.kb_page, 'progress_text'):
            self.kb_page.progress_text.append(message)
        
        # 在RAG查询页面显示提示
        if hasattr(self, 'rag_page') and hasattr(self.rag_page, 'response_text'):
            self.rag_page.response_text.append(message)

# 主函数
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# 设置文本框的复制粘贴快捷键
def setup_text_edit_shortcuts(text_edit, read_only=True):
    """为文本框设置复制粘贴快捷键"""
    if read_only:
        # 只读文本框支持Ctrl+C复制
        copy_action = QAction("复制", text_edit)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(lambda: text_edit.copy())
        text_edit.addAction(copy_action)
        
        # 添加全选快捷键
        select_all_action = QAction("全选", text_edit)
        select_all_action.setShortcut(QKeySequence("Ctrl+A"))  # 修复全选快捷键
        select_all_action.triggered.connect(text_edit.selectAll)
        text_edit.addAction(select_all_action)
    else:
        # 可编辑文本框支持Ctrl+X剪切、Ctrl+C复制、Ctrl+V粘贴
        cut_action = QAction("剪切", text_edit)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(lambda: text_edit.cut())
        text_edit.addAction(cut_action)
        
        copy_action = QAction("复制", text_edit)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(lambda: text_edit.copy())
        text_edit.addAction(copy_action)
        
        paste_action = QAction("粘贴", text_edit)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(lambda: text_edit.paste())
        text_edit.addAction(paste_action)
        
        # 添加全选快捷键
        select_all_action = QAction("全选", text_edit)
        select_all_action.setShortcut(QKeySequence("Ctrl+A"))  # 修复全选快捷键
        select_all_action.triggered.connect(text_edit.selectAll)
        text_edit.addAction(select_all_action)

if __name__ == "__main__":
    main()