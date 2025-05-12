"""
Android端分词算法的Python实现

该模块实现了与Android端相同的分词算法，用于对比测试
"""
import re
import json
import numpy as np
import os

class BertTokenizer:
    def __init__(self, vocab_file=None):
        # 特殊token ID - 与Android端保持一致
        self.cls_token_id = 0  # <s>
        self.sep_token_id = 2  # </s>
        self.unk_token_id = 3  # <unk>
        self.pad_token_id = 1  # <pad>
        self.mask_token_id = 250001  # <mask>
        
        # 特殊token字符串
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        
        # 词汇表
        self.vocab = {}
        self.id_to_token = {}
        
        # 配置
        self.add_prefix_space = True
        self.do_lower_case = False
        self.use_consistent_tokenization = True
        self.debug_mode = False
        
        # 如果提供了词汇表文件，加载它
        if vocab_file:
            self.load_vocab(vocab_file)
    
    def load_vocab(self, vocab_file):
        """加载词汇表文件"""
        print(f"加载词汇表: {vocab_file}")
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 尝试解析为JSON
            try:
                data = json.loads(content)
                
                # 打印数据结构，帮助调试
                print(f"词汇表JSON结构的键: {list(data.keys())}")
                
                # 处理不同格式的词汇表
                if "model" in data and isinstance(data["model"], dict):
                    if "vocab" in data["model"]:
                        vocab_data = data["model"]["vocab"]
                        if isinstance(vocab_data, dict):
                            # 字典格式的词汇表
                            self.vocab = {k: int(v) for k, v in vocab_data.items()}
                            print(f"从字典格式加载词汇表，大小: {len(self.vocab)}")
                        elif isinstance(vocab_data, list):
                            # 列表格式的词汇表 [token, score]
                            self.vocab = {item[0]: i for i, item in enumerate(vocab_data)}
                            print(f"从列表格式加载词汇表，大小: {len(self.vocab)}")
                            # 打印前几个词汇
                            print(f"前5个词汇: {list(self.vocab.items())[:5]}")
                        else:
                            print(f"警告: model.vocab格式不支持，类型: {type(vocab_data)}")
                elif "vocab" in data and isinstance(data["vocab"], dict):
                    self.vocab = {k: int(v) for k, v in data["vocab"].items()}
                else:
                    # 尝试其他可能的结构
                    for key in data.keys():
                        if isinstance(data[key], dict):
                            print(f"尝试使用键 '{key}' 中的数据")
                            self.vocab = {k: int(v) for k, v in data[key].items() if isinstance(v, (int, str))}
                            if self.vocab:
                                break
            except json.JSONDecodeError:
                print("JSON解析失败，尝试按行解析")
                # 尝试按行解析
                lines = content.strip().split('\n')
                for i, line in enumerate(lines):
                    token = line.strip()
                    if token:
                        self.vocab[token] = i
            
            # 如果词汇表为空，尝试从其他文件加载
            if not self.vocab:
                # 尝试从tokenizer_config.json加载
                config_file = os.path.join(os.path.dirname(vocab_file), "tokenizer_config.json")
                if os.path.exists(config_file):
                    print(f"尝试从配置文件加载: {config_file}")
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # 检查是否有词汇表信息
                    if "vocab_file" in config:
                        vocab_file_path = os.path.join(os.path.dirname(vocab_file), config["vocab_file"])
                        if os.path.exists(vocab_file_path):
                            print(f"尝试从词汇表文件加载: {vocab_file_path}")
                            with open(vocab_file_path, 'r', encoding='utf-8') as f:
                                self.vocab = json.load(f)
            
            # 构建反向映射
            if self.vocab:
                self.id_to_token = {v: k for k, v in self.vocab.items()}
                print(f"词汇表加载完成，大小: {len(self.vocab)}")
                
                # 打印部分词汇表内容
                sample_keys = list(self.vocab.keys())[:10]
                sample_tokens = [(k, self.vocab[k]) for k in sample_keys]
                print(f"词汇表样本: {sample_tokens}")
                
                # 确保特殊token的ID是正确的
                if "<s>" in self.vocab:
                    self.cls_token_id = self.vocab["<s>"]
                    print(f"设置CLS token ID: {self.cls_token_id}")
                if "</s>" in self.vocab:
                    self.sep_token_id = self.vocab["</s>"]
                    print(f"设置SEP token ID: {self.sep_token_id}")
                if "<unk>" in self.vocab:
                    self.unk_token_id = self.vocab["<unk>"]
                    print(f"设置UNK token ID: {self.unk_token_id}")
                if "<pad>" in self.vocab:
                    self.pad_token_id = self.vocab["<pad>"]
                    print(f"设置PAD token ID: {self.pad_token_id}")
                if "<mask>" in self.vocab:
                    self.mask_token_id = self.vocab["<mask>"]
                    print(f"设置MASK token ID: {self.mask_token_id}")
                
                return True
            else:
                print("警告: 词汇表为空")
                return False
                
        except Exception as e:
            print(f"加载词汇表失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_cjk_char(self, c):
        """判断字符是否为CJK字符"""
        code = ord(c)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F
    
    def longest_match_tokenize(self, text):
        """使用最长匹配算法进行分词"""
        tokens = []
        
        # 处理输入文本
        processed_text = text
        if self.add_prefix_space and not text.startswith(" "):
            processed_text = " " + text
            if self.debug_mode:
                print(f"添加空格前缀: {processed_text}")
        
        position = 0
        
        while position < len(processed_text):
            # 尝试找到最长匹配的token
            longest_token = None
            longest_length = 0
            max_look_ahead = min(15, len(processed_text) - position)  # 最大前向查找长度
            
            # 对于当前位置，尝试不同长度的子字符串
            for end_pos in range(position + 1, min(position + max_look_ahead + 1, len(processed_text) + 1)):
                substring = processed_text[position:end_pos]
                
                # 检查原始子字符串
                if substring in self.vocab and len(substring) > longest_length:
                    longest_token = substring
                    longest_length = len(substring)
                
                # 检查添加▁前缀的子字符串（对于非空格开头的token）
                if not substring.startswith(" ") and not substring.startswith("▁"):
                    with_prefix = "▁" + substring
                    if with_prefix in self.vocab and len(with_prefix) > longest_length:
                        longest_token = with_prefix
                        longest_length = len(with_prefix)
            
            # 如果找到匹配的token
            if longest_token is not None:
                tokens.append(longest_token)
                position += longest_length
                if longest_token.startswith("▁") and len(longest_token) > 1:
                    position -= 1  # 调整位置，因为▁前缀不在原始文本中
                if self.debug_mode:
                    print(f"找到token: '{longest_token}', 长度: {longest_length}")
            else:
                # 如果当前字符是CJK字符，单独处理
                current_char = processed_text[position]
                if self.is_cjk_char(current_char):
                    cjk_token = current_char
                    
                    # 尝试直接查找
                    if cjk_token in self.vocab:
                        tokens.append(cjk_token)
                        if self.debug_mode:
                            print(f"找到CJK token: '{cjk_token}'")
                    else:
                        # 尝试添加▁前缀
                        with_prefix = "▁" + cjk_token
                        if with_prefix in self.vocab:
                            tokens.append(with_prefix)
                            if self.debug_mode:
                                print(f"找到带前缀的CJK token: '{with_prefix}'")
                        else:
                            # 未知token
                            if self.debug_mode:
                                print(f"未知CJK字符: '{cjk_token}', 使用UNK")
                            tokens.append(self.unk_token)
                elif current_char == ' ':
                    # 空格处理
                    space_token = "▁"
                    if space_token in self.vocab:
                        tokens.append(space_token)
                        if self.debug_mode:
                            print(f"找到空格token: '{space_token}'")
                else:
                    # 其他未知字符
                    char_token = current_char
                    if self.debug_mode:
                        print(f"未知字符: '{char_token}', 使用UNK")
                    tokens.append(self.unk_token)
                position += 1
        
        return tokens
    
    def tokenize(self, text):
        """对文本进行分词，返回token ID和attention mask"""
        token_ids = []
        
        try:
            # 详细记录输入文本
            if self.debug_mode:
                print(f"开始分词处理，输入文本: '{text}', 长度: {len(text)}")
            
            # 添加起始标记
            token_ids.append(self.cls_token_id)
            if self.debug_mode:
                print(f"添加起始标记: {self.cls_token_id}")
            
            # 处理文本
            if text:
                # 如果需要，转换为小写
                if self.do_lower_case:
                    text = text.lower()
                    if self.debug_mode:
                        print(f"转换为小写: {text}")
                
                # 使用一致性分词策略
                if self.use_consistent_tokenization:
                    if self.debug_mode:
                        print("使用一致性分词策略")
                    
                    # 使用优化的最长匹配算法
                    tokens = self.longest_match_tokenize(text)
                    
                    # 打印分词结果
                    if self.debug_mode:
                        token_debug = " ".join([f"'{t}'" for t in tokens])
                        print(f"分词结果 ({len(tokens)} tokens): {token_debug}")
                    
                    # 将token转换为ID
                    for token in tokens:
                        if token in self.vocab:
                            token_id = self.vocab[token]
                            token_ids.append(token_id)
                            if self.debug_mode:
                                print(f"Token: '{token}' -> ID: {token_id}")
                        else:
                            # 对于未知token，尝试查找前缀
                            if token.startswith("▁") and len(token) > 1:
                                without_prefix = token[1:]
                                if without_prefix in self.vocab:
                                    token_id = self.vocab[without_prefix]
                                    token_ids.append(token_id)
                                    if self.debug_mode:
                                        print(f"Token(无前缀): '{without_prefix}' -> ID: {token_id}")
                                    continue
                            
                            # 如果是中文字符，尝试添加前缀再查找
                            if len(token) == 1 and self.is_cjk_char(token[0]):
                                with_prefix = "▁" + token
                                if with_prefix in self.vocab:
                                    token_id = self.vocab[with_prefix]
                                    token_ids.append(token_id)
                                    if self.debug_mode:
                                        print(f"CJK Token(带前缀): '{with_prefix}' -> ID: {token_id}")
                                    continue
                            
                            # 未知token
                            if self.debug_mode:
                                print(f"未知token: '{token}', 使用UNK ID: {self.unk_token_id}")
                            token_ids.append(self.unk_token_id)
            
            # 添加结束标记
            token_ids.append(self.sep_token_id)
            if self.debug_mode:
                print(f"添加结束标记: {self.sep_token_id}")
            
            # 创建attention_mask
            attention_mask = [1] * len(token_ids)
            
            return [token_ids, attention_mask]
        
        except Exception as e:
            print(f"分词处理失败: {str(e)}")
            # 返回最基本的序列：只包含[CLS]和[SEP]
            return [[self.cls_token_id, self.sep_token_id], [1, 1]]
    
    def __call__(self, text):
        """便捷调用方法"""
        result = self.tokenize(text)
        return {
            'input_ids': np.array([result[0]], dtype=np.int64),
            'attention_mask': np.array([result[1]], dtype=np.int64)
        }
    
    def convert_ids_to_tokens(self, ids):
        """将ID列表转换为token列表"""
        return [self.id_to_token.get(id, self.unk_token) for id in ids]
