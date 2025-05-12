import os
import json
import pickle
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QTextEdit, QPushButton, QComboBox, QMessageBox, QMenu, QAction)
from PyQt5.QtCore import Qt, QEvent

# 导入知识库目录常量
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_bases")
# 导入SQLite向量存储
from sqlite_vectorstore import SQLiteVectorStore
# 从localrag_app导入SentenceTransformerEmbeddings
from localrag_app import SentenceTransformerEmbeddings

def get_knowledge_bases():
    """获取所有知识库列表"""
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
        return []
    
    return [d for d in os.listdir(KNOWLEDGE_BASE_DIR) 
            if os.path.isdir(os.path.join(KNOWLEDGE_BASE_DIR, d))]

class KnowledgeNotePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            if obj == self.kb_combo:
                self.refresh_kb_list()
        return super().eventFilter(obj, event)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 标题文本框
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("标题:"))
        self.title_edit = QLineEdit()
        self.title_edit.setContextMenuPolicy(Qt.CustomContextMenu)
        self.title_edit.customContextMenuRequested.connect(self.show_title_context_menu)
        title_layout.addWidget(self.title_edit)
        layout.addLayout(title_layout)
        
        # 内容文本框
        layout.addWidget(QLabel("内容:"))
        self.content_edit = QTextEdit()
        self.content_edit.setContextMenuPolicy(Qt.CustomContextMenu)
        self.content_edit.customContextMenuRequested.connect(self.show_content_context_menu)
        layout.addWidget(self.content_edit)
        
        # 添加进度文本框
        layout.addWidget(QLabel("添加进度:"))
        self.progress_edit = QTextEdit()
        self.progress_edit.setReadOnly(True)
        self.progress_edit.setMaximumHeight(100)
        layout.addWidget(self.progress_edit)
        
        # 知识库选择
        kb_layout = QHBoxLayout()
        kb_layout.addWidget(QLabel("知识库:"))
        self.kb_combo = QComboBox()
        # 添加下拉菜单点击事件，自动刷新知识库列表
        self.kb_combo.installEventFilter(self)
        kb_layout.addWidget(self.kb_combo)
        layout.addLayout(kb_layout)
        
        # 添加到知识库按钮
        self.add_button = QPushButton("添加到知识库")
        self.add_button.clicked.connect(self.add_to_knowledge_base)
        layout.addWidget(self.add_button)
        
        self.setLayout(layout)
        
        # 初始化知识库列表
        self.refresh_kb_list()
    
    def show_title_context_menu(self, position):
        menu = QMenu()
        copy_action = QAction("复制", self)
        copy_action.triggered.connect(lambda: self.title_edit.copy())
        paste_action = QAction("粘贴", self)
        paste_action.triggered.connect(lambda: self.title_edit.paste())
        cut_action = QAction("剪切", self)
        cut_action.triggered.connect(lambda: self.title_edit.cut())
        select_all_action = QAction("全选", self)
        select_all_action.triggered.connect(self.title_edit.selectAll)
        menu.addAction(copy_action)
        menu.addAction(paste_action)
        menu.addAction(cut_action)
        menu.addAction(select_all_action)
        menu.exec_(self.title_edit.mapToGlobal(position))
    
    def show_content_context_menu(self, position):
        menu = QMenu()
        copy_action = QAction("复制", self)
        copy_action.triggered.connect(lambda: self.content_edit.copy())
        paste_action = QAction("粘贴", self)
        paste_action.triggered.connect(lambda: self.content_edit.paste())
        cut_action = QAction("剪切", self)
        cut_action.triggered.connect(lambda: self.content_edit.cut())
        select_all_action = QAction("全选", self)
        select_all_action.triggered.connect(self.content_edit.selectAll)
        menu.addAction(copy_action)
        menu.addAction(paste_action)
        menu.addAction(cut_action)
        menu.addAction(select_all_action)
        menu.exec_(self.content_edit.mapToGlobal(position))
    
    def refresh_kb_list(self):
        """刷新知识库列表"""
        self.kb_combo.clear()
        knowledge_bases = get_knowledge_bases()
        if knowledge_bases:
            self.kb_combo.addItems(knowledge_bases)
        else:
            QMessageBox.warning(self, "错误", "未找到知识库，请先在'构建知识库'页面创建知识库")
    
    def add_to_knowledge_base(self):
        """将笔记添加到知识库"""
        # 清空进度显示
        self.progress_edit.clear()
        self.progress_edit.append("开始添加笔记到知识库...")
        
        # 获取标题和内容
        title = self.title_edit.text().strip()
        content = self.content_edit.toPlainText().strip()
        
        # 验证输入
        if not title:
            self.progress_edit.append("错误: 请输入标题")
            QMessageBox.warning(self, "错误", "请输入标题")
            return
        
        if not content:
            self.progress_edit.append("错误: 请输入内容")
            QMessageBox.warning(self, "错误", "请输入内容")
            return
        
        # 获取选择的知识库
        if self.kb_combo.count() == 0:
            self.progress_edit.append("错误: 没有可用的知识库，请先创建知识库")
            QMessageBox.warning(self, "错误", "没有可用的知识库，请先创建知识库")
            return
        
        kb_name = self.kb_combo.currentText()
        if not kb_name:
            self.progress_edit.append("错误: 请选择知识库")
            QMessageBox.warning(self, "错误", "请选择知识库")
            return
        
        try:
            # 构建知识库路径
            self.progress_edit.append(f"正在处理知识库: {kb_name}")
            kb_dir = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
            db_path = os.path.join(kb_dir, "vectorstore.db")
            
            # 检查知识库是否存在
            if not os.path.exists(db_path):
                error_msg = f"知识库 {kb_name} 不存在或未正确构建"
                self.progress_edit.append(f"错误: {error_msg}")
                QMessageBox.warning(self, "错误", error_msg)
                return
            
            # 加载嵌入模型
            self.progress_edit.append("正在加载嵌入模型...")
            
            # 从元数据中获取嵌入模型名称
            metadata_path = os.path.join(kb_dir, "metadata.json")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            embedding_model_name = metadata.get("embedding_model", "all-MiniLM-L6-v2")
            embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
            model_path = os.path.join(embeddings_dir, embedding_model_name)
            
            # 加载嵌入模型
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_path)
            embeddings = SentenceTransformerEmbeddings(model)
            
            # 加载向量存储
            self.progress_edit.append("正在加载知识库向量存储...")
            vectorstore = SQLiteVectorStore(
                embedding_function=embeddings,
                db_path=db_path,
                collection_name=kb_name
            )
            
            # 构建文档内容（标题-内容配对）
            self.progress_edit.append("正在准备文档内容...")
            document_content = f"{title}\n\n{content}"
            
            # 使用知识库的词嵌入模型添加文档
            self.progress_edit.append("正在使用词嵌入模型处理文档...")
            vectorstore.add_texts([document_content])
            
            self.progress_edit.append(f"成功: 笔记已成功添加到知识库 {kb_name}")
            QMessageBox.information(self, "成功", f"笔记已成功添加到知识库 {kb_name}")
            
            # 清空输入框
            self.title_edit.clear()
            self.content_edit.clear()
            
        except Exception as e:
            error_msg = f"添加笔记到知识库失败: {str(e)}"
            self.progress_edit.append(f"错误: {error_msg}")
            QMessageBox.critical(self, "错误", error_msg)