import os
import sys
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                            QPushButton, QTextEdit, QFileDialog, QMessageBox, QCheckBox,
                            QRadioButton, QButtonGroup, QGroupBox, QFrame, QSizePolicy,
                            QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QFont

# 检查PyTorch是否可用
def torch_available():
    try:
        import torch
        return True, torch
    except ImportError:
        return False, None

# 检查PyTorch可用性
TORCH_AVAILABLE, torch = torch_available()

# 导入现有的模型转换页面
from model_conversion_page import ModelConversionPage

class EmbeddingConversionThread(QThread):
    progress_updated = pyqtSignal(str)
    completed = pyqtSignal(bool, str)
    
    def __init__(self, model_path, output_dir=None, use_cuda=False):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.use_cuda = use_cuda
    
    def update_progress(self, msg):
        self.progress_updated.emit(msg)
    
    def run(self):
        try:
            # 导入转换函数
            from convert_embedding_to_onnx import convert_embedding_model_to_onnx
            
            # 执行转换
            output_dir = convert_embedding_model_to_onnx(
                self.model_path, 
                self.output_dir, 
                self.use_cuda
            )
            
            if output_dir is not None:
                self.completed.emit(True, f"嵌入模型已成功转换为ONNX格式，保存在: {output_dir}")
            else:
                self.completed.emit(False, "转换失败，未能获取输出目录")
        except Exception as e:
            import traceback
            error_msg = f"转换失败: {str(e)}\n{traceback.format_exc()}"
            self.progress_updated.emit(f"[ERROR] {error_msg}")
            self.completed.emit(False, error_msg)

class EmbeddingConversionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("嵌入模型转换 (Sentence Transformers → ONNX)")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)
        
        # 模型选择部分
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型路径:"))
        
        self.model_path_edit = QTextEdit()
        self.model_path_edit.setMaximumHeight(60)
        self.model_path_edit.setPlaceholderText("输入嵌入模型路径或从embeddings目录选择")
        model_layout.addWidget(self.model_path_edit)
        
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_button)
        
        scan_button = QPushButton("扫描本地模型")
        scan_button.clicked.connect(self.scan_local_models)
        model_layout.addWidget(scan_button)
        
        layout.addLayout(model_layout)
        
        # 输出目录部分
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        
        self.output_dir_edit = QTextEdit()
        self.output_dir_edit.setMaximumHeight(60)
        self.output_dir_edit.setPlaceholderText("(可选) 输入输出目录，默认为模型路径 + '_onnx'")
        output_layout.addWidget(self.output_dir_edit)
        
        browse_output_button = QPushButton("浏览...")
        browse_output_button.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(browse_output_button)
        
        layout.addLayout(output_layout)
        
        # 选项部分
        options_layout = QHBoxLayout()
        
        self.cuda_checkbox = QCheckBox("使用CUDA加速")
        if TORCH_AVAILABLE:
            import torch
            self.cuda_checkbox.setEnabled(torch.cuda.is_available())
            if not torch.cuda.is_available():
                self.cuda_checkbox.setToolTip("CUDA不可用")
        else:
            self.cuda_checkbox.setEnabled(False)
            self.cuda_checkbox.setToolTip("PyTorch未安装")
        
        options_layout.addWidget(self.cuda_checkbox)
        options_layout.addStretch()
        
        layout.addLayout(options_layout)
        
        # 转换按钮
        self.convert_button = QPushButton("转换")
        self.convert_button.clicked.connect(self.convert_model)
        self.convert_button.setMinimumHeight(40)
        layout.addWidget(self.convert_button)
        
        # 进度显示
        layout.addWidget(QLabel("转换进度:"))
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        layout.addWidget(self.progress_text)
        
        self.setLayout(layout)
    
    def browse_model(self):
        embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir, exist_ok=True)
        
        model_path = QFileDialog.getExistingDirectory(
            self, "选择嵌入模型目录", embeddings_dir
        )
        
        if model_path:
            self.model_path_edit.setText(model_path)
    
    def browse_output_dir(self):
        embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir, exist_ok=True)
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "选择输出目录", embeddings_dir
        )
        
        if output_dir:
            self.output_dir_edit.setText(output_dir)
    
    def scan_local_models(self):
        embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir, exist_ok=True)
            self.progress_text.append("创建了embeddings目录，请将嵌入模型放入该目录")
            return
        
        # 扫描目录
        models = []
        for item in os.listdir(embeddings_dir):
            item_path = os.path.join(embeddings_dir, item)
            if os.path.isdir(item_path):
                # 检查是否是Sentence Transformer模型（包含config_sentence_transformers.json文件）
                if os.path.exists(os.path.join(item_path, "config_sentence_transformers.json")):
                    models.append(item_path)
                # 检查是否已经是ONNX模型（包含transformer_model.onnx文件）
                elif os.path.exists(os.path.join(item_path, "transformer_model.onnx")):
                    continue  # 跳过已经转换的ONNX模型
        
        if models:
            # 显示模型选择对话框
            model, ok = QMessageBox.question(
                self, 
                "选择模型", 
                f"找到 {len(models)} 个嵌入模型，是否使用第一个: {os.path.basename(models[0])}?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            
            if ok == QMessageBox.Yes:
                self.model_path_edit.setText(models[0])
                self.progress_text.append(f"已选择模型: {models[0]}")
            else:
                # 显示所有找到的模型
                model_list = "\n".join([f"- {os.path.basename(m)}" for m in models])
                self.progress_text.append(f"找到以下嵌入模型:\n{model_list}\n请手动选择一个")
        else:
            self.progress_text.append("未找到可用的嵌入模型，请先下载或放置模型到embeddings目录")
    
    def convert_model(self):
        model_path = self.model_path_edit.toPlainText().strip()
        output_dir = self.output_dir_edit.toPlainText().strip() or None
        use_cuda = self.cuda_checkbox.isChecked()
        
        if not model_path:
            QMessageBox.warning(self, "错误", "请选择嵌入模型路径")
            return
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", f"模型路径不存在: {model_path}")
            return
        
        # 检查是否已经是ONNX模型
        if os.path.exists(os.path.join(model_path, "transformer_model.onnx")):
            reply = QMessageBox.question(
                self, 
                "确认", 
                "该模型似乎已经是ONNX格式，是否继续转换?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # 检查是否安装了必要的依赖
        try:
            import torch
            import sentence_transformers
            import onnx
            import onnxruntime
        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else str(e)
            QMessageBox.warning(
                self, 
                "依赖缺失", 
                f"缺少必要的依赖: {missing_lib}\n\n请安装以下依赖:\npip install torch sentence-transformers onnx onnxruntime"
            )
            return
        
        # 清空进度显示
        self.progress_text.clear()
        self.progress_text.append(f"[INFO] 开始转换嵌入模型: {model_path}")
        
        # 禁用转换按钮，避免重复点击
        self.convert_button.setEnabled(False)
        self.convert_button.setText("转换中...")
        
        # 开始转换
        self.conversion_thread = EmbeddingConversionThread(
            model_path, 
            output_dir, 
            use_cuda
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
            msg_box.setText("嵌入模型转换完成!")
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
            msg_box.setText("嵌入模型转换过程遇到错误")
            msg_box.setInformativeText(message)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

class ModelConversionHub(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 添加大语言模型转换页面
        self.llm_page = ModelConversionPage()
        self.tab_widget.addTab(self.llm_page, "大语言模型转换")
        
        # 添加嵌入模型转换页面
        self.embedding_page = EmbeddingConversionPage()
        self.tab_widget.addTab(self.embedding_page, "嵌入模型转换")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = ModelConversionHub()
    window.setWindowTitle("模型转换中心")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
