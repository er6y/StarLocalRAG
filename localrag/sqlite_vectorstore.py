import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterable
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings

class SQLiteVectorStore(VectorStore):
    """使用SQLite作为存储引擎的向量数据库实现"""
    
    def __init__(
        self,
        embedding_function: Embeddings,
        db_path: str,
        collection_name: str = "documents",
        distance_strategy: str = "cosine"
    ):
        """初始化SQLite向量数据库
        
        Args:
            embedding_function: 用于生成向量的嵌入函数
            db_path: SQLite数据库文件路径
            collection_name: 集合名称，用于区分不同的文档集合
            distance_strategy: 距离计算策略，支持"cosine"和"euclidean"
        """
        self.embedding_function = embedding_function
        self.db_path = db_path
        self.collection_name = collection_name
        self.distance_strategy = distance_strategy
        
        # 创建数据库和表
        self._create_tables()
    
    def _create_tables(self):
        """创建必要的数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建文档表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            embedding BLOB
        )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection ON documents(collection)')
        
        conn.commit()
        conn.close()
    
    def _vector_to_blob(self, vector: List[float]) -> bytes:
        """将向量转换为二进制数据"""
        return np.array(vector, dtype=np.float32).tobytes()
    
    def _blob_to_vector(self, blob: bytes) -> List[float]:
        """将二进制数据转换为向量"""
        return np.frombuffer(blob, dtype=np.float32).tolist()
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """向数据库添加文本
        
        Args:
            texts: 要添加的文本列表
            metadatas: 文本对应的元数据列表
            
        Returns:
            添加的文档ID列表
        """
        if not texts:
            return []
        
        # 生成向量嵌入
        embeddings = self.embedding_function.embed_documents(list(texts))
        
        # 准备元数据
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # 连接数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 插入数据
        ids = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            embedding_blob = self._vector_to_blob(embedding)
            
            cursor.execute(
                "INSERT INTO documents (collection, content, metadata, embedding) VALUES (?, ?, ?, ?)",
                (self.collection_name, text, metadata_json, embedding_blob)
            )
            ids.append(str(cursor.lastrowid))
        
        conn.commit()
        conn.close()
        
        return ids
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """添加文档对象列表
        
        Args:
            documents: 要添加的文档对象列表
            
        Returns:
            添加的文档ID列表
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """相似度搜索
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            
        Returns:
            相似度最高的k个文档
        """
        # 生成查询向量
        query_embedding = self.embedding_function.embed_query(query)
        
        # 连接数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有文档向量
        cursor.execute(
            "SELECT id, content, metadata, embedding FROM documents WHERE collection = ?",
            (self.collection_name,)
        )
        
        results = []
        for doc_id, content, metadata_json, embedding_blob in cursor.fetchall():
            # 转换向量和计算相似度
            doc_embedding = self._blob_to_vector(embedding_blob)
            
            # 计算相似度
            if self.distance_strategy == "cosine":
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
            else:  # euclidean
                similarity = self._euclidean_distance(query_embedding, doc_embedding)
            
            # 解析元数据
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            # 将相似度添加到元数据中，方便后续调试
            metadata["similarity_score"] = similarity
            
            results.append((similarity, doc_id, content, metadata))
        
        conn.close()
        
        # 排序并返回前k个结果
        if self.distance_strategy == "cosine":
            # 余弦相似度越高越好
            results.sort(reverse=True)
        else:
            # 欧几里得距离越小越好
            results.sort()
        
        top_k_results = results[:k]
        
        # 转换为Document对象
        documents = []
        for similarity, doc_id, content, metadata in top_k_results:
            # 将相似度添加到元数据中
            metadata["similarity_score"] = similarity
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # 避免除以零
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """计算欧几里得距离"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        return np.linalg.norm(vec1 - vec2)
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """删除指定ID的文档
        
        Args:
            ids: 要删除的文档ID列表
            
        Returns:
            操作是否成功
        """
        if not ids:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 将字符串ID转换为整数
        int_ids = [int(id_) for id_ in ids]
        
        # 使用参数化查询删除文档
        placeholders = ','.join(['?'] * len(int_ids))
        cursor.execute(
            f"DELETE FROM documents WHERE id IN ({placeholders}) AND collection = ?",
            int_ids + [self.collection_name]
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    def get(self, ids: Optional[List[str]] = None, **kwargs: Any) -> List[Document]:
        """获取指定ID的文档
        
        Args:
            ids: 要获取的文档ID列表
            
        Returns:
            文档对象列表
        """
        if not ids:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 将字符串ID转换为整数
        int_ids = [int(id_) for id_ in ids]
        
        # 使用参数化查询获取文档
        placeholders = ','.join(['?'] * len(int_ids))
        cursor.execute(
            f"SELECT id, content, metadata FROM documents WHERE id IN ({placeholders}) AND collection = ?",
            int_ids + [self.collection_name]
        )
        
        documents = []
        for doc_id, content, metadata_json in cursor.fetchall():
            metadata = json.loads(metadata_json) if metadata_json else {}
            documents.append(Document(page_content=content, metadata=metadata))
        
        conn.close()
        
        return documents
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        db_path: str,
        collection_name: str = "documents",
        **kwargs: Any,
    ) -> "SQLiteVectorStore":
        """从文档列表创建向量存储
        
        Args:
            documents: 文档列表
            embedding: 嵌入函数
            db_path: 数据库路径
            collection_name: 集合名称
            
        Returns:
            SQLiteVectorStore实例
        """
        instance = cls(
            embedding_function=embedding,
            db_path=db_path,
            collection_name=collection_name,
            **kwargs
        )
        instance.add_documents(documents)
        return instance
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        db_path: str = None,
        collection_name: str = "documents",
        **kwargs: Any,
    ) -> "SQLiteVectorStore":
        """从文本列表创建向量存储
        
        Args:
            texts: 文本列表
            embedding: 嵌入函数
            metadatas: 元数据列表
            db_path: 数据库路径
            collection_name: 集合名称
            
        Returns:
            SQLiteVectorStore实例
        """
        instance = cls(
            embedding_function=embedding,
            db_path=db_path,
            collection_name=collection_name,
            **kwargs
        )
        instance.add_texts(texts, metadatas)
        return instance
    
    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """保存向量存储到本地
        
        Args:
            folder_path: 保存的文件夹路径
            index_name: 索引名称
        """
        # 确保目录存在
        os.makedirs(folder_path, exist_ok=True)
        
        # 复制数据库文件
        import shutil
        target_path = os.path.join(folder_path, f"{index_name}.db")
        shutil.copy2(self.db_path, target_path)
    
    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embedding: Embeddings,
        index_name: str = "index",
        collection_name: str = "documents",
        **kwargs: Any,
    ) -> "SQLiteVectorStore":
        """从本地加载向量存储
        
        Args:
            folder_path: 文件夹路径
            embedding: 嵌入函数
            index_name: 索引名称
            collection_name: 集合名称
            
        Returns:
            SQLiteVectorStore实例
        """
        db_path = os.path.join(folder_path, f"{index_name}.db")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        return cls(
            embedding_function=embedding,
            db_path=db_path,
            collection_name=collection_name,
            **kwargs
        )
