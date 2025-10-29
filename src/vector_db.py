"""
向量数据库模块
使用 ChromaDB 存储和检索文献向量
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """向量数据库管理类"""
    
    def __init__(self, db_path: str, 
                 embedding_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化向量数据库
        
        Args:
            db_path: 数据库路径
            embedding_model: 嵌入模型名称
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="literature",
            metadata={"description": "学术文献向量数据库"}
        )
        
        # 加载嵌入模型
        logger.info(f"正在加载嵌入模型: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        logger.info("嵌入模型加载完成")
    
    def add_documents(self, documents: List[Dict[str, any]], 
                     chunk_size: int = 500,
                     chunk_overlap: int = 100):
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表，每个文档包含 'pdf_name', 'full_text', 'metadata' 等
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        logger.info(f"正在添加 {len(documents)} 个文档到向量数据库")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_idx, doc in enumerate(documents):
            if 'error' in doc or not doc.get('full_text'):
                logger.warning(f"跳过文档 {doc.get('pdf_name', doc_idx)}: 无有效内容")
                continue
            
            # 将文本分块
            text = doc['full_text']
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            
            # 生成嵌入
            embeddings = self.embedder.encode(chunks, show_progress_bar=True)
            
            # 准备元数据
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc['pdf_name']}_chunk_{chunk_idx}"
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                
                metadata = {
                    'pdf_name': doc['pdf_name'],
                    'pdf_path': doc.get('pdf_path', ''),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'total_pages': doc.get('total_pages', 0),
                    'successful_pages': doc.get('metadata', {}).get('successful_pages', 0)
                }
                all_metadatas.append(metadata)
        
        # 批量添加到数据库
        if all_chunks:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logger.info(f"成功添加 {len(all_chunks)} 个文本块到数据库")
        else:
            logger.warning("没有可添加的文本块")
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        将文本分块
        
        Args:
            text: 原始文本
            chunk_size: 块大小（字符数）
            chunk_overlap: 重叠大小（字符数）
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # 尝试在句子边界分割
            if end < len(text):
                # 寻找最后一个句号、问号或换行符
                for delimiter in ['\n\n', '.\n', '.\n ', '. ', '\n']:
                    last_delim = chunk.rfind(delimiter)
                    if last_delim > chunk_size // 2:  # 至少保留一半内容
                        chunk = chunk[:last_delim + len(delimiter)]
                        end = start + last_delim + len(delimiter)
                        break
            
            chunks.append(chunk.strip())
            
            # 移动到下一个块，考虑重叠
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def search(self, query: str, n_results: int = 5, 
              filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            搜索结果列表，每个结果包含 'text', 'metadata', 'distance'
        """
        # 生成查询向量
        query_embedding = self.embedder.encode([query])[0]
        
        # 构建查询条件
        where = filter_metadata if filter_metadata else None
        
        # 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )
        
        # 格式化结果
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'id': results['ids'][0][i]
                })
        
        return formatted_results
    
    def get_all_papers(self) -> List[str]:
        """获取所有论文名称"""
        # 获取所有文档
        all_docs = self.collection.get()
        
        # 提取唯一的论文名称
        paper_names = set()
        if all_docs['metadatas']:
            for metadata in all_docs['metadatas']:
                paper_names.add(metadata.get('pdf_name', 'unknown'))
        
        return sorted(list(paper_names))
    
    def get_paper_content(self, pdf_name: str) -> str:
        """
        获取指定论文的完整内容
        
        Args:
            pdf_name: PDF 文件名（不含扩展名）
            
        Returns:
            论文的完整文本
        """
        # 获取该论文的所有块
        results = self.collection.get(
            where={'pdf_name': pdf_name}
        )
        
        if not results['documents']:
            return ""
        
        # 按 chunk_index 排序
        chunks_with_index = [
            (int(results['metadatas'][i].get('chunk_index', 0)), results['documents'][i])
            for i in range(len(results['documents']))
        ]
        chunks_with_index.sort(key=lambda x: x[0])
        
        # 合并文本
        full_text = '\n\n'.join([chunk for _, chunk in chunks_with_index])
        return full_text
    
    def delete_paper(self, pdf_name: str) -> bool:
        """
        删除指定论文
        
        Args:
            pdf_name: PDF 文件名
            
        Returns:
            是否成功删除
        """
        try:
            # 获取该论文的所有 ID
            results = self.collection.get(
                where={'pdf_name': pdf_name}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"已删除论文: {pdf_name}")
                return True
            else:
                logger.warning(f"未找到论文: {pdf_name}")
                return False
        except Exception as e:
            logger.error(f"删除论文时出错: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        count = self.collection.count()
        papers = self.get_all_papers()
        
        return {
            'total_chunks': count,
            'total_papers': len(papers),
            'papers': papers
        }


if __name__ == '__main__':
    # 测试代码
    db = VectorDatabase('./test_db')
    print(db.get_stats())
