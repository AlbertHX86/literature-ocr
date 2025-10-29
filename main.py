#!/usr/bin/env python3
"""
文献总结与分析系统 - 主程序
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import logging

from src.pdf_processor import PDFProcessor
from src.pdf_processor_hf import PDFProcessorHF
from src.vector_db import VectorDatabase
from src.rag_system import RAGSystem
from src.utils import save_processing_results, format_answer, format_papers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiteratureSummarySystem:
    """文献总结与分析系统主类"""
    
    def __init__(self, db_path: str = './literature_db'):
        """初始化系统"""
        self.db_path = db_path
        self.vector_db = None
        self.rag_system = None
    
    def init_database(self):
        """初始化向量数据库"""
        if self.vector_db is None:
            self.vector_db = VectorDatabase(self.db_path)
            logger.info(f"向量数据库已初始化: {self.db_path}")
    
    def init_rag(self):
        """初始化 RAG 系统"""
        self.init_database()
        if self.rag_system is None:
            self.rag_system = RAGSystem(self.vector_db)
            logger.info("RAG 系统已初始化")
    
    def process_pdfs(self, pdf_folder: str, output_dir: Optional[str] = None,
                    use_hf_api: bool = False) -> int:
        """
        处理 PDF 文件夹
        
        Args:
            pdf_folder: PDF 文件夹路径
            output_dir: 输出目录
            use_hf_api: 是否使用 Hugging Face API（云端处理）
            
        Returns:
            处理的 PDF 数量
        """
        logger.info(f"开始处理 PDF 文件夹: {pdf_folder}")
        
        # 初始化处理器
        processor = None
        if use_hf_api:
            logger.info("尝试使用 Hugging Face API 进行 PDF 处理（云端）")
            try:
                processor_hf = PDFProcessorHF()
                # 测试 API 是否可用 - 尝试处理第一个文件的第一页
                logger.info("测试 Hugging Face API 连接...")
                # 先尝试初始化，实际处理时再测试
                processor = processor_hf
            except Exception as e:
                logger.warning(f"Hugging Face API 初始化失败: {e}")
                logger.info("自动切换到本地模式...")
                use_hf_api = False
                processor = PDFProcessor()
        
        if processor is None:
            logger.info("使用本地 DeepSeek-OCR 模型进行 PDF 处理")
            processor = PDFProcessor()
        
        # 处理所有 PDF，如果 API 失败则自动回退
        try:
            results = processor.process_folder(pdf_folder, output_dir=output_dir)
        except Exception as e:
            if use_hf_api and ("404" in str(e) or "Not Found" in str(e)):
                logger.warning("Hugging Face API 不可用，切换到本地模式...")
                logger.info("注意：首次运行需要下载模型（约几GB），请稍候...")
                processor = PDFProcessor()
                results = processor.process_folder(pdf_folder, output_dir=output_dir)
            else:
                raise
        
        # 保存处理结果
        if output_dir:
            output_path = Path(output_dir) / 'processing_results.json'
            save_processing_results(results, str(output_path))
        
        # 添加到向量数据库
        self.init_database()
        self.vector_db.add_documents(results)
        
        logger.info(f"完成处理 {len(results)} 个 PDF 文件")
        return len(results)
    
    def query(self, question: str) -> str:
        """
        回答用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            格式化的答案
        """
        self.init_rag()
        
        answer_dict = self.rag_system.answer_question(question)
        return format_answer(answer_dict)
    
    def review(self, topic: str, output_file: Optional[str] = None) -> str:
        """
        生成文献综述
        
        Args:
            topic: 主题
            output_file: 输出文件路径
            
        Returns:
            综述文本
        """
        self.init_rag()
        
        review_text = self.rag_system.generate_review(topic)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(review_text)
            logger.info(f"综述已保存到: {output_path}")
        
        return review_text
    
    def search(self, query: str) -> str:
        """
        搜索相关论文
        
        Args:
            query: 查询文本
            
        Returns:
            格式化的论文列表
        """
        self.init_rag()
        
        papers = self.rag_system.find_relevant_papers(query)
        return format_papers(papers)
    
    def list_papers(self) -> List[str]:
        """列出所有论文"""
        self.init_database()
        return self.vector_db.get_all_papers()
    
    def get_paper_content(self, pdf_name: str) -> str:
        """获取论文内容"""
        self.init_database()
        return self.vector_db.get_paper_content(pdf_name)
    
    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        self.init_database()
        return self.vector_db.get_stats()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='文献总结与分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 初始化文献库
  python main.py init --pdf_folder ./papers --db_path ./literature_db
  
  # 查询问题
  python main.py query "什么是 transformer 架构？"
  
  # 生成综述
  python main.py review --topic "深度学习" --output review.md
  
  # 搜索论文
  python main.py search "计算机视觉"
  
  # 列出所有论文
  python main.py list
  
  # 交互式问答
  python main.py chat
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # init 命令
    init_parser = subparsers.add_parser('init', help='初始化文献库')
    init_parser.add_argument('--pdf_folder', required=True, help='PDF 文件夹路径')
    init_parser.add_argument('--db_path', default='./literature_db', help='数据库路径')
    init_parser.add_argument('--output_dir', help='输出目录')
    init_parser.add_argument('--use_hf_api', action='store_true', 
                           help='使用 Hugging Face API 进行 PDF 处理（无需本地模型）')
    
    # query 命令
    query_parser = subparsers.add_parser('query', help='回答问题')
    query_parser.add_argument('question', help='问题')
    query_parser.add_argument('--db_path', default='./literature_db', help='数据库路径')
    
    # review 命令
    review_parser = subparsers.add_parser('review', help='生成文献综述')
    review_parser.add_argument('--topic', required=True, help='主题')
    review_parser.add_argument('--output', help='输出文件路径')
    review_parser.add_argument('--db_path', default='./literature_db', help='数据库路径')
    
    # search 命令
    search_parser = subparsers.add_parser('search', help='搜索相关论文')
    search_parser.add_argument('query', help='查询文本')
    search_parser.add_argument('--db_path', default='./literature_db', help='数据库路径')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出所有论文')
    list_parser.add_argument('--db_path', default='./literature_db', help='数据库路径')
    
    # stats 命令
    stats_parser = subparsers.add_parser('stats', help='显示统计信息')
    stats_parser.add_argument('--db_path', default='./literature_db', help='数据库路径')
    
    # chat 命令
    chat_parser = subparsers.add_parser('chat', help='交互式问答')
    chat_parser.add_argument('--db_path', default='./literature_db', help='数据库路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 创建系统实例
    system = LiteratureSummarySystem(db_path=args.db_path)
    
    try:
        if args.command == 'init':
            count = system.process_pdfs(
                pdf_folder=args.pdf_folder,
                output_dir=args.output_dir,
                use_hf_api=getattr(args, 'use_hf_api', False)
            )
            print(f"\n✅ 成功处理 {count} 个 PDF 文件")
            print(f"数据库位置: {args.db_path}")
        
        elif args.command == 'query':
            answer = system.query(args.question)
            print(answer)
        
        elif args.command == 'review':
            review = system.review(args.topic, output_file=args.output)
            print("\n" + "=" * 60)
            print("文献综述")
            print("=" * 60)
            print(review)
            if args.output:
                print(f"\n✅ 综述已保存到: {args.output}")
        
        elif args.command == 'search':
            papers = system.search(args.query)
            print(papers)
        
        elif args.command == 'list':
            papers = system.list_papers()
            if papers:
                print(f"\n共有 {len(papers)} 篇论文:\n")
                for i, paper in enumerate(papers, 1):
                    print(f"  {i}. {paper}")
            else:
                print("数据库中暂无论文。")
        
        elif args.command == 'stats':
            stats = system.get_stats()
            print("\n数据库统计信息:")
            print("=" * 60)
            print(f"总文本块数: {stats['total_chunks']}")
            print(f"总论文数: {stats['total_papers']}")
            if stats['papers']:
                print("\n论文列表:")
                for i, paper in enumerate(stats['papers'], 1):
                    print(f"  {i}. {paper}")
        
        elif args.command == 'chat':
            print("\n" + "=" * 60)
            print("文献分析系统 - 交互式问答")
            print("=" * 60)
            print("输入 'quit' 或 'exit' 退出\n")
            
            while True:
                try:
                    question = input("请输入您的问题: ").strip()
                    if not question:
                        continue
                    
                    if question.lower() in ['quit', 'exit', '退出']:
                        print("再见！")
                        break
                    
                    answer = system.query(question)
                    print("\n" + answer + "\n")
                    
                except KeyboardInterrupt:
                    print("\n\n再见！")
                    break
                except Exception as e:
                    logger.error(f"处理问题时出错: {e}")
                    print(f"错误: {e}\n")
    
    except Exception as e:
        logger.error(f"执行命令时出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
