"""
PDF 处理模块
使用 DeepSeek-OCR 提取 PDF 中的文本、表格和图像
"""

import os
import torch
import tempfile
from typing import List, Dict, Optional
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from pdf2image import convert_from_path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """使用 DeepSeek-OCR 处理 PDF 文件"""
    
    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-OCR', 
                 device: Optional[str] = None):
        """
        初始化 PDF 处理器
        
        Args:
            model_name: DeepSeek-OCR 模型名称
            device: 设备 (cuda/cpu)，如果为 None 则自动选择
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"正在加载 DeepSeek-OCR 模型: {model_name}")
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            )
        except:
            # 如果 flash_attention_2 不可用，使用默认实现
            logger.warning("Flash Attention 2 不可用，使用默认实现")
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True
            )
        
        self.model = self.model.eval().to(self.device)
        if self.device == 'cuda':
            self.model = self.model.to(torch.bfloat16)
        
        logger.info("模型加载完成")
    
    def process_pdf(self, pdf_path: str, 
                   output_dir: Optional[str] = None,
                   prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
                   image_size: int = 640,
                   base_size: int = 1024,
                   save_results: bool = False) -> Dict[str, any]:
        """
        处理单个 PDF 文件
        
        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录
            prompt: OCR 提示词
            image_size: 图像大小
            base_size: 基础大小
            save_results: 是否保存结果
            
        Returns:
            包含提取内容的字典
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        
        logger.info(f"正在处理 PDF: {pdf_path.name}")
        
        # 将 PDF 转换为图像
        try:
            images = convert_from_path(str(pdf_path), dpi=200)
        except Exception as e:
            logger.error(f"PDF 转换失败: {e}")
            raise
        
        results = {
            'pdf_name': pdf_path.stem,
            'pdf_path': str(pdf_path),
            'total_pages': len(images),
            'pages': [],
            'full_text': '',
            'metadata': {}
        }
        
        # 处理每一页
        all_texts = []
        for page_num, image in enumerate(images, 1):
            logger.info(f"处理第 {page_num}/{len(images)} 页")
            
            # 转换为 RGB
            image_rgb = image.convert('RGB')
            
            # 使用 DeepSeek-OCR 提取内容
            try:
                output_path = None
                if save_results and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir, 
                        f"{pdf_path.stem}_page_{page_num}.md"
                    )
                
                # 将图像保存为临时文件（DeepSeek-OCR API 需要文件路径）
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_image_path = tmp_file.name
                    image_rgb.save(temp_image_path, 'PNG')
                
                try:
                    res = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=temp_image_path,
                        output_path=output_path,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=True,
                        save_results=save_results,
                        test_compress=True
                    )
                    
                    # 提取文本内容 (根据实际 API 调整)
                    # infer 方法可能返回字符串或字典，需要根据实际情况调整
                    if isinstance(res, str):
                        page_text = res
                    elif isinstance(res, dict):
                        page_text = res.get('text', res.get('result', ''))
                    else:
                        page_text = str(res)
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
                all_texts.append(page_text)
                
                results['pages'].append({
                    'page_num': page_num,
                    'text': page_text,
                    'has_content': len(page_text.strip()) > 0
                })
                
            except Exception as e:
                logger.error(f"处理第 {page_num} 页时出错: {e}")
                results['pages'].append({
                    'page_num': page_num,
                    'text': '',
                    'has_content': False,
                    'error': str(e)
                })
        
        # 合并所有页面的文本
        results['full_text'] = '\n\n'.join(all_texts)
        results['metadata'] = {
            'successful_pages': sum(1 for p in results['pages'] if p['has_content']),
            'failed_pages': sum(1 for p in results['pages'] if not p['has_content'])
        }
        
        logger.info(f"PDF 处理完成: {pdf_path.name}")
        logger.info(f"成功处理 {results['metadata']['successful_pages']}/{results['total_pages']} 页")
        
        return results
    
    def process_folder(self, folder_path: str, 
                      output_dir: Optional[str] = None,
                      **kwargs) -> List[Dict[str, any]]:
        """
        批量处理文件夹中的所有 PDF
        
        Args:
            folder_path: PDF 文件夹路径
            output_dir: 输出目录
            **kwargs: 传递给 process_pdf 的其他参数
            
        Returns:
            处理结果列表
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        # 查找所有 PDF 文件
        pdf_files = list(folder_path.glob('*.pdf')) + list(folder_path.glob('*.PDF'))
        
        if not pdf_files:
            logger.warning(f"文件夹中未找到 PDF 文件: {folder_path}")
            return []
        
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf(
                    str(pdf_file),
                    output_dir=output_dir,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"处理 {pdf_file.name} 失败: {e}")
                results.append({
                    'pdf_name': pdf_file.stem,
                    'pdf_path': str(pdf_file),
                    'error': str(e)
                })
        
        return results


def test_processor():
    """测试函数"""
    processor = PDFProcessor()
    # 测试代码
    pass


if __name__ == '__main__':
    test_processor()
