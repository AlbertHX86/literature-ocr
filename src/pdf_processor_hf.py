"""
PDF 处理模块 - Hugging Face API 版本
使用 Hugging Face Inference API 进行 OCR（可选，如果本地处理不可用）
"""

import os
import base64
import requests
from typing import List, Dict, Optional
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessorHF:
    """使用 Hugging Face API 处理 PDF 文件（云端版本）"""
    
    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-OCR',
                 api_key: Optional[str] = None):
        """
        初始化 PDF 处理器（使用 Hugging Face API）
        
        Args:
            model_name: DeepSeek-OCR 模型名称
            api_key: Hugging Face API Key（可选）
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv('HUGGINGFACE_API_KEY')
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        logger.info(f"使用 Hugging Face Inference API: {model_name}")
        if not self.api_key:
            logger.warning("未设置 HUGGINGFACE_API_KEY，某些模型可能需要")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将 PIL Image 转换为 base64 字符串"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def _call_hf_api(self, image_base64: str, prompt: str) -> str:
        """调用 Hugging Face API"""
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        # 注意：DeepSeek-OCR 可能需要特殊的 API 调用格式
        # 这里提供一个通用的实现，可能需要根据实际 API 调整
        payload = {
            'inputs': {
                'image': image_base64,
                'prompt': prompt
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                # 根据实际 API 响应格式解析
                if isinstance(result, dict):
                    return result.get('text', result.get('generated_text', str(result)))
                elif isinstance(result, list) and len(result) > 0:
                    return result[0].get('text', str(result[0]))
                return str(result)
            elif response.status_code == 503:
                logger.info("模型正在加载，等待 30 秒后重试...")
                import time
                time.sleep(30)
                return self._call_hf_api(image_base64, prompt)
            else:
                error_text = response.text[:200] if response.text else "无错误信息"
                raise Exception(f"API 错误 {response.status_code}: {error_text}")
        except Exception as e:
            logger.error(f"Hugging Face API 调用失败: {e}")
            # 对于 404 错误，说明模型可能不支持 Inference API
            if "404" in str(e) or "Not Found" in str(e):
                logger.warning("DeepSeek-OCR 可能不支持 Hugging Face Inference API。建议使用本地模式。")
            raise
    
    def process_pdf(self, pdf_path: str,
                   output_dir: Optional[str] = None,
                   prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
                   save_results: bool = False) -> Dict[str, any]:
        """
        处理单个 PDF 文件
        
        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录
            prompt: OCR 提示词
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
            logger.info(f"处理第 {page_num}/{len(images)} 页（使用 Hugging Face API）")
            
            try:
                # 转换为 RGB
                image_rgb = image.convert('RGB')
                
                # 转换为 base64
                image_base64 = self._image_to_base64(image_rgb)
                
                # 调用 Hugging Face API
                try:
                    page_text = self._call_hf_api(image_base64, prompt)
                    all_texts.append(page_text)
                except Exception as api_error:
                    # 如果是 404 错误，说明 API 不支持，抛出异常让上层处理
                    if "404" in str(api_error) or "Not Found" in str(api_error):
                        raise Exception("DeepSeek-OCR 不支持 Hugging Face Inference API，请使用本地模式")
                    raise
                
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
            'failed_pages': sum(1 for p in results['pages'] if not p['has_content']),
            'processing_method': 'huggingface_api'
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
