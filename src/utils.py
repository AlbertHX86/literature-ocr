"""
工具函数
"""

import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_processing_results(results: List[Dict], output_file: str):
    """保存处理结果到 JSON 文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理结果已保存到: {output_path}")


def load_processing_results(input_file: str) -> List[Dict]:
    """从 JSON 文件加载处理结果"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        logger.warning(f"文件不存在: {input_path}")
        return []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    logger.info(f"从 {input_path} 加载了 {len(results)} 个结果")
    return results


def format_answer(answer_dict: Dict) -> str:
    """格式化答案输出"""
    output = []
    
    output.append("=" * 60)
    output.append("答案:")
    output.append("=" * 60)
    answer = answer_dict.get('answer', '')
    output.append(answer)
    output.append("")
    
    # 显示置信度
    if answer_dict.get('confidence'):
        confidence = answer_dict['confidence']
        confidence_emoji = {'high': '✅', 'medium': '⚠️', 'low': '❌'}.get(confidence, '')
        output.append(f"{confidence_emoji} 置信度: {confidence}")
        output.append("")
    
    # 显示来源文献
    if answer_dict.get('sources'):
        output.append("📚 来源文献（这些是回答的唯一信息来源）:")
        for i, source in enumerate(answer_dict['sources'], 1):
            output.append(f"  {i}. {source}")
    
    # 显示警告
    if answer_dict.get('warning'):
        output.append("")
        output.append(f"⚠️  警告: {answer_dict['warning']}")
    
    output.append("")
    output.append("💡 提示: 所有回答严格基于您提供的文献内容，不会使用外部知识。")
    
    return '\n'.join(output)


def format_papers(paper_list: List[Dict]) -> str:
    """格式化论文列表输出"""
    if not paper_list:
        return "未找到相关论文。"
    
    output = []
    output.append(f"找到 {len(paper_list)} 篇相关论文:\n")
    
    for i, paper in enumerate(paper_list, 1):
        output.append(f"{i}. {paper['name']}")
        if paper.get('path'):
            output.append(f"   路径: {paper['path']}")
        if paper.get('snippets'):
            output.append(f"   相关片段: {paper['snippets'][0]['text'][:200]}...")
        output.append("")
    
    return '\n'.join(output)
