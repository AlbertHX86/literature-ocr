"""
RAG 问答系统
基于检索增强生成实现问答功能
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI 库未安装")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama 库未安装")

try:
    import requests
    REQUESTS_AVAILABLE = True
except:
    REQUESTS_AVAILABLE = False
    logger.warning("requests 库未安装，无法使用 Hugging Face API")


class RAGSystem:
    """RAG 问答系统 - 严格基于检索文档，避免幻觉"""
    
    def __init__(self, vector_db, 
                 use_openai: bool = False,
                 use_ollama: bool = False,
                 use_huggingface: bool = True,
                 ollama_model: str = 'llama3.1',
                 openai_model: str = 'gpt-4o-mini',
                 hf_model: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                 hf_api_key: Optional[str] = None):
        """
        初始化 RAG 系统
        
        Args:
            vector_db: VectorDatabase 实例
            use_openai: 是否使用 OpenAI API
            use_ollama: 是否使用本地 Ollama
            use_huggingface: 是否使用 Hugging Face Inference API（推荐，无需本地运行）
            ollama_model: Ollama 模型名称
            openai_model: OpenAI 模型名称
            hf_model: Hugging Face 模型名称
            hf_api_key: Hugging Face API Key（可选，某些模型需要）
        """
        self.vector_db = vector_db
        
        # 初始化 LLM - 优先级：HF > OpenAI > Ollama
        self.use_huggingface = use_huggingface and REQUESTS_AVAILABLE
        self.use_openai = use_openai and OPENAI_AVAILABLE and not self.use_huggingface
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE and not self.use_huggingface
        
        if self.use_huggingface:
            self.hf_model = hf_model
            self.hf_api_key = hf_api_key or os.getenv('HUGGINGFACE_API_KEY')
            self.hf_api_url = f"https://api-inference.huggingface.co/models/{hf_model}"
            logger.info(f"使用 Hugging Face Inference API: {hf_model}")
            if not self.hf_api_key:
                logger.warning("未设置 HUGGINGFACE_API_KEY，某些模型可能需要")
        
        if self.use_openai:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                self.openai_model = openai_model
                logger.info("使用 OpenAI API")
            else:
                logger.warning("未设置 OPENAI_API_KEY")
                self.use_openai = False
        
        if self.use_ollama:
            self.ollama_model = ollama_model
            logger.info(f"使用 Ollama 模型: {ollama_model}")
        
        if not (self.use_huggingface or self.use_openai or self.use_ollama):
            logger.warning("未配置任何 LLM，将使用模板模式（功能有限）")
    
    def answer_question(self, question: str, n_context: int = 5, 
                       min_relevance_threshold: float = 0.7) -> Dict[str, any]:
        """
        回答用户问题 - 严格基于检索到的文档，避免幻觉
        
        Args:
            question: 用户问题
            n_context: 使用的上下文文档数量
            min_relevance_threshold: 最小相关性阈值（距离越小越相关）
            
        Returns:
            包含答案和相关文档的字典
        """
        # 1. 检索相关文档
        logger.info(f"正在检索相关文档: {question}")
        relevant_docs = self.vector_db.search(question, n_results=n_context)
        
        if not relevant_docs:
            return {
                'answer': '抱歉，未在文献库中找到相关的内容。请尝试重新表述您的问题，或确认相关文献是否已添加到库中。',
                'sources': [],
                'context': '',
                'confidence': 'low',
                'warning': '未找到相关文档'
            }
        
        # 2. 检查相关性 - 过滤掉不够相关的文档
        filtered_docs = []
        for doc in relevant_docs:
            distance = doc.get('distance', 1.0)
            # ChromaDB 使用余弦距离，越小越相关
            # 转换为相关性分数 (0-1)，距离越小分数越高
            relevance_score = 1.0 - min(distance, 1.0)
            if relevance_score >= min_relevance_threshold:
                filtered_docs.append(doc)
        
        if not filtered_docs:
            return {
                'answer': f'抱歉，找到的文献内容与问题相关性较低（阈值: {min_relevance_threshold}）。请尝试更具体的问题，或扩大搜索范围。',
                'sources': [],
                'context': '',
                'confidence': 'low',
                'warning': '文档相关性不足'
            }
        
        # 3. 构建上下文，明确标注来源
        context_parts = []
        source_papers = set()
        
        for doc in filtered_docs:
            paper_name = doc['metadata']['pdf_name']
            text = doc['text']
            # 明确标注来源，防止模型编造
            context_parts.append(f"[来源论文: {paper_name}]\n{text}")
            source_papers.add(paper_name)
        
        context = '\n\n---\n\n'.join(context_parts)
        
        # 4. 生成答案 - 使用强化的防幻觉提示词
        prompt = self._build_anti_hallucination_prompt(question, context, list(source_papers))
        
        if self.use_huggingface:
            answer = self._generate_with_huggingface(prompt)
        elif self.use_openai:
            answer = self._generate_with_openai(prompt)
        elif self.use_ollama:
            answer = self._generate_with_ollama(prompt)
        else:
            answer = self._generate_template(question, context)
        
        # 5. 后处理：验证答案中是否包含来源信息
        confidence = 'high' if any(paper in answer for paper in source_papers) else 'medium'
        
        return {
            'answer': answer,
            'sources': sorted(list(source_papers)),
            'context': context,
            'num_sources': len(source_papers),
            'confidence': confidence,
            'num_filtered_docs': len(filtered_docs)
        }
    
    def generate_review(self, topic: str, n_papers: int = 10) -> str:
        """
        生成文献综述
        
        Args:
            topic: 主题
            n_papers: 使用的论文数量
            
        Returns:
            文献综述文本
        """
        logger.info(f"正在生成文献综述: {topic}")
        
        # 检索相关文档
        relevant_docs = self.vector_db.search(topic, n_results=n_papers * 3)
        
        if not relevant_docs:
            return f"抱歉，未找到关于 '{topic}' 的相关文献。"
        
        # 按论文分组
        papers_content = {}
        for doc in relevant_docs:
            paper_name = doc['metadata']['pdf_name']
            if paper_name not in papers_content:
                papers_content[paper_name] = []
            papers_content[paper_name].append(doc['text'])
        
        # 选择前 n_papers 篇论文
        selected_papers = list(papers_content.keys())[:n_papers]
        
        # 构建综述上下文
        review_parts = []
        for paper_name in selected_papers:
            paper_text = '\n\n'.join(papers_content[paper_name])
            review_parts.append(f"论文: {paper_name}\n内容摘要:\n{paper_text[:2000]}...")
        
        review_context = '\n\n---\n\n'.join(review_parts)
        
        # 生成综述
        prompt = f"""请基于以下文献内容，生成关于"{topic}"的学术文献综述。

要求：
1. 总结各文献的主要观点和方法
2. 分析不同文献之间的联系和差异
3. 指出当前研究的趋势和未来方向
4. 使用学术写作风格，结构清晰

文献内容：
{review_context}

请生成文献综述："""

        if self.use_openai:
            review = self._generate_with_openai(prompt)
        elif self.use_ollama:
            review = self._generate_with_ollama(prompt)
        else:
            review = self._generate_template_review(topic, review_context)
        
        return review
    
    def find_relevant_papers(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        查找相关论文
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            论文列表，包含论文名称和相关性摘要
        """
        logger.info(f"正在查找相关论文: {query}")
        
        # 检索相关文档
        relevant_docs = self.vector_db.search(query, n_results=n_results * 3)
        
        if not relevant_docs:
            return []
        
        # 按论文分组
        papers_score = {}
        for doc in relevant_docs:
            paper_name = doc['metadata']['pdf_name']
            if paper_name not in papers_score:
                papers_score[paper_name] = {
                    'name': paper_name,
                    'path': doc['metadata'].get('pdf_path', ''),
                    'snippets': [],
                    'score': 0
                }
            papers_score[paper_name]['snippets'].append({
                'text': doc['text'][:300],
                'distance': doc.get('distance', 1.0)
            })
            # 使用最小距离（最高相似度）作为分数
            current_score = papers_score[paper_name]['score']
            new_score = doc.get('distance', 1.0)
            if current_score == 0 or new_score < current_score:
                papers_score[paper_name]['score'] = new_score
        
        # 按分数排序
        sorted_papers = sorted(
            papers_score.values(),
            key=lambda x: x['score']
        )[:n_results]
        
        return sorted_papers
    
    def _build_anti_hallucination_prompt(self, question: str, context: str, source_papers: List[str]) -> str:
        """构建防幻觉提示词 - 严格要求只基于提供的文档"""
        papers_list = '\n'.join([f"- {paper}" for paper in source_papers])
        
        return f"""你是一个学术文献分析助手。请基于以下**来自指定文件夹的论文内容**回答问题。

**重要规则（严格遵守）：**
1. 你的回答**必须完全基于**下面提供的文献内容，不能使用任何外部知识
2. 如果文献中没有相关信息，**明确说明**"在提供的文献中未找到相关信息"
3. **不要编造、推测或补充**任何文献中没有明确提到的内容
4. 每个观点或事实都必须引用具体的来源论文
5. 如果文献内容不足以回答问题，直接说明，不要猜测

**文献来源（这些是唯一可用的信息来源）：**
{papers_list}

**文献内容：**
{context}

**用户问题：**
{question}

**请基于上述文献内容回答（如果文献中没有答案，请明确说明）：**
"""
    
    def _build_prompt(self, question: str, context: str) -> str:
        """构建标准提示词（向后兼容）"""
        return self._build_anti_hallucination_prompt(question, context, [])

    def _generate_with_huggingface(self, prompt: str) -> str:
        """使用 Hugging Face Inference API 生成答案"""
        try:
            headers = {}
            if self.hf_api_key:
                headers['Authorization'] = f'Bearer {self.hf_api_key}'
            
            payload = {
                'inputs': prompt,
                'parameters': {
                    'temperature': 0.7,
                    'max_new_tokens': 2000,
                    'return_full_text': False
                }
            }
            
            response = requests.post(
                self.hf_api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # 处理不同的响应格式
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        return result[0]['generated_text']
                    elif isinstance(result[0], dict) and 'text' in result[0]:
                        return result[0]['text']
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text']
                return str(result)
            else:
                error_msg = f"HF API 错误 {response.status_code}: {response.text}"
                logger.error(error_msg)
                # 如果模型正在加载，等待后重试一次
                if response.status_code == 503:
                    logger.info("模型正在加载，等待 10 秒后重试...")
                    import time
                    time.sleep(10)
                    return self._generate_with_huggingface(prompt)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Hugging Face API 调用失败: {e}")
            # 提取问题用于模板生成
            question = prompt.split('**用户问题：**')[-1].split('**请基于')[0].strip() if '**用户问题：**' in prompt else ''
            context = prompt.split('**文献内容：**')[1].split('**用户问题：**')[0].strip() if '**文献内容：**' in prompt else ''
            return self._generate_template(question, context)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """使用 OpenAI 生成答案"""
        try:
            system_prompt = '你是一个学术文献分析助手。必须严格基于用户提供的文献内容回答问题，不能添加文献中没有的信息。'
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.3,  # 降低温度以减少幻觉
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            question = prompt.split('**用户问题：**')[-1].split('**请基于')[0].strip() if '**用户问题：**' in prompt else ''
            context = prompt.split('**文献内容：**')[1].split('**用户问题：**')[0].strip() if '**文献内容：**' in prompt else ''
            return self._generate_template(question, context)
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """使用 Ollama 生成答案"""
        try:
            system_prompt = '你是一个学术文献分析助手。必须严格基于用户提供的文献内容回答问题，不能添加文献中没有的信息。'
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.3,  # 降低温度以减少幻觉
                    'num_predict': 2000
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama 调用失败: {e}")
            question = prompt.split('**用户问题：**')[-1].split('**请基于')[0].strip() if '**用户问题：**' in prompt else ''
            context = prompt.split('**文献内容：**')[1].split('**用户问题：**')[0].strip() if '**文献内容：**' in prompt else ''
            return self._generate_template(question, context)
    
    def _generate_template(self, question: str, context: str) -> str:
        """使用模板生成答案（当没有 LLM 时）"""
        return f"""根据文献内容，关于"{question}"的信息如下：

{context[:1000]}{'...' if len(context) > 1000 else ''}

注意：这是基于检索到的文献片段生成的简单回答。为了获得更准确和完整的答案，建议安装并配置 LLM（OpenAI 或 Ollama）。"""
    
    def _generate_template_review(self, topic: str, context: str) -> str:
        """使用模板生成综述"""
        return f"""关于"{topic}"的文献综述：

基于检索到的文献，以下是相关研究的概述：

{context[:3000]}{'...' if len(context) > 3000 else ''}

注意：这是基于检索到的文献片段生成的简单综述。为了获得更准确和完整的综述，建议安装并配置 LLM（OpenAI 或 Ollama）。"""
