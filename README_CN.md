# 文献总结与分析系统

基于 DeepSeek-OCR 的学术文献处理、检索和分析工具，支持从 PDF 文件中提取文本，构建向量数据库，并进行智能问答和文献综述生成。

## ✨ 功能特性

- 📄 **PDF 处理**: 使用 DeepSeek-OCR 提取 PDF 中的文本、表格和图像
- 🔍 **智能检索**: 基于向量数据库的语义搜索，快速找到相关文献
- 📝 **文献综述**: 自动生成指定主题的学术文献综述
- 💬 **问答系统**: 对文献库进行自然语言问答，严格基于文献内容
- 🔒 **防幻觉机制**: 所有回答仅基于您提供的文献，不会编造信息

## 📋 系统要求

- Python 3.8 或更高版本
- 操作系统：macOS、Linux 或 Windows
- 内存：建议 8GB 以上（本地模式）
- 存储空间：本地模式需要约 10GB（用于存储模型）

## 🚀 快速开始

### 方式一：本地模式（推荐用于生产环境）

#### 1. 克隆仓库

```bash
git clone https://github.com/AlbertHX86/literature-ocr.git
cd literature-ocr
```

#### 2. 安装系统依赖

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**Windows:**
下载并安装 [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)

#### 3. 安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# DeepSeek-OCR 额外依赖
pip install addict easydict
```

#### 4. 验证安装

```bash
python test_installation.py
```

#### 5. 配置环境变量（可选）

创建 `.env` 文件：

```env
# Hugging Face API Key（用于 RAG 问答）
HUGGINGFACE_API_KEY=your_token_here

# 或者使用 OpenAI API
# OPENAI_API_KEY=your_openai_key_here
```

**获取 Hugging Face API Key:**
1. 访问 https://huggingface.co
2. 注册/登录账号
3. 访问 https://huggingface.co/settings/tokens
4. 创建新的 Access Token（选择 "Read" 权限即可）

### 方式二：云端模式（实验性，当前不支持）

注意：DeepSeek-OCR 目前不支持标准的 Hugging Face Inference API，建议使用本地模式。

## 📖 使用指南

### 1. 初始化文献库

将您的 PDF 文件放入一个文件夹，然后运行：

```bash
python main.py init --pdf_folder /path/to/your/pdfs --db_path ./literature_db
```

**参数说明:**
- `--pdf_folder`: PDF 文件所在文件夹路径
- `--db_path`: 向量数据库保存路径（默认：`./literature_db`）
- `--output_dir`: 可选，处理结果输出目录

**示例:**
```bash
python main.py init --pdf_folder "/Users/albert/Desktop/Related paper" --db_path ./literature_db
```

**注意:**
- 首次运行会自动下载 DeepSeek-OCR 模型（约几GB），需要一些时间
- 处理时间取决于 PDF 文件数量和大小
- 建议使用 GPU 加速（如果有的话）

### 2. 查询问题

```bash
python main.py query "什么是 transformer 架构？"
```

系统会基于您提供的文献内容回答，并标注来源论文。

### 3. 生成文献综述

```bash
python main.py review --topic "深度学习在计算机视觉中的应用" --output review.md
```

**参数说明:**
- `--topic`: 综述主题
- `--output`: 输出文件路径（可选）

### 4. 搜索相关论文

```bash
python main.py search "神经网络优化方法"
```

### 5. 查看所有论文

```bash
python main.py list
```

### 6. 查看数据库统计

```bash
python main.py stats
```

### 7. 交互式问答

```bash
python main.py chat
```

进入交互模式，可以连续提问。输入 `quit` 或 `exit` 退出。

## 🔧 项目结构

```
literature-ocr/
├── main.py                 # 主程序入口
├── src/                    # 源代码目录
│   ├── pdf_processor.py   # PDF 处理模块（本地模式）
│   ├── pdf_processor_hf.py # PDF 处理模块（云端模式）
│   ├── vector_db.py       # 向量数据库模块
│   ├── rag_system.py      # RAG 问答系统
│   └── utils.py           # 工具函数
├── requirements.txt       # Python 依赖列表
├── test_installation.py   # 安装验证脚本
├── README.md             # 英文文档
├── README_CN.md          # 中文文档
├── QUICKSTART.md         # 快速开始指南
└── CLOUD_SETUP.md        # 云端设置指南
```

## ⚙️ 配置说明

### 环境变量

创建 `.env` 文件（从 `.env.example` 复制）：

```env
# Hugging Face API Key（用于 RAG 问答）
HUGGINGFACE_API_KEY=your_huggingface_token_here

# 或使用 OpenAI API（付费）
# OPENAI_API_KEY=your_openai_key_here
```

### 自定义配置

#### 调整文本分块大小

编辑 `src/vector_db.py`：

```python
# 在 add_documents 方法中
chunk_size=500,      # 增大以保留更多上下文
chunk_overlap=100    # 重叠大小
```

#### 更换嵌入模型

编辑 `src/vector_db.py`：

```python
# 在 __init__ 方法中
embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
```

推荐模型：
- `sentence-transformers/all-MiniLM-L6-v2` - 英文，速度快
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - 多语言支持

## 🔒 防幻觉机制

本系统采用多层防护机制，确保回答严格基于您提供的文献：

1. **相关性过滤**: 自动过滤低相关性文档（默认阈值 0.7）
2. **严格提示词**: 明确要求模型只使用提供的文献内容
3. **来源标注**: 每个答案都标注来源论文
4. **置信度显示**: 显示答案的置信度等级（high/medium/low）
5. **明确告知**: 如果文献中没有答案，明确说明而非猜测

## 🐛 常见问题

### Q: 模型下载失败怎么办？

**A:** 
- 检查网络连接
- 使用 VPN（如果在某些地区）
- 手动下载模型到 `~/.cache/huggingface/hub/`

### Q: 内存不足怎么办？

**A:**
- 使用云端 API（如果支持）
- 减小 `chunk_size` 参数
- 分批处理 PDF 文件
- 使用 CPU 模式（虽然较慢）

### Q: PDF 处理很慢怎么办？

**A:**
- 确保使用 GPU（如果有）
- 减小 `image_size` 参数（默认 640）
- 处理大量文件时建议分批进行

### Q: 如何添加新的 PDF？

**A:** 只需要再次运行 `init` 命令，系统会自动检测新文件并更新数据库。

### Q: transformers 版本冲突？

**A:** DeepSeek-OCR 可能需要特定版本的 transformers。尝试：

```bash
pip install transformers>=4.51.1
```

如果仍有问题，查看 DeepSeek-OCR 官方文档要求。

### Q: 本地模式需要 GPU 吗？

**A:** 不需要，但 GPU 可以显著加速处理速度。如果没有 GPU，会自动使用 CPU（速度较慢）。

## 📚 更多信息

- [快速开始指南](QUICKSTART.md)
- [云端设置指南](CLOUD_SETUP.md)
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

感谢以下项目和团队：

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) - OCR 模型
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [Sentence Transformers](https://www.sbert.net/) - 文本嵌入

---

**提示**: 如果遇到问题，请先查看 [常见问题](#常见问题) 部分，或提交 Issue。
