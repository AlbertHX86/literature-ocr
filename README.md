# 文献总结与分析系统

基于 DeepSeek-OCR 的学术文献处理、检索和分析工具。

## 功能特性

- 📄 **PDF 处理**: 使用 DeepSeek-OCR 提取 PDF 中的文本、表格和图像
- 🔍 **智能检索**: 基于向量数据库的语义搜索
- 📝 **文献综述**: 自动生成文献综述
- 💬 **问答系统**: 对文献库进行自然语言问答

## 安装

### 方式一：云端模式（推荐，无需本地模型）

1. 安装依赖:

```bash
pip install -r requirements.txt
```

2. 安装 Poppler（PDF 转换必需）:

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:** 下载并安装 [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)

3. 配置 Hugging Face API Key（免费）:

- 访问 https://huggingface.co/settings/tokens
- 创建 Access Token
- 在 `.env` 文件中设置 `HUGGINGFACE_API_KEY`

4. 验证安装:

```bash
python test_installation.py
```

### 方式二：本地模式（需要 GPU 和本地模型）

1. 安装依赖（同上）
2. 安装 DeepSeek-OCR 模型（首次运行会自动下载，约几GB）
3. 安装 flash-attention (可选，用于加速):

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

## 使用方法

### 1. 初始化文献库

**云端模式（推荐，无需本地模型）:**
```bash
python main.py init --pdf_folder /path/to/pdfs --db_path ./literature_db --use_hf_api
```

**本地模式:**
```bash
python main.py init --pdf_folder /path/to/pdfs --db_path ./literature_db
```

### 2. 查询文献

```bash
python main.py query "transformer architecture in computer vision"
```

### 3. 生成文献综述

```bash
python main.py review --topic "deep learning" --output review.md
```

### 4. 查找相关文献

```bash
python main.py search "quantum computing applications"
```

### 5. 交互式问答

```bash
python main.py chat
```

## 项目结构

```
literature summary/
├── main.py                 # 主程序入口
├── src/
│   ├── pdf_processor.py   # PDF 处理模块
│   ├── vector_db.py       # 向量数据库模块
│   ├── rag_system.py      # RAG 问答系统
│   └── utils.py           # 工具函数
├── requirements.txt       # 依赖列表
└── README.md             # 说明文档
```

## 配置

创建 `.env` 文件（从 `.env.example` 复制）:

```env
# 推荐：使用 Hugging Face API（免费）
HUGGINGFACE_API_KEY=your_huggingface_token_here

# 或者使用 OpenAI API（付费）
# OPENAI_API_KEY=your_openai_key_here
```

获取 Hugging Face API Key: https://huggingface.co/settings/tokens

## 注意事项

### 防幻觉机制

- **所有回答严格基于您提供的文献内容**，不会使用外部知识
- 系统会检查文档相关性，过滤低相关性内容
- 如果文献中没有相关信息，会明确告知
- 每个答案都会标注来源文献

### 推荐配置

- **云端模式**（`--use_hf_api`）：无需本地 GPU，所有人都可以使用
- 使用 Hugging Face Inference API（免费）
- 首次使用需要处理 PDF（会调用 API，可能需要一些时间）

### 其他注意事项

- 本地模式首次使用需要下载 DeepSeek-OCR 模型（约几GB）
- 本地模式建议使用 GPU 加速处理
- PDF 处理时间取决于文档大小和数量
- 确保已安装 Poppler 库（pdf2image 的依赖）

## 详细文档

更多使用说明请参考 [QUICKSTART.md](QUICKSTART.md)

## 项目参考

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
