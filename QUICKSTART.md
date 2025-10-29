# 快速开始指南

## 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 如果需要 GPU 加速，安装 PyTorch CUDA 版本
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

## 2. 安装 Poppler（PDF 转换需要）

### macOS
```bash
brew install poppler
```

### Ubuntu/Debian
```bash
sudo apt-get install poppler-utils
```

### Windows
下载并安装 [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)

## 3. 配置环境变量（可选）

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
# 编辑 .env 文件，添加你的 API 密钥（如果需要）
```

## 4. 使用示例

### 初始化文献库

将 PDF 文件放入一个文件夹，然后初始化数据库：

```bash
python main.py init --pdf_folder /path/to/your/pdfs --db_path ./literature_db
```

### 查询问题

```bash
python main.py query "什么是 transformer 架构？"
```

### 生成文献综述

```bash
python main.py review --topic "深度学习在计算机视觉中的应用" --output review.md
```

### 搜索相关论文

```bash
python main.py search "神经网络"
```

### 查看所有论文

```bash
python main.py list
```

### 查看统计信息

```bash
python main.py stats
```

### 交互式问答

```bash
python main.py chat
```

## 5. 常见问题

### Q: 模型下载失败怎么办？
A: DeepSeek-OCR 模型会从 Hugging Face 自动下载，如果网络问题，可以手动下载到本地。

### Q: 内存不足怎么办？
A: 可以减小 `chunk_size` 参数，或在 `vector_db.py` 中调整处理参数。

### Q: PDF 处理很慢怎么办？
A: 
- 确保使用 GPU（如果可用）
- 减小 `image_size` 参数（默认 640）
- 批量处理可以中断后继续

### Q: 如何添加新的 PDF？
A: 只需要再次运行 `init` 命令，系统会自动检测新文件并更新数据库。

## 6. 高级用法

### 自定义嵌入模型

在 `src/vector_db.py` 中修改：

```python
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
```

### 调整文本分块大小

在 `src/vector_db.py` 的 `add_documents` 方法中修改：

```python
chunk_size=500,  # 增大以保留更多上下文
chunk_overlap=100  # 重叠大小
```

### 使用不同的 LLM

编辑 `src/rag_system.py` 或在 `.env` 中配置模型名称。
