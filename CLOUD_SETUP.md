# 云端模式设置指南

本系统现在支持**完全云端运行**，无需本地 GPU 或模型！

## 🚀 快速开始（云端模式）

### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 Poppler（必需）

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

### 3. 获取 Hugging Face API Key（免费）

1. 访问 https://huggingface.co
2. 注册/登录账号
3. 访问 https://huggingface.co/settings/tokens
4. 创建新的 Access Token（选择 "Read" 权限即可）

### 4. 配置环境变量

创建 `.env` 文件：

```env
HUGGINGFACE_API_KEY=your_token_here
```

### 5. 开始使用

```bash
# 使用云端 API 处理 PDF
python main.py init --pdf_folder ./papers --use_hf_api

# 查询问题（自动使用 Hugging Face API）
python main.py query "什么是 transformer？"

# 交互式问答
python main.py chat
```

## ✨ 云端模式优势

- ✅ **无需本地 GPU**：所有计算在云端完成
- ✅ **无需下载模型**：不需要几GB的本地模型文件
- ✅ **所有人都可以使用**：只要有 Hugging Face 账号即可
- ✅ **免费使用**：Hugging Face Inference API 免费层足够大多数使用
- ✅ **严格防幻觉**：所有回答只基于您提供的文献内容

## 🔒 防幻觉机制

系统采用多层防幻觉机制：

1. **相关性过滤**：自动过滤低相关性文档（默认阈值 0.7）
2. **严格提示词**：明确告诉模型只能使用提供的文献内容
3. **来源标注**：每个回答都明确标注来源论文
4. **置信度显示**：显示答案的置信度等级
5. **明确告知**：如果文献中没有答案，明确说明而非猜测

## 📊 成本说明

- Hugging Face Inference API：
  - 免费层：每月充足的请求次数
  - 适合个人和小团队使用
  - 超出免费额后按使用量付费

## 🔧 本地模式（可选）

如果您有 GPU 且想要离线使用，可以不使用 `--use_hf_api` 参数，系统会使用本地模型（需要首次下载模型）。

## 常见问题

**Q: API 调用失败怎么办？**
A: 检查网络连接和 API Key 是否正确。某些模型可能需要等待加载。

**Q: 处理速度慢怎么办？**
A: 云端 API 第一次调用可能需要等待模型加载（30-60秒），后续会快很多。

**Q: 如何更换模型？**
A: 在代码中修改 `hf_model` 参数，或使用支持的其他模型。
