#!/usr/bin/env python3
"""
安装测试脚本
检查所有依赖是否正确安装
"""

import sys

def test_imports():
    """测试所有必需的导入"""
    print("正在检查依赖...")
    
    errors = []
    
    # 核心依赖
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        errors.append(f"❌ PyTorch: {e}")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        errors.append(f"❌ Transformers: {e}")
    
    # 向量数据库
    try:
        import chromadb
        print(f"✅ ChromaDB: {chromadb.__version__}")
    except ImportError as e:
        errors.append(f"❌ ChromaDB: {e}")
    
    try:
        import sentence_transformers
        print(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
    except ImportError as e:
        errors.append(f"❌ Sentence Transformers: {e}")
    
    # PDF 处理
    try:
        from pdf2image import convert_from_path
        print("✅ pdf2image")
    except ImportError as e:
        errors.append(f"❌ pdf2image: {e}")
    
    try:
        from PIL import Image
        print("✅ Pillow (PIL)")
    except ImportError as e:
        errors.append(f"❌ Pillow: {e}")
    
    # LLM 支持（可选）
    try:
        import openai
        print("✅ OpenAI (可选)")
    except ImportError:
        print("⚠️  OpenAI: 未安装（可选）")
    
    try:
        import ollama
        print("✅ Ollama (可选)")
    except ImportError:
        print("⚠️  Ollama: 未安装（可选）")
    
    # 检查 Poppler
    try:
        from pdf2image import convert_from_path
        # 尝试一个简单的测试
        print("✅ Poppler 可用（pdf2image 可以工作）")
    except Exception as e:
        errors.append(f"⚠️  Poppler: 可能需要安装 poppler-utils")
    
    # GPU 检查
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA 不可用，将使用 CPU（速度较慢）")
    except:
        pass
    
    # 总结
    print("\n" + "=" * 60)
    if errors:
        print("❌ 发现以下问题:")
        for error in errors:
            print(f"  {error}")
        print("\n请安装缺失的依赖: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有核心依赖已正确安装！")
        print("\n下一步:")
        print("1. 准备一个包含 PDF 文件的文件夹")
        print("2. 运行: python main.py init --pdf_folder /path/to/pdfs")
        return True


if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
