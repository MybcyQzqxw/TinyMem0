#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型下载工具
提供从各种源下载预训练模型的通用功能
"""

import os
import sys
from typing import Optional


def download_embedding_model(model_id: str = 'AI-ModelScope/bge-small-zh-v1.5', 
                             cache_dir: str = './embedding_models',
                             source: str = 'modelscope') -> str:
    """
    下载嵌入模型（通用函数）
    
    支持从多个源下载模型：
    - ModelScope: 中国区友好，适合下载中文模型
    - HuggingFace: 国际主流模型库
    
    Args:
        model_id: 模型ID（格式依赖于source）
        cache_dir: 本地缓存目录
        source: 下载源 ('modelscope' 或 'huggingface')
        
    Returns:
        下载后的模型本地路径
        
    Raises:
        ImportError: 缺少必要的依赖包
        RuntimeError: 下载失败
        
    Examples:
        >>> # 从ModelScope下载
        >>> path = download_embedding_model('AI-ModelScope/bge-small-zh-v1.5')
        
        >>> # 从HuggingFace下载
        >>> path = download_embedding_model(
        ...     'sentence-transformers/all-MiniLM-L6-v2',
        ...     source='huggingface'
        ... )
    """
    # 检查模型是否已存在
    model_path = os.path.join(cache_dir, model_id)
    if os.path.exists(model_path):
        print(f"✅ 模型已存在: {model_path}")
        return model_path
    
    print(f"📥 开始下载模型: {model_id}")
    print(f"📁 下载目录: {cache_dir}")
    print(f"🌐 下载源: {source}")
    
    try:
        if source == 'modelscope':
            return _download_from_modelscope(model_id, cache_dir)
        elif source == 'huggingface':
            return _download_from_huggingface(model_id, cache_dir)
        else:
            raise ValueError(f"不支持的下载源: {source}")
            
    except ImportError as e:
        print(f"❌ 错误: 缺少必要的依赖包")
        print(f"详细信息: {e}")
        if source == 'modelscope':
            print("请运行: pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
        elif source == 'huggingface':
            print("请运行: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise RuntimeError(f"模型下载失败: {e}")


def _download_from_modelscope(model_id: str, cache_dir: str) -> str:
    """从ModelScope下载模型（内部函数）"""
    from modelscope import snapshot_download
    
    downloaded_path = snapshot_download(
        model_id=model_id,
        cache_dir=cache_dir,
        revision='master'
    )
    
    print(f"✅ 模型下载完成: {downloaded_path}")
    return downloaded_path


def _download_from_huggingface(model_id: str, cache_dir: str) -> str:
    """从HuggingFace下载模型（内部函数）"""
    from huggingface_hub import snapshot_download
    
    downloaded_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir=os.path.join(cache_dir, model_id)
    )
    
    print(f"✅ 模型下载完成: {downloaded_path}")
    return downloaded_path


def download_llm_model(model_id: str,
                      cache_dir: str = './models',
                      model_type: str = 'gguf') -> str:
    """
    下载LLM模型（GGUF或其他格式）
    
    Args:
        model_id: 模型ID
        cache_dir: 缓存目录
        model_type: 模型类型 ('gguf', 'pytorch', etc.)
        
    Returns:
        下载后的模型路径
    """
    # 未来可以扩展支持直接下载GGUF模型
    # 当前作为占位符
    raise NotImplementedError("LLM模型下载功能即将推出")


def check_model_exists(model_id: str, cache_dir: str) -> bool:
    """
    检查模型是否已下载
    
    Args:
        model_id: 模型ID
        cache_dir: 缓存目录
        
    Returns:
        True if exists, False otherwise
    """
    model_path = os.path.join(cache_dir, model_id)
    return os.path.exists(model_path)
