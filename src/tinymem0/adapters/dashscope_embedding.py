#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashscope（阿里云）Embedding适配器
提供与阿里云文本嵌入API交互的特定功能
"""

from typing import List


def extract_embedding_from_response(response) -> List[float]:
    """
    从Dashscope嵌入API响应中提取向量
    
    Args:
        response: Dashscope嵌入API响应对象
        
    Returns:
        向量嵌入列表
    """
    if not response or response.status_code != 200:
        return []
    
    # 检查响应结构，兼容不同的返回格式
    if hasattr(response.output, 'embeddings') and response.output.embeddings:
        return response.output.embeddings[0].embedding
    elif hasattr(response.output, 'data') and response.output.data:
        return response.output.data[0].embedding
    else:
        print("无法获取embedding")
        return []
