#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TinyMem0适配器模块
提供与特定API服务（如阿里云Dashscope）交互的适配器
"""

from .dashscope_llm import (
    extract_llm_response_content,
    call_llm_with_prompt,
    handle_llm_error
)

from .dashscope_embedding import (
    extract_embedding_from_response
)

__all__ = [
    # LLM适配器
    'extract_llm_response_content',
    'call_llm_with_prompt',
    'handle_llm_error',
    # Embedding适配器
    'extract_embedding_from_response',
]
