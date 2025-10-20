#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM通用工具模块
提供与大语言模型交互的通用工具，不依赖特定API服务
"""

from .local_backend import LocalLLM, get_local_llm
from .json_parser import parse_json_response, extract_json_from_text, safe_json_loads

__all__ = [
    'LocalLLM',
    'get_local_llm',
    'parse_json_response',
    'extract_json_from_text',
    'safe_json_loads',
]
