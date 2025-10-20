#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompts模块 - 集中管理所有系统Prompt定义

本模块包含：
- 事实提取Prompt (fact_extraction.py)
- 记忆处理Prompt (memory_processing.py)  
- 问答系统Prompt (question_answering.py)
"""

from .fact_extraction import FACT_EXTRACTION_PROMPT
from .memory_processing import MEMORY_PROCESSING_PROMPT
from .question_answering import QA_SYSTEM_PROMPT, build_qa_prompt

__all__ = [
    'FACT_EXTRACTION_PROMPT',
    'MEMORY_PROCESSING_PROMPT',
    'QA_SYSTEM_PROMPT',
    'build_qa_prompt',
]
