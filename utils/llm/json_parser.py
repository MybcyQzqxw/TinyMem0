#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON解析工具
提供通用的JSON解析和提取功能，特别是从LLM响应中解析JSON
"""

import json
import re
from typing import List, Dict, Optional, Any, Union


def parse_json_response(response_text: str, expected_key: Optional[str] = None) -> Union[List, Dict, Any]:
    """
    解析LLM的JSON响应，支持多种格式提取
    
    这是一个健壮的JSON解析器，能够处理：
    1. 标准JSON格式
    2. 包含额外文本的JSON（从文本中提取JSON）
    3. 多行JSON
    4. JSON数组
    5. 作为回退方案，提取引号内的内容
    
    Args:
        response_text: LLM返回的文本
        expected_key: 期望的JSON键名，如果指定则返回该键的值
        
    Returns:
        解析后的数据（列表、字典或其他类型）
        
    Examples:
        >>> parse_json_response('{"facts": ["fact1", "fact2"]}', "facts")
        ['fact1', 'fact2']
        
        >>> parse_json_response('Some text {"data": [1, 2]} more text')
        {'data': [1, 2]}
    """
    if not response_text:
        return []
        
    # 方法1: 尝试直接解析JSON
    try:
        data = json.loads(response_text)
        if expected_key and isinstance(data, dict):
            return data.get(expected_key, [])
        return data
    except json.JSONDecodeError:
        pass
    
    # 方法2: 查找包含特定键的JSON对象（支持多行）
    if expected_key:
        try:
            json_match = re.search(
                r'\{[^{}]*"' + expected_key + r'"[^{}]*\}', 
                response_text, 
                re.DOTALL
            )
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return data.get(expected_key, [])
        except Exception:
            pass
    
    # 方法3: 查找任意JSON对象
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            if expected_key and isinstance(data, dict):
                return data.get(expected_key, [])
            return data
    except Exception:
        pass
    
    # 方法4: 查找JSON数组
    try:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    
    # 方法5: 回退方案 - 尝试从文本中提取引号内容（针对特定键）
    if expected_key == 'facts':
        try:
            # 查找所有引号内的内容
            facts = re.findall(r'"([^"]+)"', response_text)
            if facts:
                print(f"警告: 从非JSON格式中提取了 {len(facts)} 个{expected_key}")
                return facts
        except Exception:
            pass
    
    print(f"警告: 无法找到有效JSON内容: {response_text[:200]}...")
    return []


def extract_json_from_text(text: str) -> Optional[Union[Dict, List]]:
    """
    从文本中提取第一个有效的JSON对象或数组
    
    Args:
        text: 包含JSON的文本
        
    Returns:
        提取的JSON对象/数组，如果未找到返回None
    """
    # 尝试提取JSON对象
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    
    # 尝试提取JSON数组
    try:
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    
    return None


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    安全的JSON加载，失败时返回默认值
    
    Args:
        json_str: JSON字符串
        default: 解析失败时的默认返回值
        
    Returns:
        解析的JSON对象或默认值
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default
