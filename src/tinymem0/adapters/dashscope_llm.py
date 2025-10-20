#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashscope（阿里云）LLM适配器
提供与阿里云通义千问API交互的特定功能
"""

import os
from typing import Optional, List, Dict, Any
from dashscope import Generation


def extract_llm_response_content(response) -> Optional[str]:
    """
    从Dashscope LLM响应中提取内容
    
    Args:
        response: Dashscope API响应对象
        
    Returns:
        提取的响应内容，如果失败返回None
    """
    if not response or response.status_code != 200:
        return None
    
    # 检查响应结构，兼容不同的返回格式
    if hasattr(response.output, 'choices') and response.output.choices:
        return response.output.choices[0].message.content
    elif hasattr(response.output, 'text'):
        return response.output.text
    else:
        print("无法获取响应内容")
        return None


def call_llm_with_prompt(model: str, system_prompt: str, user_content: str) -> Optional[str]:
    """
    调用LLM并处理响应
    支持阿里云API和本地GGUF模型
    
    Args:
        model: 模型名称
        system_prompt: 系统提示词
        user_content: 用户内容
        
    Returns:
        处理后的响应内容
    """
    # 检查是否使用本地模型
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    
    if use_local:
        try:
            from utils.llm import get_local_llm
            llm = get_local_llm()
            
            # 检查是否需要JSON输出，如果是则降低温度以获得更确定性的输出
            temp = 0.3 if ('json' in system_prompt.lower() or 'JSON' in system_prompt) else 0.7
            
            return llm.chat(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_content}
                ],
                max_tokens=1024,
                temperature=temp
            )
        except Exception as e:
            print(f"本地LLM调用异常: {e}")
            return None
    else:
        # 使用阿里云API
        try:
            # 构建符合类型的消息列表
            messages: List[Dict[str, Any]] = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content}
            ]
            response = Generation.call(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                result_format='message'
            )
            
            return extract_llm_response_content(response)
        except Exception as e:
            print(f"LLM调用异常: {e}")
            return None


def handle_llm_error(response, operation_name: str = "操作"):
    """
    处理Dashscope LLM错误
    
    Args:
        response: LLM响应对象
        operation_name: 操作名称
    """
    error_msg = f"{operation_name}失败: {response.status_code if response else 'No response'}"
    print(error_msg)
    if response and hasattr(response, 'message'):
        print(f"错误信息: {response.message}")
