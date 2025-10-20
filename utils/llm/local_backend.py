#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本地LLM后端模块
支持使用llama-cpp-python加载GGUF模型文件
提供本地大语言模型的推理接口
"""

import os
from typing import Optional, List, Dict
from llama_cpp import Llama


class LocalLLM:
    """本地GGUF模型加载器和推理引擎"""
    
    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 4096, n_gpu_layers: int = -1, verbose: bool = False):
        """
        初始化本地LLM
        
        Args:
            model_path: GGUF模型文件路径，如果为None则从环境变量LOCAL_MODEL_PATH读取
            n_ctx: 上下文窗口大小
            n_gpu_layers: 使用GPU的层数，-1表示全部使用GPU
            verbose: 是否输出详细日志
        """
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH")
        
        if not self.model_path:
            raise ValueError("请设置model_path参数或LOCAL_MODEL_PATH环境变量")
        
        if not os.path.exists(self.model_path):
            # 尝试在models目录下查找
            models_dir = "models"
            if os.path.exists(models_dir):
                potential_path = os.path.join(models_dir, self.model_path)
                if os.path.exists(potential_path):
                    self.model_path = potential_path
                else:
                    raise FileNotFoundError(f"找不到模型文件: {self.model_path}")
            else:
                raise FileNotFoundError(f"找不到模型文件: {self.model_path}")
        
        print(f"正在加载模型: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose
        )
        print("模型加载完成")
    
    def generate(self, 
                 system_prompt: str, 
                 user_content: str, 
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 stop: Optional[List[str]] = None) -> Optional[str]:
        """
        生成回复
        
        Args:
            system_prompt: 系统提示词
            user_content: 用户输入内容
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            stop: 停止词列表
            
        Returns:
            生成的文本
        """
        try:
            # 如果system_prompt包含JSON要求，添加额外的JSON约束
            enhanced_system = system_prompt
            if 'json' in system_prompt.lower() or 'JSON' in system_prompt:
                enhanced_system = system_prompt + "\n\nIMPORTANT: You MUST respond with valid JSON only. Do not include any explanatory text before or after the JSON object."
            
            # 构建完整的prompt
            prompt = f"""<|im_start|>system
{enhanced_system}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["<|im_end|>"],
                echo=False
            )
            
            if response and 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['text'].strip()
            return None
            
        except Exception as e:
            print(f"LLM生成异常: {e}")
            return None
    
    def chat(self, messages: List[Dict[str, str]], 
             max_tokens: int = 512,
             temperature: float = 0.7,
             top_p: float = 0.9) -> Optional[str]:
        """
        聊天接口
        
        Args:
            messages: 消息列表，每个消息包含role和content
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            
        Returns:
            生成的文本
        """
        try:
            # 增强JSON输出约束：如果system消息包含JSON要求，添加额外约束
            enhanced_messages = messages.copy()
            if enhanced_messages and enhanced_messages[0]['role'] == 'system':
                system_content = enhanced_messages[0]['content']
                if 'json' in system_content.lower() or 'JSON' in system_content:
                    # 添加更强的JSON输出约束
                    enhanced_messages[0]['content'] = system_content + "\n\nCRITICAL: You MUST respond ONLY with valid JSON. Start with { and end with }. Do NOT include any text before or after the JSON object. Do NOT add explanations or comments."
            
            response = self.llm.create_chat_completion(
                messages=enhanced_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            if response and 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['message']['content'].strip()
            return None
            
        except Exception as e:
            print(f"LLM聊天异常: {e}")
            return None


# 全局LLM实例（单例模式）
_llm_instance: Optional[LocalLLM] = None


def get_local_llm(model_path: Optional[str] = None, 
                  n_ctx: int = 4096, 
                  n_gpu_layers: int = -1) -> LocalLLM:
    """
    获取全局LLM实例（单例模式）
    避免重复加载模型，节省内存
    
    Args:
        model_path: 模型路径
        n_ctx: 上下文窗口
        n_gpu_layers: GPU层数
        
    Returns:
        LocalLLM实例
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )
    return _llm_instance
