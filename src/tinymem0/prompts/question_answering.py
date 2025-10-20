#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问答系统的Prompt定义
用于基于记忆记录回答用户问题
"""

# QA系统的System Prompt
QA_SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on given memory records."

def build_qa_prompt(memory_context: str, question: str) -> str:
    """
    构建问答的用户Prompt
    
    Args:
        memory_context: 记忆记录的上下文
        question: 用户问题
        
    Returns:
        完整的QA Prompt
    """
    return f"""Based on the following memory records, answer the question as accurately and concisely as possible. If the answer cannot be determined from the given memories, respond with "No information available".

Memory Records:
{memory_context}

Question: {question}
Answer:"""
