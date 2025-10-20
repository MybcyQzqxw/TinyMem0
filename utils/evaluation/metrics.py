#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评测相关工具函数
提供各种评测指标计算函数
"""

from typing import List, Set, Dict


def calculate_f1(precision: float, recall: float) -> float:
    """
    计算F1分数
    
    Args:
        precision: 精确率
        recall: 召回率
        
    Returns:
        F1分数
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_precision_recall(retrieved: Set[str], 
                               relevant: Set[str]) -> Dict[str, float]:
    """
    计算精确率和召回率
    
    Args:
        retrieved: 检索到的项目集合
        relevant: 相关的项目集合
        
    Returns:
        包含precision, recall, f1的字典
    """
    if not retrieved:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not relevant:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # 计算交集
    true_positives = len(retrieved & relevant)
    
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(relevant) if relevant else 0.0
    f1 = calculate_f1(precision, recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_mrr(retrieved_list: List[str], 
                  relevant_set: Set[str]) -> float:
    """
    计算平均倒数排名 (Mean Reciprocal Rank)
    
    Args:
        retrieved_list: 检索结果列表（有序）
        relevant_set: 相关项目集合
        
    Returns:
        MRR分数
    """
    for rank, item in enumerate(retrieved_list, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def calculate_recall_at_k(retrieved_list: List[str], 
                          relevant_set: Set[str], 
                          k: int) -> float:
    """
    计算Recall@K
    
    Args:
        retrieved_list: 检索结果列表（有序）
        relevant_set: 相关项目集合
        k: 前K个结果
        
    Returns:
        Recall@K分数
    """
    if not relevant_set:
        return 0.0
    
    # 取前K个结果
    top_k = set(retrieved_list[:k])
    
    # 计算召回率
    true_positives = len(top_k & relevant_set)
    recall = true_positives / len(relevant_set)
    
    return recall


def normalize_text(text: str) -> str:
    """
    标准化文本，用于比较
    
    Args:
        text: 输入文本
        
    Returns:
        标准化后的文本
    """
    import re
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 转小写
    text = text.lower().strip()
    return text
