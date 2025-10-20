#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TinyMem0 与 LoCoMo 基准测试的适配器
用于在 LoCoMo 数据集上评估 TinyMem0 记忆系统的性能
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_system import MemorySystem
from locomo.task_eval.evaluation import eval_question_answering, f1_score, exact_match_score
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class TinyMem0LoCoMoAdapter:
    """
    TinyMem0-LoCoMo 适配器
    负责将LoCoMo对话数据输入TinyMem0记忆系统，并回答LoCoMo的QA问题
    """
    
    def __init__(self, memory_system: MemorySystem, user_id: str = "locomo_user", agent_id: str = "locomo_agent"):
        self.memory_system = memory_system
        self.user_id = user_id
        self.agent_id = agent_id
        
    def load_conversation_into_memory(self, conversation_data: Dict[str, Any]) -> None:
        """
        将LoCoMo对话数据加载到TinyMem0记忆系统中
        
        Args:
            conversation_data: LoCoMo格式的对话数据
        """
        print(f"Loading conversation {conversation_data.get('sample_id', 'unknown')} into memory...")
        
        # 获取所有session，按时间顺序处理
        sessions = []
        for key in conversation_data['conversation'].keys():
            if key.startswith('session_') and not key.endswith('_date_time'):
                session_num = int(key.split('_')[1])
                date_key = f"session_{session_num}_date_time"
                sessions.append((session_num, conversation_data['conversation'].get(date_key, ""), conversation_data['conversation'][key]))
        
        # 按session编号排序
        sessions.sort(key=lambda x: x[0])
        
        # 逐个处理每个session中的对话
        for session_num, date_time, dialogs in sessions:
            print(f"Processing session {session_num} ({date_time})...")
            
            for dialog_idx, dialog in enumerate(dialogs):
                speaker = dialog['speaker']
                text = dialog['text']
                
                # 构建完整的对话文本
                conversation_text = f"{speaker}: {text}"
                
                # 如果有图片描述，加入到对话中
                if 'blip_caption' in dialog and dialog['blip_caption']:
                    conversation_text += f" [分享了图片: {dialog['blip_caption']}]"
                
                # 构建metadata，包含session和dialog信息用于evidence追踪
                extra_metadata = {
                    'session_id': session_num,
                    'dialog_id': dialog_idx + 1,  # dialog从1开始编号
                    'date_time': date_time
                }
                
                # 将对话输入到记忆系统
                try:
                    self.memory_system.write_memory(
                        conversation=conversation_text,
                        user_id=self.user_id,
                        agent_id=self.agent_id,
                        extra_metadata=extra_metadata
                    )
                except Exception as e:
                    print(f"Error processing dialog {dialog.get('dia_id', 'unknown')}: {e}")
                    continue
    
    def answer_question(self, question: str, context_limit: int = 10) -> Tuple[str, List[str]]:
        """
        基于记忆系统回答问题
        
        Args:
            question: 要回答的问题
            context_limit: 检索的记忆数量限制
            
        Returns:
            (回答文本, 检索到的evidence列表)
        """
        try:
            # 从记忆系统中检索相关记忆
            memories = self.memory_system.search_memory(
                query=question,
                user_id=self.user_id,
                agent_id=self.agent_id,
                limit=context_limit
            )
            
            if not memories:
                return "No information available", []
            
            # 提取evidence信息（从memory的metadata中）
            retrieved_evidence = []
            for memory in memories:
                # 尝试从metadata中提取session和dialog信息
                if 'metadata' in memory:
                    meta = memory['metadata']
                    if 'session_id' in meta and 'dialog_id' in meta:
                        evidence_id = f"D{meta['session_id']}:{meta['dialog_id']}"
                        retrieved_evidence.append(evidence_id)
            
            # 构建context
            context_parts = []
            for memory in memories:
                context_parts.append(f"Memory: {memory['text']} (similarity: {memory['score']:.3f})")
            
            context = "\n".join(context_parts)
            
            # 使用LLM回答问题
            from util import call_llm_with_prompt
            
            qa_prompt = f"""Based on the following memory records, answer the question as accurately and concisely as possible. If the answer cannot be determined from the given memories, respond with "No information available".

Memory Records:
{context}

Question: {question}
Answer:"""
            
            answer = call_llm_with_prompt(
                self.memory_system.llm_model,
                "You are a helpful assistant that answers questions based on given memory records.",
                qa_prompt
            )
            
            return (answer.strip() if answer else "No information available", retrieved_evidence)
            
        except Exception as e:
            print(f"Error answering question '{question}': {e}")
            return "No information available", []
    
    def evaluate_qa_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个LoCoMo样本的QA任务
        
        Args:
            sample: LoCoMo样本数据
            
        Returns:
            包含预测结果和评估指标的字典
        """
        # 先加载对话到记忆系统
        self.load_conversation_into_memory(sample)
        
        # 回答所有问题
        results = []
        for qa_item in sample['qa']:
            question = qa_item['question']
            ground_truth = str(qa_item['answer'])
            category = qa_item.get('category', 1)
            gold_evidence = qa_item.get('evidence', [])
            
            # 获取模型回答和检索的evidence
            prediction, retrieved_evidence = self.answer_question(question)
            
            # 计算QA评估指标
            if category in [2, 3, 4]:  # single-hop, temporal, open-domain
                f1 = f1_score(prediction, ground_truth)
            elif category == 1:  # multi-hop
                f1 = self._multi_answer_f1(prediction, ground_truth)
            elif category == 5:  # adversarial
                f1 = 1.0 if 'no information available' in prediction.lower() or 'not mentioned' in prediction.lower() else 0.0
            else:
                f1 = f1_score(prediction, ground_truth)
            
            em = exact_match_score(prediction, ground_truth)
            
            # 计算Evidence评估指标
            evidence_metrics = self._calculate_evidence_metrics(retrieved_evidence, gold_evidence)
            recall_at_5 = self._calculate_recall_at_k(retrieved_evidence, gold_evidence, k=5)
            recall_at_10 = self._calculate_recall_at_k(retrieved_evidence, gold_evidence, k=10)
            mrr = self._calculate_mrr(retrieved_evidence, gold_evidence)
            
            result_item = {
                'question': question,
                'answer': ground_truth,
                'prediction': prediction,
                'category': category,
                'f1_score': f1,
                'exact_match': em,
                'evidence': gold_evidence,
                'retrieved_evidence': retrieved_evidence,
                # Evidence评估指标
                'evidence_precision': evidence_metrics['precision'],
                'evidence_recall': evidence_metrics['recall'],
                'evidence_f1': evidence_metrics['f1'],
                'recall_at_5': recall_at_5,
                'recall_at_10': recall_at_10,
                'mrr': mrr
            }
            results.append(result_item)
        
        return {
            'sample_id': sample.get('sample_id', 'unknown'),
            'qa_results': results
        }
    
    def _multi_answer_f1(self, prediction: str, ground_truth: str) -> float:
        """
        计算多答案的F1分数（用于category 1的问题）
        """
        from locomo.task_eval.evaluation import f1
        return f1(prediction, ground_truth)
    
    def _calculate_evidence_metrics(self, retrieved_evidence: List[str], gold_evidence: List[str]) -> Dict[str, float]:
        """
        计算Evidence Precision, Recall, F1
        
        Args:
            retrieved_evidence: 系统检索出的evidence列表
            gold_evidence: 标准答案的evidence列表
            
        Returns:
            包含precision, recall, f1的字典
        """
        if not retrieved_evidence or not gold_evidence:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 转换为集合以便计算交集
        retrieved_set = set(retrieved_evidence)
        gold_set = set(gold_evidence)
        
        # 计算交集
        common = retrieved_set & gold_set
        
        if len(common) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 计算precision和recall
        precision = len(common) / len(retrieved_set)
        recall = len(common) / len(gold_set)
        
        # 计算F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_recall_at_k(self, retrieved_evidence: List[str], gold_evidence: List[str], k: int = 5) -> float:
        """
        计算Recall@K：前K条检索结果中是否包含gold evidence
        
        Args:
            retrieved_evidence: 系统检索出的evidence列表（按相似度排序）
            gold_evidence: 标准答案的evidence列表
            k: 取前K条
            
        Returns:
            Recall@K分数（0或1）
        """
        if not gold_evidence:
            return 1.0  # 如果没有gold evidence，默认为1
        
        if not retrieved_evidence:
            return 0.0
        
        # 取前K条
        top_k = retrieved_evidence[:k]
        top_k_set = set(top_k)
        gold_set = set(gold_evidence)
        
        # 检查是否有交集
        return 1.0 if len(top_k_set & gold_set) > 0 else 0.0
    
    def _calculate_mrr(self, retrieved_evidence: List[str], gold_evidence: List[str]) -> float:
        """
        计算MRR（Mean Reciprocal Rank）：gold evidence在检索结果中的排名倒数
        
        Args:
            retrieved_evidence: 系统检索出的evidence列表（按相似度排序）
            gold_evidence: 标准答案的evidence列表
            
        Returns:
            MRR分数
        """
        if not gold_evidence or not retrieved_evidence:
            return 0.0
        
        gold_set = set(gold_evidence)
        
        # 找到第一个命中的位置
        for rank, evidence in enumerate(retrieved_evidence, start=1):
            if evidence in gold_set:
                return 1.0 / rank
        
        return 0.0  # 没有命中
    


class LoCoMoEvaluator:
    """
    LoCoMo基准测试评估器
    """
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.samples = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载LoCoMo数据"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_tinymem0(self, output_file: str, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        使用TinyMem0评估LoCoMo基准测试
        
        Args:
            output_file: 结果输出文件路径
            sample_ids: 要评估的样本ID列表（None表示评估所有样本）
            
        Returns:
            评估结果统计
        """
        # 初始化TinyMem0记忆系统
        memory_system = MemorySystem(
            log_level="info",
            log_mode="plain"
        )
        
        # 创建适配器
        adapter = TinyMem0LoCoMoAdapter(memory_system)
        
        # 筛选要评估的样本
        eval_samples = self.samples
        if sample_ids:
            eval_samples = [s for s in self.samples if s.get('sample_id') in sample_ids]
        
        print(f"Evaluating {len(eval_samples)} samples...")
        
        all_results = []
        all_f1_scores = []
        all_em_scores = []
        all_evidence_precision = []
        all_evidence_recall = []
        all_evidence_f1 = []
        all_recall_at_5 = []
        all_recall_at_10 = []
        all_mrr = []
        category_stats = {}
        
        for sample in tqdm(eval_samples, desc="Evaluating samples"):
            try:
                # 为每个样本创建新的记忆系统实例（避免样本间干扰）
                sample_id = sample.get('sample_id', 'unknown')
                sample_memory_system = MemorySystem(
                    collection_name=f"memories_{sample_id}",
                    log_level="warn",  # 减少日志输出
                    qdrant_path=f"./qdrant_data_{sample_id}"  # 为每个样本使用独立的数据目录
                )
                sample_adapter = TinyMem0LoCoMoAdapter(sample_memory_system)
                
                # 评估样本
                result = sample_adapter.evaluate_qa_sample(sample)
                all_results.append(result)
                
                # 收集统计信息
                for qa_result in result['qa_results']:
                    all_f1_scores.append(qa_result['f1_score'])
                    all_em_scores.append(qa_result['exact_match'])
                    all_evidence_precision.append(qa_result['evidence_precision'])
                    all_evidence_recall.append(qa_result['evidence_recall'])
                    all_evidence_f1.append(qa_result['evidence_f1'])
                    all_recall_at_5.append(qa_result['recall_at_5'])
                    all_recall_at_10.append(qa_result['recall_at_10'])
                    all_mrr.append(qa_result['mrr'])
                    
                    category = qa_result['category']
                    if category not in category_stats:
                        category_stats[category] = {
                            'f1_scores': [], 'em_scores': [], 
                            'evidence_precision': [], 'evidence_recall': [], 'evidence_f1': [],
                            'recall_at_5': [], 'recall_at_10': [], 'mrr': [],
                            'count': 0
                        }
                    
                    category_stats[category]['f1_scores'].append(qa_result['f1_score'])
                    category_stats[category]['em_scores'].append(qa_result['exact_match'])
                    category_stats[category]['evidence_precision'].append(qa_result['evidence_precision'])
                    category_stats[category]['evidence_recall'].append(qa_result['evidence_recall'])
                    category_stats[category]['evidence_f1'].append(qa_result['evidence_f1'])
                    category_stats[category]['recall_at_5'].append(qa_result['recall_at_5'])
                    category_stats[category]['recall_at_10'].append(qa_result['recall_at_10'])
                    category_stats[category]['mrr'].append(qa_result['mrr'])
                    category_stats[category]['count'] += 1
                
            except Exception as e:
                print(f"Error evaluating sample {sample.get('sample_id', 'unknown')}: {e}")
                continue
        
        # 计算总体统计
        overall_stats = {
            'total_questions': len(all_f1_scores),
            'average_f1': sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0,
            'average_em': sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0,
            'average_evidence_precision': sum(all_evidence_precision) / len(all_evidence_precision) if all_evidence_precision else 0,
            'average_evidence_recall': sum(all_evidence_recall) / len(all_evidence_recall) if all_evidence_recall else 0,
            'average_evidence_f1': sum(all_evidence_f1) / len(all_evidence_f1) if all_evidence_f1 else 0,
            'average_recall_at_5': sum(all_recall_at_5) / len(all_recall_at_5) if all_recall_at_5 else 0,
            'average_recall_at_10': sum(all_recall_at_10) / len(all_recall_at_10) if all_recall_at_10 else 0,
            'average_mrr': sum(all_mrr) / len(all_mrr) if all_mrr else 0,
        }
        
        # 计算分类别统计
        for category, stats in category_stats.items():
            stats['average_f1'] = sum(stats['f1_scores']) / len(stats['f1_scores']) if stats['f1_scores'] else 0
            stats['average_em'] = sum(stats['em_scores']) / len(stats['em_scores']) if stats['em_scores'] else 0
            stats['average_evidence_precision'] = sum(stats['evidence_precision']) / len(stats['evidence_precision']) if stats['evidence_precision'] else 0
            stats['average_evidence_recall'] = sum(stats['evidence_recall']) / len(stats['evidence_recall']) if stats['evidence_recall'] else 0
            stats['average_evidence_f1'] = sum(stats['evidence_f1']) / len(stats['evidence_f1']) if stats['evidence_f1'] else 0
            stats['average_recall_at_5'] = sum(stats['recall_at_5']) / len(stats['recall_at_5']) if stats['recall_at_5'] else 0
            stats['average_recall_at_10'] = sum(stats['recall_at_10']) / len(stats['recall_at_10']) if stats['recall_at_10'] else 0
            stats['average_mrr'] = sum(stats['mrr']) / len(stats['mrr']) if stats['mrr'] else 0
        
        # 保存结果
        output_data = {
            'evaluation_time': datetime.now().isoformat(),
            'data_file': self.data_file,
            'total_samples': len(eval_samples),
            'overall_stats': overall_stats,
            'category_stats': category_stats,
            'detailed_results': all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # 打印统计信息
        print(f"\n=== TinyMem0 LoCoMo Evaluation Results ===")
        print(f"Total Questions: {overall_stats['total_questions']}")
        print(f"\n## QA Metrics ##")
        print(f"Overall F1 Score: {overall_stats['average_f1']:.4f}")
        print(f"Overall Exact Match: {overall_stats['average_em']:.4f}")
        print(f"\n## Evidence/Retrieval Metrics ##")
        print(f"Evidence Precision: {overall_stats['average_evidence_precision']:.4f}")
        print(f"Evidence Recall: {overall_stats['average_evidence_recall']:.4f}")
        print(f"Evidence F1: {overall_stats['average_evidence_f1']:.4f}")
        print(f"Recall@5: {overall_stats['average_recall_at_5']:.4f}")
        print(f"Recall@10: {overall_stats['average_recall_at_10']:.4f}")
        print(f"MRR: {overall_stats['average_mrr']:.4f}")
        print(f"\n## By Category ##")
        for category, stats in category_stats.items():
            print(f"Category {category}: F1={stats['average_f1']:.4f}, EM={stats['average_em']:.4f}, Ev-F1={stats['average_evidence_f1']:.4f}, MRR={stats['average_mrr']:.4f} ({stats['count']} questions)")
        
        print(f"\nDetailed results saved to: {output_file}")
        
        return output_data

def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyMem0 on LoCoMo benchmark")
    parser.add_argument('--data-file', type=str, default='locomo/data/locomo10.json',
                       help='Path to LoCoMo data file')
    parser.add_argument('--output-file', type=str, default='tinymem0_locomo_results.json',
                       help='Output file for evaluation results')
    parser.add_argument('--sample-ids', type=str, nargs='*',
                       help='Specific sample IDs to evaluate (default: all)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # 检查必要的环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY environment variable is required")
        return
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found")
        return
    
    # 创建评估器
    evaluator = LoCoMoEvaluator(args.data_file)
    
    # 限制评估样本数量（用于测试）
    sample_ids = args.sample_ids
    if args.num_samples and not sample_ids:
        sample_ids = [s['sample_id'] for s in evaluator.samples[:args.num_samples]]
    
    # 运行评估
    results = evaluator.evaluate_tinymem0(args.output_file, sample_ids)
    
    return results

if __name__ == "__main__":
    main()