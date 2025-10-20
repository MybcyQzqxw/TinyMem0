#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记忆系统评测脚本
用于在 LoCoMo 数据集上评测各种记忆系统的性能，包括QA和Evidence检索指标

支持的模型系统：
- tinymem0: TinyMem0 记忆系统
- (未来可扩展其他模型)

使用方法：
    python scripts/evaluate_system.py --model tinymem0 --num-samples 5
"""

import json
import os
import sys
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import argparse
from importlib import import_module

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from locomo.task_eval.evaluation import eval_question_answering, f1_score, exact_match_score
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模型系统映射表
AVAILABLE_MODELS = {
    'tinymem0': {
        'name': 'TinyMem0',
        'description': 'TinyMem0 记忆系统 (支持本地LLM和阿里云API)',
        'module': 'tinymem0',
        'class': 'MemorySystem'
    },
    # 未来可以添加更多模型，例如：
    # 'mem0ai': {
    #     'name': 'Mem0.ai',
    #     'description': 'Mem0.ai 官方实现',
    #     'module': 'mem0ai',
    #     'class': 'MemorySystem'
    # },
}

class MemorySystemEvaluator:
    """
    记忆系统评测器
    负责将对话数据输入记忆系统，并评估QA和Evidence检索性能
    """
    
    def __init__(self, memory_system: Any, user_id: str = "eval_user", agent_id: str = "eval_agent"):
        """
        Args:
            memory_system: 任意记忆系统实例（支持 write_memory 和 search_memory 接口）
            user_id: 评测用户ID
            agent_id: 评测代理ID
        """
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
            from tinymem0.adapters import call_llm_with_prompt
            from tinymem0.prompts import QA_SYSTEM_PROMPT, build_qa_prompt
            
            qa_prompt = build_qa_prompt(context, question)
            
            answer = call_llm_with_prompt(
                self.memory_system.llm_model,
                QA_SYSTEM_PROMPT,
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
        return float(f1(prediction, ground_truth))
    
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
    
    def evaluate_memory_system(self, model_name: str, output_file: str, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        评估指定的记忆系统模型
        
        Args:
            model_name: 模型名称 (例如: 'tinymem0')
            output_file: 结果输出文件路径
            sample_ids: 要评估的样本ID列表（None表示评估所有样本）
            
        Returns:
            评估结果统计
        """
        # 动态加载模型系统
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}")
        
        model_config = AVAILABLE_MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"评估模型: {model_config['name']}")
        print(f"描述: {model_config['description']}")
        print(f"{'='*60}\n")
        
        # 动态导入模型类
        try:
            module = import_module(model_config['module'])
            MemorySystemClass = getattr(module, model_config['class'])
        except (ImportError, AttributeError) as e:
            print(f"Error: Failed to load model {model_name}: {e}")
            print(f"Please ensure the module '{model_config['module']}' is properly installed in src/")
            return {}
        
        # 初始化记忆系统
        memory_system = MemorySystemClass(
            log_level="info",
            log_mode="plain"
        )
        
        # 创建评测器
        evaluator = MemorySystemEvaluator(memory_system)
        
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
                sample_memory_system = MemorySystemClass(
                    collection_name=f"memories_{sample_id}",
                    log_level="warn",  # 减少日志输出
                    qdrant_path=f"./qdrant_data_{sample_id}"  # 为每个样本使用独立的数据目录
                )
                sample_evaluator = MemorySystemEvaluator(sample_memory_system)
                
                # 评估样本
                result = sample_evaluator.evaluate_qa_sample(sample)
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
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="在 LoCoMo 数据集上评测记忆系统模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用的模型系统：
  tinymem0    - TinyMem0 记忆系统 (支持本地LLM和阿里云API)

示例用法：
  # 评测 TinyMem0 模型，使用默认数据集，评测5个样本
  python scripts/evaluate_system.py --model tinymem0 --num-samples 5
  
  # 评测所有样本，保存到指定文件
  python scripts/evaluate_system.py --model tinymem0 --output-file results.json
  
  # 评测指定的样本ID
  python scripts/evaluate_system.py --model tinymem0 --sample-ids sample_001 sample_002
        """
    )
    
    # 必选参数（除非使用 --list-models）
    parser.add_argument('--model', '-m', type=str,
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='要评测的模型系统名称')
    
    # 可选参数
    parser.add_argument('--data-file', type=str, default='locomo/data/locomo10.json',
                       help='LoCoMo 数据文件路径 (默认: locomo/data/locomo10.json)')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                       help='评测结果输出文件 (默认: {model}_locomo_results.json)')
    parser.add_argument('--sample-ids', type=str, nargs='*',
                       help='指定要评测的样本ID列表 (默认: 评测所有样本)')
    parser.add_argument('--num-samples', '-n', type=int, default=None,
                       help='限制评测的样本数量，用于快速测试 (默认: 评测所有样本)')
    parser.add_argument('--list-models', action='store_true',
                       help='列出所有可用的模型系统')
    
    args = parser.parse_args()
    
    # 如果请求列出模型，显示后退出
    if args.list_models:
        print("\n可用的记忆系统模型：")
        print("=" * 70)
        for model_id, config in AVAILABLE_MODELS.items():
            print(f"\n{model_id:15} - {config['name']}")
            print(f"{'':15}   {config['description']}")
            print(f"{'':15}   模块: {config['module']}.{config['class']}")
        print("\n" + "=" * 70)
        return
    
    # 如果不是列出模型，则 --model 参数是必需的
    if not args.model:
        parser.error("评测模式下 --model/-m 参数是必需的。使用 --list-models 查看可用模型。")
    
    # 设置默认输出文件名
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"{args.model}_locomo_results_{timestamp}.json"
    
    # 检查必要的环境变量
    use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    
    print("\n" + "=" * 70)
    print("记忆系统评测配置")
    print("=" * 70)
    print(f"模型系统: {AVAILABLE_MODELS[args.model]['name']}")
    print(f"数据文件: {args.data_file}")
    print(f"输出文件: {args.output_file}")
    
    if use_local_llm:
        # 使用本地LLM，检查模型路径
        if not os.getenv("LOCAL_MODEL_PATH"):
            print("\n❌ 错误: 使用本地LLM时必须设置 LOCAL_MODEL_PATH 环境变量")
            print("请在 .env 文件中配置：LOCAL_MODEL_PATH=/path/to/your/model.gguf")
            return
        model_path = os.getenv("LOCAL_MODEL_PATH")
        if model_path and not os.path.exists(model_path):
            print(f"\n❌ 错误: 找不到模型文件: {model_path}")
            return
        print(f"LLM模式: 本地模型")
        print(f"模型路径: {model_path}")
    else:
        # 使用阿里云API，检查API密钥
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("\n❌ 错误: 使用阿里云API时必须设置 DASHSCOPE_API_KEY 环境变量")
            print("请在 .env 文件中配置：DASHSCOPE_API_KEY=your_api_key")
            return
        print(f"LLM模式: 阿里云 Dashscope API")

    # --- Ensure embedding model exists (if needed) ---------------------------------
    def _ensure_embedding_available(project_root: str) -> bool:
        """Ensure LOCAL_EMBEDDING_MODEL is set or download a default embedding model.

        Returns True if embedding model exists/was downloaded and env var is set, False otherwise.
        """
        local_emb = os.getenv("LOCAL_EMBEDDING_MODEL")
        if local_emb and os.path.exists(local_emb):
            print(f"Embedding: 使用 LOCAL_EMBEDDING_MODEL={local_emb}")
            return True

        # try to find any model under ./embedding_models
        default_dir = os.path.join(project_root, 'embedding_models')
        if os.path.exists(default_dir) and any(os.scandir(default_dir)):
            # try to find a reasonable model directory (avoid .lock or temp files)
            def _find_model_dir(base_dir: str):
                model_files = ('pytorch_model.bin', 'model.safetensors', 'tf_model.h5', 'model.ckpt.index', 'flax_model.msgpack')
                for root, dirs, files in os.walk(base_dir):
                    # skip hidden/temp dirs
                    if os.path.basename(root).startswith('.'):
                        continue
                    for mf in model_files:
                        if mf in files:
                            return root
                # fallback: return first non-hidden child
                for entry in os.scandir(base_dir):
                    if not entry.name.startswith('.'):
                        return entry.path
                return None

            found = _find_model_dir(default_dir)
            if found:
                print(f"Embedding: 在 {default_dir} 发现模型，使用: {found}")
                os.environ['LOCAL_EMBEDDING_MODEL'] = found
                return True

        # attempt to run the download_embedding.py script non-interactively
        download_script = os.path.join(project_root, 'scripts', 'download_embedding.py')
        if os.path.exists(download_script):
            print("\n⚙️ 未检测到本地 embedding 模型，正在自动下载默认 embedding 模型到 ./embedding_models（可能需要网络）...")
            try:
                subprocess.run([sys.executable, download_script, '--model-id', '1', '--cache-dir', default_dir], check=True)
            except Exception as e:
                print(f"❌ 自动下载 embedding 失败: {e}")
                return False

            # after download, find the actual model dir containing model files (avoid .lock)
            def _find_model_dir(base_dir: str):
                model_files = ('pytorch_model.bin', 'model.safetensors', 'tf_model.h5', 'model.ckpt.index', 'flax_model.msgpack')
                for root, dirs, files in os.walk(base_dir):
                    if os.path.basename(root).startswith('.'):
                        continue
                    for mf in model_files:
                        if mf in files:
                            return root
                # fallback: pick the largest non-hidden directory
                candidates = [entry for entry in os.scandir(base_dir) if entry.is_dir() and not entry.name.startswith('.')]
                if candidates:
                    candidates.sort(key=lambda e: e.stat().st_size if e.is_file() else 0, reverse=True)
                    return candidates[0].path
                return None

            found = _find_model_dir(default_dir)
            if found:
                os.environ['LOCAL_EMBEDDING_MODEL'] = found
                print(f"✅ embedding 已下载并设置 LOCAL_EMBEDDING_MODEL={found}")
                return True
        else:
            print("⚠️ 未找到 scripts/download_embedding.py，无法自动下载 embedding。")

        print("❌ embedding 模型不可用，请手动运行 scripts/download_embedding.py 下载或在 .env 中配置 LOCAL_EMBEDDING_MODEL")
        return False


    # --- Ensure local LLM model exists (if using local LLM) -----------------------
    def _ensure_local_llm(project_root: str) -> bool:
        """Ensure LOCAL_MODEL_PATH exists. If not, try to find a GGUF under ./models and set it.

        If a local model is still not found, suggest running scripts/download_model.py.
        """
        if os.getenv("USE_LOCAL_LLM", "false").lower() != "true":
            return True

        model_path = os.getenv("LOCAL_MODEL_PATH")
        if model_path and os.path.exists(model_path):
            print(f"LLM: 使用 LOCAL_MODEL_PATH={model_path}")
            return True

        # search repo models/ for .gguf
        models_dir = os.path.join(project_root, 'models')
        if os.path.exists(models_dir):
            for fn in os.listdir(models_dir):
                if fn.lower().endswith('.gguf'):
                    candidate = os.path.join(models_dir, fn)
                    print(f"LLM: 在 {models_dir} 发现 GGUF 模型: {candidate}，将临时使用该路径。")
                    os.environ['LOCAL_MODEL_PATH'] = candidate
                    return True

        # no local model found — try helper script
        helper = os.path.join(project_root, 'scripts', 'download_model.py')
        if os.path.exists(helper):
            print("\n⚙️ 未检测到本地 LLM 模型，尝试运行 scripts/download_model.py 来帮助配置（不会自动下载大模型）。")
            try:
                subprocess.run([sys.executable, helper], check=False)
            except Exception:
                pass

        model_path = os.getenv("LOCAL_MODEL_PATH")
        if model_path and os.path.exists(model_path):
            print(f"LLM: 已配置 LOCAL_MODEL_PATH={model_path}")
            return True

        print("❌ 本地 LLM 模型仍不可用。请将 GGUF 模型放入项目的 models/ 目录，或在 .env 中设置 LOCAL_MODEL_PATH。")
        return False

    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        print(f"\n❌ 错误: 找不到数据文件 {args.data_file}")
        print("请确保 LoCoMo 数据集已下载并放在正确的位置")
        return
    
    print("=" * 70 + "\n")
    
    # 确保 embedding 可用
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not _ensure_embedding_available(project_root):
        return

    # 确保本地LLM（如使用）可用或已提示用户
    if not _ensure_local_llm(project_root):
        return

    # 创建评估器
    evaluator = LoCoMoEvaluator(args.data_file)
    
    # 限制评估样本数量（用于测试）
    sample_ids = args.sample_ids
    if args.num_samples and not sample_ids:
        sample_ids = [s['sample_id'] for s in evaluator.samples[:args.num_samples]]
        print(f"📊 将评测前 {args.num_samples} 个样本\n")
    elif sample_ids:
        print(f"📊 将评测指定的 {len(sample_ids)} 个样本\n")
    else:
        print(f"📊 将评测所有 {len(evaluator.samples)} 个样本\n")
    
    # 运行评估
    try:
        results = evaluator.evaluate_memory_system(args.model, args.output_file, sample_ids)
        
        if results:
            print("\n✅ 评测完成！")
            return results
        else:
            print("\n❌ 评测失败")
            return None
            
    except KeyboardInterrupt:
        print("\n\n⚠️  评测被用户中断")
        return None
    except Exception as e:
        print(f"\n❌ 评测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()