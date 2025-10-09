#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行TinyMem0在LoCoMo基准测试上的评估脚本
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def run_tinymem0_evaluation():
    """运行TinyMem0 LoCoMo评估的主函数"""
    
    parser = argparse.ArgumentParser(description="Run TinyMem0 evaluation on LoCoMo benchmark")
    parser.add_argument('--data-file', type=str, default='locomo/data/locomo10.json',
                       help='Path to LoCoMo data file')
    parser.add_argument('--output-dir', type=str, default='evaluation_outputs',
                       help='Directory to save evaluation results')
    parser.add_argument('--sample-ids', type=str, nargs='*',
                       help='Specific sample IDs to evaluate (default: all)')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to evaluate (for testing, default: 1)')
    parser.add_argument('--log-level', type=str, default='info',
                       choices=['debug', 'info', 'warn', 'error'],
                       help='Log level for memory system')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_file = os.path.join(args.output_dir, 'tinymem0_locomo_results.json')
    
    # 检查必要的环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY environment variable is required")
        print("Please set it in your .env file or environment variables")
        return
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found")
        return
    
    print("=== TinyMem0 LoCoMo Evaluation ===")
    print(f"Data file: {args.data_file}")
    print(f"Output file: {output_file}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Log level: {args.log_level}")
    
    try:
        # 导入评估模块
        from tinymem0_locomo_adapter import LoCoMoEvaluator
        
        # 创建评估器
        evaluator = LoCoMoEvaluator(args.data_file)
        
        # 限制评估样本数量（用于测试）
        sample_ids = args.sample_ids
        if args.num_samples and not sample_ids:
            sample_ids = [s['sample_id'] for s in evaluator.samples[:args.num_samples]]
        
        print(f"Starting evaluation of {len(sample_ids) if sample_ids else 'all'} samples...")
        
        # 运行评估
        results = evaluator.evaluate_tinymem0(output_file, sample_ids)
        
        print("\n=== Evaluation Complete ===")
        print(f"Results saved to: {output_file}")
        
        return results
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required packages are installed")
        return
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    run_tinymem0_evaluation()