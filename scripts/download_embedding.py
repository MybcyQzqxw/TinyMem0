#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
嵌入模型下载脚本
提供交互式和命令行两种方式下载嵌入模型
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_manager import download_embedding_model


# 预定义的常用模型列表
AVAILABLE_MODELS = {
    '1': {
        'id': 'AI-ModelScope/bge-small-zh-v1.5',
        'name': 'BGE-Small-ZH v1.5',
        'size': '~400MB',
        'dims': 512,
        'lang': '中文优化',
        'description': '推荐：中文检索效果好，体积小'
    },
    '2': {
        'id': 'AI-ModelScope/bge-base-zh-v1.5',
        'name': 'BGE-Base-ZH v1.5',
        'size': '~800MB',
        'dims': 768,
        'lang': '中文优化',
        'description': '更好的效果，体积适中'
    },
    '3': {
        'id': 'AI-ModelScope/bge-large-zh-v1.5',
        'name': 'BGE-Large-ZH v1.5',
        'size': '~1.5GB',
        'dims': 1024,
        'lang': '中文优化',
        'description': '最佳效果，体积较大'
    },
    '4': {
        'id': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'name': 'Multilingual MiniLM',
        'size': '~450MB',
        'dims': 384,
        'lang': '多语言',
        'description': '支持50+种语言'
    },
    '5': {
        'id': 'sentence-transformers/all-MiniLM-L6-v2',
        'name': 'All-MiniLM-L6-v2',
        'size': '~90MB',
        'dims': 384,
        'lang': '英文优化',
        'description': '最小体积，英文任务'
    }
}


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🚀 TinyMem0 嵌入模型下载工具")
    print("=" * 60)
    print()


def print_models():
    """打印可用模型列表"""
    print("📋 可用的嵌入模型：\n")
    for key, model in AVAILABLE_MODELS.items():
        print(f"[{key}] {model['name']}")
        print(f"    模型ID: {model['id']}")
        print(f"    大小: {model['size']} | 维度: {model['dims']} | 语言: {model['lang']}")
        print(f"    说明: {model['description']}")
        print()


def interactive_download():
    """交互式下载模式"""
    print_banner()
    print_models()
    
    while True:
        choice = input("请选择要下载的模型 [1-5] (输入 'q' 退出): ").strip()
        
        if choice.lower() == 'q':
            print("👋 退出下载")
            return
        
        if choice not in AVAILABLE_MODELS:
            print("❌ 无效的选择，请重新输入\n")
            continue
        
        model = AVAILABLE_MODELS[choice]
        print(f"\n✅ 你选择了: {model['name']}")
        print(f"📦 模型ID: {model['id']}")
        print(f"💾 大小: {model['size']}")
        
        # 询问下载目录
        default_cache = './embedding_models'
        cache_dir = input(f"\n📁 下载目录 [默认: {default_cache}]: ").strip()
        if not cache_dir:
            cache_dir = default_cache
        
        # 确认下载
        confirm = input(f"\n确认下载到 {cache_dir}? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            try:
                print(f"\n⏳ 开始下载 {model['name']}...")
                downloaded_path = download_embedding_model(
                    model_id=model['id'],
                    cache_dir=cache_dir
                )
                
                print(f"\n✅ 下载成功！")
                print(f"📁 模型路径: {downloaded_path}")
                print(f"\n📝 请在 .env 文件中配置:")
                print(f"   LOCAL_EMBEDDING_MODEL={downloaded_path}")
                print(f"   EMBEDDING_DIM={model['dims']}")
                
                # 询问是否继续下载
                cont = input("\n是否继续下载其他模型? [y/N]: ").strip().lower()
                if cont not in ['y', 'yes']:
                    print("👋 完成下载")
                    return
                else:
                    print()
                    print_models()
                    
            except Exception as e:
                print(f"\n❌ 下载失败: {e}")
                cont = input("\n是否重试或选择其他模型? [y/N]: ").strip().lower()
                if cont not in ['y', 'yes']:
                    return
                else:
                    print()
                    print_models()
        else:
            print("❌ 取消下载\n")


def command_line_download(args):
    """命令行下载模式"""
    print_banner()
    
    model_id = args.model_id
    cache_dir = args.cache_dir
    
    # 如果是数字，从预定义列表中获取
    if model_id in AVAILABLE_MODELS:
        model = AVAILABLE_MODELS[model_id]
        model_id = model['id']
        print(f"📦 使用预定义模型: {model['name']}")
        print(f"🔗 模型ID: {model_id}")
    else:
        print(f"📦 自定义模型ID: {model_id}")
    
    print(f"📁 下载目录: {cache_dir}")
    
    try:
        print(f"\n⏳ 开始下载...")
        downloaded_path = download_embedding_model(
            model_id=model_id,
            cache_dir=cache_dir
        )
        
        print(f"\n✅ 下载成功！")
        print(f"📁 模型路径: {downloaded_path}")
        
        # 如果是预定义模型，显示配置建议
        if args.model_id in AVAILABLE_MODELS:
            model = AVAILABLE_MODELS[args.model_id]
            print(f"\n📝 建议配置 (.env 文件):")
            print(f"   LOCAL_EMBEDDING_MODEL={downloaded_path}")
            print(f"   EMBEDDING_DIM={model['dims']}")
        else:
            print(f"\n📝 请在 .env 文件中配置:")
            print(f"   LOCAL_EMBEDDING_MODEL={downloaded_path}")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='TinyMem0 嵌入模型下载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式模式（推荐）
  python scripts/download_embedding.py
  
  # 下载预定义模型
  python scripts/download_embedding.py --model-id 1
  
  # 下载自定义模型
  python scripts/download_embedding.py --model-id AI-ModelScope/bge-small-zh-v1.5
  
  # 指定下载目录
  python scripts/download_embedding.py --model-id 1 --cache-dir ./models
        """
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        help='模型ID (1-5使用预定义模型，或指定完整模型ID)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./embedding_models',
        help='模型下载目录 (默认: ./embedding_models)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的预定义模型'
    )
    
    args = parser.parse_args()
    
    # 如果只是列出模型
    if args.list:
        print_banner()
        print_models()
        return
    
    # 如果指定了model-id，使用命令行模式
    if args.model_id:
        command_line_download(args)
    else:
        # 否则使用交互式模式
        interactive_download()


if __name__ == "__main__":
    main()
