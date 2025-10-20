#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
辅助脚本：帮助在本地项目中查找或提示放置 LLM GGUF 模型。
- 不会尝试从远程下载大型 LLM（这通常需要手动授权/镜像）。
- 如果在 `models/` 目录中发现 `.gguf` 文件，会建议并可将其写入 `.env`。
"""

import os
from pathlib import Path

project_root = Path(__file__).parent.parent
models_dir = project_root / 'models'
env_file = project_root / '.env'


def find_gguf_models():
    if not models_dir.exists():
        return []
    return [p for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() == '.gguf']


def write_env_local_model(path: str):
    # Append or update LOCAL_MODEL_PATH in .env
    lines = []
    if env_file.exists():
        lines = env_file.read_text(encoding='utf-8').splitlines()

    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith('LOCAL_MODEL_PATH='):
            new_lines.append(f'LOCAL_MODEL_PATH={path}')
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f'LOCAL_MODEL_PATH={path}')

    env_file.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
    print(f"✅ 已将 LOCAL_MODEL_PATH 写入 {env_file}: {path}")


if __name__ == '__main__':
    models = find_gguf_models()
    if not models:
        print("⚠️ 未在 ./models 目录中发现 .gguf 模型。请将模型文件放入该目录，或在 .env 中手动设置 LOCAL_MODEL_PATH。")
        print("示例: LOCAL_MODEL_PATH=./models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf")
    else:
        print("🔎 在 ./models 目录中发现以下 GGUF 模型：")
        for i, p in enumerate(models, start=1):
            print(f"  [{i}] {p}")

        choice = None
        if len(models) == 1:
            choice = 1
        else:
            try:
                choice = int(input("请输入要写入 .env 的序号（或回车取消）: ").strip() or '0')
            except Exception:
                choice = 0

        if choice and 1 <= choice <= len(models):
            selected = models[choice - 1]
            write_env_local_model(str(selected))
        else:
            print("已取消写入 .env。您仍然可以手动将模型路径添加到 .env 或将模型重命名为 .gguf 放入 models/。")
