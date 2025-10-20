# 评测系统快速参考

## 🚀 快速开始

### 1️⃣ 查看可用模型
```bash
python scripts/evaluate_system.py --list-models
```

### 2️⃣ 快速测试（5个样本）
```bash
python scripts/evaluate_system.py --model tinymem0 --num-samples 5
```

### 3️⃣ 完整评测
```bash
python scripts/evaluate_system.py --model tinymem0
```

## 📋 常用命令

```bash
# 查看帮助
python scripts/evaluate_system.py --help

# 列出所有模型
python scripts/evaluate_system.py --list-models

# 评测指定样本数
python scripts/evaluate_system.py -m tinymem0 -n 10

# 自定义输出文件
python scripts/evaluate_system.py -m tinymem0 -o my_results.json

# 评测指定样本ID
python scripts/evaluate_system.py -m tinymem0 --sample-ids sample_001 sample_002

# 使用自定义数据集
python scripts/evaluate_system.py -m tinymem0 --data-file locomo/data/custom.json
```

## ⚙️ 环境配置

### 本地 LLM 模式（推荐）

`.env` 文件：
```env
USE_LOCAL_LLM=true
LOCAL_MODEL_PATH=/path/to/your/model.gguf
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

### 阿里云 API 模式

`.env` 文件：
```env
USE_LOCAL_LLM=false
DASHSCOPE_API_KEY=your_api_key_here
```

## 📊 评测指标

### QA 指标
- **F1 Score**: Token级别F1分数
- **Exact Match (EM)**: 完全匹配率

### Evidence 检索指标
- **Evidence F1**: 证据检索F1分数
- **Evidence Precision**: 证据精确率
- **Evidence Recall**: 证据召回率
- **Recall@5**: 前5个结果召回率
- **Recall@10**: 前10个结果召回率
- **MRR**: 平均倒数排名

## 🔧 添加新模型

### 步骤 1: 创建模型目录
```bash
mkdir -p src/my_model
```

### 步骤 2: 实现接口
`src/my_model/__init__.py`:
```python
class MemorySystem:
    def __init__(self, **kwargs): ...
    def write_memory(self, conversation, user_id=None, agent_id=None): ...
    def search_memory(self, query, user_id=None, agent_id=None, limit=5): ...
```

### 步骤 3: 注册模型
编辑 `scripts/evaluate_system.py`：
```python
AVAILABLE_MODELS = {
    'tinymem0': {...},
    'my_model': {
        'name': 'My Model',
        'description': '模型描述',
        'module': 'my_model',
        'class': 'MemorySystem'
    }
}
```

### 步骤 4: 运行评测
```bash
python scripts/evaluate_system.py --model my_model -n 5
```

## 📁 项目结构

```
TinyMem0/
├── src/                    # 模型实现
│   └── tinymem0/          # TinyMem0 模型
├── scripts/               # 工具脚本
│   └── evaluate_system.py # 评测脚本 ⭐
├── utils/                 # 通用工具
│   ├── llm/              # LLM工具
│   ├── evaluation/       # 评测指标
│   └── model_manager/    # 模型管理
└── locomo/               # 评测数据集
```

## 🎯 参数速查

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--model` | `-m` | 模型名称 | 必需 |
| `--data-file` | | 数据文件路径 | `locomo/data/locomo10.json` |
| `--output-file` | `-o` | 输出文件路径 | `{model}_locomo_results_{timestamp}.json` |
| `--num-samples` | `-n` | 评测样本数 | 全部 |
| `--sample-ids` | | 指定样本ID | 全部 |
| `--list-models` | | 列出所有模型 | - |
| `--help` | `-h` | 显示帮助 | - |

## 💡 小技巧

1. **快速测试**: 使用 `-n 5` 先测试少量样本
2. **查看模型**: 使用 `--list-models` 确认模型已注册
3. **保存结果**: 使用 `-o` 指定有意义的文件名
4. **调试模式**: 在 `.env` 中设置 `LOG_LEVEL=debug`

## 📚 相关文档

- [评测系统详细指南](EVALUATION_GUIDE.md)
- [项目架构说明](PROJECT_ARCHITECTURE.md)
- [项目主文档](../README.md)

## 🆘 常见问题

**Q: 如何查看有哪些模型？**  
A: `python scripts/evaluate_system.py --list-models`

**Q: 如何快速测试？**  
A: `python scripts/evaluate_system.py -m tinymem0 -n 5`

**Q: 如何使用本地模型？**  
A: 在 `.env` 中设置 `USE_LOCAL_LLM=true` 和模型路径

**Q: 结果保存在哪里？**  
A: 默认保存在 `{model}_locomo_results_{timestamp}.json`

**Q: 如何评测自己的模型？**  
A: 参考"添加新模型"部分的步骤
