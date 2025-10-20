# 评测系统使用指南

本指南介绍如何使用 `scripts/evaluate_system.py` 在 LoCoMo 数据集上评测不同的记忆系统模型。

## 快速开始

### 1. 查看可用模型

```bash
python scripts/evaluate_system.py --list-models
```

### 2. 查看帮助信息

```bash
python scripts/evaluate_system.py --help
```

### 3. 快速测试（评测少量样本）

```bash
# 评测 TinyMem0，只测试 5 个样本
python scripts/evaluate_system.py --model tinymem0 --num-samples 5
```

### 4. 完整评测

```bash
# 评测 TinyMem0，所有样本
python scripts/evaluate_system.py --model tinymem0
```

## 命令行参数说明

### 必选参数

- `--model`, `-m`: 要评测的模型系统名称
  - 可选值: `tinymem0` (未来可扩展更多模型)
  - 示例: `--model tinymem0`

### 可选参数

- `--data-file`: LoCoMo 数据文件路径
  - 默认: `locomo/data/locomo10.json`
  - 示例: `--data-file locomo/data/locomo_full.json`

- `--output-file`, `-o`: 评测结果输出文件路径
  - 默认: `{model}_locomo_results_{timestamp}.json`
  - 示例: `--output-file my_results.json`

- `--num-samples`, `-n`: 限制评测的样本数量（用于快速测试）
  - 默认: 评测所有样本
  - 示例: `--num-samples 10`

- `--sample-ids`: 指定要评测的样本ID列表
  - 默认: 评测所有样本
  - 示例: `--sample-ids sample_001 sample_002 sample_003`

- `--list-models`: 列出所有可用的模型系统
  - 示例: `--list-models`

## 使用示例

### 示例 1: 快速测试（5个样本）

```bash
python scripts/evaluate_system.py --model tinymem0 --num-samples 5
```

### 示例 2: 评测所有样本并保存结果

```bash
python scripts/evaluate_system.py \
    --model tinymem0 \
    --output-file tinymem0_full_results.json
```

### 示例 3: 评测指定样本

```bash
python scripts/evaluate_system.py \
    --model tinymem0 \
    --sample-ids sample_001 sample_005 sample_010
```

### 示例 4: 使用自定义数据文件

```bash
python scripts/evaluate_system.py \
    --model tinymem0 \
    --data-file locomo/data/custom_dataset.json \
    --num-samples 20
```

## 环境配置

评测前需要配置环境变量（在 `.env` 文件中）：

### 使用本地 LLM（推荐）

```env
# 启用本地LLM
USE_LOCAL_LLM=true

# 本地模型路径
LOCAL_MODEL_PATH=/path/to/your/model.gguf

# 本地嵌入模型（可选，默认使用 sentence-transformers）
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 使用阿里云 API

```env
# 禁用本地LLM
USE_LOCAL_LLM=false

# 阿里云 API 密钥
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

## 评测指标说明

评测系统会计算以下指标：

### QA 指标

- **F1 Score**: 预测答案与标准答案的 F1 分数
- **Exact Match (EM)**: 精确匹配分数

### Evidence 检索指标

- **Evidence F1**: 检索证据与标准证据的 F1 分数
- **Evidence Precision**: 检索证据的精确率
- **Evidence Recall**: 检索证据的召回率
- **Recall@5**: 前5个结果中包含正确证据的比例
- **Recall@10**: 前10个结果中包含正确证据的比例
- **MRR (Mean Reciprocal Rank)**: 平均倒数排名

## 输出文件格式

评测结果保存为 JSON 格式，包含：

```json
{
  "metadata": {
    "evaluation_time": "2025-10-20T14:30:00",
    "model": "tinymem0",
    "data_file": "locomo/data/locomo10.json",
    "total_samples": 100
  },
  "overall_metrics": {
    "average_f1": 0.75,
    "average_em": 0.68,
    "average_evidence_f1": 0.72,
    "average_recall_at_5": 0.85,
    "average_recall_at_10": 0.92,
    "average_mrr": 0.78
  },
  "category_metrics": {
    "1": {
      "average_f1": 0.80,
      "average_em": 0.72,
      "count": 50
    }
  },
  "sample_results": [
    {
      "sample_id": "sample_001",
      "category": "1",
      "f1_score": 0.85,
      "em_score": 1.0,
      "evidence_metrics": { ... }
    }
  ]
}
```

## 添加新模型系统

要添加新的记忆系统模型进行评测：

### 1. 在 `src/` 下创建模型目录

```bash
mkdir -p src/my_memory_system
```

### 2. 实现模型类

在 `src/my_memory_system/__init__.py` 中实现记忆系统类，需要提供以下接口：

```python
class MemorySystem:
    def __init__(self, log_level="info", log_mode="plain", **kwargs):
        """初始化记忆系统"""
        pass
    
    def write_memory(self, conversation: str, user_id: str = None, agent_id: str = None):
        """写入对话到记忆系统"""
        pass
    
    def search_memory(self, query: str, user_id: str = None, agent_id: str = None, limit: int = 5):
        """搜索记忆，返回相关记忆列表"""
        pass
```

### 3. 在评测脚本中注册模型

编辑 `scripts/evaluate_system.py`，在 `AVAILABLE_MODELS` 字典中添加：

```python
AVAILABLE_MODELS = {
    'tinymem0': { ... },
    'my_model': {
        'name': 'My Memory System',
        'description': '我的记忆系统实现',
        'module': 'my_memory_system',
        'class': 'MemorySystem'
    },
}
```

### 4. 运行评测

```bash
python scripts/evaluate_system.py --model my_model --num-samples 5
```

## 故障排除

### 问题 1: 找不到数据文件

**错误信息**: `Error: Data file locomo/data/locomo10.json not found`

**解决方法**: 确保 LoCoMo 数据集已下载到 `locomo/data/` 目录

### 问题 2: 模型加载失败

**错误信息**: `Failed to load model tinymem0`

**解决方法**: 
1. 检查模型目录是否在 `src/` 下
2. 检查 `__init__.py` 是否正确导出 `MemorySystem` 类
3. 运行 `pip install -e .` 安装项目

### 问题 3: 环境变量未设置

**错误信息**: `LOCAL_MODEL_PATH environment variable is required`

**解决方法**: 在 `.env` 文件中配置必要的环境变量

### 问题 4: 内存不足

**解决方法**: 
1. 使用 `--num-samples` 限制评测样本数量
2. 减小批处理大小
3. 使用更小的模型

## 性能优化建议

1. **使用本地 LLM**: 比云 API 更快，成本更低
2. **GPU 加速**: 配置 `n_gpu_layers` 参数利用 GPU
3. **批量处理**: 调整批处理大小以平衡速度和内存
4. **缓存结果**: 保存中间结果，避免重复计算

## 更多信息

- [TinyMem0 项目主页](../README.md)
- [LoCoMo 数据集说明](../locomo/README.md)
- [开发指南](./DEVELOPMENT.md)
