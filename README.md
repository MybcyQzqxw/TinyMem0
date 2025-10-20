# 极简化记忆系统 - 学习项目

> **⚠️ 学习项目声明**: 这是一个用于学习和研究目的的开源项目，旨在探索和实现基于向量数据库的智能记忆系统。本项目仅供学习交流使用，不适用于生产环境。

## 项目概述

这是一个基于Qdrant向量数据库和Qwen大语言模型的智能记忆系统**学习项目**，支持记忆的写入、检索和处理。通过这个项目，您可以学习：

- 向量数据库的实际应用
- 大语言模型在记忆系统中的应用
- 语义搜索的实现原理
- 记忆冲突处理机制
- 智能记忆系统的架构设计

## 学习目标

通过本项目的学习，您将掌握：

1. **向量数据库集成**: 学习如何使用Qdrant进行向量存储和检索
2. **LLM应用开发**: 了解如何将大语言模型集成到实际应用中
3. **语义搜索实现**: 掌握基于向量相似度的智能搜索技术
4. **记忆冲突处理**: 学习如何处理和解决记忆系统中的冲突
5. **系统架构设计**: 理解智能记忆系统的整体架构设计

## 功能特性

### 1. 记忆写入

- **信息提取**: 使用LLM从用户对话中提取相关事实和偏好
- **相关记忆检索**: 通过语义搜索找到相关的已有记忆
- **记忆处理**: 智能决定添加、更新、删除或保持记忆不变
- **冲突处理**: 防止记忆冲突，保持数据一致性

### 2. 记忆搜索

- **元数据检索**: 支持通过user_id和agent_id快速过滤
- **语义检索**: 基于向量相似度的智能搜索
- **相似度阈值**: 可配置的相似度过滤

### 3. 记忆处理

- **ADD**: 添加新记忆
- **UPDATE**: 更新现有记忆
- **DELETE**: 删除冲突或过时的记忆
- **NONE**: 保持记忆不变

## 学习环境搭建

### 1. 环境变量配置

复制 `.env.example` 文件为 `.env` 并根据实际情况修改配置：

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

然后编辑 `.env` 文件，配置必要的环境变量（详见"本地LLM配置"章节）。

### 2. 安装依赖

```bash
# 安装所有依赖（包含评测所需）
pip install -r requirements.txt
```

### 3. 下载嵌入模型（使用本地LLM时需要）

使用交互式脚本下载嵌入模型：

```bash
# 推荐：交互式模式，有友好界面
python scripts/download_embedding.py

# 或使用命令行模式，快速下载预定义模型
python scripts/download_embedding.py --model-id 1

# 查看所有可用模型
python scripts/download_embedding.py --list
```

详见 `scripts/README.md` 获取更多信息。

### 5. 配置Qdrant

系统使用Qdrant的本地文件存储模式，无需Docker服务：

```bash
# 数据将存储在 ./qdrant_data 目录中
# 系统会自动创建该目录
```

### 6. 配置API密钥（仅在使用阿里云API时需要）

**注意**: 使用阿里云API模式时需配置密钥，使用本地LLM时可跳过：

```export
DASHSCOPE_API_KEY=your_actual_api_key_here
```

## 学习使用方法

### 安装

使用标准的Python包安装方式：

```bash
# 开发模式安装（推荐用于学习）
pip install -e .

# 或直接安装
pip install .
```

### 基本使用示例

```python
from tinymem0 import MemorySystem

# 初始化记忆系统
memory_system = MemorySystem()

# 写入记忆
memory_system.write_memory(
    conversation="我叫张三，是一名软件工程师，我喜欢看电影",
    user_id="user_001",
    agent_id="agent_001"
)

# 搜索记忆
results = memory_system.search_memory(
    query="张三的职业",
    user_id="user_001",
    agent_id="agent_001"
)

for result in results:
    print(f"记忆: {result['text']}, 相似度: {result['score']}")
```

### 运行学习示例

```bash
python examples/basic_usage.py
```

## 学习架构分析

### 文件结构

```text
TinyMem0/                           # 项目根目录
├── src/                            # 源代码目录（标准Python项目结构）
│   └── tinymem0/                   # 主包
│       ├── __init__.py             # 包初始化，导出主要接口
│       ├── memory_system.py        # 核心记忆系统 - 学习重点
│       ├── prompts/                # Prompt定义目录 - 集中管理所有Prompt
│       │   ├── __init__.py         # 统一导出接口
│       │   ├── fact_extraction.py  # 事实提取Prompt
│       │   ├── memory_processing.py # 记忆处理Prompt
│       │   └── question_answering.py # 问答系统Prompt
│       └── utils/                  # 工具模块 - 通用功能
│           ├── __init__.py         # 统一导出接口
│           ├── llm_utils.py        # LLM调用和JSON解析
│           ├── local_llm_backend.py # 本地GGUF模型后端
│           ├── embedding_utils.py  # 嵌入向量工具
│           └── evaluation_utils.py # 评测指标计算
├── scripts/                        # 实用脚本
│   ├── __init__.py
│   ├── download_embedding.py       # 嵌入模型下载工具
│   └── evaluate_system.py          # 记忆系统评测脚本
├── examples/                       # 使用示例
│   └── basic_usage.py              # 基础使用示例 - 学习如何使用
├── locomo/                         # 评测数据集
│   └── ...                         # 评测相关文件
├── .env.example                    # 环境变量模板
├── requirements.txt                # 统一依赖管理
├── setup.py                        # 安装配置（支持pip install）
└── README.md                       # 项目文档
```

### 核心学习组件

1. **MemorySystem**: 主记忆系统类 - 学习系统设计
   - `extract_facts()`: 从对话中提取事实 - 学习LLM应用
   - `search_memories()`: 语义搜索记忆 - 学习向量搜索
   - `process_memory()`: 处理记忆冲突 - 学习冲突解决
   - `write_memory()`: 记忆写入主流程 - 学习系统流程

2. **向量数据库**: Qdrant - 学习向量数据库应用
   - 存储记忆的向量表示
   - 支持语义搜索
   - 元数据过滤

3. **大语言模型**: Qwen - 学习LLM集成
   - 事实提取
   - 记忆冲突处理
   - 文本向量化

### 学习工作流程

1. **记忆写入流程**:

   ```text
   用户对话 → 提取事实 → 检索相关记忆 → 处理冲突 → 更新数据库
   ```

2. **记忆搜索流程**:

   ```text
   查询 → 向量化 → 语义搜索 → 过滤 → 返回结果
   ```

## 学习API参考

### MemorySystem类

#### 初始化

```python
MemorySystem(collection_name: str = "memories")
```

#### 主要方法

- `write_memory(conversation, user_id=None, agent_id=None)`: 写入记忆
- `search_memory(query, user_id=None, agent_id=None, limit=5)`: 搜索记忆
- `extract_facts(conversation)`: 提取事实
- `add_memory(text, metadata=None)`: 添加记忆
- `update_memory(memory_id, new_text, metadata=None)`: 更新记忆
- `delete_memory(memory_id)`: 删除记忆

## 学习Prompt管理

### Prompt模块

系统使用专门的`prompt.py`模块来管理所有的prompt定义，这是学习LLM应用的重要部分：

- **FACT_EXTRACTION_PROMPT**: 用于从用户对话中提取事实和偏好的prompt
- **MEMORY_PROCESSING_PROMPT**: 用于处理记忆冲突和更新的prompt
- **PromptManager**: 提供统一的prompt访问接口

## 学习建议

1. **从example.py开始**: 先运行示例代码，理解基本用法
2. **阅读源码**: 深入理解memory_system.py的实现原理
3. **修改参数**: 尝试调整相似度阈值、搜索限制等参数
4. **扩展功能**: 基于现有代码添加新的记忆处理逻辑
5. **性能优化**: 学习如何优化向量搜索和LLM调用

## 学习资源

- [Qdrant官方文档](https://qdrant.tech/documentation/)
- [Qwen模型介绍](https://github.com/QwenLM/Qwen)
- [向量数据库应用指南](https://www.pinecone.io/learn/)

## 本地LLM配置

### 环境变量配置

TinyMem0支持两种LLM模式：阿里云API或本地GGUF模型。通过环境变量`USE_LOCAL_LLM`控制：

```bash
# Windows PowerShell
$env:USE_LOCAL_LLM = "true"  # 使用本地LLM
$env:USE_LOCAL_LLM = "false" # 使用阿里云API（默认）

# Linux/Mac
export USE_LOCAL_LLM=true
export USE_LOCAL_LLM=false
```

### 本地LLM相关环境变量

```bash
# 必需：本地GGUF模型路径
$env:LOCAL_MODEL_PATH = "path/to/your/model.gguf"

# 可选：本地嵌入模型名称（默认：BAAI/bge-small-zh-v1.5）
$env:LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 可选：嵌入向量维度（默认：512用于本地模型，1536用于阿里云）
$env:EMBEDDING_DIM = "512"
```

### 本地LLM依赖安装

```bash
# 安装llama-cpp-python（支持GGUF模型）
pip install llama-cpp-python>=0.2.0

# 安装sentence-transformers（本地嵌入模型）
pip install sentence-transformers>=2.2.0
```

### 推荐的本地模型

- **LLM模型**: Qwen2-7B-Instruct-GGUF, Llama3-8B-Chinese-Chat-GGUF
- **嵌入模型**: BAAI/bge-small-zh-v1.5（中文优化，512维）

## LoCoMo评测指标说明

### QA任务评测指标

- **Token-level F1 Score**: 预测答案与参考答案的token级别F1分数
- **Exact Match (EM)**: 预测答案与参考答案完全匹配的比例

### Evidence检索评测指标

- **Evidence Precision**: 检索到的对话中包含相关证据的比例
- **Evidence Recall**: 参考证据中被成功检索到的比例
- **Evidence F1**: Evidence Precision和Recall的调和平均数
- **Recall@5**: 前5个检索结果中包含相关证据的比例
- **Recall@10**: 前10个检索结果中包含相关证据的比例
- **MRR (Mean Reciprocal Rank)**: 第一个相关证据出现位置的倒数的平均值

### 运行LoCoMo评测

本项目提供统一的评测脚本 `scripts/evaluate_system.py`，支持评测不同的记忆系统模型。

#### 快速开始

```bash
# 1. 查看可用的模型系统
python scripts/evaluate_system.py --list-models

# 2. 快速测试（评测5个样本）
python scripts/evaluate_system.py --model tinymem0 --num-samples 5

# 3. 完整评测（所有样本）
python scripts/evaluate_system.py --model tinymem0
```

#### 配置环境（使用本地LLM）

```bash
# Windows PowerShell
$env:USE_LOCAL_LLM = "true"
$env:LOCAL_MODEL_PATH = "path/to/model.gguf"
$env:LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# Linux/Mac
export USE_LOCAL_LLM=true
export LOCAL_MODEL_PATH=path/to/model.gguf
export LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

#### 高级用法

```bash
# 评测指定样本
python scripts/evaluate_system.py --model tinymem0 --sample-ids sample_001 sample_002

# 自定义输出文件
python scripts/evaluate_system.py --model tinymem0 -o my_results.json

# 使用自定义数据集
python scripts/evaluate_system.py --model tinymem0 --data-file locomo/data/custom.json
```

详细使用说明请参考 [评测系统使用指南](docs/EVALUATION_GUIDE.md)。

评测结果将保存为 JSON 文件，包含所有 QA 和 Evidence/Retrieval 指标的详细统计信息。

## 贡献学习

欢迎提交学习心得、代码改进建议或问题报告！

---

**免责声明**: 本项目仅用于学习和研究目的，使用者需自行承担使用风险。
