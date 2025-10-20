# 项目结构说明

## 整体架构

```
TinyMem0/
├── src/                          # 源代码目录 - 各种记忆系统模型实现
│   └── tinymem0/                # TinyMem0 记忆系统
│       ├── __init__.py
│       ├── memory_system.py     # 核心记忆系统实现
│       ├── prompts/             # Prompt 模板
│       │   ├── fact_extraction.py
│       │   ├── memory_processing.py
│       │   └── question_answering.py
│       └── adapters/            # API 适配器（特定于TinyMem0）
│           ├── dashscope_llm.py      # 阿里云 LLM 适配
│           └── dashscope_embedding.py # 阿里云 Embedding 适配
│
├── scripts/                      # 工具脚本
│   ├── evaluate_system.py       # 🎯 统一评测脚本（核心）
│   └── download_embedding.py    # 嵌入模型下载工具
│
├── utils/                        # 通用工具库（跨项目复用）
│   ├── llm/                     # LLM 相关工具
│   │   ├── local_backend.py    # 本地 GGUF 模型加载
│   │   └── json_parser.py      # JSON 解析工具
│   ├── evaluation/              # 评测指标
│   │   └── metrics.py          # F1, Precision, Recall, MRR 等
│   └── model_manager/           # 模型管理
│       └── downloader.py       # 模型下载工具
│
├── locomo/                       # LoCoMo 评测框架
│   ├── data/                    # 评测数据集
│   └── task_eval/               # 评测工具
│
└── docs/                         # 文档
    └── EVALUATION_GUIDE.md      # 评测系统使用指南
```

## 设计理念

### 1. 模块化设计

- **src/**: 存放各种记忆系统模型的实现
  - 每个模型系统独立一个子目录（如 `tinymem0/`）
  - 模型之间完全解耦，互不影响
  - 未来可轻松添加新模型（如 `mem0ai/`, `langchain_memory/` 等）

- **utils/**: 通用工具库，可跨项目复用
  - 与具体业务逻辑无关
  - 纯工具函数，无 API 依赖
  - 可以独立提取到其他项目使用

- **scripts/**: 独立的工具脚本
  - `evaluate_system.py`: 统一评测入口，支持所有模型
  - `download_embedding.py`: 嵌入模型下载工具

### 2. 统一评测接口

`scripts/evaluate_system.py` 是核心评测脚本，通过 `--model` 参数选择要评测的模型：

```bash
python scripts/evaluate_system.py --model tinymem0 --num-samples 5
```

**工作原理**:

1. 通过 `AVAILABLE_MODELS` 字典维护模型注册表
2. 动态导入指定模型的类（`importlib.import_module`）
3. 使用统一的 `MemorySystemEvaluator` 进行评测
4. 输出标准化的评测结果

### 3. 模型接口标准

所有记忆系统模型必须实现以下接口：

```python
class MemorySystem:
    def __init__(self, log_level="info", log_mode="plain", **kwargs):
        """初始化"""
        pass
    
    def write_memory(self, conversation: str, user_id: Optional[str] = None, 
                     agent_id: Optional[str] = None, extra_metadata: Optional[Dict] = None):
        """写入对话到记忆"""
        pass
    
    def search_memory(self, query: str, user_id: Optional[str] = None, 
                      agent_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """搜索记忆"""
        pass
```

## 如何添加新模型

### 步骤 1: 创建模型目录

```bash
mkdir -p src/my_model
```

### 步骤 2: 实现 MemorySystem 类

`src/my_model/__init__.py`:

```python
from .memory_system import MemorySystem

__all__ = ['MemorySystem']
```

`src/my_model/memory_system.py`:

```python
from typing import Optional, List, Dict

class MemorySystem:
    def __init__(self, log_level="info", log_mode="plain", **kwargs):
        # 实现初始化逻辑
        pass
    
    def write_memory(self, conversation: str, user_id: Optional[str] = None, 
                     agent_id: Optional[str] = None, extra_metadata: Optional[Dict] = None):
        # 实现写入逻辑
        pass
    
    def search_memory(self, query: str, user_id: Optional[str] = None, 
                      agent_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        # 实现搜索逻辑
        return []
```

### 步骤 3: 注册模型

编辑 `scripts/evaluate_system.py`，在 `AVAILABLE_MODELS` 字典中添加：

```python
AVAILABLE_MODELS = {
    'tinymem0': { ... },
    'my_model': {
        'name': 'My Memory System',
        'description': '我的记忆系统实现',
        'module': 'my_model',
        'class': 'MemorySystem'
    },
}
```

### 步骤 4: 运行评测

```bash
# 查看新模型是否注册成功
python scripts/evaluate_system.py --list-models

# 评测新模型
python scripts/evaluate_system.py --model my_model --num-samples 5
```

## 目录职责说明

### src/ - 模型实现层

- **职责**: 各种记忆系统模型的具体实现
- **特点**: 
  - 每个模型独立目录
  - 包含模型特定的 Prompt、适配器等
  - 实现统一的 MemorySystem 接口

### utils/ - 通用工具层

- **职责**: 跨项目可复用的通用工具
- **特点**:
  - 无业务逻辑依赖
  - 纯工具函数
  - 可独立提取使用

- **子模块**:
  - `llm/`: LLM 相关（本地模型加载、JSON 解析）
  - `evaluation/`: 评测指标（F1, MRR, Recall 等）
  - `model_manager/`: 模型下载和管理

### scripts/ - 脚本工具层

- **职责**: 独立的命令行工具
- **特点**:
  - 可直接运行的 Python 脚本
  - 提供命令行参数解析
  - 独立功能，互不依赖

- **核心脚本**:
  - `evaluate_system.py`: 统一评测入口（最重要）
  - `download_embedding.py`: 嵌入模型下载

### locomo/ - 评测框架层

- **职责**: LoCoMo 评测数据集和工具
- **特点**:
  - 包含标准评测数据
  - 提供评测指标计算
  - 独立的评测框架

## 核心工作流程

### 评测流程

```
用户执行命令
    ↓
scripts/evaluate_system.py (解析参数)
    ↓
动态加载模型类 (from src/{model_name})
    ↓
创建 MemorySystemEvaluator
    ↓
加载 LoCoMo 数据集
    ↓
对每个样本:
    ├── write_memory (加载对话)
    ├── search_memory (检索相关记忆)
    └── 计算评测指标
    ↓
汇总统计结果
    ↓
保存 JSON 文件
```

### 模型使用流程

```
初始化 MemorySystem
    ↓
配置环境 (本地 LLM / API)
    ↓
写入对话 (write_memory)
    ├── 提取事实
    ├── 检索相关记忆
    └── 处理冲突
    ↓
搜索记忆 (search_memory)
    ├── 向量化查询
    ├── 语义搜索
    └── 返回结果
```

## 环境配置

### 环境变量 (.env)

```env
# LLM 模式选择
USE_LOCAL_LLM=true                    # true=本地模型, false=阿里云API

# 本地 LLM 配置
LOCAL_MODEL_PATH=/path/to/model.gguf  # GGUF 模型路径
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5  # 嵌入模型

# 阿里云 API 配置（USE_LOCAL_LLM=false 时需要）
DASHSCOPE_API_KEY=your_api_key_here
```

### 依赖安装

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或安装为可编辑包
pip install -e .
```

## 扩展性说明

### 当前支持的模型

1. **tinymem0**: TinyMem0 记忆系统
   - 支持本地 LLM (GGUF) 和阿里云 API
   - 使用 Qdrant 向量数据库
   - 实现完整的记忆冲突处理

### 未来可扩展的模型

- **mem0ai**: Mem0.ai 官方实现
- **langchain_memory**: LangChain 记忆模块
- **custom_models**: 自定义记忆系统

### 扩展点

1. **新增模型**: 在 `src/` 下添加新目录
2. **新增评测指标**: 在 `utils/evaluation/` 中实现
3. **新增工具**: 在 `scripts/` 或 `utils/` 中添加
4. **新增数据集**: 在 `locomo/data/` 中添加

## 最佳实践

### 1. 代码组织

- ✅ 模型实现放在 `src/{model_name}/`
- ✅ 通用工具放在 `utils/`
- ✅ 独立脚本放在 `scripts/`
- ❌ 不要在 `utils/` 中放业务逻辑
- ❌ 不要在 `src/` 中放通用工具

### 2. 接口设计

- ✅ 遵循统一的 MemorySystem 接口
- ✅ 使用类型注解 (`typing`)
- ✅ 提供详细的文档字符串
- ❌ 不要破坏接口兼容性

### 3. 评测脚本

- ✅ 使用 `--model` 参数选择模型
- ✅ 提供 `--num-samples` 快速测试
- ✅ 保存详细的 JSON 结果
- ❌ 不要硬编码模型路径

## 相关文档

- [README.md](../README.md) - 项目主文档
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - 评测系统详细使用指南
- [scripts/README.md](../scripts/README.md) - 脚本工具说明

## 问题反馈

如有问题或建议，欢迎提 Issue 或 PR！
