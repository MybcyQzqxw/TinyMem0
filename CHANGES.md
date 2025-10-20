# TinyMem0 本地化改造日志

## 改造概述

将TinyMem0从阿里云API依赖改造为支持本地GGUF模型，并增强LoCoMo评测指标实现。

### 改造目标

1. ✅ 支持本地GGUF模型替代阿里云通义千问API
2. ✅ 支持本地嵌入模型替代阿里云文本嵌入API
3. ✅ 通过环境变量灵活切换API/本地模式
4. ✅ 实现完整的LoCoMo评测指标（QA + Evidence/Retrieval）
5. ✅ 充分复用locomo文件夹中的评测代码

## 核心改动清单

### 1. 新增文件

#### `local_llm.py`
- **功能**: 封装llama-cpp-python，提供本地GGUF模型推理接口
- **核心类**: `LocalLLM`
  - `generate()`: 文本生成方法
  - `chat()`: 对话生成方法
- **单例函数**: `get_local_llm()` 获取全局模型实例
- **配置参数**:
  - `n_ctx=4096`: 上下文窗口大小
  - `n_gpu_layers=-1`: GPU加速（全部层）
  - `temperature=0.7`: 采样温度
  - `top_p=0.9`: nucleus采样参数

#### `.env.example`
- **功能**: 环境变量配置模板
- **包含配置**:
  - `USE_LOCAL_LLM`: 本地/API模式切换
  - `LOCAL_MODEL_PATH`: GGUF模型文件路径
  - `LOCAL_EMBEDDING_MODEL`: 本地嵌入模型名称
  - `EMBEDDING_DIM`: 嵌入向量维度
  - `DASHSCOPE_API_KEY`: 阿里云API密钥（API模式）

#### `CHANGES.md`
- **功能**: 本改造日志文档

### 2. 修改文件

#### `util.py`
**改动位置**: `call_llm_with_prompt()` 函数

```python
# 新增导入
from local_llm import get_local_llm

# 新增逻辑
use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
if use_local_llm:
    llm = get_local_llm()
    return llm.generate(prompt)
else:
    # 原API调用逻辑
```

#### `memory_system.py`
**改动1**: `get_embeddings()` 方法支持本地嵌入模型

```python
use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
if use_local_llm:
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
else:
    # 原阿里云API调用
```

**改动2**: `_init_collection()` 方法支持动态向量维度

```python
default_dim = 512 if os.getenv("USE_LOCAL_LLM", "false").lower() == "true" else 1536
self.vector_dim = int(os.getenv("EMBEDDING_DIM", default_dim))
```

**改动3**: `write_memory()` 方法新增 `extra_metadata` 参数

```python
def write_memory(self, conversation, user_id=None, agent_id=None, extra_metadata=None):
    # ...
    metadata = {
        "user_id": user_id,
        "agent_id": agent_id,
        "timestamp": timestamp,
        **(extra_metadata or {})  # 合并额外metadata
    }
```

#### `tinymem0_locomo_adapter.py`
**改动1**: `load_conversation_into_memory()` 添加metadata追踪

```python
extra_metadata = {
    "session_id": session_id,
    "dialog_id": dialog_id
}
self.memory_system.write_memory(
    conversation, 
    extra_metadata=extra_metadata
)
```

**改动2**: `answer_question()` 返回evidence列表

```python
def answer_question(self, question: str, context_limit: int = 10) -> tuple[str, List[str]]:
    # ...检索记忆
    retrieved_evidence = [
        f"{mem['metadata']['session_id']}_{mem['metadata']['dialog_id']}"
        for mem in memories
    ]
    return answer, retrieved_evidence
```

**改动3**: 新增Evidence评测指标方法

- `_calculate_evidence_metrics()`: 计算Precision/Recall/F1
- `_calculate_recall_at_k()`: 计算Recall@K (K=5,10)
- `_calculate_mrr()`: 计算MRR (Mean Reciprocal Rank)

**改动4**: `evaluate_qa_sample()` 收集所有指标

```python
# QA指标
qa_metrics = {
    "token_f1": f1_score(predicted, reference),
    "exact_match": exact_match_score(predicted, reference)
}

# Evidence指标
evidence_metrics = self._calculate_evidence_metrics(retrieved_evidence, gold_evidence)
recall_5 = self._calculate_recall_at_k(retrieved_evidence, gold_evidence, k=5)
recall_10 = self._calculate_recall_at_k(retrieved_evidence, gold_evidence, k=10)
mrr = self._calculate_mrr(retrieved_evidence, gold_evidence)
```

**改动5**: `evaluate_tinymem0()` 汇总并打印所有指标

```python
# 打印QA Metrics
print(f"Token-level F1: {avg_f1:.4f}")
print(f"Exact Match: {avg_em:.4f}")

# 打印Evidence/Retrieval Metrics
print(f"Evidence Precision: {avg_evidence_precision:.4f}")
print(f"Evidence Recall: {avg_evidence_recall:.4f}")
print(f"Evidence F1: {avg_evidence_f1:.4f}")
print(f"Recall@5: {avg_recall_5:.4f}")
print(f"Recall@10: {avg_recall_10:.4f}")
print(f"MRR: {avg_mrr:.4f}")
```

#### `requirements.txt` & `requirements_locomo.txt`
**新增依赖**:
```
llama-cpp-python>=0.2.0
sentence-transformers>=2.2.0
```

#### `README.md`
**新增章节**:
- "## 本地LLM配置": 详细说明环境变量配置和模型安装
- "## LoCoMo评测指标说明": 说明所有QA和Evidence/Retrieval指标

## 技术细节

### 本地LLM架构

```
┌─────────────────────────────────────────────────┐
│              TinyMem0 Application               │
├─────────────────────────────────────────────────┤
│  util.py: call_llm_with_prompt()                │
│    ├─ if USE_LOCAL_LLM == true                  │
│    │   └─ local_llm.py: LocalLLM.generate()     │
│    │       └─ llama-cpp-python: Llama           │
│    │           └─ GGUF模型文件                   │
│    └─ else                                       │
│        └─ dashscope.Generation.call()           │
│            └─ 阿里云通义千问API                  │
└─────────────────────────────────────────────────┘
```

### 嵌入模型架构

```
┌─────────────────────────────────────────────────┐
│         memory_system.py: MemorySystem          │
├─────────────────────────────────────────────────┤
│  get_embeddings()                               │
│    ├─ if USE_LOCAL_LLM == true                  │
│    │   └─ SentenceTransformer                   │
│    │       └─ BAAI/bge-small-zh-v1.5 (512维)    │
│    └─ else                                       │
│        └─ dashscope.TextEmbedding               │
│            └─ 阿里云文本嵌入 (1536维)            │
└─────────────────────────────────────────────────┘
```

### Evidence匹配机制

```
1. 对话加载时: 为每条对话添加 session_id + dialog_id metadata
2. 问题回答时: 检索相关记忆，提取其metadata中的 session_id_dialog_id
3. Evidence评测: 将检索到的 session_id_dialog_id 与参考答案的evidence列表对比
4. 指标计算:
   - Precision: 检索结果中相关evidence的比例
   - Recall: 参考evidence中被检索到的比例
   - F1: Precision和Recall的调和平均
   - Recall@K: 前K个结果中包含相关evidence的比例
   - MRR: 第一个相关evidence的排名倒数
```

## 环境变量配置说明

### 必需配置（本地LLM模式）

```bash
# Windows PowerShell
$env:USE_LOCAL_LLM = "true"
$env:LOCAL_MODEL_PATH = "path/to/model.gguf"

# Linux/Mac
export USE_LOCAL_LLM=true
export LOCAL_MODEL_PATH=path/to/model.gguf
```

### 可选配置

```bash
# 本地嵌入模型（默认: BAAI/bge-small-zh-v1.5）
$env:LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 嵌入向量维度（默认: 本地512, API 1536）
$env:EMBEDDING_DIM = "512"
```

### API模式配置

```bash
$env:USE_LOCAL_LLM = "false"
$env:DASHSCOPE_API_KEY = "your_api_key_here"
```

## 推荐模型

### LLM模型（GGUF格式）

- **Qwen2-7B-Instruct-GGUF**: 中文优化，推理速度快
- **Llama3-8B-Chinese-Chat-GGUF**: 中文能力强，社区支持好
- **ChatGLM3-6B-GGUF**: 轻量级，适合资源受限环境

### 嵌入模型

- **BAAI/bge-small-zh-v1.5** (推荐): 512维，中文优化，检索效果好
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**: 384维，多语言支持

## 评测指标说明

### QA任务指标

| 指标 | 说明 | 计算方法 | 取值范围 |
|------|------|----------|----------|
| Token-level F1 | token级别的F1分数 | 2 * (P * R) / (P + R) | [0, 1] |
| Exact Match (EM) | 完全匹配的比例 | 预测==参考 ? 1 : 0 | {0, 1} |

### Evidence检索指标

| 指标 | 说明 | 计算方法 | 取值范围 |
|------|------|----------|----------|
| Evidence Precision | 检索结果的准确率 | 相关evidence数 / 检索总数 | [0, 1] |
| Evidence Recall | 参考evidence的召回率 | 检索到的相关evidence数 / 参考总数 | [0, 1] |
| Evidence F1 | Precision和Recall的调和平均 | 2 * (P * R) / (P + R) | [0, 1] |
| Recall@5 | 前5个结果的召回率 | 前5个中相关evidence数 / 参考总数 | [0, 1] |
| Recall@10 | 前10个结果的召回率 | 前10个中相关evidence数 / 参考总数 | [0, 1] |
| MRR | 平均倒数排名 | 1 / 第一个相关evidence的排名 | (0, 1] |

## 测试验证清单

- [ ] 配置.env文件（复制.env.example）
- [ ] 安装requirements.txt依赖
- [ ] 安装requirements_locomo.txt依赖
- [ ] 下载GGUF模型文件
- [ ] 设置LOCAL_MODEL_PATH环境变量
- [ ] 运行tinymem0_locomo_adapter.py
- [ ] 验证QA指标输出正确
- [ ] 验证Evidence指标输出正确
- [ ] 检查qdrant_data目录创建正常
- [ ] 测试API模式切换（可选）

## 已知问题

1. **首次运行时**本地嵌入模型会自动下载，需要网络连接
2. **GGUF模型加载**需要足够的系统内存（7B模型约需8GB）
3. **GPU加速**需要安装CUDA版本的llama-cpp-python

## 后续优化建议

1. **性能优化**: 
   - 添加模型量化选项（Q4_0, Q5_0等）
   - 实现批量推理提升吞吐量
   
2. **功能增强**:
   - 支持更多本地LLM后端（如vLLM, TGI）
   - 添加模型缓存机制减少加载时间
   
3. **评测扩展**:
   - 支持更多LoCoMo子任务
   - 添加评测结果可视化

## 参考资料

- [llama-cpp-python文档](https://llama-cpp-python.readthedocs.io/)
- [sentence-transformers文档](https://www.sbert.net/)
- [LoCoMo基准测试](https://github.com/lm-sys/LoCoMo)
- [GGUF模型格式说明](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

**改造完成时间**: 2024年
**改造作者**: GitHub Copilot
**项目版本**: TinyMem0 v2.0 (本地化版本)
