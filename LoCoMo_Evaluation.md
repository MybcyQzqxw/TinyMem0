
# TinyMem0 LoCoMo 基准评测说明

本说明文档适用于本项目下的 LoCoMo 长期会话记忆基准评测，内容与当前项目结构、依赖、入口、输出完全一致。

---

## 1. 评测入口与核心文件

- **评测主入口脚本**：`run_locomo_eval.py`
- **记忆系统适配器**：`tinymem0_locomo_adapter.py`
- **LoCoMo数据集**：`locomo/data/locomo10.json`
- **依赖清单**：`requirements_locomo.txt`（需结合主`requirements.txt`）

---

## 2. 支持的评测指标

评测覆盖以下核心指标：

- **Token-level F1 Score**：主指标，反映预测与参考答案的重合度。
- **Exact Match (EM)**：预测与参考答案完全一致的比例。
- **按类别统计**：每类问题（多跳、时序、开放域、单跳、对抗）分别统计F1/EM。

---

## 3. 环境准备

- **API密钥**：在项目根目录的`.env`文件中：

  ```env
  DASHSCOPE_API_KEY=你的密钥
  ```

- **安装依赖**：

  ```bash
  pip install -r requirements_locomo.txt
  ```

  如需主系统依赖，先装`requirements.txt`，再装`requirements_locomo.txt`。

---

## 4. 评测运行方法

**快速评测（1个样本）**：
**快速评测（1个样本）**：

```bash
python run_locomo_eval.py --num-samples 1
```

**批量评测（如3个样本）**：
**批量评测（如3个样本）**：

```bash
python run_locomo_eval.py --num-samples 3
```

**全量评测（10个样本）**：
**全量评测（10个样本）**：

```bash
python run_locomo_eval.py --num-samples 10
```

**自定义输出目录**：
**自定义输出目录**：

1. **控制台输出**：

   - 评测进度（tqdm进度条）
   - 总体F1/EM分数
   - 各类别详细统计

2. **结果文件**：

   - 默认保存于：`evaluation_outputs/tinymem0_locomo_results.json`
   - 可通过`--output-dir`自定义目录
   - 结果文件内容示例：

     ```json
     {
       "evaluation_time": "2025-10-09T21:30:00",
       "data_file": "locomo/data/locomo10.json",
       "total_samples": 10,
       "overall_stats": {
         "total_questions": 45,
         "average_f1": 0.3245,
         "average_em": 0.2000
       },
       "category_stats": {
         "1": {"average_f1": 0.35, "average_em": 0.22, "count": 18},
         "2": {"average_f1": 0.31, "average_em": 0.18, "count": 15}
       },
       "detailed_results": [
         {
           "sample_id": "sample_0",
           "qa_results": [
             {
               "question": "When did Caroline go to the LGBTQ support group?",
               "answer": "7 May 2023",
               "prediction": "May 7, 2023",
               "f1_score": 0.8,
               "exact_match": 0.0,
               "category": 2
             }
           ]
         }
       ]
     }
     ```

## 5. 输出与结果说明

1. **控制台输出**：
   - 评测进度（tqdm进度条）
   - 总体F1/EM分数
   - 各类别详细统计

2. **结果文件**：
   - 默认保存于：`evaluation_outputs/tinymem0_locomo_results.json`
   - 可通过`--output-dir`自定义目录
   - 结果文件内容示例：

     ```json
     {
       "evaluation_time": "2025-10-09T21:30:00",
       "data_file": "locomo/data/locomo10.json",
       "total_samples": 10,
       "overall_stats": {
         "total_questions": 45,
         "average_f1": 0.3245,
         "average_em": 0.2000
       },
       "category_stats": {
         "1": {"average_f1": 0.35, "average_em": 0.22, "count": 18},
         "2": {"average_f1": 0.31, "average_em": 0.18, "count": 15}
       },
       "detailed_results": [
         {
           "sample_id": "sample_0",
           "qa_results": [
             {
               "question": "When did Caroline go to the LGBTQ support group?",
               "answer": "7 May 2023",
               "prediction": "May 7, 2023",
               "f1_score": 0.8,
               "exact_match": 0.0,
               "category": 2
             }
           ]
         }
       ]
     }
     ```

---

## 6. 其他说明

- 评测数据集、评测指标、输出格式均与LoCoMo官方一致。
- 评测结果仅保存于指定目录，不会污染项目根目录。
- 依赖如有缺失，请补充到`requirements_locomo.txt`并重新安装。
- TinyMem0为学习型记忆系统，性能低于商业大模型属正常现象。
