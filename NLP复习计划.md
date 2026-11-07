# NLP 复习计划（为大模型微调做准备）

> **目标**：搞懂 Transformer / GPT / HF Trainer，能读懂本 repo 的 `01_Qwen3.py`、`02_sft_demo.py`、`03_model_generate.py`、`04_dpo.py`、`05_sft_demo.py`
>
> **时间**：10/12（周一）– 10/28（周三），只在**周一、二、三、五**学习，每天 **2 小时**，共 **11 天，22 小时**

## 使用说明

- **视频目录**：`~/Desktop/SVNLP/尚硅谷大模型技术之NLP/4.视频/`
- **代码目录**：`~/PycharmProjects/NLP/translation/`、`~/PycharmProjects/NLP/review_hf/`
- **重敲代码**放在 `~/PycharmProjects/NLP/retype/`：新建空文件，**不看原文**写，写完再对照原文
- **倍速规则**（时间按这个规则算过，排得很满，请严格遵守）：
  - ⭐⭐⭐ = **1 倍速**，必须认真看
  - ⭐ = **1.5 倍速**
  - 不标 = **2 倍速**
- 括号里写的是**按倍速换算后的实际耗时**
- 📖 读书属于可选项。时间不够就跳过，书留着当查阅资料用
- 当天没做完的部分，优先挪到周四或周末补；实在补不上，就用 **10/28 的缓冲时间**

## 日程总览

| 日期 | 主题 | 视频 | 动手 |
|---|---|---|---|
| 10/12 周一 | 自注意力 | 1h45 | 自测 |
| 10/13 周二 | Mask、位置编码、nn.Transformer API | 1h50 | 自测 |
| 10/14 周三 | API、DataLoader、模型定义 | 1h25 | 读自己的代码 |
| 10/16 周五 | 训练循环 | 1h05 | **重敲①** |
| 10/19 周一 | 自回归预测 | 1h20 | **重敲②** + 修 bug |
| 10/20 周二 | 手写多头注意力 | 55min | **重敲③**（最难） |
| 10/21 周三 | GPT、BERT、HF 分词器 | 1h50 | 自测 |
| 10/23 周五 | HF Datasets、手写训练循环 | 1h20 | **重敲④** |
| 10/26 周一 | Trainer 参数、Trainer 改造 | 1h40 | 修 bug + 总自测 |
| 10/27 周二 | 补 Trainer 参数视频、自定义 loss | 30min | **重敲⑤** |
| 10/28 周三 | 缓冲 + 预习 Qwen3 | — | 补进度 / 读 `01_Qwen3.py` |

---

## 10/12 周一｜自注意力（共约 2h）

- [ ] `day_02/13-NLP-Seq2Seq-推理机制` ⭐（16min）：自回归生成、贪心解码、束搜索
- [ ] `day_02/21-NLP-Transformer-整体结构` ⭐⭐⭐（14min）
- [ ] `day_02/23-NLP-Transformer-编码器-自注意力子层-上` ⭐⭐⭐（36min）
- [ ] `day_03/01-NLP-Transformer-编码器-自注意力子层-回顾`（5min）
- [ ] `day_03/02-NLP-Transformer-编码器-自注意力子层-下` ⭐⭐⭐（13min）
- [ ] `day_03/03-NLP-Transformer-编码器-前馈神经网络`（2min）
- [ ] `day_03/04-NLP-Transformer-编码器-维度梳理` ⭐⭐⭐（6min）
- [ ] `day_03/05-NLP-Transformer-编码器-残差连接`（5min）
- [ ] `day_03/06-NLP-Transformer-编码器-层归一化`（7min）
- [ ] 📖 可选：读书 7.2.3.2（p.40–46）

**自测（10min）**
- Q/K/V 分别是怎么来的？为什么要除以 √d_k？
- softmax 是对注意力矩阵的行做还是列做？
- 多头注意力：`(B, T, d_model)` 拆成多头时形状怎么变？合并回来之后为什么还要乘 W_o？
- LayerNorm 是在哪个维度上做归一化的？

## 10/13 周二｜Mask、位置编码、nn.Transformer API（共约 2h）

- [ ] `day_03/07-NLP-Transformer-编码器-位置编码-上`（11min）
- [ ] `day_03/08-NLP-Transformer-编码器-位置编码-下`（15min）
- [ ] `day_03/09-NLP-Transformer-编码器-小结`（2min）
- [ ] `day_03/10-NLP-Transformer-解码器-概述`（4min）
- [ ] `day_03/11-NLP-Transformer-解码器-Mask自注意力子层` ⭐⭐⭐（11min）
- [ ] `day_03/12-NLP-Transformer-解码器-小结`（6min）
- [ ] 读笔记 `day_03/缩放注意力.md`（10min），用来代替 41 分钟的 `13-缩放注意力系数` 视频
- [ ] `day_03/14-NLP-Transformer-训练和推理机制` ⭐⭐⭐（14min）
- [ ] `day_04/01-NLP-Transformer-API-概述`（4min）
- [ ] `day_04/02-NLP-Transformer-API-构造参数`（9min）
- [ ] `day_04/03-NLP-Transformer-API-forward-上` ⭐⭐⭐（26min）

**自测（5min）**
- 因果 mask 为什么是加 `-inf` 的上三角矩阵？加完之后经过 softmax 会变成什么？
- 为什么训练时能一次并行算完整个序列，推理时却只能一步一步来？

## 10/14 周三｜API、DataLoader、模型定义（共约 2h）

- [ ] `day_04/04-NLP-Transformer-API-forward-下` ⭐⭐⭐（15min）
- [ ] `day_04/05-NLP-Transformer-API-encoder&decoder` ⭐（5min）
- [ ] `day_05/01-NLP-Transformer-翻译案例-DataLoader` ⭐（28min）
- [ ] `day_05/02-NLP-Transformer-翻译案例-模型定义-组件声明`（9min）
- [ ] `day_05/03-NLP-Transformer-翻译案例-模型定义-encode方法`（20min）
- [ ] `day_05/04-NLP-Transformer-翻译案例-模型定义-decode&forward方法` ⭐（7min）
- [ ] 📖 书 p.66–72 中 4 种 mask 参数的表格（10min）
- [ ] 💻 读自己写的 `translation/src/dataset.py` 和 `model.py`（15min）

**自测**
- `src_key_padding_mask`、`tgt_key_padding_mask`、`tgt_mask`、`memory_key_padding_mask` 各自用在哪个注意力里？形状分别是什么？
- `dataset.py` 里 `en[:-1]` 和 `en[1:]` 为什么要这样切？

## 10/16 周五｜训练循环 + 重敲①（共约 2h）

- [ ] `day_05/05-NLP-Transformer-翻译案例-模型训练-准备工作`（16min）
- [ ] `day_05/06-NLP-Transformer-翻译案例-模型训练-核心逻辑` ⭐⭐⭐（30min）
- [ ] 💻 **重敲①**（40min）：`collate_fn`（用 `pad_sequence`）+ `train_one_epoch`（错开一位、`ignore_index`、两处 reshape）
- [ ] 🐛 修 `translation/src/train.py:66`：`'$H'` → `'%H'`
- [ ] `day_06/07-NLP-Transformer-哈佛实现-多头注意力-上` ⭐⭐⭐（22min），为下周二手写多头注意力预热

**自测**
- 对照 `Finetune/02_sft_demo.py:288` 的 `labels = data_tensor[:,1:]`，说清楚它和翻译案例的错位是同一件事
- `model.decode` 没传 `tgt_key_padding_mask`，为什么结果仍然是对的？（提示：pad 在右边 + 因果 mask）如果换成左 padding，会出什么问题？

## 10/19 周一｜自回归预测 + 重敲②（共约 2h）

- [ ] `day_06/02-NLP-Transformer-翻译案例-模型预测-核心逻辑-上` ⭐⭐⭐（46min）
- [ ] `day_06/03-NLP-Transformer-翻译案例-模型预测-核心逻辑-下` ⭐⭐⭐（34min）
- [ ] 💻 **重敲②**（35min）：批量贪心生成 `predict_batch`，用 `is_finish` 张量标记结束，最后按 eos 截断
- [ ] 🐛 修 `translation/src/evaluate.py:16`：`predict_batch` 少传了 `device`
- [ ] 🐛 修 `translation/src/evaluate.py:24`：`corpus_bleu` 的参数顺序反了，应该是 `(references, predictions)`

**自测**
- 为什么只取 `decoder_output[:, -1, :]`？
- 每一步都要把整个前缀重新输入一遍，会浪费什么？KV Cache 是怎么解决这个问题的？
- 对照 `Finetune/03_model_generate.py`：为什么要把 `input_ids` 那一段截掉？

## 10/20 周二｜手写多头注意力 + 重敲③（共约 2h）⚠️ 最难的一天

- [ ] `day_06/08-NLP-Transformer-哈佛实现-多头注意力-中` ⭐⭐⭐（21min）
- [ ] `day_06/09-NLP-Transformer-哈佛实现-多头注意力-下` ⭐⭐⭐（14min）
- [ ] 💻 **重敲③**（60min）：手写 `MultiHeadAttention(nn.Module)`，同时支持因果 mask 和 padding mask
  - 验收：用 `torch.allclose` 和 `F.scaled_dot_product_attention(q, k, v, is_causal=True)` 对比结果一致
- [ ] `day_06/10-NLP-预训练模型-概述`（7min）
- [ ] `day_06/11-NLP-预训练模型-分类` ⭐（14min）

## 10/21 周三｜GPT、BERT、HF 分词器（共约 2h）

- [ ] `day_07/01-NLP-预训练模型-GPT-模型结构` ⭐⭐⭐（14min）
- [ ] `day_07/02-NLP-预训练模型-GPT-预训练` ⭐⭐⭐（7min）
- [ ] `day_07/03-NLP-预训练模型-GPT-微调` ⭐⭐⭐（24min）
- [ ] `day_07/04-NLP-预训练模型-BERT-模型结构`（15min）
- [ ] `day_07/05-NLP-预训练模型-BERT-预训练`（7min）
- [ ] `day_07/06-NLP-预训练模型-BERT-微调` ⭐（14min）
- [ ] `day_08/03-NLP-预训练模型-HF-分词器-使用-上` ⭐（11min）
- [ ] `day_08/04-NLP-预训练模型-HF-分词器-使用-下` ⭐（16min）
- [ ] `day_08/05-NLP-预训练模型-HF-分词器&模型-配套使用`（4min）

**自测**
- GPT 预训练和 `translation/` 里 decoder 的训练有什么相同点？去掉了哪两样东西？
- 为什么 BERT 做分类用 `[CLS]`，GPT 做分类却用最后一个 token？
- Qwen 属于三类预训练模型中的哪一类？

## 10/23 周五｜HF Datasets、手写训练循环 + 重敲④（共约 2h）

- [ ] `day_08/08-NLP-预训练模型-HF-Datasets-数据预处理-上` ⭐（16min）
- [ ] `day_08/09-NLP-预训练模型-HF-Datasets-数据预处理-下`（7min）
- [ ] `day_08/11-NLP-预训练模型-HF-Datasets-集成DataLoader` ⭐（15min）
- [ ] `day_08/13-NLP-预训练模型-案例-数据预处理`（7min）
- [ ] `day_08/16-NLP-预训练模型-案例-训练-核心逻辑` ⭐⭐⭐（4min）
- [ ] `day_08/17-NLP-预训练模型-案例-训练-BatchSize选择策略` ⭐（5min）
- [ ] 💻 **重敲④**（35min）：把 `CrossEntropyLoss(ignore_index=pad)` 改成手写的 `log_softmax → gather → mask → sum / mask.sum()`
  - 验收：结果和 `CrossEntropyLoss` 一致；然后读 `Finetune/02_sft_demo.py:150` 的 `compute_loss`
- [ ] `day_09/04-NLP-预训练模型-HF-Trainer-概述` ⭐⭐⭐（13min）
- [ ] `day_09/09-NLP-预训练模型-HF-Trainer-TrainingArguments参数-TrainingDurationandBatchSize` ⭐⭐⭐（15min）

**自测**
- `review_hf/src/train.py` 里没有写 loss 函数，loss 是从哪里来的？
- 换成因果语言模型（CausalLM）后，传进去的 labels 还需要自己错开一位吗？

## 10/26 周一｜Trainer 参数、Trainer 改造 + 总复习（共约 2h）

- [ ] `day_09/10-NLP-预训练模型-HF-Trainer-TrainingArguments参数-LR&Scheduler` ⭐⭐⭐（8min）
- [ ] `day_09/12-NLP-预训练模型-HF-Trainer-TrainingArguments参数-TrainingStability` ⭐（8min）
- [ ] `day_09/13-NLP-预训练模型-HF-Trainer-TrainingArguments参数-MixedPrecisionTraining` ⭐⭐⭐（28min）
- [ ] `day_09/14-NLP-预训练模型-HF-Trainer-TrainingArguments参数-GradientCheckpointing` ⭐⭐⭐（3min）
- [ ] `day_09/15-…-Evaluation`、`15-…-Logging`、`16-…-Checkpointing`、`17-…-BestModelTrack`（合计 7min）
- [ ] `day_10/01-NLP-预训练模型-案例-Trainer改造-上` ⭐⭐⭐（9min）
- [ ] `day_10/02-NLP-预训练模型-案例-Trainer改造-下` ⭐⭐⭐（38min）
- [ ] 🐛 修 `review_hf/src/train_hf.py:5`：在 `import config`（第 7 行）之前就用了 `config.LOGS_DIR`，会报 `NameError`
- [ ] ✅ **总自测（15min）**：不看代码，向自己讲清楚 `02_sft_demo.py` 的完整流程：数据 → padding → labels 错位 → assistant mask → loss → 学习率调度

**自测**
- 总 step 数怎么算？（`train_hf.py` 里注释的 `50000 * 3 / (4 * 2)` 是怎么来的？）
- bf16 和 gradient checkpointing 分别省的是什么？各自要付出什么代价？

---

## 10/27 周二｜Trainer 收尾 + 重敲⑤（共约 2h）

- [ ] `day_09/05-NLP-预训练模型-HF-Trainer-Trainer参数-一`（5min）
- [ ] `day_09/06-NLP-预训练模型-HF-Trainer-Trainer参数-二`（7min）
- [ ] `day_09/07-NLP-预训练模型-HF-Trainer-Trainer参数-三`（约 6min）
- [ ] `day_09/11-NLP-预训练模型-HF-Trainer-TrainingArguments参数-Optimizer`（8min）
- [ ] 💻 **重敲⑤**（55min）：把 `review_hf/src/train.py` 的手写训练循环改写成 `Trainer` 版，加上 `compute_metrics` 和 `EarlyStoppingCallback`
  - 验收：准确率和手写版接近（约 0.94）。如果本机跑太慢，用 `dataset.select(range(2000))` 取一个小子集跑通流程就行
- [ ] 💻 读 `title_classifier/src/training/train.py` 的 `compute_loss_func`，对照 `Finetune/04_dpo.py` 的 `compute_loss`（15min）：两者都是往训练里塞自定义损失
- [ ] 📖 对照本 repo 的 `SFTConfig参数.md`，把这两周认识的参数标出来（15min）

## 10/28 周三｜缓冲 + 预习 Qwen3（共约 2h）

**先补前面没做完的部分。** 补完之后，按下面顺序用剩下的时间：

- [ ] 💻 读 `Finetune/01_Qwen3.py`（约 60min）：标出和标准 Transformer **不一样**的地方（RoPE、RMSNorm、SwiGLU、GQA、qk_norm），以及**一样**的地方（因果 mask、残差、多头注意力）。不一样的部分不用当天弄懂，标出来留到微调课时补
- [ ] 💻 把 `predict_batch` 的 argmax 换成 temperature + top-k 采样（20min）
- [ ] 💻 在 10/20 写的 MultiHeadAttention 外面包一层 Pre-Norm Block（Attention + FFN + 残差）（40min）

## 仍然没排进去的（有空再看）

- [ ] 🎬 `day_02/12-Seq2Seq-训练机制`、`day_03/13-缩放注意力系数`、`day_04/06-翻译案例-需求说明&思路分析`

## 明确跳过的内容

| 内容 | 原因 |
|---|---|
| `day_01` 全部（导论、Word2Vec、RNN） | 和微调关系不大。只有 `08/09-英文分词`（BPE）有空可以看 |
| `day_02/01–11、14–18`（RNN、Seq2Seq 结构、Attention 评分函数） | 历史演进，概念知道就行 |
| `day_02/done/` 里的 16、19、20、22 | 已经看过 |
| `day_04/07–16`、`day_05/07–10`、`day_06/01、04、05`、`day_08/01、02、06、07、10、12、14、15、18–22`、`day_09/01–03`、`day_10/03` | 工程细节、内容重复、AutoDL |
| `NLP/review/`、`NLP/translation-transformer/` | 分别和 `review_hf/`、书上代码重复 |

## 书里没讲、微调前还要补的知识

读 `01_Qwen3.py` 时会遇到下面这些，等开始微调课程时再补：

- [ ] RoPE 旋转位置编码（对应配置里的 `rope_theta`）
- [ ] RMSNorm（对应 `rms_norm_eps`）
- [ ] SwiGLU（对应 `hidden_act: silu`）
- [ ] GQA（对应 `num_key_value_heads`）
- [ ] KV Cache
- [ ] Chat Template（`apply_chat_template`），以及“只对 assistant 部分算 loss”
