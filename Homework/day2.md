今天的问题清单
[+] Q1（可选）DeepSeek-V4 的 CSA、HCA 是什么
  https://sebastianraschka.com/llm-architecture-gallery/csa-hca/
[+] Q2 SiLU、Swish 的形式
[+] Q3 SwiGLU 的结构（口述）
[+] Q4 MoE 的结构、原理，Router 怎么工作，MoE 的优势（口述）
[+] Q5 RMSNorm 的计算方式；Pre-Norm 和 Post-Norm；Pre-Norm 的问题和解决方案
[+] Q6 LLM 训练分哪几个阶段，每个阶段的训练机制，能力边界（口述）
[+] Q7 LLM 推理：生成过程是怎样的
[+] Q8 token 选择策略：两种确定性生成怎么做；随机采样有哪些控制参数，分别怎么控制（口述）
[+] Q9 推理代码：看懂、跑一遍

---

# 答案

## Q1（可选）DeepSeek-V4 的 CSA、HCA 是什么

> 来源是题目链接里的解读文章，没有核对 DeepSeek-V4 原论文。

两者都是**长上下文下的注意力压缩方法**：把较早 token 的 KV 合并成更少的条目，减少需要存储和计算的历史长度。

| | CSA（Compressed Sparse Attention） | HCA（Heavily Compressed Attention） |
|---|---|---|
| 压缩率 | 每 4 个位置合并为 1 条 | 每 128 个位置合并为 1 条 |
| 注意力方式 | 用 top-k 索引器选一部分条目，稀疏计算 | 对全部压缩条目稠密计算 |
| 特点 | 保留更多历史细节 | 更粗略，但历史更短 |

两者都为最近的 128 个 token 保留未压缩的 KV，并在网络中交替使用。代价是会损失一部分细粒度的历史信息。

## Q2 SiLU、Swish 的形式

- **SiLU：** $\text{SiLU}(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是 sigmoid。
- **Swish：** $\text{Swish}(x) = x \cdot \sigma(\beta x)$。

$\beta = 1$ 时两者完全等价。可学习的 $\beta$ 收益有限，还可能让训练不稳定，所以通常固定为 1，工程上也常混用这两个名字。

## Q3 SwiGLU 的结构（口述）

> SwiGLU 是门控 FFN，用 SiLU 作为门控函数。输入分成两路线性层：一路是主分支，另一路是门控分支，门控分支经过 SiLU 激活。两路结果逐元素相乘，再经过一个线性层映射回隐藏维度。

$$\text{FFN}(x) = W_{down}\big(\text{SiLU}(W_{gate}\,x) \odot W_{up}\,x\big)$$

代码：[01_Qwen3.py:171-174](../01_Qwen3.py#L171-L174)

## Q4 MoE 的结构、原理，Router 怎么工作，MoE 的优势（口述）

> **结构和原理：** MoE 用多个并行的 FFN 专家替代单个 FFN，再由 Router 根据 token 的特征选出少数专家参与计算。

**Router 的工作流程：**

1. **计算得分：** 用线性映射把 token 的隐藏表示转成每个专家的得分。
2. **选专家：** 选出得分最高的 k 个，其余不参与本次计算。
3. **算权重：** 对选中专家的得分做 softmax。
4. **专家计算：** 被选中的专家各自独立处理这个 token。
5. **合并输出：** 按权重对专家输出加权求和。

例子：得分为 `3.2, 0.7, 5.8, 1.6`，选中 3.2 和 5.8，softmax 后权重约为 `0.07` 和 `0.93`。

**优势：**

- **高容量、低计算：** 总参数量很大，但每次只激活少数专家。
- **分工协作：** 专家会形成功能分化，提升表达能力和泛化能力。
- **易于分布式扩展：** 专家相互独立，可以分布到不同设备上并行运行。

## Q5 RMSNorm 的计算方式；Pre-Norm 和 Post-Norm；Pre-Norm 的问题和解决方案

**RMSNorm 的计算方式：**

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \qquad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

其中 $\gamma$ 是可学习的缩放参数，$\epsilon$ 用来防止除零。它去掉了 LayerNorm 的均值标准化，计算更简单。

代码：[01_Qwen3.py:461-463](../01_Qwen3.py#L461-L463)

**三种归一化位置：**

| 方式 | 公式 | 梯度 | 特点 |
|---|---|---|---|
| Post-Norm | $y = \text{Norm}(x + F(x))$ | $\frac{\partial \text{Norm}(z)}{\partial z}\left(1 + \frac{\partial F(x)}{\partial x}\right)$，其中 $z = x + F(x)$ | 恒等路径受扰动，深层训练不稳定，现代大模型基本不再使用 |
| Pre-Norm | $y = x + F(\text{Norm}(x))$ | $1 + \frac{\partial F(\text{Norm}(x))}{\partial x}$ | 保留恒等梯度路径，是当前主流 |
| Post-Norm in Residual | $y = x + \text{Norm}(F(x))$ | $1 + \frac{\partial \text{Norm}(F(x))}{\partial x}$ | 同样保留恒等路径，近年部分模型采用 |

`01_Qwen3.py` 的 [TransformerBlock](../01_Qwen3.py#L147-L158) 用的是 Pre-Norm。

**Pre-Norm 的问题和解决方案：**

> PDF 只讲了 Post-Norm 的问题，没有明确写 Pre-Norm 的问题。以下是补充内容，需要和课堂内容核对。

- **问题：** 主路径 $x$ 全程没有被归一化，每层的 $F$ 都直接累加上去，数值尺度可能随层数增大，同时子层的输出也没有经过归一化。
- **解决方案一：** 在所有 Block 之后加 **Final Norm**。PDF 的 Qwen3 结构图里有 Final RMSNorm，代码在 [01_Qwen3.py:116](../01_Qwen3.py#L116)。
- **解决方案二：** 使用 PDF 提到的 **Post-Norm in Residual**，先把子层输出归一化再相加，控制每层累加量的大小。

## Q6 LLM 训练分哪几个阶段，每个阶段的训练机制，能力边界（口述）

> 三个阶段的模型主体不变，变的是训练数据和目标，本质上都在调整下一个 token 的概率分布。

| 阶段 | 数据 | 训练机制 | 获得的能力 | 能力边界 |
|---|---|---|---|---|
| **预训练** | 大规模自然文本 | 预测下一个 token，$-\sum_t \log p_\theta(x_t \mid x_{<t})$ | 语言规律、知识、上下文建模 | 学到的是统计规律，不是事实判断，高概率内容不一定真实；擅长续写，但不能稳定遵循指令 |
| **监督微调（SFT）** | “指令—回答”示范对话，经 Chat Template 序列化 | 读完整对话，用损失掩码只计算 Assistant 部分，$-\sum_t m_t \log p_\theta(y_t \mid y_{<t})$ | 从“续写文本”变为“响应指令” | 受基础模型能力和示范质量限制；一条样本只给一个目标回答，无法说明哪个回答更好 |
| **偏好对齐** | 同一问题的多个回答及偏好关系 | 提高优质回答相对较差回答的概率，常见方法是 RLHF 和 DPO | 更倾向于有用、真实、安全的回答 | 不会补充新知识，也不能突破基础模型的能力上限；偏好标准有偏差时，模型可能学到并放大偏差 |

偏好对齐公式 $-(\log P_\theta(y_w \mid x) - \log P_\theta(y_l \mid x))$ 只是帮助理解的抽象表示，**不是实际的损失函数**。

## Q7 LLM 推理：生成过程是怎样的

1. **构造输入：** 用 Chat Template 把消息转成序列，末尾加上 Assistant 起始标记，再分词。
2. **Prefill：** 一次前向计算并行处理全部输入，用最后一个 token 的隐藏状态算出第一个新 token 的概率分布，并建立 KV Cache。
3. **Decode：** 每轮只计算新 token 的 Q、K、V，把 K、V 追加进缓存，再用 Q 和全部 K、V 做注意力计算，得到下一个 token 的概率分布。由于后一个 token 依赖前一个，这一步只能顺序执行。
4. **选择 token：** 每一步按贪心、Beam Search 或随机采样选出下一个 token。
5. **终止：** 生成 EOS，或达到最大 token 数量时停止。

## Q8 token 选择策略：两种确定性生成怎么做；随机采样有哪些控制参数，分别怎么控制（口述）

**确定性生成：**

- **贪心搜索：** 每步选概率最高的 token，$y_t = \arg\max_i p_i$。它简单、开销低，但局部最优不一定是整体最优，长文本还可能重复。
- **Beam Search：** 每步保留多条累计得分较高的候选序列，结束后选得分高的结果。它适合翻译、摘要，但计算和存储开销更大，开放式对话较少使用。

**随机采样的控制参数：**

| 参数 | 控制什么 | 怎么控制 |
|---|---|---|
| **Temperature** | 分布的平滑程度 | $p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$。$T<1$ 更集中、更稳定，$T>1$ 更平缓、更多样，$T \to 0$ 接近贪心；它不排除低概率 token |
| **Top-K** | 候选数量 | 只保留概率最高的 K 个，重新归一化后采样；数量固定，无法适应分布变化 |
| **Top-P** | 累计概率 | 保留累计概率达到 P 的最小集合；分布集中时候选少，分散时候选多 |

## Q9 推理代码：看懂、跑一遍

用本地 `model/Qwen3-0.6B`，在 MPS 上按 PDF 的流程跑通。

Chat Template 的结果（输入长度 23 个 token）：

```text
<|im_start|>user\n请简要解释大语言模型如何生成文本。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

| 生成方式 | 结果 |
|---|---|
| 随机采样（`T=0.7, top_p=0.8, top_k=20`） | 输出形状 `[1, 177]`，新生成 154 个 token，用时 6.1 秒，以 EOS（151645）结束 |
| 贪心和 Beam Search（`num_beams=3`，各 60 token） | 都正常输出，这次只有一个词不同（“模式”和“规律”） |
| `TextStreamer`、`TextIteratorStreamer` | 都能流式输出，这两次设了 60 token 上限 |

**代码要看懂的四点：**

- **Chat Template：** `enable_thinking=False` 会预先加一个空的 `<think>` 区块，让模型直接回答。
- **返回值：** `generate()` 返回“输入 + 新生成内容”，所以要先按输入长度截取，再解码。
- **流式输出：** `TextIteratorStreamer` 需要独立线程，只是为了边生成边读取，不会加快生成。
- **参数差异：** `Qwen3-0.6B/generate.ipynb` 的 cell 3 用的是 `top_p=0.9`，而 PDF 引用的模型卡建议是 `0.8`。
