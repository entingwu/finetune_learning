今日课程重点
[ ] Q1 SmoothQuant 量化的基本原理
[ ] Q2 模型量化 GPTQ 实操
[ ] Q3 KV Cache 量化实操
[ ] Q4 量化后推理服务性能测试实操，模型性能评估实操
[ ] Q5 理解 `max_num_seqs` 和 `max_num_batched_tokens` 对整体指标的影响
[ ] Q6 理解部署时数据并行的含义，数据并行能提升推理服务哪些指标（吞吐和 TTFT），及其原因
[ ] Q7 理解部署时，为什么专家并行不能独立存在
[ ] Q8 理解并行策略配置的思路（单机多卡用张量并行，多机多卡要结合张量并行和流水线并行的原因）

---

# 答案

> 内容来自《大模型部署》PDF，括号里是章节号。讲义里没有、从论文或自己计算得到的内容，会标「补充」。

## Q1 SmoothQuant 量化的基本原理（6.1.2.2 第 6 节第 4 小节）

**1. 定位**

GPTQ 和 AWQ 只量化权重（W4A16），SmoothQuant 同时量化**权重和激活值**，典型方案是 **W8A8-INT8**。这样矩阵乘法可以直接用 INT8 整数运算，目的是提升计算效率。

**2. 要解决的问题：激活值有离群值**

- 大模型的激活值中，少数通道的数值远大于其他通道（离群值，outlier）。
- 如果这些通道和其他通道共用一个缩放因子，大值会把量化间隔拉大，大量小激活值只能被粗略表示，精度损失明显。
- 权重的分布则比较平坦，容易量化。

结论：**激活值难量化，权重容易量化。**

**3. 核心思想：把量化难度从激活值转移到权重**

对每个输入通道做等价缩放：

$$
wx = (a\,w)\cdot\frac{x}{a}
$$

对激活值大的通道取 $a>1$：激活值缩小 $a$ 倍，对应权重放大 $a$ 倍。乘积不变，所以计算结果不变，但两边的数值分布都变得容易量化了。

**4. 讲义里的数值例子**（对称 INT8，两个通道共用一个缩放因子）

| 对比项 | 平滑前 | 平滑后（第 1 通道 ÷10，对应权重 ×10） |
|---|---|---|
| 激活值 | [127, 0.63] | [12.7, 0.63] |
| 对应权重 | [0.1, 1] | [1, 1] |
| 激活量化间隔 $s_x$ | 127/127 = 1 | 12.7/127 = 0.1 |
| 0.63 量化后的值 | 1 | 0.6 |
| 0.63 的绝对误差 | **0.37** | **0.03** |

**5. 平滑系数怎么定**

- 激活值也不能缩得太多，否则权重被放大太多，权重又会变得难量化。
- 所以要用少量**校准数据**统计每个通道激活值的大小，再结合对应权重的大小确定平滑系数，在两者之间取平衡。
- 补充（SmoothQuant 论文）：对第 $j$ 个通道，
  $s_j = \dfrac{\max|X_j|^{\alpha}}{\max|W_j|^{1-\alpha}}$。
  其中 $\alpha$ 是迁移强度，一般取 0.5，表示一半难度交给权重。
- 平滑系数在部署前就确定下来，$x/a$ 这一步可以合并到前一层（如 LayerNorm）的参数里，推理时没有额外开销。

**6. 和 AWQ 的区别**

| | AWQ | SmoothQuant |
|---|---|---|
| 共同点 | 都做通道缩放 $wx=(aw)\cdot\frac{x}{a}$ | 同左 |
| 目标 | 保护重要权重，降低权重量化误差 | 缓解激活离群值，让激活值也能量化 |
| 量化对象 | 只量化权重（W4A16），激活保持 16 位 | 权重和激活都量化（W8A8） |
| 主要收益 | 节省显存 | 用 INT8 矩阵乘法提升计算效率 |

---

## Q2 模型量化 GPTQ 实操（6.1.2.3，代码见 [quantize/quantize_qwen3.py](../quantize/quantize_qwen3.py)）

**1. GPTQ 原理一句话**

按列逐步量化权重。每量化一列，就用校准数据算出这一步造成的输出误差，然后调整**还没量化**的列来补偿，使这一层的输出尽量接近量化前（误差补偿）。

**2. 用 LLM Compressor 量化的五个步骤**

| 步骤 | 做什么 | 关键代码 |
|---|---|---|
| ① 加载模型 | 读入原始 bf16 模型和 tokenizer | `AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")` |
| ② 选量化方案 | 4 位权重、16 位激活，主要目的是省显存 | `scheme="W4A16"` |
| ③ 选量化算法 | GPTQ；只量化 Linear 层，`lm_head` 不量化以保留输出层精度 | `GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])` |
| ④ 执行量化 | 准备校准数据（ultrachat_200k 取 256 条，用 chat template 转成 `text` 字段），调用 `oneshot()` | `oneshot(model, tokenizer, dataset, recipe, text_column="text", num_calibration_samples=256, max_seq_length=1024)` |
| ⑤ 保存 | 以压缩格式保存权重和量化配置，tokenizer 存到同一目录 | `model.save_pretrained(SAVE_DIR, save_compressed=True)` |

**3. 运行与部署**

```bash
pip install llmcompressor transformers datasets
python quantize_qwen3.py          # 输出到 ./model/Qwen3-8B-W4A16-GPTQ

vllm serve model/Qwen3-8B-W4A16-GPTQ --served-model-name Qwen3-8B-GPTQ
```

vLLM 会读取 `config.json` 里的量化配置，自动选择对应的 kernel，不需要额外参数。

**4. 注意点**

- 校准数据要贴近实际业务。对话模型就用对话数据，并且必须先套 chat template。
- `num_calibration_samples` 可以从 128～256 开始；`max_seq_length` 从 2048 开始，显存不够时降到 1024。
- 讲义 5.1 小节写的是「启动 Qwen3-32B」「Model loading took 61.03 GiB」，这里应该是笔误。补充（估算）：Qwen3-8B 做 W4A16 后，Linear 层约 7B 参数 × 0.5 字节 ≈ 3.5 GB，加上没有量化的 embedding 和 lm_head（bf16，约 2.5 GB），模型加载后实际占用大约 **6 GiB**，而 bf16 基线是 15.27 GiB。以自己跑出来的日志为准。

---

## Q3 KV Cache 量化实操（6.1.3，代码见 [quantize/quantization_gptq_kv.py](../quantize/quantization_gptq_kv.py)）

**1. 为什么要量化 KV Cache**

KV Cache 是显存中除权重之外最大的一块。从 BF16（2 字节）换成 FP8（1 字节）后，KV Cache 显存减半，同样的显存能放下更多 token，因此可以支持更多并发或更长上下文。

补充（按 Qwen3-8B 的 config 计算：36 层、8 个 KV 头、head_dim 128）：

| KV dtype | 每个 token 的 KV Cache | 20 GB 显存能放的 token 数 |
|---|---|---|
| BF16 | 2 × 36 × 8 × 128 × 2 B = 144 KB | 约 14.5 万 |
| FP8 | 72 KB | 约 29 万 |

**2. `--kv-cache-dtype` 的主要取值**

| 取值 | 说明 |
|---|---|
| `auto`（默认） | 跟模型精度一致，例如 BF16 |
| `fp8` / `fp8_e4m3` / `fp8_e5m2` | FP8。E4M3 精度更高，E5M2 范围更大；CUDA 上 `fp8` 等同于 `fp8_e4m3`。缩放系数可以提前校准，**没有校准时默认为 1.0**，此时直接转换类型，可能有舍入误差或数值截断 |
| `fp8/int8/int4_per_token_head` | 逐 token、逐 head **动态**计算缩放系数，不需要校准，但计算开销更大；目前依赖 Triton Attention 后端 |

**3. 实操：GPTQ W4A16 和 FP8 KV Cache 一起做**

在 Q2 的 recipe 上加一个 `kv_cache_scheme`：

```python
from compressed_tensors.quantization import QuantizationArgs

recipe = GPTQModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,          # 8 位
        type="float",        # 浮点，即 FP8
        strategy="tensor",   # 逐张量：每层的 K 和 V 各用一个 scale
        dynamic=False,       # 用校准得到的静态 scale
    ),
)
```

- `oneshot()` 在做 GPTQ 的同时，用校准数据算出每层 K 和 V 的 FP8 缩放系数，并保存到模型里。**这一步不会提前生成 KV Cache**，保存的只是缩放系数。
- `strategy="attn_head"` 表示每个 KV head 各用一个 scale，要求 FlashAttention 3 以及 Hopper 或 Blackwell 架构的 GPU。

**4. 部署**

```bash
vllm serve ./model/Qwen3-4B-W4A16-GPTQ-KVFP8 \
  --served-model-name qwen3 \
  --kv-cache-dtype fp8 \
  --max-model-len 2048
```

日志里出现 `Using fp8 data type to store kv cache` 就说明 FP8 KV Cache 已经生效。还可以对比启动日志里的 `GPU KV cache size: xxx tokens`，FP8 下应该接近 BF16 的 2 倍。

> ⚠️ 我的 [quantization_gptq_kv.py](../quantize/quantization_gptq_kv.py) 里缺少 `from compressed_tensors.quantization import QuantizationArgs`，直接运行会报 `NameError`，需要先补上这行。

---

## Q4 量化后推理服务性能测试和模型性能评估实操（4.1 / 4.2 / 6.1.2.3）

**1. 实验设计**：同一张显卡、同一套测试命令，分别测 bf16 基线和 GPTQ 量化版，只改变「是否量化」这一个因素。

**2. 推理服务性能测试（evalscope perf）**

```bash
evalscope perf \
  --url "http://127.0.0.1:8000/v1/chat/completions" \
  --parallel 20 40 \
  --number 200 200 \
  --model Qwen3-8B-GPTQ \
  --api openai --dataset openqa --stream \
  --outputs-dir ./outputs/inference_performance_test/gptq_model \
  --no-timestamp
```

需要关注的指标：

| 类别 | 指标 | 含义 |
|---|---|---|
| 吞吐 | Request Throughput | 每秒完成的请求数（req/s） |
| 吞吐 | Output Token Throughput | 每秒生成的输出 token 数（tok/s） |
| 延迟 | E2EL | 从发出请求到收到完整响应的时间 |
| 延迟 | TTFT | 从发出请求到收到第一个 token 的时间 |
| 延迟 | TPOT | 首 token 之后平均每个 token 的生成时间，$\text{TPOT}=\dfrac{\text{E2EL}-\text{TTFT}}{N_{out}-1}$ |

延迟要同时看平均值和 P50、P95、P99。生产环境中，应先定好 SLA（例如 P99 TTFT < 0.5s），再找满足 SLA 的最大并发。

**3. 模型性能评估（evalscope eval，MMLU-Pro）**

```bash
evalscope eval \
  --model Qwen3-8B-GPTQ \
  --api-url "http://127.0.0.1:8000/v1/chat/completions" --api-key EMPTY \
  --datasets mmlu_pro --limit 10 --eval-batch-size 8 \
  --generation-config '{"max_tokens":2048,"temperature":0}' \
  --work-dir ./outputs/model_performance_test/gptq_model --no-timestamp
```

**4. 讲义给出的结果**

| 指标 | Baseline (bf16) | GPTQ W4A16 |
|---|---|---|
| 模型权重显存 | 15.27 GiB | 约 6 GiB（见 Q2 说明） |
| 平均请求延迟 | 63.63 s | **23.59 s**（降低 62.9%） |
| 平均输出速率 | 213.39 tok/s | **617.14 tok/s**（2.89 倍） |
| MMLU-Pro | 57.9%（81/140） | 55.0%（77/140） |

**5. 为什么量化后变快了**（补充分析）

- **Decode 阶段受内存带宽限制**：每生成一个 token 都要把全部权重读一遍。权重从 16 GB 降到约 6 GB，每步读取的数据量少了一半多，单步更快。
- **KV Cache 空间变大**：权重省下约 9 GB 显存，全部留给 KV Cache，能同时运行的请求更多，抢占也更少。在 20 或 40 并发下，吞吐提升明显。
- **精度损失**：MMLU-Pro 下降 2.9 个百分点。不过每个子集只取 10 题（共 140 题），相差 4 题，误差范围比较大。结论是「基本持平」，真实业务中还是要用自建的垂直领域测评集来验证。

---

## Q5 `max_num_seqs` 和 `max_num_batched_tokens` 对整体指标的影响（6.2）

**1. 两个参数分别限制什么**

| 参数 | 限制的对象 | 说明 |
|---|---|---|
| `--max-num-seqs` | 每轮调度**最多同时运行的序列数** | 不限制服务能接收的请求总数，超出的请求进入等待队列 |
| `--max-num-batched-tokens` | 每轮前向计算**最多处理的 token 数** | Prefill 的输入 token 和 Decode 的新 token 都算在内；取值不能小于 `max-num-seqs` |

**2. `max_num_seqs` 的影响**

| | 调大 | 调小 |
|---|---|---|
| 吞吐 | 一批处理更多请求，GPU 利用率高，吞吐上升 | 吞吐受限 |
| TTFT | 排队的请求少，TTFT 下降 | 排队的请求多，TTFT 上升 |
| TPOT/ITL | 一批越大，单步越慢，TPOT 略有上升 | TPOT 较低 |
| 风险 | **KV Cache 不够时会触发抢占（Preemption）**：被抢占的请求回到等待队列，恢复时要重新计算上下文，吞吐和延迟反而一起变差 | 显存利用不充分 |

结论：并不是越大越好。上限由 KV Cache 容量决定。

**3. `max_num_batched_tokens` 的影响**

调度器优先安排 Decode，用剩下的 token 预算做 Prefill；放不下的部分用 Chunked Prefill 分到后面几轮。

| | 调大 | 调小 |
|---|---|---|
| TTFT | 长输入用更少轮次就能完成 Prefill，**TTFT 下降** | 长 prompt 要拆成更多轮，**TTFT 上升** |
| ITL/TPOT | 一轮里塞进大量 Prefill，单轮耗时变长，正在 Decode 的请求被拖慢，**ITL 上升** | 单轮短，**ITL 平稳** |
| 吞吐 | 一般会上升 | 可能略有下降 |

一句话：**这个参数是在 TTFT 和 ITL 之间做取舍。** 长输入场景（如 RAG、文档总结）优先保 TTFT，可以调大；交互式对话优先保 ITL，可以调小。

**4. 调优方法**（6.2.2）

1. 用接近真实业务的负载（输入长度、输出长度、到达速率）和默认参数测一遍基线，同时在 Prometheus 里观察运行请求数、等待请求数、KV Cache 使用率和抢占次数。
2. **先调 `max-num-seqs`**：如果运行请求数经常达到上限、有请求在等待、而 KV Cache 还有空间，就调大；如果频繁抢占，就调小。
3. **固定 `max-num-seqs`，再调 `max-num-batched-tokens`**：每次只改一个变量，否则分不清性能变化是哪个参数造成的。
4. 用相同负载复测，选能稳定满足 SLA 的组合，不要单纯追求最大批次。

---

## Q6 数据并行的含义，能提升哪些指标，为什么（7.1.3 / 7.2.3）

**1. 含义**

在不同 GPU（或 GPU 组）上运行**多个完整的模型副本**，请求被分配到各个副本上，各副本独立处理。

```bash
vllm serve Qwen/Qwen3-4B --data-parallel-size 2   # 2 张卡，2 个副本
```

- 每个副本都存一份完整权重，**不会降低单个副本的显存需求**，所以 DP 解决不了「单卡放不下模型」的问题。
- DP 解决的是「单卡放得下，但处理不过来」的问题。

**2. 能提升的指标及原因**

| 指标 | 是否提升 | 原因 |
|---|---|---|
| **吞吐**（req/s、tok/s） | ✅ 接近线性提升 | N 个副本并行处理请求，副本之间**不需要通信**，整体处理能力约为 N 倍 |
| **TTFT**（高负载时） | ✅ 明显下降 | TTFT = 排队时间 + Prefill 时间。高并发下 TTFT 主要来自**排队**。请求被分到 N 个副本后，每个副本的负载变为 1/N，等待队列变短，KV Cache 压力变小、抢占减少，新请求能更快进入 Prefill |
| TPOT / ITL | ⚠️ 基本不变或略有改善 | 单个请求仍然只在一个副本上跑，单步 decode 的速度由单卡的算力和带宽决定。只有在原来负载过高、批次太大时，分流后才会略好一些 |
| 低负载时的单请求延迟 | ❌ 没有提升 | 没有排队时，一个请求的计算速度不会因为多了副本而变快 |

一句话：**DP 是横向扩容，提升的是服务整体的处理能力和高峰期的排队延迟，不会让单个请求算得更快。**

---

## Q7 为什么专家并行不能单独存在（7.1.4 / 7.2.4）

**1. EP 只负责 MoE 层**

MoE 模型的每一层由两部分组成：

- **Attention 层**（以及 embedding、norm、lm_head 等）：所有 token 都要经过，属于稠密计算。
- **MoE 层（FFN 专家）**：router 把每个 token 分配给 top-k 个专家处理。

EP 的做法是把不同专家放到不同 GPU 上，token 按路由结果发送到对应的 GPU。**它只规定了专家怎么分布，没有规定 Attention 这些非 MoE 层放在哪、怎么算。**

**2. 因此必须搭配 TP 或 DP 一起用**

非 MoE 层必须有自己的并行方式：

- 用 **TP**：Attention 的权重切到多张卡上；
- 用 **DP**：每张卡上放一份完整的 Attention，各自处理不同的请求；
- 或者两者组合。

**3. vLLM 的配置方式也说明了这一点**

EP 没有单独的 `size` 参数，只有一个开关 `--enable-expert-parallel`，规模由 TP 和 DP 推导出来：

$$
EP_{size} = TP_{size} \times DP_{size}
$$

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel
# TP=1, DP=4 → EP=4：Attention 用 4 路数据并行，MoE 专家分到 4 张卡上
```

结论：EP 依附在 TP×DP 搭好的 GPU 布局上，只改变 MoE 层的分布方式，所以不能脱离 TP 或 DP 单独存在。

---

## Q8 并行策略配置思路（7.1.5 / 7.2.5）

**1. 整体决策流程**

```
单卡放得下模型吗？
├─ 放得下 → 单卡部署；吞吐不够 → 加 DP（多副本）
└─ 放不下 → 先把单个副本切分到多卡：
     ├─ 单机多卡 → TP
     └─ 多机多卡 → 机内 TP + 机间 PP
   再看吞吐是否够，不够就加 DP
（MoE 模型可以额外开启 EP）

所需 GPU 数 = TP × PP × DP
```

**2. 为什么单机多卡用 TP**

- TP 把**每一层**的权重矩阵切开，所有 GPU 同时计算同一层，然后 all-reduce 合并结果。**所有卡一直在工作，没有空闲等待。**
- 代价是通信非常频繁：每层都要通信。补充（估算）：一个标准 Transformer 层在 TP 下有 2 次 all-reduce（Attention 输出一次，MLP 输出一次）。Qwen3-8B 有 36 层，**一次前向就要做 72 次 all-reduce**。
- 单机内 GPU 之间通过 NVLink（几百 GB/s）或 PCIe 连接，带宽高、延迟低，能承受这样的通信量。
- 如果单机多卡用 PP，同一时刻只有一个阶段在计算，其余阶段在等待（流水线气泡），**GPU 利用率低**。所以单机多卡优先用 TP。

**3. 为什么多机多卡要用「机内 TP + 机间 PP」**

- 跨机器只能走网络（InfiniBand 或以太网），带宽比 NVLink 低一个数量级，延迟也高很多。
- 如果跨机器做 TP，每层两次 all-reduce 都要走网络，通信时间会远大于计算时间。
- **PP 的通信量小得多**：只在阶段边界把这一批 token 的 hidden state（tokens × hidden × 2 字节）传给下一个阶段。一次前向只需要传「阶段数 − 1」次，而且是点对点发送，不是 all-reduce。
- 所以把通信最频繁的 TP 放在机器内部（用 NVLink），把通信最少的 PP 放在机器之间（用网络）。这样通信量和带宽是匹配的。

```bash
# 例：2 台机器 × 每台 4 卡
vllm serve <model> --tensor-parallel-size 4 --pipeline-parallel-size 2
```

**4. 四种并行策略对比**

| 策略 | 切分对象 | 能否降低单副本显存 | 通信 | 主要用途 |
|---|---|---|---|---|
| TP | 每层的权重矩阵 | ✅ | 每层 all-reduce，通信量大 | 单机多卡放大模型 |
| PP | 按层分段 | ✅ | 只在阶段边界点对点传输，通信量小 | 跨机器扩展 |
| DP | 不切分，复制多份 | ❌ | 副本之间不通信 | 提升吞吐、降低排队延迟 |
| EP | MoE 层的专家 | ✅（只针对专家部分） | token 分发和收集（all-to-all） | MoE 模型，必须配合 TP 或 DP |
