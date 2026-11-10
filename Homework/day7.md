今天的问题清单
[ ] Q1 尝试使用公开测评集完成一个对模型能力的测评
[ ] Q2 vLLM 监控体系搭建（按课件说明）
[ ] Q3 🌟 推理过程中显存是怎么分配的，口述 vLLM 显存分配机制，`--gpu-memory-utilization` 参数的作用
[ ] Q4 整数对称量化和非对称量化的计算公式
[ ] Q5 理解权重量化为什么按 per-output-channel 完成
[ ] Q6 仅对权重量化 vs 对权重与激活值都量化 的区别
[ ] Q7 PTQ 和 QAT 两种量化方式的具体区别
[ ] Q8 🌟 AWQ 的核心思想（口述）
[ ] Q9 🌟 GPTQ 的核心思想（口述）

---

# 答案

> 内容来自《大模型部署》PDF，括号里是章节号。讲义里没有、从网上或源码查到的内容，会标「补充」并附来源。

## Q1 用公开测评集测评模型能力（4.2.1）

**命令**（讲义原命令基础上，为适应 Mac 本机算力做了缩减）：

```bash
source ~/venvs/vllm/bin/activate
cd Homework
evalscope eval \
  --model Qwen3-0.6B \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key EMPTY \
  --datasets mmlu_pro \
  --eval-batch-size 8 \
  --limit 3 \
  --work-dir ./day7_eval/mmlu_pro_nothink \
  --generation-config '{"max_tokens":512,"temperature":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --no-timestamp
```

| 参数 | 含义 |
|---|---|
| `--datasets mmlu_pro` | evalscope 内置的 MMLU-Pro，14 个学科子集，10 选 1 选择题 |
| `--limit 3` | 每个子集只取前 3 题（讲义用 10；Mac 上推理慢，先缩小规模） |
| `--eval-batch-size 8` | 同时发 8 个请求 |
| `--generation-config` | `max_tokens` 限制输出长度；`temperature: 0` 使结果可复现；`enable_thinking: false` 关闭 Qwen3 的思考模式，避免输出过长 |
| `--work-dir` / `--no-timestamp` | 结果写到固定目录，不加时间戳 |

**结果**（2026-11-09，Mac 本机 vLLM，Qwen3-0.6B fp16，非思考模式，总耗时约 8.5 分钟）

| 子集 | 准确率 | 子集 | 准确率 |
|---|---|---|---|
| engineering | 100% (3/3) | business | 33.3% |
| computer science | 66.7% | philosophy | 33.3% |
| physics | 66.7% | other | 33.3% |
| biology | 33.3% | math / chemistry / law | 0% |
| | | health / economics / psychology / history | 0% |
| **OVERALL** | **26.2%（11/42）** | | |

- 平均输入 1338 token：evalscope 默认用 **5-shot**，每题前面都拼了 5 道示例题
- 平均输出 144 token，平均每题耗时 75 秒（8 个请求并发，受限于 Mac 本机算力）
- 结果文件：`Homework/day7_eval/mmlu_pro_nothink/`
  - `reports/report.html`：可视化报告
  - `reports/Qwen3-0.6B/mmlu_pro.json`：汇总分数
  - `reviews/`：每题的模型输出、提取出的答案、是否答对
  - 运行日志：`Homework/day7_eval_run.log`

**结论**

- 26.2% 明显高于随机猜的 10%（MMLU-Pro 有 10 个选项），说明 0.6B 模型有一定的知识和推理能力，但整体比较弱。讲义里 Qwen3-8B 是 57.9%。
- 每个子集只有 3 题，单个子集的分数波动很大，不能用来比较学科强弱，只能看 OVERALL。要得到可靠结论，至少要用讲义的 `--limit 10`，最好跑全量。
- 关掉思考模式、限制 `max_tokens` 是为了省时间，代价是牺牲了 CoT 推理。数学、化学这类要计算的题全错，很可能和这个有关。可以对比打开思考模式的结果（`enable_thinking: true`，`max_tokens` 调到 2048 以上）。
- 测评用法：量化、微调前后用**同一套配置**各跑一次，对比分数，判断模型能力有没有下降（讲义 6.1.2.3 就是这样比较 GPTQ 和基线的）。

---

## Q2 vLLM 监控体系搭建（5.1 / 5.2）

已搭好，完整步骤见 [vllm-monitor/README.md](../vllm-monitor/README.md)。

```
vLLM (127.0.0.1:8000/metrics)
   │ Prometheus 每 5s 拉取（Pull 模式）
   ▼
Prometheus (127.0.0.1:9090) ──> Grafana (127.0.0.1:3000，导入 vLLM 官方 grafana.json)
```

- `/metrics`：vLLM 以 Prometheus 格式暴露运行指标
- Prometheus：按 `scrape_interval` 主动访问 `/metrics`，采集并存储数据
- Grafana：数据源填 `http://prometheus:9090`（两个容器在同一个 compose 网络里，用服务名访问），再导入 Dashboard
- 和讲义不一样的地方：vLLM 跑在 Mac 本机，不需要 SSH 隧道，target 直接写 `host.docker.internal:8000`

常用指标（5.1）：

| 指标 | 含义 |
|---|---|
| `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` | 累计输入 / 输出 token 数 |
| `vllm:time_to_first_token_seconds` | 首 Token 延迟（TTFT）分布 |
| `vllm:inter_token_latency_seconds` | 相邻输出 Token 间隔（ITL）分布 |
| `vllm:e2e_request_latency_seconds` | 端到端延迟分布 |
| `vllm:num_requests_running` / `_waiting` | 正在执行 / 排队的请求数 |
| `vllm:kv_cache_usage_perc` | KV Cache 使用率 |

---

## Q3 vLLM 显存分配机制，`--gpu-memory-utilization` 的作用（3.2.1 / 6.1.1）

> 口述：vLLM 的显存分成四块：模型权重、计算中间数据（激活值）、运行时资源（CUDA 上下文等）、KV Cache。启动时，先用 `--gpu-memory-utilization` 定一个显存预算（占总显存的比例）。然后加载权重，再用一次模拟前向计算，测出激活值和运行时资源的峰值。预算减去这些开销，剩下的**全部预分配给 KV Cache**，并切成固定大小的物理块。之后请求来了，就按需分配块，用 Block Table 记录逻辑块到物理块的映射，也就是 PagedAttention。

**1. 显存的四部分（3.2.1.1）**

| 部分 | 特点 | 谁来管理 |
|---|---|---|
| 模型权重 | 启动时加载，大小固定 | vLLM |
| 计算中间数据 | 前向计算的激活值、临时张量 | PyTorch |
| 运行时资源 | CUDA 上下文、计算缓冲区（CUDA Graph 等） | CUDA / PyTorch |
| KV Cache | 随请求动态增长，在多轮推理之间一直保留 | vLLM 分页管理 |

**2. 启动时的分配流程（6.1.1）**

1. `--gpu-memory-utilization` 设定预算：`预算 = 总显存 × 比例`
2. 加载模型权重
3. 模拟前向计算（profile），测出激活值和运行时资源的峰值
4. `KV Cache 显存 = 预算 − 权重 − 激活峰值 − 其他开销`
5. 把这部分显存切成固定大小的物理块（Physical Block），预先占好

补充（vLLM 0.11.0 源码 `vllm/v1/worker/gpu_worker.py`）：
```python
requested_memory = total_memory * gpu_memory_utilization
available_kv_cache_memory_bytes = requested_memory - profile_result.non_kv_cache_memory
```

**3. `--gpu-memory-utilization` 的作用**

- 含义：**当前这个 vLLM 实例**最多使用总显存的多大比例。它是单实例的上限，不考虑同一张卡上的其他进程。
- 默认值：补充，本地 vLLM 0.11.0 默认是 `0.9`。较新的 vLLM 文档里默认值已改为 `0.92`，以所装版本为准。
- 调大：KV Cache 更多，能容纳更长的上下文、更高的并发，抢占更少；但留给其他程序的显存更少，容易 OOM。
- 调小：适合一张卡上跑多个实例，或者需要给其他进程留显存。
- 注意：它只**控制分配**，不会减少权重和 KV Cache 本身需要的空间。要真正省显存，得靠量化（6.1.2 / 6.1.3）。
- 也可以用 `--kv-cache-memory-bytes` 直接指定 KV Cache 大小，设置后就不再按比例自动计算。

**4. 为什么要分页（3.2.1.2）**

- 如果连续分配：不同请求长度不同，频繁地分配和释放会产生显存碎片，还得预留最大长度，浪费空间
- 分页：KV Cache 切成固定大小的物理块，一个请求的逻辑块通过 Block Table 映射到任意空闲的物理块，不要求连续
- 只有最后一个块可能没填满，浪费很少
- 在此基础上还有 Prefix Caching：前缀相同的请求，逻辑块直接映射到已有的物理块，省掉重复的 Prefill

**5. 自己算一下：Qwen3-0.6B 每个 token 的 KV Cache 有多大**（补充，按本地 config.json 计算）

```
每 token KV 字节数 = 2(K 和 V) × 层数 × KV头数 × head_dim × 每元素字节数
                 = 2 × 28 × 8 × 128 × 2 (fp16)
                 = 114,688 B ≈ 112 KiB
```
→ 1 GiB 的 KV Cache 大约能放 9,362 个 token。`--kv-cache-dtype fp8` 每元素只占 1 字节，能放的 token 数翻倍（6.1.3）。

---

## Q4 整数对称量化和非对称量化的计算公式（6.1.2.2 第 2 节）

**非对称量化**（缩放 + 平移 + 取整）

| 步骤 | 公式 |
|---|---|
| 缩放因子 | $s = \dfrac{x_{max} - x_{min}}{q_{max} - q_{min}}$ |
| 零点 | $z = q_{min} - \dfrac{x_{min}}{s}$ |
| 量化 | $q = \mathrm{round}\left(\dfrac{w}{s} + z\right)$ |
| 反量化 | $\hat{w} = s\,(q - z)$ |

例子（讲义）：权重范围 [-1, 2]，量化成 INT4 [-8, 7]
- $s = \frac{2-(-1)}{7-(-8)} = 0.2$，$z = -8 - \frac{-1}{0.2} = -3$
- $w = 0.55$：$q = \mathrm{round}(0.55/0.2 - 3) = \mathrm{round}(-0.25) = 0$，$\hat w = 0.2 × (0 + 3) = 0.6$，误差 0.05

**对称量化**（只缩放 + 取整，零点固定为 0）

| 步骤 | 公式 |
|---|---|
| 缩放因子 | $s = \dfrac{\max(\lvert w_{min}\rvert, \lvert w_{max}\rvert)}{q_{max}}$ |
| 量化 | $q = \mathrm{round}\left(\dfrac{w}{s}\right)$ |
| 反量化 | $\hat{w} = s \times q$ |

例子（讲义）：权重范围 [-2, 2]，INT4 只用 [-7, 7]（不用 -8，保证正负两边对称），$s = 2/7 ≈ 0.286$

**对比**

| | 对称 | 非对称 |
|---|---|---|
| 参数 | 只有 $s$ | $s$ 和 $z$ |
| 适合的分布 | 正负范围接近、以 0 为中心（权重大多如此） | 偏向一侧（如 ReLU 之后的激活值） |
| 优点 | 计算简单 | 能用满整个整数范围，适应性更强 |

**浮点量化（FP8 / FP4）**：$s = \max(\lvert x_{min}\rvert,\lvert x_{max}\rvert)/F_{max}$，$q = \mathrm{Round_{FP}}(x/s)$，$\hat{x} = s × q$

---

## Q5 为什么权重量化按 per-output-channel 做（6.1.2.2 第 3 节 + 补充）

**讲义部分：量化粒度**

| 粒度 | 说明 |
|---|---|
| Per-Tensor | 整个张量共用一组量化参数 |
| Per-Channel | 每个通道单独一组 |
| Per-Group | 每组（如 128 个数）单独一组 |

讲义的例子：两个输出通道，范围分别是 [-1, 1] 和 [-10, 10]，对称 INT4
- 共用一个 scale：$s = 10/7 ≈ 1.43$，第一个通道里的 0.4 被量化成 0，误差 0.4
- 每个通道各用一个 scale：第一个通道 $s = 1/7$，0.4 被量化成 3，反量化得 0.429，误差 0.029

**原因一：数值范围（精度角度）**

权重矩阵每一行对应一个输出神经元，不同输出通道的数值范围可能差很多。共用一个 scale，会被范围最大的那个通道主导，范围小的通道就被量化得很粗。

**原因二：scale 能提到矩阵乘法外面（效率角度，补充）**

线性层 $y_j = \sum_k W_{jk}\, x_k$（$j$ 是输出通道，$k$ 是输入通道，求和沿 $k$ 进行）

- **按输出通道**：同一行 $j$ 共用 $s_j$，$s_j$ 和求和下标 $k$ 无关，可以提到求和外面：
  $$y_j = \sum_k s_j\, q_{jk}\, x_k = s_j \sum_k q_{jk}\, x_k$$
  → 先用低精度整数做完整个矩阵乘法，最后每个输出元素**只乘一次** $s_j$。反量化的开销是「每个输出一次」，不是「每次乘加一次」。
- **按输入通道**：$s_k$ 随求和下标变化，提不出来，只能在每次乘加里都乘一次 $s_k$，整数 GEMM 就用不了。

所以 per-output-channel 既能提高精度，又不破坏高效的矩阵乘法 kernel。Per-Group 是把它进一步细分到输入维度上，每 128 个一组，精度更高，代价是要存更多 scale，kernel 里也要分组反量化。W4A16 一般采用 group size 128。

来源：[Integer Quantization for Deep Learning Inference (arXiv 2004.09602)](https://arxiv.org/pdf/2004.09602)；[量化矩阵乘法布局推导 (hackmd)](https://hackmd.io/@Afn5isQ9RRKCQuXiomYR4A/H1OLIAxvGg)

---

## Q6 仅量化权重 vs 权重和激活值都量化（6.1.2.2 第 4 节）

以 $Y = WX$ 为例：

| | 仅量化权重（W4A16 / W8A16） | 权重 + 激活值都量化（W8A8） |
|---|---|---|
| 权重 | INT4 / INT8 | INT8 / FP8 |
| 激活值 | FP16 / BF16 | INT8 / FP8 |
| 计算方式 | 权重**先反量化回 FP16**，再做**浮点矩阵乘法** | 直接做**整数矩阵乘法**，结果再乘 $s_W s_X$ 恢复尺度 |
| 公式 | $Y = \mathrm{dequant}(Q_W)\,X$ | $Y ≈ (s_W Q_W)(s_X Q_X) = s_W s_X (Q_W Q_X)$ |
| 主要收益 | **省显存**：权重小，显存带宽压力小，对 Decode 这种受显存带宽限制的阶段帮助大 | **算得更快**：用上硬件的 INT8/FP8 Tensor Core，算力提升，对 Prefill、高并发帮助大 |
| 难点 | 精度损失小 | 激活值有离群值，难量化；需要确定激活值的 scale |
| 典型算法 | GPTQ、AWQ | SmoothQuant |

激活值是随输入变化的，确定 scale 有两种方式：
- **静态**：推理前用校准数据算好，推理时直接用。开销小，但可能不适应新的输入
- **动态**：推理时根据当前激活值实时计算。适应性好，但有额外开销

怎么选（6.1.2.3）：主要想省显存就选 W4A16；主要想提高吞吐就选 W8A8-INT8 / W8A8-FP8，FP8 要求 Ada（8.9）及以上的 GPU。

---

## Q7 PTQ 和 QAT 的区别（6.1.2.2 第 5 节）

| | PTQ 训练后量化 | QAT 量化感知训练 |
|---|---|---|
| 时机 | 模型训练完以后直接转换 | 在训练 / 微调过程中做 |
| 是否改参数 | 不重新训练（GPTQ 会调整权重，但不做反向传播） | 通过反向传播更新参数 |
| 数据 | 不需要数据（RTN），或只要少量校准数据（GPTQ / AWQ / SmoothQuant） | 需要训练数据 |
| 过程 | 根据权重范围（加上校准数据）定量化参数 → 转成低精度 | 前向时模拟量化 + 反量化，让量化误差参与 loss → 反向更新 → 反复迭代 → 最后转成低精度 |
| 成本 | 低，几分钟到几小时 | 高，需要训练时间和算力 |
| 精度 | 一般，低位宽时可能掉点明显 | 通常更好 |
| 应用 | 大模型部署的主流做法 | 精度要求高、愿意付出训练成本的场景 |

**校准**：用少量有代表性的输入，对比同一层量化前后的输出，据此调整量化参数，让量化后的输出尽量接近原来。

---

## Q8 AWQ 的核心思想（6.1.2.2 第 6 节）

> 口述：不同权重的量化误差，对输出的影响是不一样的。权重误差要乘上对应的输入激活值，激活值大的通道，误差就被放大。AWQ 用校准数据找出激活值大的「重要通道」（大约 1%）。它不把这些通道保留成 FP16（那样是混合精度，硬件跑起来效率低），而是把重要权重**乘上一个大于 1 的系数 a**，同时把对应的输入激活值**除以 a**，乘积不变。只要组内最大值、也就是 scale 没被抬高，放大后的权重量化得更细，折算回原来尺度后误差就变小了。最终只有权重被量化成 INT4，激活值保持 FP16。

**核心等式**：$wx = (aw)\cdot\dfrac{x}{a}$，$a > 1$

**为什么放大权重能减小误差（讲义例子，对称 INT4 分组量化）**

| | 原来 | 重要权重放大 2 倍 |
|---|---|---|
| 权重组 | [1.4, **0.28**] | [1.4, **0.56**] |
| scale | 1.4/7 = 0.2 | 组内最大值仍是 1.4，所以还是 0.2 |
| 量化结果 | 0.28 → 0.2 | 0.56 → 0.6，折算回去 0.6/2 = 0.3 |
| 误差 | 0.08 | **0.02** |

关键条件：**scale 不能跟着同比例变大**，否则放大就没用了。所以 AWQ 要用校准数据去搜索合适的放大倍数。

要点：
- 属于 PTQ，不需要训练，需要少量校准数据
- 只量化权重，典型用法是 W4A16
- 激活值只用来**决定**哪些通道要保护，本身不被量化

---

## Q9 GPTQ 的核心思想（6.1.2.2 第 6 节）

> 口述：RTN 是每个权重各自四舍五入，只让单个权重的误差最小，不管对整层输出的影响。GPTQ 的思路是**误差补偿**：按列依次量化权重矩阵，每量化完一列，就用校准数据算出这一步给整层输出带来的误差，然后**调整后面还没量化的权重**，把这个误差抵消掉。接着量化下一列，一直重复到所有列都量化完。目标是让量化后这一层的**输出**尽量接近原来，而不只是让每个权重本身接近。

**讲义的直观例子**：$y = w_1x_1 + w_2x_2$，校准数据里 $x_1 = x_2$

$w_1$ 量化后变成 $w_1 - 0.1$ → 把还没量化的 $w_2$ 调成 $w_2 + 0.1$：
$$y' = (w_1 - 0.1)x_1 + (w_2 + 0.1)x_2 = y + 0.1(x_2 - x_1) = y$$

输出没变，第一个权重的误差被第二个权重补偿掉了。

要点：
- 属于 PTQ，需要校准数据，用来衡量权重变化对输出的影响
- 只能尽量减小误差，不能完全消除：真实输入更复杂，而且调整后的权重自己也还要量化
- 比 RTN 精度高，量化过程的计算量也更大
- 适合 INT4 这类低位宽的权重量化（W4A16）

**GPTQ 和 AWQ 对比**

| | GPTQ | AWQ |
|---|---|---|
| 思路 | 误差补偿：调整还没量化的权重 | 激活感知：放大重要权重来保护它们 |
| 校准数据的用途 | 计算输出误差，决定怎么补偿 | 统计激活值大小，找重要通道 |
| 共同点 | PTQ、只量化权重、常用 W4A16 | 同左 |

讲义实测（Qwen3-8B，RTX 5090，6.1.2.3）：GPTQ W4A16 相比 BF16 基线，平均延迟从 63.63s 降到 23.59s，输出吞吐提高到 2.89 倍，MMLU-Pro 准确率从 57.9% 降到 55%，基本持平。
