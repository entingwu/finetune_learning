今日课程重点内容
[ ] Q1 vLLM 基本使用，配置工具调用，实操
[ ] Q2 工具调用的底层原理
[ ] Q3 vLLM 运行架构的三类进程
[ ] Q4 🌟 vLLM 内存管理机制：PagedAttention（口述）
[ ] Q5 vLLM 的 Prefix Caching 整体原理
[ ] Q6 vLLM 的调度机制：连续批处理的原理
[ ] Q7 vLLM 的 Attention Backend：FlashAttention
[ ] Q8 vLLM 性能测试：有哪些性能评测指标
[ ] Q9 vLLM 性能测试：实操

---

# 答案

> 内容来自《大模型部署》PDF，括号里是章节号。讲义里没有、从论文或自己计算得到的内容，会标「补充」。

## Q1 vLLM 基本使用和工具调用配置（2.1 / 2.2，代码见 [sft/inference/test_vllm_langchain.py](../sft/inference/test_vllm_langchain.py)）

**1. 安装**

```bash
pip install vllm --torch-backend=auto   # 根据 CUDA 驱动自动选择对应的 PyTorch
```

**2. 启动推理服务**

```bash
vllm serve ./model/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3
```

| 参数 | 说明 |
|---|---|
| 第一个参数 | 模型：HF Hub 上的仓库名，或本地目录 |
| `--host` | 监听地址。`127.0.0.1` 只允许本机访问，`0.0.0.0` 允许所有网卡访问（默认） |
| `--port` | 端口，默认 8000 |
| `--served-model-name` | 对外的模型名，客户端请求里的 `model` 字段要和它一致 |

**3. 开启工具调用**

要让模型自己决定是否调用工具（`tool_choice="auto"`），启动时要加两个参数：

```bash
vllm serve ./model/Qwen3-0.6B --served-model-name Qwen3-0.6B \
  --enable-auto-tool-choice \     # 允许模型自主决定是否调用工具
  --tool-call-parser hermes       # 解析模型输出中的工具调用；Qwen3 用 hermes
```

**4. 客户端调用（LangChain）**

```python
llm = init_chat_model(model="Qwen3-0.6B", model_provider="openai",
                      base_url="http://127.0.0.1:8000/v1", api_key="none",
                      use_responses_api=False)

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况。"""
    return f"{city}今天晴天，气温28℃"

llm_with_tools = llm.bind_tools([get_weather], tool_choice="auto")   # ① 绑定工具
response = llm_with_tools.invoke("帮我查询一下深圳今天的天气。")       # ② 模型返回 tool_calls
result = get_weather.invoke(response.tool_calls[0]["args"])          # ③ 客户端自己执行工具
final = llm.invoke([("user", question), response,                    # ④ 把工具结果交还给模型
                    ToolMessage(content=result, tool_call_id=response.tool_calls[0]["id"])])
```

运行方式：先按脚本开头注释里的命令启动 vLLM，再执行 `python sft/inference/test_vllm_langchain.py`。能看到先打印出 `tool_calls`，再打印最终回答，就说明工具调用配置成功。

**5. 常见问题**

| 现象 | 原因 |
|---|---|
| 报 400，提示 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser` | 启动时没有加这两个参数 |
| `tool_calls` 为空，`content` 里却出现了 `<tool_call>{...}</tool_call>` 文本 | parser 选错了，没能解析出来 |
| 请求返回 404，提示 model does not exist | 客户端的 `model` 和 `--served-model-name` 不一致 |

---

## Q2 工具调用的底层原理（2.2.2）

**核心：模型本身不会执行任何工具，它只是按照约定的格式生成了一段文本。** 「调用工具」这件事，是 vLLM 的 parser 解析这段文本，再由客户端去执行的。

**完整流程（共 5 步，一次工具调用需要请求模型 2 次）**

```
客户端                         vLLM（API Server）                       模型
  │ ① messages + tools schema ──▶ │
  │                               │ ② Chat Template 把 tools 渲染进 prompt ──▶ │
  │                               │                                           │ ③ 自回归生成
  │                               │ ◀── "<tool_call>{"name":"get_weather",     │
  │                               │      "arguments":{"city":"深圳"}}</tool_call>"
  │                               │ ④ Tool Call Parser（hermes）解析
  │ ◀── tool_calls 字段（OpenAI 格式）
  │ ⑤ 客户端执行 get_weather("深圳")
  │ ── messages + tool 结果（role=tool）──▶ 再次请求 ──▶ 模型根据结果生成最终回答
```

**每一步具体做了什么**

1. **客户端传入 tools**：每个工具的 JSON Schema，包括 name、description 和 parameters。
2. **Chat Template 渲染**：补充（以 Qwen3 的模板为例），tools 会被写进 system 部分，大致如下：
   ```
   # Tools
   You may call one or more functions to assist with the user query.
   <tools>
   {"type":"function","function":{"name":"get_weather",...}}
   </tools>
   For each function call, return a json object with function name and arguments
   within <tool_call></tool_call> XML tags.
   ```
   所以工具描述本质上是 prompt 的一部分，会占用输入 token。工具越多，prompt 越长。
3. **模型生成**：模型在训练时学过这种格式。需要调用工具时，就生成 `<tool_call>{json}</tool_call>`；不需要时，直接输出自然语言回答。
4. **Parser 解析**：`--tool-call-parser hermes` 识别 `<tool_call>` 标签，提取其中的 JSON，转成 OpenAI 格式的 `tool_calls` 字段（包含 id、name、arguments），并把这段内容从 `content` 中去掉。不同模型的输出格式不同，所以要选择对应的 parser。
5. **执行工具并回传结果**：客户端执行函数后，把结果作为 `role="tool"` 的消息、带上对应的 `tool_call_id` 发回去。补充：Chat Template 会把它渲染成 `<tool_response>...</tool_response>`，模型再根据这个结果生成最终回答。

**一句话**：工具调用 = Chat Template 把工具写进 prompt + 模型按约定格式输出 + Parser 把文本转成结构化的 `tool_calls`。真正执行工具的是客户端（Agent）。

---

## Q3 vLLM 运行架构的三类进程（3.1）

> 讲义里是「三类**进程**」（不是线程）。vLLM 采用多进程架构，进程之间通过 IPC 通信。

| 进程 | 职责 | 关键组件 |
|---|---|---|
| **API Server** | 对外入口。接收 HTTP 请求，解析参数，套用 Chat Template，**分词**后提交给 Engine Core；拿到结果后**反分词**，按协议返回，支持流式输出 | Request handling、Tokenizer、De-tokenizer、Token streaming |
| **Engine Core** | 推理引擎的核心控制。**调度**：决定每一轮计算哪些请求；管理 KV Cache；把任务下发给 Worker，再根据结果更新请求状态 | Scheduler、KV Cache Manager |
| **GPU Worker** | 在 GPU 上执行计算。启动时加载权重、分配 KV Cache；运行时准备输入，执行前向计算，返回 token 或 logits | Model Runner、Attention Backend、CUDA Graph |

**一个请求的处理流程**：用户请求 → API Server 分词 → Engine Core 调度，组成批次 → GPU Worker 前向计算 → 结果返回 Engine Core 更新状态，**这个循环反复执行**，每轮生成一个 token → API Server 反分词，流式返回 → 达到结束条件后，释放 KV Cache。

**为什么拆成多个进程**（补充）：分词、反分词和 HTTP 处理都是 CPU 密集的工作。如果和调度、GPU 计算放在同一个 Python 进程里，会受 GIL 限制互相阻塞，GPU 就得等 CPU。拆开后，CPU 的工作和 GPU 的计算可以重叠进行，GPU 利用率更高。多卡（TP）时，每张卡对应一个 GPU Worker。

---

## Q4 PagedAttention（口述版）（3.2.1.2）

**1. 要解决的问题**

KV Cache 要在多轮生成之间一直保留，并且随着生成不断变长。如果给每个请求**预先分配一段连续显存**：

- 生成长度事先不知道，只能按 `max_model_len` **预留**，大部分空间实际用不上（内部浪费）；
- 请求不断进入和结束，大小不一的空洞造成**显存碎片**（外部碎片）。

补充：PagedAttention 论文统计，传统系统中 KV Cache 显存的实际利用率只有 20%～40%。

**2. 做法：借鉴操作系统的分页内存**

| 操作系统 | vLLM |
|---|---|
| 物理页 | **物理块（Physical Block）**：把 KV Cache 显存切成固定大小的块，每块存固定数量 token 的 K 和 V（补充：默认 16 个 token） |
| 虚拟页 | **逻辑块（Logical Block）**：把一个请求的 token 序列按相同大小分块，按顺序编号 |
| 页表 | **块表（Block Table）**：记录「逻辑块 → 物理块」的映射，例如 L1→P3、L2→P1、L3→P5 |

- **按需分配**：生成到新的块时才申请一个空闲物理块，不预留。
- **不要求连续**：同一个请求的 KV 可以分散在显存各处，顺序由块表保证。
- **用完即还**：请求结束后，它的物理块立刻还给空闲池，供其他请求使用。
- **浪费很小**：只有每个请求的**最后一个块**可能没填满，最多浪费「块大小 − 1」个 token。

**3. PagedAttention 计算**

KV 分散存储之后，普通 Attention kernel 要求 KV 连续，没法直接用。PagedAttention 是一种**能直接读取非连续 KV Cache 的 Attention 计算方法**：计算时按块表找到每个逻辑块对应的物理块，依次读取 K 和 V 计算。**它不改变 Attention 的数学结果，只改变读取 KV 的方式。**

**4. 补充：算一下一个块有多大（Qwen3-0.6B：28 层、8 个 KV 头、head_dim 128，BF16）**

```
每个 token：2 (K,V) × 28 × 8 × 128 × 2 B = 114,688 B ≈ 112 KB
每个块（16 token）：112 KB × 16 ≈ 1.75 MB
```

启动日志里的 `GPU KV cache size: N tokens`，就是「KV Cache 显存 ÷ 每 token 大小」。

**5. 收益**

显存利用率接近 100%，同样的显存能同时运行更多请求，所以吞吐更高。分块管理还让 **Prefix Caching（共享块）** 和**抢占**（释放某个请求的块）变得容易实现。

**口述版**：vLLM 把 KV Cache 显存切成固定大小的物理块，每个请求的 token 按同样大小分成逻辑块，用块表记录逻辑块到物理块的映射。生成时按需分配块，不要求连续，请求结束就回收，只有最后一个块可能浪费，从而消除了预留浪费和碎片。计算 Attention 时，PagedAttention 根据块表去非连续的物理块中读取 K 和 V，数学上和普通 Attention 完全一样。

---

## Q5 Prefix Caching 整体原理（3.2.1.2）

**1. 场景**

很多请求有**相同的前缀**，例如相同的 system prompt、相同的 few-shot 示例，或者多轮对话中的历史消息。这部分的 K 和 V 完全相同，每个请求都重新做一遍 Prefill 是浪费。

> 典型例子：我们 SFT 做的意图识别任务，每条请求都带同一段很长的 system prompt（60 个意图 + 55 个槽位 + 规则），只有最后的「用户文本」不同。

**2. 原理：在 PagedAttention 的基础上共享物理块**

- 计算过的 KV 块**保留**下来。即使原请求已经结束，只要这些块没被覆盖或淘汰，后续请求仍然可以复用。
- 新请求到达时，逐块检查前缀是否和已有缓存相同。**命中时，直接把它的逻辑块映射到已有的物理块**，不重新计算，也不重复存储；没命中的部分才分配新块并计算。
- 讲义中的例子：请求 A 和 B 共享系统提示词。B 的 L1、L2 直接映射到 A 的 P3、P1，只有 L3 新分配了 P4。

**3. 补充：怎么判断「前缀相同」（vLLM V1 实现）**

- 每个**填满的块**算一个哈希：`hash(前一个块的哈希, 本块的 token ids)`。由于包含了前一个块的哈希，**哈希相同就意味着从开头到这个块的整个前缀都相同**。
- 维护一张「哈希 → 物理块」表，新请求按块逐个查表，第一次查不到就停止。
- **只有填满的块才能共享**，最后一个没填满的块不参与。所以命中粒度是块大小（16 token）。
- 每个块有**引用计数**，被多个请求共享时不会被释放。引用计数为 0 的块进入空闲队列，但内容保留，按 **LRU** 淘汰；没被淘汰之前都可以再次命中。
- V1 引擎**默认开启**，参数是 `--enable-prefix-caching` / `--no-enable-prefix-caching`。

**4. 收益和局限**

| | |
|---|---|
| ✅ 收益 | 省掉了公共前缀的 Prefill 计算，**TTFT 明显下降**，同时节省 KV Cache 显存 |
| ⚠️ 前提 | 请求之间要有**相同的前缀**，而且**必须从第一个 token 开始一致**，中间有一个 token 不同，之后的部分都无法命中 |
| ⚠️ 局限 | 前缀差异大时收益很小；缓存可能被淘汰；对 Decode 阶段没有帮助 |

**实践建议**：把固定的内容（system prompt、工具定义、few-shot 示例）**放在最前面**，把会变化的内容（用户问题、时间戳）放在后面，否则前缀一变就无法命中。

---

## Q6 连续批处理（Continuous Batching）的原理（3.2.2）

**1. 静态批处理的问题**

把若干请求组成一个固定批次，**整批都结束后**才能开始下一批。

- 不同请求的输出长度差别很大。短请求结束后，它的位置只能空等最长的那个请求，**GPU 被浪费**。
- 新请求要等整批结束才能进入，**排队时间长**，TTFT 变差。

**2. 连续批处理：以「每一轮迭代」为单位调度**（补充：出自 Orca 论文的 iteration-level scheduling）

模型的执行由很多轮前向计算组成，每轮各请求生成 1 个 token。**每轮结束后，调度器重新组织批次**：

- 已完成的请求（生成了 EOS 或达到 max_tokens）**立即移出**，释放 KV Cache；
- 未完成的请求**继续**留在批次里；
- 有空余资源时，**马上加入**等待队列中的新请求。

```
静态批处理：  [A A A A A A A A]  [D D ...]     ← D 要等 A 结束
              [B B B . . . . .]               ← B 结束后空转
              [C C C C . . . .]
连续批处理：  [A A A A A A A A]
              [B B B D D D D D]               ← B 一结束，D 就补进来
              [C C C C E E E E]
```

**3. Prefill 和 Decode 混合在同一批**

同一批中的请求可以处于不同阶段。讲义的例子：A、B 在 Decode，各需要算 1 个 token；新来的 C 在 Prefill，需要算 4 个 token。三者**拼成 6 个 token 做一次前向计算**，通过请求边界、位置信息和块表来区分，各自的 Attention 互不干扰。

**4. 配套的 Chunked Prefill（3.2.2.2）**

如果新请求的 prompt 很长（例如 1000 个 token），一次性做完 Prefill，这一轮会很慢，正在 Decode 的 A、B 都要等着，**ITL 会出现尖峰**。Chunked Prefill 把长 Prefill 分成几块，分到多轮中完成，每轮同时照顾 Decode。已经算过的块的 KV 会保留，不需要重算。

**5. 补充：调度规则（vLLM V1）**

- 每轮有一个 **token 预算**（`max_num_batched_tokens`），同时运行的请求数上限是 `max_num_seqs`。
- **Decode 优先**，剩下的预算用于 Prefill，不够就分块。
- KV Cache 不够时触发**抢占**：释放某个运行中请求的块，把它放回等待队列，恢复时重新计算。

**效果**：GPU 几乎每一轮都满载，吞吐大幅提升；新请求不用等整批结束，TTFT 下降。

---

## Q7 Attention Backend：FlashAttention（3.2.3.1，原理部分为补充）

**1. 什么是 Attention Backend**（讲义内容）

不同的 GPU、模型结构、数据类型和 KV Cache 布局，适合不同的 Attention kernel。vLLM 把 Attention 计算抽象成可替换的**后端**，启动时**自动选择兼容的实现**。常见的有 FlashAttention、FlashInfer、Triton 等。

补充：启动日志中会打印实际选中的后端，例如 `Using FLASH_ATTN backend`。也可以通过环境变量 `VLLM_ATTENTION_BACKEND` 手动指定，不同版本的写法可能不同，以官方文档为准。

**2. 标准 Attention 的瓶颈：显存读写，而不是计算**

$$
S = QK^\top,\quad P = \text{softmax}(S),\quad O = PV
$$

- 标准实现需要把 $N\times N$ 的中间矩阵 $S$ 和 $P$ **完整写回显存（HBM），再读出来**，显存占用是 $O(N^2)$。
- GPU 的片上 SRAM 很快，但只有几十到几百 KB；HBM 容量大，但慢得多。Attention 的大部分时间花在 **HBM 读写**上，属于 memory-bound。

**3. FlashAttention 的核心思想：IO-aware，分块计算，不把 N×N 矩阵写回显存**

- **Tiling（分块）**：把 Q、K、V 切成小块载入 SRAM，在 SRAM 内完成「$QK^\top$ → softmax → 乘 V」，只把最终结果 $O$ 写回 HBM。
- **Online Softmax（在线 softmax）**：softmax 需要整行的最大值和求和，但分块计算时一次只能看到一部分。做法是一边计算一边维护当前的最大值 $m$ 和分母 $l$，遇到新块时修正之前的结果：

  $$
  m^{new}=\max(m,\ \tilde m_j),\quad
  l^{new}=e^{m-m^{new}}\,l + e^{\tilde m_j-m^{new}}\,\tilde l_j,\quad
  O^{new}=e^{m-m^{new}}\,O + e^{\tilde S_j - m^{new}} V_j
  $$

  所有块处理完后，再计算 $O/l$。**结果和标准 Attention 完全相同，是精确计算，不是近似。**
- **重计算**（训练时）：反向传播不保存 $P$，用保存下来的 $m$ 和 $l$ 重新计算。用少量额外计算换来大量显存节省。

**4. 效果**

| | 标准 Attention | FlashAttention |
|---|---|---|
| 中间矩阵 | N×N 写回 HBM | 不写回，只在 SRAM 里分块计算 |
| 显存 | $O(N^2)$ | $O(N)$ |
| 速度 | 受显存带宽限制 | 读写大幅减少，通常快 2～4 倍 |
| 结果 | 精确 | **精确**（数学上等价） |

后续版本：FA2 改进了并行和任务划分；FA3 针对 Hopper（H100）做了异步和 FP8 优化。讲义中 KV Cache 逐头 FP8 量化要求 FA3，就是这个原因。

**5. 和 PagedAttention 的关系**

两者关注的点不同：**PagedAttention 解决「KV 存放在哪里」**（分块、非连续），**FlashAttention 解决「Attention 怎么算更快」**（分块计算、减少显存读写）。vLLM 的 FlashAttention 后端会传入块表，**直接从分页的 KV Cache 中读数据**进行分块计算，两者结合使用。

---

## Q8 性能评测指标（4.1.1）

**1. 吞吐（Throughput）：服务整体的处理能力**

| 指标 | 单位 | 含义 |
|---|---|---|
| 请求吞吐 Request Throughput | req/s | 每秒成功完成的请求数 |
| 输出 Token 吞吐 Output Token Throughput | tok/s | 每秒生成的输出 token 数 |

**2. 延迟（Latency）：单个请求的体验**

| 指标 | 含义 | 主要影响因素 |
|---|---|---|
| **E2EL** 端到端延迟 | 从发出请求到收到完整响应 | 输出长度 × TPOT + TTFT |
| **TTFT** 首 Token 延迟 | 从发出请求到收到第一个 token | 排队时间 + Prefill（prompt 越长越慢） |
| **TPOT** 单 Token 生成时间 | 首 token 之后平均每个 token 的时间，$\dfrac{E2EL-TTFT}{N_{out}-1}$ | Decode 速度（批越大越慢） |
| **ITL**（补充） | 相邻两个 token 之间的时间间隔 | 和 TPOT 类似，但 TPOT 是每个请求的平均值，ITL 看的是每一个间隔，能反映卡顿（例如长 Prefill 插队时出现的尖峰） |

**3. 看分位数，不只看平均值**

P50、P95、P99 分别表示 50%、95%、99% 的请求不超过这个值。**P99 反映最慢那一批用户的体验**，生产环境的 SLA 通常按 P99 来定。

**4. 其他**：成功率、超时率。高负载下请求失败的话，前面的数字都没有意义。

**5. 补充：几个指标之间的关系**

- 并发增加 → 每批更大 → **吞吐上升，TPOT 变大**；超过饱和点后，吞吐不再增长，**TTFT 因为排队急剧上升**。
- 所以「极限性能」指的是**在延迟满足 SLA 的前提下能达到的最大并发**，而不是吞吐最大时的并发（4.1.3）。
- vLLM 的 `/metrics` 中也有对应指标：`vllm:time_to_first_token_seconds`、`vllm:inter_token_latency_seconds`、`vllm:e2e_request_latency_seconds` 等，可以接入 Prometheus 和 Grafana 持续监控（见 day7 Q2）。

---

## Q9 性能测试实操（4.1.2 / 4.1.3）

**1. 安装**

```bash
pip install "evalscope[perf]"
```

**2. 启动被测服务**（本地 Mac 用 0.6B；AutoDL 上可以换成 8B）

```bash
vllm serve ./model/Qwen3-0.6B --served-model-name Qwen3-0.6B --max-model-len 4096
```

**3. 基础压测**

```bash
evalscope perf \
  --url "http://127.0.0.1:8000/v1/chat/completions" \
  --model Qwen3-0.6B \
  --api openai --dataset openqa --stream \
  --parallel 5 --number 20 \
  --warmup-num 5
```

| 参数 | 含义 |
|---|---|
| `--parallel` | 并发数，模拟同时访问的用户数。可以写多档，用空格分隔 |
| `--number` | 总请求数。档数要和 `--parallel` 一致 |
| `--dataset openqa` | 用开放问答数据作为输入 |
| `--warmup-num` | 预热请求数，不计入统计 |
| `--stream` | 流式输出。**必须开**，否则测不出 TTFT |

**4. 多档并发，找极限性能**

```bash
evalscope perf \
  --url "http://127.0.0.1:8000/v1/chat/completions" \
  --model Qwen3-0.6B --api openai --dataset openqa --stream \
  --parallel 8 16 32 64 128 \
  --number 400 400 400 400 400 \
  --outputs-dir ./outputs/perf --no-timestamp
```

**5. 怎么看结果**

1. 先定 SLA，例如 **P99 TTFT < 0.5s，P99 E2EL < 8s**。
2. 每一档关注三点：
   - 吞吐（req/s、tok/s）**是否还在增长**；
   - P95/P99 的 TTFT 和 E2EL **是否突然变差**；
   - **成功率**是否下降，是否出现超时。
3. 吞吐不再增长、而 TTFT 急剧上升的那一档就是**饱和点**。**满足 SLA 的最大并发**就是服务容量。
4. 压测时可以同时看 Grafana 上的 `num_requests_running`、`num_requests_waiting` 和 KV Cache 使用率：等待队列持续增长，说明已经超出处理能力。

**6. 我的测试结果**（运行后填写）

| 并发 | req/s | 输出 tok/s | 平均 TTFT | P99 TTFT | 平均 TPOT | 平均 E2EL | 成功率 |
|---|---|---|---|---|---|---|---|
| 8 | | | | | | | |
| 16 | | | | | | | |
| 32 | | | | | | | |
| 64 | | | | | | | |
| 128 | | | | | | | |

结论：在 SLA 为 ______ 时，最大可用并发为 ______。

> 注意：Mac 上的 vLLM 跑在 CPU 上，测出的数值只能用来熟悉流程，不能代表 GPU 上的服务性能。正式结论要在 AutoDL 的 GPU 上测。
