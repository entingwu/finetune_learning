今天的重点内容
[ ] Q1 QLoRA 训练代码、整体流程，实操一遍
[ ] Q2 数据并行的具体过程
[ ] Q3 🌟 数据并行中梯度的 AllReduce 过程：第一阶段、第二阶段分别做什么，通信量是多少
[ ] Q4 流水线并行的具体原理、问题（气泡，GPU 利用率低）、优化方案
[ ] Q5 张量并行的具体原理、问题（单层内通信多）
[ ] Q6 ZeRO 有哪些分片策略
[ ] Q7 🌟 ZeRO-3 的整体过程（要能口述）
[ ] Q8 用 Accelerate 完成一次分布式训练，整体流程实操一遍

---

# 答案

> 内容来自《大模型微调》PDF，括号里是章节号。讲义里没有、从论文或自己计算得到的内容，会标「补充」。

## Q1 QLoRA 训练代码和整体流程（4.3 / 6.4，代码见 [sft/src/train_qlora.py](../sft/src/train_qlora.py)）

**1. QLoRA 是什么**

LoRA 冻结了基础模型，只训练低秩矩阵 A、B，梯度和优化器状态都很小。但冻结的基础模型仍要以 BF16 加载到显存，模型越大，这部分越成为主要负担。

QLoRA 在 LoRA 的基础上，把**基础模型的线性层权重以 4-bit 存储**（冻结），LoRA 参数仍然以较高精度训练。

| 方式 | 静态显存（字节） | Qwen3-4B 估算（补充） |
|---|---|---|
| 全参（混合精度 + AdamW） | $18P$ | 4B × 18 ≈ 72 GB |
| LoRA | $2P + 18T$ | 约 8 GB + LoRA 部分 |
| QLoRA | $M_{4bit\ base} + 18T$ | 约 2.5～3 GB + LoRA 部分 |

（$P$ 是基础模型参数量，$T$ 是 LoRA 参数量，$T \ll P$。以上不含激活值。）

**2. 三项核心技术**

| 技术 | 解决什么问题 | 怎么做 |
|---|---|---|
| **NF4 量化** | 权重近似正态分布，集中在 0 附近，均匀量化浪费了两端的格点 | 把标准正态分布按累积概率分成 16 个等概率区间，每个区间取中位数作为量化值。格点在 0 附近密、两端稀，和权重分布匹配 |
| **双重量化** | 分块量化时每个块都有一个 FP32 缩放因子，数量很多，本身也占显存 | 把缩放因子以 256 个一组，再用 8-bit 量化一次 |
| **分页优化器** | 训练中显存有短时峰值，可能 OOM | 利用 NVIDIA 统一内存，按页管理优化器状态；显存紧张时自动把页换到 CPU 内存，用到时再换回来。**目的是防止 OOM，不是加速** |

**3. 代码流程（6.4.2）**

```python
# ① 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 4-bit 加载
    bnb_4bit_quant_type="nf4",               # NF4
    bnb_4bit_use_double_quant=True,          # 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16,   # 计算时反量化成 BF16
)

# ② 按量化配置加载模型（主要的 Linear 层变成 4-bit，embedding、norm 等保持高精度）
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config)

# ③ 训练前准备：冻结全部基础参数；未量化的 FP16/BF16 参数转成 FP32，提高数值稳定性
model = prepare_model_for_kbit_training(model)

# ④ 注入 LoRA（和普通 LoRA 一样）
peft_model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                              target_modules=[...7 个 proj...], lora_dropout=0.05))
peft_model.print_trainable_parameters()

# ⑤ SFTTrainer 训练，optim 用分页优化器
SFTConfig(..., learning_rate=1e-4, optim="paged_adamw_8bit", bf16=True, assistant_only_loss=True)
trainer.train(); trainer.save_model(...)   # 只保存 Adapter，不含 4-bit 基础权重
```

**4. 训练完之后**

- **评估或推理**：用**同样的 bnb_config** 加载 `AutoPeftModelForCausalLM.from_pretrained(adapter_path, quantization_config=bnb_config)`，保持和训练时一样的量化状态。
- **合并**：要用 **BF16/FP16 重新加载**基础模型后再 `merge_and_unload()`，因为 LoRA 的增量必须和浮点权重相加，不能直接加到 4-bit 权重上。

**5. 我的 [train_qlora.py](../sft/src/train_qlora.py) 和讲义对比，发现的问题**

| 位置 | 问题 | 影响 |
|---|---|---|
| `nub_4bit_compute_dtype=torch.bfloat16` | ⚠️ **拼写错误**，应该是 `bnb_4bit_compute_dtype` | 这个参数不会生效，计算类型会退回默认的 FP32，训练更慢、显存更大 |
| `bnb_4bit_use_double_quant=False` | 讲义里是 `True` | 没用双重量化，缩放因子多占一点显存（0.6B 模型影响不大） |
| `optim="paged_adamw_32bit"` | 讲义里是 `paged_adamw_8bit` | 都是分页优化器，32bit 的优化器状态更大，但也更精确，可以保留 |
| `learning_rate=3e-4` | 讲义里是 `1e-4` | LoRA 一般比全参学习率大 5～10 倍，3e-4 也在常用范围内 |

---

## Q2 数据并行的具体过程（7.2）

**核心思想**：每张 GPU 放一份**完整的模型副本**，不同 GPU 处理不同的数据子集，目的是提升训练吞吐。

**一个训练 step 的流程**

1. **切分数据**：一个全局 batch 被平均分给 N 张卡，每张卡拿到不同的数据（由 DistributedSampler 负责）。
2. **各自前向、反向**：每张卡用自己的模型副本和自己的数据算 loss，反向传播得到**本地梯度**。由于数据不同，各卡的梯度也不同。
3. **梯度同步（AllReduce）**：所有卡的梯度求和再取平均，每张卡都得到**相同的全局梯度**。
4. **各自更新参数**：每张卡用相同的全局梯度、相同的优化器独立更新参数，所以所有副本更新后**仍然完全一致**。

全局 batch 大小 = `per_device_train_batch_size × gradient_accumulation_steps × GPU 数`。

**优缺点**

| 优点 | 缺点 |
|---|---|
| 实现简单，接近线性加速 | 每张卡都存一份完整的模型、梯度和优化器状态，**显存冗余**，单卡必须放得下整个训练状态 |
| | 每一步都要同步梯度，模型越大通信量越大（见 Q3） |

---

## Q3 梯度 AllReduce 的过程和通信量（7.2，细节为补充）

讲义里只给了 AllReduce 的示意图。下面以最常用的 **Ring AllReduce**（NCCL 的默认实现之一）来说明。

**设定**：N 张卡连成一个环，每张卡的梯度大小为 $\Phi$，把梯度切成 N 块。

**第一阶段：Reduce-Scatter（分散归约）**

- 共 **N−1 步**。每一步，每张卡把自己的一块梯度发给下一张卡，同时收到上一张卡发来的一块，并**累加**到自己对应的那块上。
- N−1 步之后，**每张卡各自拥有 1 块已经求和完成的梯度**，并且各卡拥有的块不同。

**第二阶段：All-Gather（全收集）**

- 也是 **N−1 步**。每张卡把自己已经求和完成的那块沿环传下去，收到的块**直接覆盖**（不再累加）。
- N−1 步之后，**每张卡都拿到了全部 N 块求和后的梯度**，再除以 N 得到平均梯度。

**以 4 张卡为例**

```
初始：       GPU0 [a0 a1 a2 a3]  GPU1 [b0 b1 b2 b3]  GPU2 [c0 ...]  GPU3 [d0 ...]
Reduce-Scatter 3 步后：GPU0 持有完整的第 1 块 (a1+b1+c1+d1)，GPU1 持有第 2 块，…… 每卡一块
All-Gather 3 步后：     每张卡都持有 4 个完整的块
```

**通信量（每张卡发送的数据量）**

| 阶段 | 步数 | 每步发送 | 合计 |
|---|---|---|---|
| Reduce-Scatter | N−1 | $\Phi/N$ | $\frac{N-1}{N}\Phi$ |
| All-Gather | N−1 | $\Phi/N$ | $\frac{N-1}{N}\Phi$ |
| **总计** | 2(N−1) | | $\boxed{2\cdot\frac{N-1}{N}\Phi \approx 2\Phi}$ |

**结论**：每张卡的通信量约为梯度大小的 2 倍，**几乎和卡数无关**。所以 Ring AllReduce 扩展性好，卡数增加时每卡通信量不会增加，同时每张卡的带宽都被充分利用，不会出现单点瓶颈。

**算一下**：Qwen3-4B 用 BF16 梯度，$\Phi \approx 8$ GB。4 张卡时，每张卡每个 step 发送 $2 \times \frac{3}{4} \times 8 = 12$ GB。

---

## Q4 流水线并行：原理、问题、优化（7.3）

**1. 原理**

把模型**按层纵向切分**，不同的层放到不同的 GPU 上，每张卡负责一段（一个 stage）。目的是**降低单卡显存**，让单卡放不下的大模型也能训练。

- 前向：从 GPU 1 开始，算完把**中间激活值**传给下一张卡，一直到最后一张卡算出 loss。
- 反向：顺序相反，梯度从最后一张卡逐级传回第一张卡。
- 卡与卡之间只传阶段边界的激活值和梯度，**通信量小**。

**2. 问题：气泡（Bubble），GPU 利用率低**

各个 stage 是接力执行的。GPU 1 算完前向后，要一直等到后面所有卡都算完前向和反向，梯度传回来时才有事做。同一时刻往往只有一张卡在工作，其他卡**空闲等待**，这段空闲时间就叫气泡，会造成计算资源浪费。

补充（GPipe 论文）：设 stage 数为 $p$、微批次数为 $m$，气泡占比约为

$$
\text{bubble} = \frac{p-1}{m+p-1}
$$

不切微批次时（$m=1$），4 个 stage 的气泡占比是 $\frac{3}{4}$，也就是 75% 的时间在空转。

**3. 优化方案：微批次 + 流水线调度**

- **微批次（Micro-batch）**：把一个 batch 再切成 $m$ 个微批次。GPU 1 算完微批次 1 的前向后，不用等，立刻开始算微批次 2，同时把微批次 1 的结果交给 GPU 2。各个 stage 就能同时处理不同的微批次，像工厂流水线一样。
- 微批次越多，气泡占比越小。例如 4 个 stage、16 个微批次时，气泡占比为 $\frac{3}{19} \approx 16\%$。
- **调度策略**（补充）：
  - **GPipe**：先跑完所有微批次的前向，再跑全部反向。缺点是要同时保存 $m$ 个微批次的激活值，显存压力大。
  - **1F1B**：进入稳定阶段后，每张卡交替执行「1 次前向 + 1 次反向」。气泡和 GPipe 一样，但同时保存的激活值从 $m$ 份降到约 $p$ 份，显存更省。

---

## Q5 张量并行：原理和问题（7.4）

**1. 原理**

把**单层内部**的大矩阵乘法切开，分到多张卡上算。每张卡只持有权重的一部分，各自算出部分结果后，通过通信合并成完整结果。

以 $Y = XW$ 为例，把 $W$ **按列切**成 $[W_1, W_2]$：

$$
Y = X[W_1, W_2] = [XW_1,\ XW_2]
$$

GPU 1 算 $XW_1$，GPU 2 算 $XW_2$，再把结果拼起来。

补充（Megatron-LM 在 Transformer 中的切法）：

| 模块 | 切法 | 通信 |
|---|---|---|
| MLP | 第一个 Linear（up/gate）**按列切**，每卡得到一部分中间结果，激活函数可以各卡单独算；第二个 Linear（down）**按行切**，每卡得到一个部分和 | 前向结束时做 **1 次 AllReduce** 求和 |
| Attention | 按**注意力头**切，每卡算几个头；输出投影 o_proj 按行切 | 前向结束时做 **1 次 AllReduce** |

所以每个 Transformer 层**前向需要 2 次 AllReduce，反向也需要 2 次**。

**2. 问题：层内通信太频繁**

- 每一层都要通信，而且下一步计算必须等通信完成（通信在关键路径上，无法跳过）。
- 补充：例如 Qwen3-4B 有 36 层，一次前向加反向就要做 $36 \times 4 = 144$ 次 AllReduce，每次传输的数据量是 $b \times s \times h$（batch × 序列长度 × 隐藏维度）。
- 所以张量并行**依赖高带宽、低延迟的互联**（NVLink），一般**只在单机内部使用**，跨机器的网络带宽撑不住。
- TP 规模也不宜太大（通常不超过单机卡数，例如 8）。切得越细，每张卡的计算量越小，但通信量不变，通信占比会越来越高。

**3. 四种并行策略对比**

| 并行方式 | 切什么 | 降低单卡显存 | 通信特点 |
|---|---|---|---|
| 数据并行 | 切数据，每卡一份完整模型 | ❌ | 每步做一次梯度 AllReduce |
| 流水线并行 | 按层切 | ✅ | 只在阶段边界点对点传输，量小；但有气泡 |
| 张量并行 | 切层内的矩阵 | ✅ | 每层都要 AllReduce，次数多；需要 NVLink |
| 专家并行 | 切 MoE 的专家 | ✅（专家部分） | token 分发和收集（All-to-All） |

---

## Q6 ZeRO 的分片策略（7.6）

**1. 动机**

普通数据并行中，每张卡都保存一整套模型状态，N 张卡就存了 N 份，**冗余很大**。ZeRO 的核心思想是**分片存储、按需加载**：把模型状态切成 N 份，每张卡只存 1 份，计算需要时再临时从其他卡拿过来，用完立即释放。

**2. 模型状态由三部分组成**

| 状态 | 说明 |
|---|---|
| 参数 Parameters | 各层的权重与偏置 |
| 梯度 Gradients | 反向传播得到的梯度 |
| 优化器状态 Optimizer States | 例如 Adam 的一阶矩、二阶矩（混合精度时还有 FP32 参数副本），**占比最大** |

**3. 三个阶段**

| 阶段 | 分片内容 | 单卡显存（补充，按 ZeRO 论文） | 通信量（补充） |
|---|---|---|---|
| 不用 ZeRO（普通 DP） | 不分片 | $16\Psi$ | $2\Psi$ |
| **ZeRO-1** | 优化器状态 | $4\Psi + \frac{12\Psi}{N}$ | $2\Psi$（和 DP 相同） |
| **ZeRO-2** | 优化器状态 + 梯度 | $2\Psi + \frac{14\Psi}{N}$ | $2\Psi$（和 DP 相同） |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | $\frac{16\Psi}{N}$ | $3\Psi$（DP 的 1.5 倍） |

（$\Psi$ 是参数量，$N$ 是卡数。按论文的混合精度 Adam 计算：FP16 参数 2 字节 + FP16 梯度 2 字节 + 优化器状态 12 字节，包括 FP32 参数副本、一阶矩、二阶矩。讲义 5.4.1 把梯度按 FP32 计算，合计 18 字节，思路相同。）

**补充：算一下 Qwen3-4B 用 4 张卡，单卡静态显存**

| 方案 | 单卡显存 |
|---|---|
| 普通 DP | 16 × 4 = **64 GB**（单张 32GB 的 5090 放不下） |
| ZeRO-1 | 16 + 12 = **28 GB** |
| ZeRO-2 | 8 + 14 = **22 GB** |
| ZeRO-3 | 64 / 4 = **16 GB** |

**4. 怎么选**

阶段越高，显存越省，但通信越多：

- 显存够用就选 **ZeRO-2**：通信量和普通 DP 一样，相当于免费省显存。
- 模型参数本身也放不下时，才用 **ZeRO-3**。
- 还不够的话，可以再开启 **offload**，把优化器状态或参数放到 CPU 内存（我的 [ds_zero3.json](../sft/ds_zero3.json) 就把 optimizer offload 到了 CPU）。

---

## Q7 ZeRO-3 的整体过程（口述版）

> 以 N 张卡、一个 L 层的模型为例。

**初始状态**：模型的参数、梯度、优化器状态都被**均分成 N 份**，每张卡只保存自己那 1/N。每张卡拿到不同的数据，这一点和数据并行一样。

**① 前向传播（逐层进行）**

- 要算第 $l$ 层时，所有卡先做一次 **All-Gather**，把这一层的完整参数拼出来。
- 每张卡用完整参数和**自己的数据**计算这一层。
- 算完后，**立即释放**其他卡的参数分片，只保留自己那 1/N，然后进入下一层。
- 这样显存中任何时刻只有「1 层的完整参数」加上「所有层的 1/N 分片」。

**② 反向传播（逐层倒序）**

- 算第 $l$ 层的梯度时，参数已经在前向时释放了，所以要**再做一次 All-Gather** 拿回完整参数。
- 各卡用自己的数据算出这一层的完整梯度（各卡的梯度不同）。
- 做 **Reduce-Scatter**：所有卡的梯度求和并平均，每张卡**只保留自己负责那 1/N 的平均梯度**，其余部分丢弃。
- 释放完整参数，进入前一层。

**③ 参数更新**

- 每张卡只用**自己那 1/N 的梯度和 1/N 的优化器状态**，更新**自己那 1/N 的参数**。
- 各卡负责的分片不同，合起来就是一次完整的更新。不需要额外通信，下一步前向时会再通过 All-Gather 拿到最新参数。

**④ 通信量（补充）**

前向 All-Gather $\Psi$ + 反向 All-Gather $\Psi$ + 梯度 Reduce-Scatter $\Psi$ = **$3\Psi$**，是普通 DP（$2\Psi$）的 1.5 倍。用多 50% 的通信，换来单卡显存降到 $\frac{1}{N}$。

**一句话**：ZeRO-3 = 数据并行 + 「用到哪层就 All-Gather 哪层，用完就扔」，梯度用 Reduce-Scatter 分片，每张卡只更新自己的那份。

> 保存模型时，分片要先合并回完整权重。[ds_zero3.json](../sft/ds_zero3.json) 里的 `stage3_gather_16bit_weights_on_model_save: true` 就是做这件事的，否则 `save_model` 保存的是分片，加载不了。

---

## Q8 用 Accelerate 完成分布式训练（8.1～8.3）

**1. Accelerate 的定位**

Accelerate **不实现**分布式算法。它把 DDP、FSDP、DeepSpeed 等工具统一封装起来，用同一套配置和启动方式调用。TRL 的 SFTTrainer 已经集成了 Accelerate，所以**训练代码几乎不用改**。

| 工具 | 支持的并行方式 |
|---|---|
| PyTorch DDP | 数据并行 |
| PyTorch FSDP | 全分片数据并行（类似 ZeRO-3） |
| DeepSpeed | ZeRO、流水线并行、专家并行 |
| Megatron-LM | 张量并行、流水线并行、专家并行 |

**2. 三个步骤**

**① 准备训练脚本**（[sft/src/train_zero.py](../sft/src/train_zero.py)）

和单卡的 SFTTrainer 脚本基本一样，只需注意：

- 加载模型时**不要写 `device_map="auto"`**，设备分配交给 Accelerate。
- 全参训练 Qwen3-4B 时要开 `gradient_checkpointing=True`，用重新计算换显存。
- `per_device_train_batch_size` 指的是**每张卡**的 batch 大小。

**② 生成 Accelerate 配置**

```bash
accelerate config --config_file accelerate_zero3.yaml
# 依次回答：This machine → multi-GPU → 1 台机器 → 使用 DeepSpeed: yes
#          → ZeRO stage: 3 → offload 设置 → 梯度累积步数 → GPU 数 → bf16
```

我的配置（[sft/accelerate_zero3.yaml](../sft/accelerate_zero3.yaml) 和 [ds_zero3.json](../sft/ds_zero3.json)）：

```yaml
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 2                 # 2 张卡就是 2 个进程
deepspeed_config:
  deepspeed_config_file: ds_zero3.json   # stage 3，optimizer offload 到 CPU
  zero3_init_flag: true          # 加载模型时直接分片，避免每个进程先加载一份完整模型
```

ds_zero3.json 里的 `"auto"` 会自动沿用 SFTConfig 里对应的 batch size、梯度累积、梯度裁剪和 bf16 设置，避免两边配置不一致。

**③ 启动**

```bash
cd sft
accelerate launch --config_file accelerate_zero3.yaml src/train_zero.py
```

Accelerate 会读取配置，**按 `num_processes` 启动多个进程**，每个进程绑定一张卡，并初始化 DeepSpeed ZeRO-3。每个进程运行同一份脚本，数据切分、梯度同步和参数分片由 SFTTrainer 和 Accelerate 自动完成。

**3. 检查是否运行正常**

- 运行 `nvidia-smi`：每张卡都应该有一个 python 进程，显存占用大致相同。
- 日志里每个进程会各打印一次初始化信息，但训练日志和 tensorboard 只由主进程写入。
- 训练完后用 [evaluate.py](../sft/src/evaluate.py) 加载 `outputs/zero/best`，在测试集上计算 Joint Exact Match，和单卡结果对比。
