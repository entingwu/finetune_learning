# vLLM 部署笔记（Mac 本地）

> 状态：2026-11-07 在 Mac（Apple Silicon，macOS 26）上实测跑通。`Qwen3-0.6B` 启动约 1 分钟，短问题回答约 3 秒；`test_openai.py`、`test_langchain.py` 都能正常返回。

## 0. 快速开始

```bash
source ~/venvs/vllm/bin/activate
cd /Users/entingwu/PycharmProjects/Finetune

vllm serve ./model/Qwen3-0.6B \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name qwen3 \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096
```

看到 `Application startup complete` 就可以用了。另开一个终端测试：

```bash
source ~/venvs/vllm/bin/activate
cd /Users/entingwu/PycharmProjects/Finetune
curl http://127.0.0.1:8000/v1/models    # 列表里有 "qwen3" 就对了
python test_openai.py
python test_langchain.py
```

停止服务：在运行 `vllm serve` 的终端按 `Ctrl+C`；后台启动的用 `lsof -ti:8000 | xargs kill`。

---

## 1. 环境说明

| 环境 | 位置 | 用途 | 关键版本 |
|---|---|---|---|
| `Finetune` | `Finetune/.venv` | 训练和实验，**不要装 vLLM** | torch 2.13.0、numpy 2.5.1 |
| `vllm` | `~/venvs/vllm` | 只用来跑 vLLM 推理和客户端测试 | vLLM 0.11.0、torch 2.8.0、transformers 4.57.6 |

为什么分开：装 vLLM 0.11.0 会把 torch 降到 2.8.0、numpy 降到 2.2.6，还会装 70 多个依赖，会影响训练环境。

vllm 环境里另外装了 `langchain-openai`（给 `test_langchain.py` 用），`openai` 是 vLLM 自带的依赖。

## 2. 切换环境

```bash
deactivate                          # 退出当前环境（比如 Finetune）
source ~/venvs/vllm/bin/activate    # 激活 vllm 环境，提示符变成 (vllm)
which python                        # 确认：/Users/entingwu/venvs/vllm/bin/python
```

提示符前如果还有 conda 环境（`(base)`、`(finetune_qwen)`），再运行 `conda deactivate`。

**必须用 `source` 激活后再运行 `vllm`**：`~/venvs/vllm/bin/activate` 末尾加了一行 `export CPLUS_INCLUDE_PATH=...`，用来绕过 Mac 编译器头文件的问题（见第 6 节）。直接用绝对路径 `~/venvs/vllm/bin/vllm` 启动的话，这行不会生效，预热阶段会报错。

如果启动时还报 `'algorithm' file not found`，先运行 `echo $CPLUS_INCLUDE_PATH`：输出是空的，就说明这个终端是在激活脚本修改之前激活的，重新 `source ~/venvs/vllm/bin/activate` 一次就好（或者新开一个终端）。

## 3. 在 VS Code 里选 vllm 解释器

`~/venvs` 不在 VS Code 自动搜索的位置，所以列表里默认看不到。

- **一次性**：`Cmd+Shift+P` → Python: Select Interpreter → Enter interpreter path... → 输入 `/Users/entingwu/venvs/vllm/bin/python`
- **永久**：在用户设置（JSON）里加 `"python.venvPath": "~/venvs"`

选好以后，新打开的终端会自动激活 vllm 环境，已经开着的终端要关掉重开。

## 4. 启动参数说明

| 参数 | 说明 |
|---|---|
| `./model/Qwen3-0.6B` | 模型路径，是相对路径，必须先 `cd` 到 Finetune 目录 |
| `--host 127.0.0.1` / `--port 8000` | 只允许本机访问，端口 8000 |
| `--served-model-name qwen3` | 客户端请求时用的模型名，必须和代码里的 `model=` 一致，否则报 404 |
| `--dtype float16` | Mac 上的 vLLM 不支持 bf16（模型默认是 bf16）；还报错就改成 `float32` |
| `--max-model-len 4096` | Qwen3 默认最长 40960 个 token，按这个长度预留 KV 缓存太占内存 |
| `--max-num-batched-tokens 4096` | Mac（ARM）上不支持分块预填充，这个值必须 ≥ `max-model-len` |
| `VLLM_CPU_KVCACHE_SPACE`（环境变量） | KV 缓存占多少 GB 内存，默认 4；想调就在启动前 `export VLLM_CPU_KVCACHE_SPACE=8` |

启动日志里这几条是正常的，不用管：

- `Casting torch.bfloat16 to torch.float16`：`--dtype float16` 要求的转换
- `Chunked prefill is not supported for ARM`：所以才需要 `--max-num-batched-tokens`
- `Triton not installed`、`Pin memory is not supported on CPU`、`NUMA is not enabled`：CPU 模式下本来就没有这些功能

## 5. 客户端脚本

两个脚本都已经改成 `model="qwen3"`，要用 vllm 环境运行：

| 脚本 | 用的库 | 说明 |
|---|---|---|
| `test_openai.py` | `openai` | 用 `enable_thinking: False` 关闭了思考模式，直接输出答案 |
| `test_langchain.py` | `langchain-openai` | 没关思考模式，输出里会先有一段 `<think>...</think>` 推理过程 |

0.6B 模型很小，回答里会有事实错误（比如说 LangChain 是 Facebook 开发的）。这是模型能力的问题，不是部署出错。

## 6. 踩过的坑

| # | 什么时候 | 报错 | 原因 | 解决 |
|---|---|---|---|---|
| 1 | 安装 | `pip install` 只打印 usage | `--torch-backend` 是 `uv pip` 的参数，普通 `pip` 不认识 | 用 `uv pip install` |
| 2 | 安装（编译） | `fatal error: 'cstddef' file not found` | Command Line Tools 里的 `usr/include/c++/v1` 只剩 3 个旧文件，编译器找到这个目录就不再去 SDK 里找完整的标准库头文件 | `export CPLUS_INCLUDE_PATH=$(xcrun --show-sdk-path)/usr/include/c++/v1` |
| 3 | 重新安装 | `ninja --version ... no such file or directory` | 上次失败的编译在 uv 缓存里留下了过期的 CMake 配置 | `uv cache clean vllm` 后再装 |
| 4 | 启动 | `max_num_batched_tokens (2048) is smaller than max_model_len (4096)` | Mac 上关闭了分块预填充，一批 token 数必须能放下最长的请求 | 加 `--max-num-batched-tokens 4096`，或把 `--max-model-len` 降到 2048 |
| 5 | 启动 | `Qwen2Tokenizer has no attribute all_special_tokens_extended` | uv 装了 transformers 5.x，vLLM 0.11.0 只兼容 4.x | `uv pip install --python ~/venvs/vllm/bin/python "transformers<5"` |
| 6 | 启动（模型加载完、预热时） | `CppCompileError ... 'algorithm' file not found` | 和第 2 条同一个原因：vLLM 在 CPU 上会用 torch 编译器现场编译 C++ | 已写进 `~/venvs/vllm/bin/activate`，激活环境就自动生效 |
| 7 | 运行客户端 | 404 / model not found | 脚本里写的 `model="Qwen3-8B"` 和服务端名字 `qwen3` 不一致 | 已改成 `model="qwen3"` |

**第 2、6 条的根本修复**：重装 Command Line Tools（需要输密码）：

```bash
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```

修好以后，可以把 `~/venvs/vllm/bin/activate` 末尾那几行 `CPLUS_INCLUDE_PATH` 删掉。

## 7. 从零重建 vllm 环境

```bash
# 1. 建环境
uv venv ~/venvs/vllm --python 3.12 --prompt vllm

# 2. 装 vLLM（Mac 上要从源码编译，需要先设头文件路径）
export CPLUS_INCLUDE_PATH=$(xcrun --show-sdk-path)/usr/include/c++/v1
uv pip install --python ~/venvs/vllm/bin/python vllm "transformers<5" --torch-backend=auto

# 3. 客户端依赖
uv pip install --python ~/venvs/vllm/bin/python langchain-openai

# 4. 把头文件路径写进激活脚本，以后启动时自动生效
cat >> ~/venvs/vllm/bin/activate <<'EOF'

# Mac 的 Command Line Tools 里 c++/v1 头文件残缺，指向 SDK 里完整的 libc++ 头文件
export CPLUS_INCLUDE_PATH="$(xcrun --show-sdk-path)/usr/include/c++/v1"
EOF
```

如果编译中途失败，重试前先运行 `uv cache clean vllm`（第 6 节第 3 条）。

## 8. 在 AutoDL（GPU）上部署

Mac 上 vLLM 只能用 CPU：0.6B 可以试，4B 及以上会很慢。正式推理放到服务器上：

```bash
uv venv ~/vllm-env --python 3.12 && source ~/vllm-env/bin/activate
uv pip install vllm --torch-backend=auto     # Linux + CUDA 有现成的安装包，不需要编译
CUDA_VISIBLE_DEVICES=0 vllm serve ~/autodl-tmp/sft/outputs/zero/best --served-model-name qwen3
```

- 推理只需要一张卡：Qwen3-4B 的 bf16 权重约 8GB，一张 24GB 卡足够。
- 服务器上可以去掉 `--dtype float16` 和 `--max-num-batched-tokens`，直接用 bf16 和默认设置。
- 加载 ZeRO-3 训练的模型前，先确认 `outputs/zero/best` 里有完整的 `model*.safetensors`（训练时开了 `zero3_save_16bit_model: true`）。
- 服务器上很可能也会遇到第 6 节第 5 条（transformers 5.x 不兼容），同样装 `"transformers<5"` 解决。
