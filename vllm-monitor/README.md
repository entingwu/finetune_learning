# vLLM 监控（Prometheus + Grafana）

对应讲义第 5 章「vLLM 运行监控」。用 Docker Compose 在本机启动 Prometheus 和 Grafana，采集并展示 vLLM 的 `/metrics`。

## 架构

```
vLLM (Mac 本机 127.0.0.1:8000/metrics)
   │  Prometheus 每 5s 拉取（Pull）
   ▼
Prometheus 容器 (127.0.0.1:9090)  ──查询──>  Grafana 容器 (127.0.0.1:3000)
```

- Prometheus：定时访问 `/metrics`，存储时序数据
- Grafana：连接 Prometheus，用 Dashboard 画图

## 目录文件

| 文件 | 作用 |
|---|---|
| `prometheus.yml` | 采集配置：每 5s 抓取 `host.docker.internal:8000/metrics` |
| `docker-compose.yaml` | 定义 prometheus、grafana 两个容器 |
| `grafana.json` | vLLM 官方 Dashboard（来自 vllm 仓库 `examples/observability/prometheus_grafana/grafana.json`） |
| `metrics.txt` | 一份 `curl /metrics` 的原始输出样例 |
| `perf_results/` | evalscope 压测结果，每次一个时间戳子目录（见文末「压测记录」） |

## 使用步骤

### 1. 启动 vLLM

```bash
vllm serve ./model/Qwen3-0.6B --host 127.0.0.1 --port 8000 \
  --served-model-name Qwen3-0.6B --dtype float16 \
  --max-model-len 4096 --max-num-batched-tokens 4096
```

确认指标接口可用：

```bash
curl http://127.0.0.1:8000/metrics | grep '^vllm:'
```

### 2. 启动监控（需先打开 Docker Desktop）

```bash
cd vllm-monitor
docker compose up -d      # 启动
docker compose ps         # 查看状态
docker compose down       # 停止
```

### 3. 验证采集

打开 http://127.0.0.1:9090/targets ，`vllm` 显示 **UP** 即正常。

若为 DOWN，依次检查：
1. vLLM 是否在运行：`curl http://127.0.0.1:8000/metrics`
2. `prometheus.yml` 里的端口是否正确
3. 改完 `prometheus.yml` 后需 `docker compose restart prometheus`

### 4. Grafana

打开 http://127.0.0.1:3000 （默认账号 `admin` / `admin`）。

已配置好（存在 `grafana-data` 卷里，`down` 后不会丢）：
- 数据源：Connections → Data sources → Prometheus，URL 填 `http://prometheus:9090`
  （两个容器在同一 compose 网络，用服务名访问，**不能填 localhost**）
- Dashboard：Dashboards → New → Import，上传 `grafana.json`，选择 Prometheus 数据源

面板直达：http://127.0.0.1:3000/d/b281712d-8bff-41ef-9f3f-71ad43c05e9b/vllm

## 常用 PromQL

在 Prometheus 的 Query 页面或 Grafana 面板里使用：

| 想看什么 | PromQL |
|---|---|
| 正在执行 / 排队的请求数 | `vllm:num_requests_running` / `vllm:num_requests_waiting` |
| KV Cache 使用率 | `vllm:kv_cache_usage_perc` |
| 输出 token 吞吐 (tok/s) | `rate(vllm:generation_tokens_total[1m])` |
| 输入 token 吞吐 (tok/s) | `rate(vllm:prompt_tokens_total[1m])` |
| P99 首 Token 延迟 (TTFT) | `histogram_quantile(0.99, sum by (le)(rate(vllm:time_to_first_token_seconds_bucket[1m])))` |
| P95 Token 间延迟 (ITL) | `histogram_quantile(0.95, sum by (le)(rate(vllm:inter_token_latency_seconds_bucket[1m])))` |
| P95 端到端延迟 | `histogram_quantile(0.95, sum by (le)(rate(vllm:e2e_request_latency_seconds_bucket[1m])))` |
| 请求抢占速率 | `rate(vllm:num_preemptions_total[1m])` |
| 前缀缓存命中率 | `rate(vllm:prefix_cache_hits_total[5m]) / rate(vllm:prefix_cache_queries_total[5m])` |

要点：
- `_total` 结尾是**累计计数器 (counter)**，要套 `rate()` 才得到每秒速率
- 延迟类是**直方图 (histogram)**，对 `_bucket` 用 `histogram_quantile` 求分位数

## 配合压测观察

一边压测一边看 Dashboard（时间范围选 Last 15 minutes）：

```bash
source ~/venvs/vllm/bin/activate   # evalscope 装在这个环境里，不激活会 command not found
cd vllm-monitor
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen3-0.6B --api openai --dataset openqa --stream \
  --parallel 4 --number 40 \
  --outputs-dir ./perf_results
```

- `--outputs-dir`：结果写到 `perf_results/<时间戳>/<模型名>/`
- 想跑快些可加 `--max-tokens 256`，否则 Qwen3 会输出很长的思考过程（见下方记录）

调度参数调优（讲义 6.2）时重点看：
- running 经常顶到 `--max-num-seqs` 且 waiting > 0、KV Cache 还有余量 → 可尝试调大 `--max-num-seqs`
- 抢占次数持续上升 → KV Cache 不够，`--max-num-seqs` 偏大
- 调大 `--max-num-batched-tokens` → 观察 TTFT 是否下降、ITL 是否变差

## 与讲义的差异

| 讲义 | 本目录 | 原因 |
|---|---|---|
| vLLM 在云 GPU，SSH 隧道转发到本地 18000 | vLLM 在 Mac 本机，直接抓 8000 | 不需要隧道 |
| `targets: ["host.docker.internal:18000"]` | `targets: ["host.docker.internal:8000"]` | 同上 |
| Dashboard 路径 `examples/online_serving/...` | `examples/observability/prometheus_grafana/grafana.json` | vllm 仓库已调整目录 |

若以后换成云端 GPU 部署，先建隧道，再把 `prometheus.yml` 端口改回 18000：

```bash
# Mac 上 Docker Desktop 可通过 host.docker.internal 访问本机 127.0.0.1，监听本机即可
ssh -N -L 18000:127.0.0.1:8000 root@<云服务器地址> -p <SSH端口>
```

## 注意

- 数据持久化：Prometheus 指标存在命名卷 `vllm-monitor_prometheus-data`（默认保留 15 天），Grafana 配置存在 `vllm-monitor_grafana-data`，`docker compose down` 后都不会丢。**`docker compose down -v` 会连卷一起删掉**，慎用。
- 端口都绑定在 `127.0.0.1`，只有本机能访问。

## 压测记录

### 2026-11-09 · Qwen3-0.6B · Mac 本机 · 并发 4 × 40 请求

- 结果目录：[`perf_results/20261109_102319/Qwen3-0.6B/`](perf_results/20261109_102319/Qwen3-0.6B/)
  - `perf_report.html`：图形化报告（浏览器打开）
  - `performance_summary.txt`：文字汇总
  - `parallel_4_number_40/`：原始数据（每请求明细 `benchmark_data.db`、分位数 `benchmark_percentile.json` 等）
- vLLM 启动参数：`--dtype float16 --max-model-len 4096 --max-num-batched-tokens 4096`
- 压测命令：`--parallel 4 --number 40 --dataset openqa --stream`（未限制 max_tokens）

| 指标 | avg | p50 | p99 |
|---|---|---|---|
| 端到端延迟 | 60.8 s | 48.1 s | 210.6 s |
| TTFT | 276 ms | 257 ms | 482 ms |
| TPOT | 88 ms | 84 ms | 142 ms |
| 输出长度 | 664 tok | 588 tok | 2048 tok |

- 成功率 100%，总耗时 687 s，RPS 0.06，输出吞吐 **38.7 tok/s**（单请求 decode 约 11 tok/s）

结论：
- TTFT 很低：输入仅约 30 token，Prefill 开销可忽略
- 延迟主要来自**输出过长**：Qwen3 默认带思考过程，平均 664 token，最长写满 2048
- 瓶颈是 **Mac 本机算力**：Grafana 上 KV Cache 使用率仅约 7%、waiting 始终为 0，调度参数（`--max-num-seqs` 等）不是瓶颈
