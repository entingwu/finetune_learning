"""
➜  Finetune git:(main) ✗ source /Users/entingwu/venvs/vllm/bin/activate
(vllm) ➜  Finetune git:(main) ✗ evalscope perf \                              
--url "http://127.0.0.1:8000/v1/chat/completions" \
--parallel 5 \
--model Qwen3-0.6B \
--number 20 \
--api openai \
--dataset openqa \
--warmup-num 5 \
--stream


/Users/entingwu/PycharmProjects/Finetune/outputs/20261108_001834/Qwen3-0.6B/perf_report.html


evalscope perf \
--url "http://127.0.0.1:8000/v1/chat/completions" \
--parallel 8 16 32 64 128 \
--model Qwen3-0.6B \
--number 400 400 400 400 400 \
--api openai \
--dataset openqa \
--stream
"""
