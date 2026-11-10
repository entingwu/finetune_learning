from langchain.chat_models import init_chat_model

"""
1. vllm serve ./model/Qwen3-0.6B --host 127.0.0.1 --port 8000 --served-model-name Qwen3-0.6B \
    --dtype float16 --max-model-len 4096 --max-num-batched-tokens 4096
2. /Users/entingwu/venvs/vllm/bin/python sft/inference/test_vllm.py
"""

llm = init_chat_model(
  model="Qwen3-0.6B", # Qwen3-0.6B
  model_provider="openai",
  base_url="http://127.0.0.1:8000/v1",
  temperature=0.7,
  max_tokens=1280,
  use_responses_api=False,
  api_key="none",
)

response = llm.invoke("请介绍大语言模型")
print(response.content)