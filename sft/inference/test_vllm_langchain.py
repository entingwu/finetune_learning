from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

"""
vllm serve ./model/Qwen3-0.6B --host 127.0.0.1 --port 8000 --served-model-name Qwen3-0.6B \
    --dtype float16 --max-model-len 4096 --max-num-batched-tokens 4096 \
    --enable-auto-tool-choice --tool-call-parser hermes

/Users/entingwu/venvs/vllm/bin/python sft/inference/test_vllm_langchain.py
"""
# 1. 初始化模型
llm = init_chat_model(
  model="Qwen3-0.6B", # Qwen3-0.6B
  model_provider="openai",
  base_url="http://127.0.0.1:8000/v1",
  temperature=0.7,
  max_tokens=1280,
  use_responses_api=False,
  api_key="none",
)

# 2. 定义工具
@tool
def get_weather(city: str) -> str:
  """查询指定城市的天气情况。"""
  return f"{city}今天晴天,气温28℃,空气质量良好。"

# 3. 将工具绑定到模型
llm_with_tools = llm.bind_tools(
  [get_weather],
  tool_choice="auto",
)

# 4. 用户提问
question = "帮我查询一下深圳今天的天气。"
response = llm_with_tools.invoke(question)

# 5. 查看模型生成的工具调用信息
print("模型生成的工具调用：")
print(response.tool_calls)

# 6. 执行工具，并将结果返回给模型
if response.tool_calls:
  tool_messages = []
  for call in response.tool_calls:
    if call["name"] == "get_weather":
      result = get_weather.invoke(call["args"])
      tool_messages.append(
        ToolMessage(
          content=str(result),
          tool_call_id=call["id"],
        )
      )

  # 7. 将工具执行结果交给模型，生成最终回答
  final_response = llm.invoke(
    [
      ("user", question),
      response,
      *tool_messages
    ]
  )
  print("\n最终回答：")
  print(final_response.content)
else:
  print("\n模型直接回答：")
  print(response.content)



  
