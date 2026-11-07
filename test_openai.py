from openai import OpenAI

# 连接本地
client = OpenAI(
    base_url="http://localhost:8000/v1/",
    api_key="none"# 占位符，可忽略
)

# 多轮对话
response = client.chat.completions.create(
    model="Qwen3-8B",# 指定模型,必须与启动vllm时指定的名字一致
    messages=[
        {"role": "user", "content": "量子计算对RSA加密的威胁与应对策略,请展示你的推理过程在think标签中"}
    ],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}  # 关键参数
    }
)

# print(response)