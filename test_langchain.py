from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    model="qwen3",
    base_url="http://localhost:8000/v1/",
    api_key="none",
    temperature=0)

# 构建消息（支持多角色）
messages = [
    SystemMessage(content="你是一个专业的技术助手，回答需简洁准确"),
    HumanMessage(content="什么是LangChain")
]

# 调用模型获取响应
response = llm.invoke(messages)
print(response.content)