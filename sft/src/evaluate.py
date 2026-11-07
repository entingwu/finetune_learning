from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

# 把 sft/ 加入搜索路径，这样直接运行脚本和 python -m src.evaluate 都能找到 src 包
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.message import to_message
from tqdm import tqdm
import json

BASE_PATH = Path(__file__).parents[1]
MODEL_PATH = Path(__file__).parents[2]

"""
python -m src.evaluate
"""
def evaluate(model, tokenizer):
  """
  Args:
      model: 需要评测的模型
      tokenizer: 对应的分词器
  """
  model.eval()

  # 1. 加载数据集
  dataset = load_dataset("json", data_files=str(BASE_PATH / "data" / "test.jsonl"))["train"]

  # 2. 转换数据格式，将数据格式转换为message list格式
  dataset = dataset.map(to_message, batched=False, remove_columns=["text"], fn_kwargs={"include_assistant_message": False})
  # print(dataset[:1])

  correct_nums = 0
  eval_batch_size = 4
  # 3. 遍历数据集，每次拿一个小批次数据出来
  for i in tqdm(range(0, len(dataset), eval_batch_size)):
    batch_data = dataset[i: i + eval_batch_size]

    # 3.1 使用tokenizer对其进行格式化（apply_chat_template）,再对其进行分词
    batch_text = tokenizer.apply_chat_template(
      batch_data["messages"], 
      tokenize = False, 
      add_generation_prompt = True, 
      enable_thinking = False)

    # 此处需要padding，并且需要进行左填充
    batch_tokenized_result = tokenizer(
      batch_text, return_tensors = "pt", padding = True, padding_side = "left"
    ).to(model.device) # 沿着左边进行填充

    # 3.2 调用模型的model.generate方法，让模型做自回归生成
    batch_generated_result = model.generate(**batch_tokenized_result, max_new_tokens = 128, do_sample = False, num_beams = 1)

    # 3.3 解析模型生成的结果，将token id解析成文本
    input_length = batch_tokenized_result["input_ids"].shape[1]
    generated_ids = batch_generated_result[:, input_length:]
    batch_decoded_result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # 3.4 拿数据集当中，真实的意图和槽位和模型生成的做一个对比，记录这个批次当中，模型生成的正确的数据有多少。
    intent = batch_data["intent"]
    slots = batch_data["slots"]
    for sample_intent, sample_slots, model_result in zip(intent, slots, batch_decoded_result):
      real_result = {"intent": sample_intent, "slots": sample_slots} # 真实答案
      try:
        model_result = json.loads(model_result)
      except Exception:
        model_result = {}
      if real_result == model_result:
        correct_nums += 1

  # 4. 使用correct_nums / total_nums, 得到模型生成的正确的比例。
  accuracy = correct_nums / len(dataset)
  return accuracy

# python -m src.evaluate
if __name__ == "__main__":
  # 基线测试: 0.025218560860793545
  # model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH / "model" / "Qwen3-0.6B"), dtype="auto", device_map="auto")
  # tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH / "model" / "Qwen3-0.6B"))
  # accuracy = evaluate(model, tokenizer)
  # print('基线测评结果', accuracy)

  # 全参微调: 0.6395427034297243
  # model = AutoModelForCausalLM.from_pretrained(str(BASE_PATH / "outputs" / "full" / "best"),dtype = "auto",device_map = "auto")
  # tokenizer = AutoTokenizer.from_pretrained(str(BASE_PATH / "outputs" / "full" / "best"))
  # accuracy = evaluate(model,tokenizer)
  # print("全参微调评测结果：",accuracy)

  # Lora微调: 0.6355077336919973
  # model = AutoModelForCausalLM.from_pretrained(str(BASE_PATH / "outputs" / "lora" / "merged"), dtype="auto", device_map="auto")
  # tokenizer = AutoTokenizer.from_pretrained(str(BASE_PATH / "outputs" / "lora" / "merged"))
  # accuracy = evaluate(model, tokenizer)
  # print('LoRA微调评测结果', accuracy)

  # QLora微调: 0.6126429051782112
  model = AutoModelForCausalLM.from_pretrained(str(BASE_PATH / "outputs" / "Qlora" / "merged"), dtype="auto", device_map="auto")
  tokenizer = AutoTokenizer.from_pretrained(str(BASE_PATH / "outputs" / "Qlora" / "merged"))
  accuracy = evaluate(model, tokenizer)
  print('QLoRA微调评测结果', accuracy)

  # ZeRO: 0.657
  # model = AutoModelForCausalLM.from_pretrained(str(BASE_PATH / "outputs" / "zero" / "checkpoint-720"), device_map="auto")
  # tokenizer = AutoTokenizer.from_pretrained(str(BASE_PATH / "outputs" / "zero" / "checkpoint-720"))
  # accuracy = evaluate(model, tokenizer)
  # print('ZeRO训练评测结果', accuracy)
