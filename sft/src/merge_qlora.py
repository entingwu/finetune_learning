from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from pathlib import Path

BASE_PATH = Path(__file__).parents[1]

adapter_path = str(BASE_PATH / "outputs" / "Qlora" / "best")

# 1. 加载基座模型和A，B适配器
model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, torch_dtype="auto", device_map="auto")
# 2. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 3. 将适配器和基座模型合并到一起去
merged_model = model.merge_and_unload()

# 4. 保存合并后的模型权重和相关配置文件
merged_model.save_pretrained(str(BASE_PATH / "outputs" / "Qlora" / "merged"))

# 5. trainer.save_model()
tokenizer.save_pretrained(str(BASE_PATH / "outputs" / "Qlora" / "merged"))