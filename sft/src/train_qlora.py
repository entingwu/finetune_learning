import time
import sys
import os
import torch
from pathlib import Path

# 把 sft/ 加入搜索路径，这样直接运行脚本和 python -m src.evaluate 都能找到 src 包
sys.path.insert(0, str(Path(__file__).parents[1]))

from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.message import to_message
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

BASE_PATH = Path(__file__).parents[1]
MODEL_PATH = Path(__file__).parents[2]

# AutoDL 上写到 /root/tf-logs；本地 /root 不可写，改写到 outputs/tf-logs
TF_LOG_ROOT = Path("/root/tf-logs") if os.access("/root", os.W_OK) else BASE_PATH / "outputs" / "tf-logs"
os.environ["TENSORBOARD_LOGGING_DIR"] = str(TF_LOG_ROOT / time.strftime("%Y%m%d-%H%M%S"))

def train():
  """
  QLoRA微调
  """
  # 1. 加载数据集：加载训练集和验证集
  dataset_dict = load_dataset("json", data_files={"train": str(BASE_PATH / "data" / "train.jsonl"), "validation": str(BASE_PATH / "data" / "validation.jsonl")})

  # 2. 转换数据格式，将数据转换为message list格式
  # 训练时必须包含 assistant 消息，否则模型没有可学习的答案（labels）
  dataset_dict = dataset_dict.map(to_message, remove_columns=["text", "intent", "slots"], fn_kwargs={"include_assistant_message": True})

  # 3. 构造SFTConfig 配置参数
  config = SFTConfig(
    max_length=10240,
    assistant_only_loss=True,
    # 模型自带模板没有 {% generation %}，assistant_only_loss 需要用这个模板生成 mask
    chat_template_path=str(MODEL_PATH / "new_chat_template.jinja"),
    output_dir=str(BASE_PATH / "outputs" / "Qlora"),
    per_device_train_batch_size=8,
    num_train_epochs=1,
    # learning rate
    learning_rate=3e-4, # LoRA微调一般需要使用全参微调10倍的学习率
    warmup_steps=0.1,
    # Optimizer
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    bf16=True,
    bf16_full_eval=False,
    # Gradient Checkpointing
    gradient_checkpointing=False,
    report_to = "tensorboard",
    # Optimizer
    optim = "paged_adamw_32bit", # QLoRA需要使用分页优化器
    # Logging
    logging_strategy="steps",
    logging_steps=50,
    # Eval
    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=8,
    # Checkpoint saving
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    greater_is_better=False,
  )

  # 4. 加载模型和tokenizer
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    nub_4bit_compute_dtype=torch.bfloat16,
  )

  # 5. 加载模型和tokenizer
  model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH / "model" / "Qwen3-0.6B"), quantization_config=bnb_config)
  tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH / "model" / "Qwen3-0.6B"))

  # 6. 构造LoRA Config
  lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r = 8,
    lora_alpha = 16,
    target_modules = "all-linear",
    lora_dropout = 0.05,
  )

  # 7. 调用prepare_model_for_kbit_training
  model = prepare_model_for_kbit_training(model)
  # 8. 调用get_peft_model, 注入LoRA组件
  peft_model = get_peft_model(model, lora_config)
  print('peft', peft_model)
  peft_model.print_trainable_parameters()
  # lora.Linear4bit

  # 9. 构造SFTTrainer实例
  trainer = SFTTrainer(
    model=peft_model,
    args=config,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    processing_class=tokenizer,
  )

  # 10. trainer.train()
  trainer.train()

  # 11. trainer.save_model() 保存最佳模型
  trainer.save_model(str(BASE_PATH / "outputs" / "Qlora" / "best"))

# python -m src.train_lora

if __name__ == "__main__":
  train()