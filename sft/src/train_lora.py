import time
import sys
import os
from pathlib import Path

# 把 sft/ 加入搜索路径，这样直接运行脚本和 python -m src.evaluate 都能找到 src 包
sys.path.insert(0, str(Path(__file__).parents[1]))
# 云服务器(AutoDL等)上用 /root/tf-logs；本地(macOS 的 /root 只读)则写到 sft/tf-logs
_tb_root = Path("/root/tf-logs") if Path("/root/tf-logs").is_dir() else Path(__file__).parents[1] / "tf-logs"
os.environ["TENSORBOARD_LOGGING_DIR"] = str(_tb_root / time.strftime("%Y%m%d-%H%M%S"))

from datasets import load_dataset

from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.message import to_message
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

BASE_PATH = Path(__file__).parents[1]
MODEL_PATH = Path(__file__).parents[2]

def train():
  """
  LoRA微调
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
    output_dir=str(BASE_PATH / "outputs" / "lora"),
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
  model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH / "model" / "Qwen3-0.6B"), dtype="auto")
  tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH / "model" / "Qwen3-0.6B"))

  # 5. 构造LoRA Config
  lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r = 8,
    lora_alpha = 16,
    target_modules = "all-linear",
    lora_dropout = 0.05,
  )

  # 6. 调用ft_modelled，对模型进行封装
  peft_model = get_peft_model(model, lora_config)
  peft_model.print_trainable_parameters()

  # 7. 构造SFTTrainer实例
  trainer = SFTTrainer(
    model=peft_model,
    args=config,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    processing_class=tokenizer,
  )

  # 8. trainer.train()
  trainer.train()

  # 9. trainer.save_model() 保存最佳模型
  trainer.save_model(str(BASE_PATH / "outputs" / "lora" / "best"))

# python -m src.train_lora

if __name__ == "__main__":
  train()