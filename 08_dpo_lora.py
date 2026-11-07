from peft import LoraConfig
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset, DatasetDict
## 1.1 数据加载
train_data:DatasetDict = load_dataset("./data/ultrafeedback_binarized.jsonl", split="train_prefs")
test_data:DatasetDict = load_dataset("./data/ultrafeedback_binarized.jsonl", split="test_prefs")
train_data = train_data.remove_columns(column_names=['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])
test_data = test_data.remove_columns(column_names=['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])

# 2、构造DPOConfig
from trl.trainer.dpo_config import DPOConfig

config = DPOConfig(
    per_device_train_batch_size=4,
    per_device_eval_batch_size= 4,
    gradient_accumulation_steps=8,
    max_steps=300,
    logging_strategy="steps",
    logging_steps=10,
    report_to="tensorboard",
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    eval_strategy="steps",
    eval_steps=50,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    output_dir="./finetuned/05_sft_demo",
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length= 650,
    # assistant_only_loss=True,
    chat_template_path="./new_chat_template.jinja"
)

from peft import LoraConfig,get_peft_model

model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B/")
# 2.1、构造LoraConfig
lora_config = LoraConfig(
    r = 16,
    lora_alpha = 16,
    # target_modules = ["q_proj","v_proj"]
    # target_modules = ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]
    target_modules = "all-linear",
    lora_dropout = 0.05,
    task_type = "CAUSAL_LM"
)

# 2.2、获取peft_model
peft_model = get_peft_model(model,lora_config)



from trl.trainer.dpo_trainer import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 3、构造trainer
trainer = DPOTrainer(
    model=peft_model,
    args=config,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,
)

# 7、训练和保存
trainer.train()
# 保存模型参数，和Tokenizer相关的配置，从而使得，后面，可以通过加载这个路径，得到model和tokenizer
trainer.save_model("./finetuned/08_dpo_lora")