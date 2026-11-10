from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from compressed_tensors.quantization import QuantizationArgs

# 1. 配置
MODEL_ID = "../model/Qwen3-8B"
DATASET_ID = "../data/ultrachat_200k"
SAVE_DIR = "../model/Qwen3-8B-GPTQ-KVFP8" # W4A16

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 1024

# 2. 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

dataset = load_dataset(
    DATASET_ID,
    split="test_sft",
)

# 只使用 256 条样本
dataset = dataset.select(range(NUM_CALIBRATION_SAMPLES))

# 4. 将 messages 转换为 Qwen3 的 chat template 文本
def preprocess(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
)

# 5. 配置 GPTQ W4A16
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        dynamic=False,
    ),
)

# 6. 执行量化
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    recipe=recipe,
    text_column="text",
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    max_seq_length=MAX_SEQ_LENGTH,
)

# 7. 保存量化模型
model.save_pretrained(
    SAVE_DIR,
    save_compressed=True,
)
tokenizer.save_pretrained(SAVE_DIR)

print(f"量化模型已保存至：{SAVE_DIR}")
