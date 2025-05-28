from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# Model & dataset setup
model_name = "tiiuae/falcon-rw-1b"
dataset_path = "scripts/google_fresher_qa.json"

# Load dataset
dataset = load_dataset("json", data_files=dataset_path)
dataset = dataset["train"].train_test_split(test_size=0.1)

# Merge question and answer into one field
def preprocess(example):
    example["text"] = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
    return example

dataset = dataset.map(preprocess)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # prevent padding issues

# Load model in float32 (for CPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,      # ✅ Use float32 for CPU
    device_map={"": "cpu"}          # ✅ Force CPU usage
)

# Apply LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ Updated training arguments for CPU
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,     # Smaller batch size for CPU
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",            # Updated from 'evaluation_strategy'
    save_strategy="epoch",
    num_train_epochs=1,               # Start with 1 epoch to test
    logging_dir="./logs",
    learning_rate=2e-5,
    bf16=False,
    fp16=False,                       # ❌ Disable fp16 for CPU
    save_total_limit=1,
)

# Formatting function
def formatting_func(example):
    return example["text"]

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    formatting_func=formatting_func,
)

# Train
trainer.train()

# ✅ Save the trained LoRA adapter
model.save_pretrained("fresher-chatbot-lora")
