from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Load the fine-tuned LoRA model directory (use absolute path)
lora_path = "fresher-chatbot-lora"  # <-- change this if your LoRA model path differs

# Load PEFT config to get the base model name
config = PeftConfig.from_pretrained(lora_path)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, lora_path)

# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Move model to CPU
model = model.to("cpu")
model.eval()

# Inference function
def chat_with_bot(question):
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the part after the prompt
    generated_text = decoded_output[len(prompt):].strip()

    # Remove extra sections if model repeats prompt or adds more
    split_tokens = ["###", "Question:", "Answer:"]
    for token in split_tokens:
        if token in generated_text:
            generated_text = generated_text.split(token)[0].strip()
    
    return generated_text

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = chat_with_bot(user_input)
        print(f"Bot: {answer}")
