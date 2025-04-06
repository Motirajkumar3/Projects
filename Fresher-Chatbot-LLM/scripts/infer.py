from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Load the fine-tuned LoRA model directory (use absolute path)
lora_path = "C:/Users/motir/Desktop/fresher-chatbot-llm/fresher-chatbot-lora" # <--change your file location

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

    # Extract answer portion after prompt
    if "### Answer:" in decoded_output:
        answer = decoded_output.split("### Answer:")[-1].strip()
        # Cut off anything after another "###", if it appears
        answer = answer.split("###")[0].strip()
    else:
        answer = decoded_output.strip()

    return answer

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = chat_with_bot(user_input)
        print(f"Bot: {answer}")
