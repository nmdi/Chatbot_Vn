from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Chọn mô hình sinh văn bản tiếng Việt (GPT-2 đã fine-tuned)
MODEL_NAME = "VietAI/gpt2-news-title"

# Load model & tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

def generate_reply(prompt: str, max_length: int = 50) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            temperature=0.7
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
