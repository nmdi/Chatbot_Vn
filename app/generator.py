from transformers import AutoTokenizer, AutoModelForCausalLM

# Tên model tiếng Việt thay thế
MODEL_NAME = "vblagoje/gpt2-medium-vietnamese"

# Tải tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Hàm tạo phản hồi từ prompt đầu vào
def generate_reply(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        num_return_sequences=1
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply
