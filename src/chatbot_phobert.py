import json
import torch
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googletrans import Translator
import random
import os

# --- CẤU HÌNH ---
MODEL_PATH = "models/phobert_intent_classifier.pt"
LABEL_PATH = "models/label_encoder.json"
INTENTS_PATH = "data/intents.json"
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DỊCH TỰ ĐỘNG ---
translator = Translator()

def translate_to_vietnamese(text):
    try:
        translated = translator.translate(text, dest="vi")
        print(f"[Dịch EN->VI]: {text} ➜ {translated.text}")
        return translated.text
    except Exception as e:
        print("⚠️ Lỗi dịch:", e)
        return text

# --- LOAD TOKENIZER + MODEL ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- LOAD LABELS + INTENTS ---
with open(INTENTS_PATH, encoding="utf-8") as f:
    intents = json.load(f)["intents"]

with open(LABEL_PATH, encoding="utf-8") as f:
    label_map = json.load(f)
    idx2label = {v: k for k, v in label_map.items()}

# --- TIỀN XỬ LÝ ---
def preprocess_input(text):
    text = translate_to_vietnamese(text)
    text = ViTokenizer.tokenize(text.lower())
    return text

# --- DỰ ĐOÁN INTENT ---
def predict_intent(text):
    processed_text = preprocess_input(text)
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_label_id = torch.argmax(probs, dim=1).item()
    pred_label = idx2label[pred_label_id]
    confidence = probs[0][pred_label_id].item()
    return pred_label, confidence

# --- CHỌN CÂU TRẢ LỜI ---
def get_response(intent_tag):
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "Mình chưa hiểu ý bạn lắm."
