import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi import ViTokenizer
import numpy as np
from utils import load_label_encoder

MODEL_PATH = os.path.join("models", "phobert_intent_classifier.pt")
ENCODER_PATH = os.path.join("models", "label_encoder.json")
INTENT_PATH = os.path.join("data", "intents.json")
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 64

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Load label encoder + intent data
label_encoder = load_label_encoder(ENCODER_PATH)
with open(INTENT_PATH, encoding="utf-8") as f:
    intent_data = json.load(f)

def predict_intent(text: str):
    text = ViTokenizer.tokenize(text.lower())
    encoding = tokenizer(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoding)
        probs = torch.nn.functional.softmax(output.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    
    intent_id = predicted.item()
    intent = list(label_encoder.keys())[list(label_encoder.values()).index(intent_id)]
    return intent, confidence.item()

def get_response(intent: str) -> str:
    for item in intent_data["intents"]:
        if item["tag"] == intent:
            return np.random.choice(item["responses"])
    return "Mình chưa hiểu rõ ý bạn, bạn nói lại được không?"
