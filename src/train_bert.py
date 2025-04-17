import os, json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from pyvi import ViTokenizer

# ⚙️ Config
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 64
BATCH_SIZE = 8
EPOCHS = 4

# 📁 Đường dẫn dựa trên vị trí src/train_model.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))               # src/
BASE_DIR = os.path.dirname(CURRENT_DIR)                                # Chatbot_Vn/
APP_DIR = os.path.join(BASE_DIR, "app")
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phobert_intent_classifier.pt")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.json")
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")

# 🔧 Tạo thư mục models nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)

# ⚡ Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load & xử lý dữ liệu
with open(INTENTS_PATH, encoding="utf-8") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(ViTokenizer.tokenize(pattern.lower()))
        labels.append(intent["tag"])

# 2. Encode label
df = pd.DataFrame({"text": texts, "label": labels})
label_encoder = LabelEncoder()
df['label_enc'] = label_encoder.fit_transform(df['label'])

# Lưu label encoder dưới dạng dict: {label: id}
label_map = {label: int(i) for i, label in enumerate(label_encoder.classes_)}
with open(ENCODER_PATH, "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

# 3. Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PhoDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# 4. Tạo dataloader
X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label_enc'], test_size=0.2, stratify=df['label_enc'])
train_data = PhoDataset(X_train.tolist(), y_train.tolist())
val_data = PhoDataset(X_val.tolist(), y_val.tolist())

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# 5. Load PhoBERT model
num_labels = df['label_enc'].nunique()
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=5e-5)

# 6. Huấn luyện
print("👉 Bắt đầu huấn luyện PhoBERT...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"✅ Loss epoch {epoch+1}: {total_loss / len(train_loader):.4f}")

    # 🎯 Đánh giá trên tập validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"🎯 Accuracy validation: {acc:.2f}%")

# 7. Lưu mô hình
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Mô hình đã được lưu tại: {MODEL_PATH}")
print(f"✅ Label encoder đã được lưu tại: {ENCODER_PATH}")
