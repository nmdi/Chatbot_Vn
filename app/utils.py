import json

def load_label_encoder(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
