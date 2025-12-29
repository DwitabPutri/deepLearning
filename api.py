import re
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = FastAPI(title="POS Tagging API")

MODEL_DIR = "/content/drive/MyDrive/deep_learning/checkpoint-625"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

# tokenizer berbasis regex (kata, angka desimal, tanda baca)
def tokenize_words(sentence: str):
    return re.findall(r"\d+\.\d+|\w+|[^\w\s]", sentence)

def predict_pos(sentence: str):
    # tokenisasi linguistik
    words = tokenize_words(sentence)

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # softmax → probabilities
    probs = F.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(outputs.logits, dim=2)

    id2label = model.config.id2label
    word_ids = inputs.word_ids(batch_index=0)

    previous_word_idx = None
    results = []

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue

        label_id = predictions[0][idx].item()
        confidence = probs[0][idx][label_id].item()

        results.append({
            "word": words[word_idx],
            "label": id2label[label_id],
            "confidence": round(confidence, 4)
        })

        previous_word_idx = word_idx

    return results

@app.get("/")
def home():
    return {
        "message": "POS Tagging API is running",
        "usage": "/predict?sentence=input",
        "docs": "/docs"
    }

@app.get("/predict")
def predict(sentence: str):
    return {"result": predict_pos(sentence)}
