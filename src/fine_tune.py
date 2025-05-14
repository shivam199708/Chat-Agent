import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

# === CONFIGURATION ===
DATA_PATH = "../feedback_log.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = "../models/fine_tuned_model"

# === LOAD FEEDBACK LOG ===
def load_training_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            context = entry["context"].strip()
            response = entry["response"].strip()
            label = 1.0 if entry["feedback"] == "up" else 0.0
            examples.append(InputExample(texts=[context, response], label=label))
    return examples

# === TRAIN MODEL ===
def fine_tune_model():
    print("✅ Loading base model...")
    model = SentenceTransformer(MODEL_NAME)

    print("📄 Loading feedback data...")
    train_examples = load_training_examples(DATA_PATH)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    print("🚀 Starting fine-tuning...")
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(OUTPUT_DIR)
    print(f"✅ Fine-tuned model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    fine_tune_model()
