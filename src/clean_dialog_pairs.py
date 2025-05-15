import json
import pandas as pd

# === CONFIGURATION ===
INPUT_PATH = "../data/processed/dialog_pairs.jsonl"
OUTPUT_PATH = "../data/processed/cleaned_dialog_pairs.jsonl"

# === LOAD AND CLEAN ===
def clean_dialog_pairs(input_path, output_path, min_context_len=4, min_response_len=4):
    cleaned = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            context_len = len(entry["context"].split())
            response_len = len(entry["response"].split())

            # Filter out short or vague examples
            if context_len >= min_context_len and response_len >= min_response_len:
                cleaned.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Cleaned dataset saved to: {output_path}")
    print(f"🧹 Retained {len(cleaned)} out of {sum(1 for _ in open(input_path, 'r'))} examples.")

if __name__ == "__main__":
    clean_dialog_pairs(INPUT_PATH, OUTPUT_PATH)
