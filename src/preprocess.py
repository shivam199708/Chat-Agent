import os
import pandas as pd
import json

def preprocess_ubuntu_dialogs(
    input_path: str = "data/raw/train.parquet", 
    output_path: str = "data/processed/dialog_pairs.jsonl"
):
    """
    Preprocess Ubuntu dialogues:
    - Use INSTRUCTION as context, RESPONSE as response.
    - Save to JSONL format.
    """
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df)} dialogues from {input_path}")

    pairs = []

    for _, row in df.iterrows():
        context = row.get("INSTRUCTION", "").strip()
        response = row.get("RESPONSE", "").strip()

        if not context or not response:
            continue

        pairs.append({
            "context": context,
            "response": response
        })

    print(f"✅ Extracted {len(pairs)} context-response pairs")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"✅ Saved preprocessed data to {output_path}")

if __name__ == "__main__":
    preprocess_ubuntu_dialogs()
