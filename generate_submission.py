import json
import pandas as pd
from pathlib import Path

RESULTS_FILE = "pipeline/outputs/results_small_submission_nochunk_clean.json"
OUTPUT_CSV = "pipeline/outputs/submission_nochunk_clean.csv"

def generate_csv():
    print(f"Loading {RESULTS_FILE}...")
    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found!")
        return

    # Convert to standard format
    # Submission usually requires specific columns. Assuming just id and answer based on qid
    # If qid is "test_0001", extracting just id if needed, or keeping qid.
    # Usually valid submission format is often "id,answer".
    
    rows = []
    for entry in data:
        rows.append({
            "qid": entry["qid"],
            "answer": entry["predicted"]
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by ID just in case
    df = df.sort_values(by="qid")
    
    print(f"Generated {len(df)} rows.")
    print(df.head())
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_csv()
