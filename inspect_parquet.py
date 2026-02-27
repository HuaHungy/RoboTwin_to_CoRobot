
import pandas as pd
import os
import sys

def inspect_parquet(path):
    print(f"\n--- Inspecting: {path} ---")
    try:
        df = pd.read_parquet(path)
        print("Columns:", df.columns.tolist())
        print("\nDtypes:\n", df.dtypes)
        print("\nFirst row sample:")
        # Print a sample but avoid printing huge binary blobs
        first_row = df.iloc[0].to_dict()
        for k, v in first_row.items():
            if isinstance(v, bytes) and len(v) > 100:
                print(f"{k}: <bytes len={len(v)}>")
            else:
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    paths = sys.argv[1:]
    for p in paths:
        inspect_parquet(p)
