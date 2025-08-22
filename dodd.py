import pandas as pd
from datetime import datetime

# Input & Output files
INPUT_FILE = r"D:\Scientist\Machine learning\tt1.csv"
OUTPUT_FILE = r"D:\Scientist\Machine learning\tt1_final.csv"

# Convert Unix timestamp → readable datetime
def convert_timestamp(ts):
    try:
        ts = int(ts)
        if ts > 0:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        else:
            return "N/A"
    except:
        return "N/A"

def fix_csv():
    # Load CSV
    df = pd.read_csv(INPUT_FILE)

    # Convert time columns if they exist
    if "time" in df.columns:
        df["time"] = df["time"].apply(convert_timestamp)

    if "block_time" in df.columns:
        df["block_time"] = df["block_time"].apply(convert_timestamp)

    # Keep only last 500 transactions max
    df = df.head(500)

    # Save cleaned CSV
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Fixed CSV saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    fix_csv()
