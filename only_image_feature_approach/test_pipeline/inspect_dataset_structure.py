import pandas as pd
import io
from PIL import Image

DATA_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\skippd_train_aligned_v13_with_time_features.parquet"

try:
    print(f"Loading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    print("\n--- DataFrame Info ---")
    print(df.info())
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Column Inspection ---")
    for col in df.columns:
        print(f"Column: {col}")
        sample = df[col].iloc[0]
        print(f"  Type: {type(sample)}")
        print(f"  Sample: {str(sample)[:100]}...") # Truncate long strings/bytes
        
        if col == "image" and isinstance(sample, dict):
            print("  Image Dict Keys:", sample.keys())
            if 'bytes' in sample:
                print(f"  Image Bytes Length: {len(sample['bytes'])}")

except Exception as e:
    print(f"Error reading file: {e}")
