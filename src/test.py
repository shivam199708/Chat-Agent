import pandas as pd

df = pd.read_parquet("data/raw/train.parquet")
print(df.columns)
print(df.head(2))
