import pandas as pd

file_path = "Datasets/GSE195832/processed/processed_expression.csv"

df = pd.read_csv(file_path)

print("Shape:", df.shape)
print(df.head())

X = df.drop(columns=["sample"])
samples = df["sample"]

print("X shape:", X.shape)
print("Samples:", len(samples))