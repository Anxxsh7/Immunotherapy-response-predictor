import pandas as pd

file_path = "Datasets/GSE195832/processed/processed_expression.csv"

df = pd.read_csv(file_path)

df = df.set_index("sample")

output_path = "Datasets/GSE195832/processed/processed_expression_repo_format.csv"

df.to_csv(output_path)

print(df.shape)
print(output_path)