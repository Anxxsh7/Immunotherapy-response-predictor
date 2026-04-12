import pandas as pd
import numpy as np
import os

file_path = "Datasets/GSE195832/supplementary/GSE195832_fpkm_sample.txt.gz"

df = pd.read_csv(file_path, sep="\t", compression="gzip")

df = df.set_index("gene_id").T
df = df.reset_index().rename(columns={"index": "sample"})

gene_cols = df.columns.drop("sample")
df[gene_cols] = np.log2(df[gene_cols] + 1)

out_dir = "Datasets/GSE195832/processed"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "processed_expression.csv")
df.to_csv(out_path, index=False)

print(df.shape)
print(out_path)