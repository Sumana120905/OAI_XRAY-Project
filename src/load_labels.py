import pandas as pd

df = pd.read_csv("data/labels/oai_labels.txt", sep="\t", low_memory=False)

df = df[["subjectkey", "visit", "side", "xrkl"]]

df = df[df["xrkl"].str.strip().isin(["0","1","2","3","4"])]

df["xrkl"] = df["xrkl"].str.strip().astype(int)

print(df.head())

print("\nClean KL Grade counts:")
print(df["xrkl"].value_counts())

df.to_csv("data/labels/clean_labels.csv", index=False)

print("\nSaved → data/labels/clean_labels.csv")