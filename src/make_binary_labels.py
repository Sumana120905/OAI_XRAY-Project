import pandas as pd

df = pd.read_csv("data/labels/clean_labels.csv")

df["binary_label"] = df["xrkl"].apply(lambda x: 0 if x <= 1 else 1)

print(df["binary_label"].value_counts())

df.to_csv("data/labels/binary_labels.csv", index=False)

print("\nSaved → data/labels/binary_labels.csv")