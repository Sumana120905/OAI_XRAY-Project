import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/labels/clean_labels.csv")

subjects = df["subjectkey"].unique()

train, temp = train_test_split(subjects, test_size=0.3, random_state=42, shuffle=True)

val, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

def assign_split(subject):
    if subject in train:
        return "train"
    elif subject in val:
        return "val"
    else:
        return "test"

df["split"] = df["subjectkey"].apply(assign_split)

assert set(train).isdisjoint(val)
assert set(train).isdisjoint(test)
assert set(val).isdisjoint(test)

print(df.groupby("split")["xrkl"].value_counts())

df.to_csv("data/labels/final_labels_split.csv", index=False)

print("\nSaved → data/labels/final_labels_split.csv")