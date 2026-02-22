import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/labels/clean_labels.csv")

print("Total Samples: ", len(df))
print("\nKL Distribution: ")

val = df["xrkl"].value_counts().sort_index()

print(val)

val.plot(kind="bar")
plt.xlabel("KL Grade")
plt.ylabel("Count")
plt.title("KL Grade Distribution")
plt.show()