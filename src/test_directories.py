import os
import pandas as pd

root_dir = "data/raw_packages/Package_1243859"

data = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        2
        if file.isdigit():

            image_path = os.path.join(root, file)

            parts = root.split(os.sep)

            barcode = parts[-1]

            data.append({
                "image_path" : image_path,
                "barcode" : barcode
            })

image_df = pd.DataFrame(data)

labels = pd.read_csv("data/labels/final_labels_split.csv")

labels["barcode"] = labels["barcode"].astype(str)
labels["barcode"] = labels["barcode"].astype(str).str[-8:]
image_df["barcode"] = image_df["barcode"].astype(str)

merged = image_df.merge(labels, on="barcode")

final_df = merged[["image_path", "xrkl", "split"]]

final_df.to_csv("data/labels/image_labels.csv", index=False)

            