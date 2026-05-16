import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import torchvision.transforms as T

class Access_Dataset(Dataset):
    
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.resize = T.Resize((224, 224), antialias=True)

    def __len__(self):
        return len(self.df)

    def load_image(self, dicom_file_path):
        ds = pydicom.dcmread(dicom_file_path)
        img = ds.pixel_array.astype(np.float32)

        # Min-max normalization
        min_val = img.min()
        max_val = img.max()

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)

        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

        # Resize
        img = self.resize(img).squeeze(0)   # (1, 224, 224)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label = int(row["xrkl"])

        image = self.load_image(image_path)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
