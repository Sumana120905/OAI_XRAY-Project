import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch

resize_transform = T.Resize((224, 224), antialias=True)

def load_image(dicom_file_path):

    ds = pydicom.dcmread(dicom_file_path)

    pixel_array = ds.pixel_array

    img = pixel_array.astype(np.float32)

    min_val = img.min()
    max_val = img.max()

    if max_val > min_val:

        img = (img - min_val) / (max_val - min_val)

    else:
        img = img

    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    
    resized_img = resize_transform(img_tensor).squeeze(0)

    return resized_img

dcm_file_path = "data/raw_packages/Package_1243859/results%2FP001/0.E.1/9000622/20050613/00831903/001"

processed_tensor = load_image(dcm_file_path)

plt.imshow(processed_tensor.squeeze(0), cmap='gray')
plt.title("DICOM Image")
plt.axis('off')
plt.show()