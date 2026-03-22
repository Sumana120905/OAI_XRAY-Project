import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch

dicom_file_path = "data/raw_packages/Package_1243859/results%2FP001/0.E.1/9000099/20050531/00839603/001"
ds = pydicom.dcmread(dicom_file_path)

pixel_array = ds.pixel_array

'''print("Shape:", pixel_array.shape)
print("Min:", pixel_array.min())
print("Max:", pixel_array.max())

plt.imshow(pixel_array, cmap='gray')
plt.title("DICOM Image")
plt.axis('off')
plt.show()'''

image = pixel_array.astype(np.float32)

min_val = image.min()
max_val = image.max()

if max_val > min_val:
    image = (image - min_val) / (max_val - min_val) # Max-Min Normalization
else:
    image = image  


img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

resize_transform = T.Resize((224, 224), antialias=True)  # Resizing

resized_img = resize_transform(img_tensor)

resized_img = resized_img.squeeze(0)

print("Resized shape:", resized_img.shape)
print("Min/Max:", resized_img.min().item(), resized_img.max().item())



# This is a test practice code for initial preprocessing on an image ( DICOM -> Pixel_array -> Normalization -> Resizing)




