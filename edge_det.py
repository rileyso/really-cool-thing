from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
IMAGE_PATH = Path("images") / "bedroom_001.jpg"
DATASET_PATH = Path("D:\Datasets\Aachen-Day-Night\db")
image = cv2.imread(str(IMAGE_PATH))

fig, ax = plt.subplots(1, 3, figsize=(15,5))
ax[0].imshow(image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("gray image shape:", image_gray.shape)
# Image Gradient How quickly the brightness changes
im_edges = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
ax[1].imshow(im_edges, cmap='gray')

# Full edge detection pipeline, gradient internally 
edges = cv2.Canny(image_gray, 100, 100)
ax[2].imshow(edges, cmap='gray')
# plt.show()


# Load Pretrained Resnet-18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
# Remove the final classfication layer
model.fc = torch.nn.Identity()
model.eval()
preprocess = weights.transforms()