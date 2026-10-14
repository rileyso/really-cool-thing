from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pickle

DATASET_PATH = Path("really-cool-thing/data")

## Setup
# Load Pretrained Resnet-18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
# Remove the final classfication layer
model.fc = torch.nn.Identity()
model.eval()
preprocess = weights.transforms()

def extract_features(model, image_path):
    # prepare image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print("image shape:", image.shape)
    image = Image.fromarray(image)
    image_tensor = preprocess(image)
    # Add batch dimensions
    # [3, 224, 224] -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)
    
    # Extract Features
    with torch.no_grad():
        features = model(image_tensor)
    # Remove batch dimension
    features = features.squeeze(0)
    features_array = features.numpy()
    return features_array

def main():
    db = {}
    # 4479 images
    for image_path in DATASET_PATH.rglob("*.jpg"):
        # print(image_path)
        filename = image_path.name
        features_array = extract_features(model, image_path)
        db[filename] = features_array
    # print(db)
    with open("n216-feats.pkl", "wb") as f:
        pickle.dump(db, f)

if __name__ == "__main__":
    main()
    

    



