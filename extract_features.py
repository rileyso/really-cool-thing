from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pickle

# DATASET_PATH = Path("really-cool-thing/data_train")
DATASET_PATH = Path("C:/Users/Riley/OneDrive - The University of Sydney (Students)/data_train")

## Setup
# Load Pretrained Resnet-18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
# Stop before adaptive average pooling and the classifier so spatial layout remains.
model = torch.nn.Sequential(*list(model.children())[:-2])
model.eval()
preprocess = weights.transforms()

def extract_features(model, preprocess, image_path):
    # prepare image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print("image shape:", image.shape)
    image = Image.fromarray(image)
    image_tensor = preprocess(image)
    # Add batch dimensions
    # [3, 224, 224] -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)

    # Use the same device as the model (CUDA when available).
    image_tensor = image_tensor.to(next(model.parameters()).device)
    with torch.no_grad():
        features = model(image_tensor)
    features_array = features.squeeze(0).cpu().numpy()
    return features_array

def main():
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    db = {}
    count= 0
    for image_path in DATASET_PATH.rglob("*.jpg"):
        filename = image_path.name
        print(filename, count)
        count+=1
        features_array = extract_features(model, preprocess, image_path)
        db[filename] = features_array
    print(len(db))
    with open("aero-feats.pkl", "wb") as f:
        pickle.dump(db, f)

if __name__ == "__main__":
    main()