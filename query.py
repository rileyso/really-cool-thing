from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pickle
from extract_features import extract_features
from sklearn.metrics.pairwise import cosine_similarity

# DATASET_PATH = Path("D:\Datasets\Aachen-Day-Night\db")
DATASET_PATH = Path("C:/Users/Riley/Uni/AMME5710/really-cool-thing/data")

 
# QUERY_PATH = Path("D:/Datasets/Aachen-Day-Night/query/day/nexus4/IMG_20130210_163148.jpg")
QUERY_PATH = Path("C:/Users/Riley/Uni/AMME5710/really-cool-thing/query/IMG_20261012_125222_1.jpg")
# QUERY_PATH = DATASET_PATH / "998.jpg"

with open("n216-feats.pkl", "rb") as f:
    db = pickle.load(f)
    
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()
preprocess = weights.transforms()

query_features = extract_features(model, QUERY_PATH)

results = []
for filename, db_features in db.items():
    image_path = DATASET_PATH / filename
    similarity = cosine_similarity(
        query_features.reshape(1, -1),
        db_features.reshape(1, -1)
    )[0, 0]
    results.append((image_path, similarity))

results.sort(key=lambda x: x[1], reverse=True)
candidates = results[:3]

for image_path, similarity in candidates:
    print(image_path, similarity)
    

fig, ax = plt.subplots(1, 4, figsize=(24, 5))

query_image = cv2.imread(str(QUERY_PATH))
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

ax[0].imshow(query_image)
ax[0].set_title("Query")
ax[0].axis("off")

for i, (image_path, similarity) in enumerate(candidates):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax[i + 1].imshow(image)
    ax[i + 1].set_title(f"Match {i+1}\n{similarity:.3f}")
    ax[i + 1].axis("off")

plt.tight_layout()
plt.show()

