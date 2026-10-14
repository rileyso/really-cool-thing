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
DATASET_PATH = Path("data")

# QUERY_PATH = Path("D:/Datasets/Aachen-Day-Night/query/day/nexus4/IMG_20130210_163148.jpg")
QUERY_PATH = Path("query/n216_door_im8_72.jpg")
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
    
fig, ax = plt.subplots(1, 4, figsize=(12, 5))

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
# plt.show()

image_0 = QUERY_PATH
image_1 = candidates[0][0]
image_2 = candidates[1][0]
image_3 = candidates[2][0]

print(f"query: {image_0}")

# Load image and extract ORB features
query = cv2.imread(str(image_0), cv2.IMREAD_GRAYSCALE)

candidate_1 = cv2.imread(str(image_1), cv2.IMREAD_GRAYSCALE)
candidate_2 = cv2.imread(str(image_2), cv2.IMREAD_GRAYSCALE)
candidate_3 = cv2.imread(str(image_3), cv2.IMREAD_GRAYSCALE)

# Extract features
orb = cv2.ORB_create(nfeatures=1000) # "nfeatures" controls the sensitivity of how many features we want to extract
kp1, des1 = orb.detectAndCompute(query,None) # second parameter is a mask image (only extract points inside mask)
kp2, des2 = orb.detectAndCompute(candidate_1,None)

kp3, des3 = orb.detectAndCompute(candidate_2,None)
kp4, des4 = orb.detectAndCompute(candidate_3,None)

# print('found %d and %d feature points ...'%(len(kp1),len(kp2)))
# kp: keypoints (image location of feature), des: descriptor: ORB feature descriptor for each keypoint

# draw located keypoints
queryc = cv2.cvtColor(query, cv2.COLOR_GRAY2RGB)
candidate_1c = cv2.cvtColor(candidate_1, cv2.COLOR_GRAY2RGB)
candidate_2c = cv2.cvtColor(candidate_2, cv2.COLOR_GRAY2RGB)
candidate_3c = cv2.cvtColor(candidate_3, cv2.COLOR_GRAY2RGB)

for k in kp1:
    cv2.circle(queryc,(int(k.pt[0]), int(k.pt[1])),3,(0,255,0),-1)
for k in kp2:
    cv2.circle(candidate_1c,(int(k.pt[0]), int(k.pt[1])),3,(0,255,0),-1)
for k in kp3:
    cv2.circle(candidate_2c,(int(k.pt[0]), int(k.pt[1])),3,(0,255,0),-1)
for k in kp4:
    cv2.circle(candidate_3c,(int(k.pt[0]), int(k.pt[1])),3,(0,255,0),-1)

plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.imshow(queryc,'grey')
plt.title(f'Left Image {image_0}')
plt.gca().set_axis_off()

plt.subplot(142)
plt.imshow(candidate_1c,'grey')
plt.title(f'Right Image{image_1}')
plt.gca().set_axis_off()

plt.subplot(143)
plt.imshow(candidate_2c,'grey')
plt.title(f'Right Image{image_2}')
plt.gca().set_axis_off()

plt.subplot(144)
plt.imshow(candidate_3c,'grey')
plt.title(f'Right Image{image_3}')
plt.gca().set_axis_off()
plt.show()

