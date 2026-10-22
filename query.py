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
import heapq
import networkx as nx
import math
DATASET_PATH = Path("C:/Users/Riley/OneDrive - The University of Sydney (Students)/data_train")
# QUERY_PATH = Path("IMG_20261012_125222_1.jpg")
QUERY_PATH = Path("C:/Users/Riley/OneDrive - The University of Sydney (Students)/query/IMG_20261012_125222_1.jpg")
# QUERY_PATH = Path("c:/Users/Riley/OneDrive - The University of Sydney (Students)/query/20261014_031622580_iOS.jpg")

DESTINATION = ("stairwell","bottom")
DESTINATION = ("n203","door")
class Node:
    def __init__(self, room, in_room, x, y):
        self.room = room
        self.in_room = in_room
        self.x = int(x)
        self.y = int(y)

        # f = g + h
        self.f = float('inf')
        self.g = float('inf') # cost to start from this cell
        self.h = 0 # heuristic - manhattan

def load_resnet18():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess

def main():
    with open("features/aero-feats.pkl", "rb") as f:
        db = pickle.load(f)

    model, preprocess = load_resnet18()
    query_features = extract_features(model, preprocess, QUERY_PATH)


    # Filter Similarity By Cosine Distance
    results = []
    for filename, db_features in db.items():
        foldername = "_".join(filename.split("_")[:2])
        image_path = DATASET_PATH / foldername / filename
        similarity = cosine_similarity(
            query_features.reshape(1, -1),
            db_features.reshape(1, -1)
        )[0, 0]
        results.append((image_path, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    candidates = results[:3]

    print("candidate confidence rankings: ")

    for image_path, similarity in candidates:
        print(image_path, similarity)
        
    

    query_image = cv2.imread(str(QUERY_PATH))
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    # fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    # ax[0].imshow(query_image)
    # ax[0].set_title("Query")
    # ax[0].axis("off")

    # for i, (image_path, similarity) in enumerate(candidates):
    #     image = cv2.imread(str(image_path))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     ax[i + 1].imshow(image)
    #     ax[i + 1].set_title(f"Match {i+1}\n{similarity:.3f}")
    #     ax[i + 1].axis("off")

    # plt.tight_layout()
    # # plt.show()

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

    # plt.figure(figsize=(15, 5))
    # plt.subplot(141)
    # plt.imshow(queryc,'grey')
    # plt.title(f'Left Image {image_0}')
    # plt.gca().set_axis_off()

    # plt.subplot(142)
    # plt.imshow(candidate_1c,'grey')
    # plt.title(f'Right Image{image_1}')
    # plt.gca().set_axis_off()

    # plt.subplot(143)
    # plt.imshow(candidate_2c,'grey')
    # plt.title(f'Right Image{image_2}')
    # plt.gca().set_axis_off()

    # plt.subplot(144)
    # plt.imshow(candidate_3c,'grey')
    # plt.title(f'Right Image{image_3}')
    # plt.gca().set_axis_off()

    # plt.figure(figsize=(6, 6))
    bev = cv2.imread("usyd_map.png")
    

    print(f"final candidate: {candidates[0][0]}")
    final = str(candidates[0][0])
    final = final.split('\\')[-1]
    final = final.split('/')[-1]
    final = final.split('.')[0]
    final = final.split('_')
    print(final)

    try:
        with open("roomloc_coords.txt", 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("Error: the file doesn't exist")
    node_list = []
    for line in content.splitlines():
        line = line.split(',')
        name = line[0]
        in_room = line[1]
        xloc = line[2]
        yloc = line[3]
        node = Node(name, in_room, xloc, yloc)
        node_list.append(node)

    # Circle setup
    radius = 5
    color = (255, 0, 255)
    thickness = -1

    source = next(node for node in node_list
                  if final[0] == node.room and final[1] == node.in_room)
    destination = next(node for node in node_list
                       if DESTINATION[0] == node.room and DESTINATION[1] == node.in_room)

    print(f"destination {destination.room} coords:")
    print(destination.x, destination.y)

    # Keep connections explicit so they can be reviewed and corrected later.
    connections = [
        ("n203_door", "corridor_n203"),
        ("corridor_n203", "corridor_n210"),
        ("corridor_n210", "corridor_stairs"),
        ("corridor_stairs", "corridor_n209"),
        ("corridor_n209", "corridor_n233"),
        ("corridor_n233", "corridor_n222"),
        ("corridor_stairs", "stairwell_top"),
        ("stairwell_top", "stairwell_bottom"),
        ("stairwell_bottom", "stairwell_middle"),
        ("corridor_n233", "n216_door"),
        ("n216_door", "n216_middle"),
        ("n216_middle", "n216_front"),
        ("n214_front", "n214_middleright"),
        ("n214_middleright", "n214_middleleft"),
        ("n214_middleleft", "corridor_n209"),
    ]

    graph = nx.Graph()
    node_by_name = {
        f"{node.room}_{node.in_room}": node
        for node in node_list
    }

    for name, node in node_by_name.items():
        graph.add_node(name, pos=(node.x, node.y))

    for node_a, node_b in connections:
        distance = math.dist(
            graph.nodes[node_a]["pos"],
            graph.nodes[node_b]["pos"]
        )
        graph.add_edge(node_a, node_b, weight=distance)

    source_name = f"{source.room}_{source.in_room}"
    destination_name = f"{destination.room}_{destination.in_room}"

    path = nx.astar_path(
        graph,
        source_name,
        destination_name,
        heuristic=lambda a, b: math.dist(
            graph.nodes[a]["pos"], graph.nodes[b]["pos"]
        ),
        weight="weight",
    )
    print("A* path:", " -> ".join(path))

    # Draw the path before the nodes so the nodes remain visible.
    for node_a, node_b in zip(path, path[1:]):
        a = node_by_name[node_a]
        b = node_by_name[node_b]
        cv2.line(
            bev,
            (a.x, bev.shape[0] - a.y),
            (b.x, bev.shape[0] - b.y),
            (0, 255, 0),
            thickness=3,
        )

    # Draw only nodes currently used by the route.
    # Other nodes stay in the graph, but remain hidden until their connections
    # are reviewed and added to an active route.
    for name in path:
        node = node_by_name[name]
        point = (node.x, bev.shape[0] - node.y)
        node_color = (255, 0, 0) if node is destination else color
        cv2.circle(bev, point, radius, node_color, thickness)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Route")

    axes[1].imshow(queryc)
    axes[1].set_title("Query")

    axes[2].imshow(candidate_1c)
    axes[2].set_title("Matched")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()