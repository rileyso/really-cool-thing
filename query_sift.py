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
import re

import easyocr

DATASET_PATH = Path("C:/Users/Riley/OneDrive - The University of Sydney (Students)/data_train")
# QUERY_PATH = Path("query/IMG_20261012_125222_1.jpg")
# QUERY_PATH = Path("C:/Users/Riley/OneDrive - The University of Sydney (Students)/data_train/n216_door/n216_door_im13_131.jpg")
# QUERY_PATH = Path("C:/Users/Riley/OneDrive - The University of Sydney (Students)/query/Screenshot 2026-10-30 132133.png")

QUERY_PATH = Path("C:/Users/Riley/OneDrive - The University of Sydney (Students)/data_train/n216_door/n216_door_im13_131.jpg")


QUERY_PATH = Path("c:/Users/Riley/OneDrive - The University of Sydney (Students)/query/IMG_3677.jpeg") #220  door
QUERY_PATH = Path("c:/Users/Riley/OneDrive - The University of Sydney (Students)/query/IMG_3679.jpeg") #233 door

DESTINATION = ("stairwell","bottom")
# DESTINATION = ("n216","door")
OCR_SCAN_LIMIT = 50
OCR_MATCH_LIMIT = 5

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
    # Match extract_features.py: preserve the convolutional spatial map.
    model = torch.nn.Sequential(*list(model.children())[:-2])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess

def detect_door_tags(image_path, reader, show_preprocessing=False):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Door tags are expected above the middle of the view.
    top_half = image[:image.shape[0] // 2, :]
    enlarged = cv2.resize(top_half, None, fx=5, fy=5,
                          interpolation=cv2.INTER_CUBIC)
    # Otsu selects one high-contrast cutoff: white door text on black.
    _, thresholded = cv2.threshold(
        enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, morph_kernel)

    if show_preprocessing:
        figure, axes = plt.subplots(1, 3, figsize=(9, 3),
                                    constrained_layout=True)
        for axis, processed_image, title in zip(
                axes,
                (top_half, thresholded, cleaned),
                ("Query top half", "Otsu threshold", "Morphological close")): 
            axis.imshow(processed_image, cmap="gray")
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"OCR preprocessing: {image_path.name}")

    detections = reader.readtext(
        cleaned,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    tag_pattern = re.compile(r"^[A-Z]\d{3}$")
    return {
        re.sub(r"[^A-Za-z0-9]", "", text).upper()
        for _, text, _ in detections
        if tag_pattern.fullmatch(re.sub(r"[^A-Za-z0-9]", "", text).upper())
    }


def main():
    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError(
            "EasyOCR is required for door-tag detection. "
            "Install it with: python -m pip install easyocr"
        ) from error

    use_gpu = torch.cuda.is_available()
    print(f"EasyOCR device: {'GPU' if use_gpu else 'CPU'}")
    ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
    query_tags = detect_door_tags(QUERY_PATH, ocr_reader,
                                  show_preprocessing=True)
    if query_tags:
        print(f"Query door tag(s): {', '.join(sorted(query_tags))}")
    else:
        print("No door tag detected by EasyOCR.")

    with open("aero-feats.pkl", "rb") as f:
        db = pickle.load(f)

    model, preprocess = load_resnet18()
    query_features = extract_features(model, preprocess, QUERY_PATH)
    # print(len(db))
    # print(db)

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
    candidates = results[:OCR_MATCH_LIMIT]

    # Search further down the CNN ranking until enough OCR tag matches are found.
    if query_tags:
        candidates = []
        for index, (image_path, similarity) in enumerate(
                results[:OCR_SCAN_LIMIT], start=1):
            candidate_tags = detect_door_tags(image_path, ocr_reader)
            print(f"Candidate {index} OCR ({image_path.name}): "
                  f"{', '.join(sorted(candidate_tags)) or 'none'}")
            if query_tags & candidate_tags:
                candidates.append((image_path, similarity))
                if len(candidates) == OCR_MATCH_LIMIT:
                    break
        if not candidates:
            raise RuntimeError(
                f"No matching door tag found in the top {OCR_SCAN_LIMIT} "
                "CNN candidates."
            )

    print("candidate confidence rankings: ")

    for image_path, similarity in candidates:
        print(image_path, similarity)
        
    query_image = cv2.imread(str(QUERY_PATH))
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, len(candidates) + 1, figsize=(15, 5))
    ax[0].imshow(query_image)
    ax[0].set_title(f"Query\n{Path(QUERY_PATH).name}")
    ax[0].axis("off")

    for i, (image_path, similarity) in enumerate(candidates):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax[i + 1].imshow(image)
        ax[i + 1].set_title(
            f"Match {i + 1}\n{Path(image_path).name}\nCNN: {similarity:.3f}"
        )
        ax[i + 1].axis("off")

    plt.tight_layout()
    # plt.show()

    image_0 = QUERY_PATH
    image_paths = [candidate[0] for candidate in candidates]

    print(f"query: {image_0}")

    # Load the query and all three candidates in grayscale for ORB.
    query = cv2.imread(str(image_0), cv2.IMREAD_GRAYSCALE)
    candidate_images = [
        cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        for image_path in image_paths
    ]
    if query is None or any(image is None for image in candidate_images):
        raise FileNotFoundError("Could not load the query or one of the candidates")

    # Extract features.
    orb = cv2.ORB_create(nfeatures=1000)
    query_kp, query_des = orb.detectAndCompute(query, None)
    candidate_features = [
        orb.detectAndCompute(image, None) for image in candidate_images
    ]

    # Draw located keypoints.
    queryc = cv2.cvtColor(query, cv2.COLOR_GRAY2RGB)
    candidate_colours = [cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                         for image in candidate_images]
    for keypoint in query_kp or []:
        cv2.circle(queryc, (int(keypoint.pt[0]), int(keypoint.pt[1])),
                   3, (0, 255, 0), -1)
    for image, (keypoints, _) in zip(candidate_colours, candidate_features):
        for keypoint in keypoints or []:
            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])),
                       3, (0, 255, 0), -1)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the query against every candidate and estimate geometric inliers.
    orb_matches = []
    orb_match_counts = []
    ransac_masks = []
    ransac_inlier_counts = []
    for index, (candidate_path, (candidate_kp, candidate_des)) in enumerate(
            zip(image_paths, candidate_features), start=1):
        if query_des is None or candidate_des is None:
            matches = []
        else:
            matches = sorted(
                bf.match(query_des, candidate_des),
                key=lambda match: match.distance,
            )

        # A homography needs at least four point correspondences.
        inlier_mask = [0] * len(matches)
        if len(matches) >= 4:
            query_points = np.float32([
                query_kp[match.queryIdx].pt for match in matches
            ]).reshape(-1, 1, 2)
            candidate_points = np.float32([
                candidate_kp[match.trainIdx].pt for match in matches
            ]).reshape(-1, 1, 2)
            _, homography_mask = cv2.findHomography(
                query_points,
                candidate_points,
                cv2.RANSAC,
                5.0,
            )
            if homography_mask is not None:
                inlier_mask = homography_mask.ravel().tolist()

        inlier_count = sum(inlier_mask)
        orb_matches.append(matches)
        orb_match_counts.append(len(matches))
        ransac_masks.append(inlier_mask)
        ransac_inlier_counts.append(inlier_count)
        print(f"Candidate {index}: {candidate_path} - "
              f"{len(matches)} ORB matches, {inlier_count} RANSAC inliers")

    best_index = max(range(len(orb_match_counts)),
                     key=orb_match_counts.__getitem__)
    best_candidate_path = image_paths[best_index]
    print(f"Highest ORB match count: Candidate {best_index + 1} - "
          f"{best_candidate_path} ({orb_match_counts[best_index]} matches, "
          f"{ransac_inlier_counts[best_index]} RANSAC inliers)")

    # Draw only RANSAC inliers; each subplot has the same allocated size.
    correspondence_fig, correspondence_axes = plt.subplots(
        1, len(image_paths), figsize=(10, 3.5), squeeze=False,
        constrained_layout=True
    )
    for index, axis in enumerate(correspondence_axes[0]):
        candidate_keypoints, _ = candidate_features[index]
        correspondence = cv2.drawMatches(
            query,
            query_kp or [],
            candidate_images[index],
            candidate_keypoints or [],
            orb_matches[index],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(80, 80, 80),
            matchesMask=ransac_masks[index],
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        # auto aspect keeps every correspondence subplot the same size.
        axis.imshow(
            cv2.cvtColor(correspondence, cv2.COLOR_BGR2RGB),
            aspect="auto",
        )
        axis.set_title(
            f"Query: {Path(QUERY_PATH).name}\n"
            f"Match {index + 1}: {Path(image_paths[index]).name}\n"
            f"{len(orb_matches[index])} matches / "
            f"{ransac_inlier_counts[index]} inliers"
        )
        axis.axis("off")
    correspondence_fig.suptitle(
        f"RANSAC inliers: {Path(QUERY_PATH).name}"
    )

    # Figure 3 temporarily disabled.
    if False:
        plt.figure(figsize=(15, 5))
        total_images = len(candidate_colours) + 1
        plt.subplot(1, total_images, 1)
        plt.imshow(queryc)
        plt.title(f"Query\n{Path(QUERY_PATH).name}")
        plt.gca().set_axis_off()

        for index, candidate_colour in enumerate(candidate_colours, start=1):
            plt.subplot(1, total_images, index + 1)
            plt.imshow(candidate_colour)
            plt.title(
                f"Match {index}\n{Path(image_paths[index - 1]).name}"
            )
            plt.gca().set_axis_off()





    # plt.figure(figsize=(6, 6))
    bev = cv2.imread("usyd_map.png")
    

    print(f"final candidate: {best_candidate_path}")
    final = str(best_candidate_path)
    final = final.split('\\')[-1]
    final = final.split('/')[-1]
    final = final.split('.')[0]
    final = final.split('_')
    # print(final)


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

    # print(f"destination {destination.room} coords:")
    # print(destination.x, destination.y)

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
    axes[1].set_title(f"Query\n{Path(QUERY_PATH).name}")

    axes[2].imshow(candidate_colours[best_index])
    axes[2].set_title(
        f"Match\n{Path(best_candidate_path).name}"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()