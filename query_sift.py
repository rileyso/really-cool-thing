import argparse
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
from door_ocr import (load_reader, detect_door_text, texts_match,
                      matching_texts)

ONEDRIVE_DIR = Path.home() / "OneDrive - The University of Sydney (Students)"
DATASET_PATH = ONEDRIVE_DIR / "data_train"
QUERY_DIR = ONEDRIVE_DIR / "query"

# Query image: uncomment one.
# QUERY_PATH = Path("query/IMG_20261012_125222_1.jpg")
# QUERY_PATH = DATASET_PATH / "n216_door" / "n216_door_im13_131.jpg"
# QUERY_PATH = QUERY_DIR / "18.png"  # was Screenshot 2026-10-30 132133.png
# QUERY_PATH = QUERY_DIR / "7.jpeg"  # 220 door, was IMG_3677.jpeg
QUERY_PATH = QUERY_DIR / "20.png"  # 233 door, was IMG_3660.jpeg

DESTINATION = ("stairwell","bottom")
# DESTINATION = ("n216","door")
OCR_SCAN_LIMIT = 100
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

def load_databases():
    with open("aero-feats.pkl", "rb") as f:
        db = pickle.load(f)

    # Door text lists for the training images, precomputed by extract_door.py.
    with open("aero-doors.pkl", "rb") as f:
        door_db = pickle.load(f)
    return db, door_db

def find_best_match(query_path, ocr_reader, model, preprocess, db, door_db,
                    verbose=False):
    """Locate one query image against the training set.

    CNN ranking, then the OCR text filter, then ORB/RANSAC picks the best of
    the surviving candidates. Prints and figures only appear when verbose.
    """
    query_path = Path(query_path)
    query_texts = detect_door_text(query_path, ocr_reader,
                                   show_preprocessing=verbose)

    query_features = extract_features(model, preprocess, query_path)
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
    method = "cnn+orb (no query text)"

    # Search further down the CNN ranking until enough OCR text matches are found.
    if query_texts:
        text_candidates = []
        for index, (image_path, similarity) in enumerate(
                results[:OCR_SCAN_LIMIT], start=1):
            candidate_texts = door_db.get(image_path.name, [])
            if verbose:
                print(f"Candidate {index} OCR ({image_path.name}): "
                      f"{candidate_texts or 'none'}")
            if texts_match(query_texts, candidate_texts):
                text_candidates.append((image_path, similarity))
                if len(text_candidates) == OCR_MATCH_LIMIT:
                    break
        if text_candidates:
            candidates = text_candidates
            method = "text+orb"
        else:
            # No door text in common: rank by the CNN alone, then ORB.
            method = "cnn+orb (no text match)"
            if verbose:
                print(f"No matching door text in the top {OCR_SCAN_LIMIT} "
                      "CNN candidates; falling back to the top CNN results.")

    if verbose:
        print("candidate confidence rankings: ")

        for image_path, similarity in candidates:
            print(image_path, similarity)

        query_image = cv2.imread(str(query_path))
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, len(candidates) + 1, figsize=(15, 5))
        ax[0].imshow(query_image)
        ax[0].set_title(f"Query\n{query_path.name}")
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

    image_paths = [candidate[0] for candidate in candidates]

    # Load the query and all three candidates in grayscale for ORB.
    query = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
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
        if verbose:
            print(f"Candidate {index}: {candidate_path} - "
                  f"{len(matches)} ORB matches, {inlier_count} RANSAC inliers")

    best_index = max(range(len(orb_match_counts)),
                     key=orb_match_counts.__getitem__)
    best_candidate_path = image_paths[best_index]

    if verbose:
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
                f"Query: {query_path.name}\n"
                f"Match {index + 1}: {Path(image_paths[index]).name}\n"
                f"{len(orb_matches[index])} matches / "
                f"{ransac_inlier_counts[index]} inliers"
            )
            axis.axis("off")
        correspondence_fig.suptitle(
            f"RANSAC inliers: {query_path.name}"
        )

    # Which query/candidate strings the text filter matched on.
    matched_texts = []
    if query_texts:
        matched_texts = matching_texts(
            query_texts, door_db.get(best_candidate_path.name, []))

    return {
        "method": method,
        "query_texts": query_texts,
        "matched_texts": matched_texts,
        "best_index": best_index,
        "best_candidate_path": best_candidate_path,
        "orb_matches": orb_match_counts[best_index],
        "ransac_inliers": ransac_inlier_counts[best_index],
        "query_colour": queryc,
        "best_colour": candidate_colours[best_index],
    }

def main(verbose=False):
    # Quiet by default: only the final route figure is shown. verbose also
    # shows the OCR, candidate and RANSAC figures and per-candidate output.
    ocr_reader = load_reader()
    db, door_db = load_databases()
    model, preprocess = load_resnet18()

    result = find_best_match(QUERY_PATH, ocr_reader, model, preprocess,
                             db, door_db, verbose=verbose)

    if result["query_texts"]:
        print(f"Query door text: {result['query_texts']}")
    else:
        print("No door text detected by EasyOCR.")
    print(f"query: {QUERY_PATH}")
    print(f"Method: {result['method']}")

    best_index = result["best_index"]
    best_candidate_path = result["best_candidate_path"]
    queryc = result["query_colour"]
    best_colour = result["best_colour"]
    print(f"Highest ORB match count: Candidate {best_index + 1} - "
          f"{best_candidate_path} ({result['orb_matches']} matches, "
          f"{result['ransac_inliers']} RANSAC inliers)")
    if result["matched_texts"]:
        print("Matched text: " + "; ".join(
            query_text if query_text == candidate_text
            else f"{query_text} ~ {candidate_text}"
            for query_text, candidate_text in result["matched_texts"]))

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

    axes[2].imshow(best_colour)
    axes[2].set_title(
        f"Match\n{Path(best_candidate_path).name}"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="show the OCR, candidate and RANSAC figures and per-candidate "
             "output, not just the final route figure")
    main(verbose=parser.parse_args().verbose)
