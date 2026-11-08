from difflib import SequenceMatcher
from pathlib import Path
import re
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch

# Rectangle candidates are cropped and OCR'd individually.
MAX_RECTANGLES = 4
MIN_RECT_AREA_FRACTION = 0.002
MAX_RECT_AREA_FRACTION = 0.25
MIN_CROP_SIZE = 20

# Free-text OCR: keep confident detections and ignore very short fragments.
MIN_TEXT_CONFIDENCE = 0.3
MIN_TEXT_LENGTH = 3
MAX_TEXTS_PER_IMAGE = 30
TEXT_MATCH_RATIO = 0.85
MIN_PARTIAL_LENGTH = 4

# Hardcoded image for manual testing (python door_ocr.py).
# IMAGE_PATH = Path("C:\Users\Riley\OneDrive - The University of Sydney (Students)\data_train\corridor_n203\corridor_n203_im2_262.jpg")
DATA_DIR = Path.home() / "OneDrive - The University of Sydney (Students)" / "query"

IMAGE_PATH = DATA_DIR / "corridor_n203" / "corridor_n203_im2_262.jpg"

IMAGE_PATH = DATA_DIR / "20.png"


def load_reader():
    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError(
            "EasyOCR is required for door-tag detection. "
            "Install it with: python -m pip install easyocr"
        ) from error

    use_gpu = torch.cuda.is_available()
    # print(f"EasyOCR device: {'GPU' if use_gpu else 'CPU'}")
    return easyocr.Reader(["en"], gpu=use_gpu)


def prepare_regions(image_path, show_preprocessing=False):
    """Returns the cleaned full-view binary image and binary rectangle crops."""
    image_path = Path(image_path)
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

    edge_maps, rectangles = find_rectangles(enlarged)
    crops = []
    crop_sources = []
    for source, corners in rectangles:
        crop = crop_rectangle(enlarged, corners)
        if min(crop.shape[:2]) < MIN_CROP_SIZE:
            continue
        # Threshold each crop on its own so the plate sets the cutoff.
        _, crop = cv2.threshold(
            crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        crops.append(crop)
        crop_sources.append(source)

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

        # Green outlines came from the strong edges, orange from faint edges.
        outline_colours = {"canny": (0, 255, 0), "faint": (255, 140, 0)}
        outlined = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2RGB)
        for source, corners in rectangles:
            cv2.polylines(outlined, [corners.astype(np.int32)], True,
                          outline_colours[source], 6)
        columns = len(edge_maps) + 1 + len(crops)
        figure, axes = plt.subplots(1, columns, squeeze=False,
                                    figsize=(3 * columns, 3),
                                    constrained_layout=True)
        for column, (name, edges) in enumerate(edge_maps.items()):
            axes[0, column].imshow(edges, cmap="gray")
            axes[0, column].set_title(f"{name} edges")
        outline_column = len(edge_maps)
        axes[0, outline_column].imshow(outlined)
        axes[0, outline_column].set_title(
            f"{len(rectangles)} rectangle candidate(s)")
        for index, (crop, source) in enumerate(zip(crops, crop_sources),
                                               start=1):
            axes[0, outline_column + index].imshow(crop, cmap="gray")
            axes[0, outline_column + index].set_title(
                f"Crop {index} ({source})")
        for axis in axes[0]:
            axis.axis("off")

    return cleaned, crops


def canny_edges(gray):
    """Strong edges: Canny with thresholds scaled to the image median."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    return cv2.Canny(blurred, int(max(0, 0.5 * median)),
                     int(min(255, 1.0 * median)))


def faint_edges(gray):
    """Faint edges: boost local contrast first, then use low Canny thresholds."""
    boosted = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(boosted, (5, 5), 0)
    median = float(np.median(blurred))
    return cv2.Canny(blurred, int(0.2 * median), int(0.5 * median))


# (name, edge detector, gap-closing kernel size). Faint outlines are broken
# up, so they need a bigger kernel to close into a contour.
EDGE_SOURCES = [
    ("canny", canny_edges, 5),
    ("faint", faint_edges, 25),
]


def find_rectangles(gray):
    """Returns {source: closed edge map} and [(source, 4 corners)] candidates.

    Each edge source contributes up to MAX_RECTANGLES; near-duplicates across
    sources are dropped, keeping the first source that found the rectangle.
    """
    frame_area = gray.shape[0] * gray.shape[1]
    edge_maps = {}
    rectangles = []
    for name, detect_edges, close_size in EDGE_SOURCES:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_size, close_size))
        # Close gaps so plate outlines become closed contours.
        edges = cv2.morphologyEx(detect_edges(gray), cv2.MORPH_CLOSE, kernel)
        edge_maps[name] = edges

        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (MIN_RECT_AREA_FRACTION * frame_area
                    <= area <= MAX_RECT_AREA_FRACTION * frame_area):
                continue
            perimeter = cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(corners) == 4 and cv2.isContourConvex(corners):
                candidates.append((area, corners.reshape(-1, 2)))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        # Drop near-duplicates (e.g. the inner and outer edge of one plate).
        kept = 0
        for _, corners in candidates:
            box = cv2.boundingRect(corners)
            if all(box_overlap(box, cv2.boundingRect(other)) < 0.8
                   for _, other in rectangles):
                rectangles.append((name, corners))
                kept += 1
            if kept == MAX_RECTANGLES:
                break
    return edge_maps, rectangles


def box_overlap(box_a, box_b):
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    return inter / (aw * ah + bw * bh - inter)


def order_points(pts):
    # Returns top-left, top-right, bottom-left, bottom-right.
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[3] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)]
    ordered[2] = pts[np.argmax(diff)]
    return ordered


def crop_rectangle(image, corners):
    top_left, top_right, bottom_left, bottom_right = order_points(
        corners.astype(np.float32))
    width = int(max(np.linalg.norm(top_right - top_left),
                    np.linalg.norm(bottom_right - bottom_left)))
    height = int(max(np.linalg.norm(bottom_left - top_left),
                     np.linalg.norm(bottom_right - top_right)))
    destination = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(
        np.float32([top_left, top_right, bottom_left, bottom_right]),
        destination,
    )
    return cv2.warpPerspective(image, matrix, (width, height))


def detect_door_tags(image_path, reader, show_preprocessing=False,
                     include_full_image=True):
    """Room numbers only (e.g. N203), read with a restricted character set."""
    cleaned, crops = prepare_regions(image_path, show_preprocessing)
    tags = set()
    if include_full_image:
        tags |= read_tags(reader, cleaned)
    for crop in crops:
        tags |= read_tags(reader, crop)
    return tags


def read_tags(reader, image):
    detections = reader.readtext(
        image,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    tag_pattern = re.compile(r"^[A-Z]\d{3}$")
    return {
        re.sub(r"[^A-Za-z0-9]", "", text).upper()
        for _, text, _ in detections
        if tag_pattern.fullmatch(re.sub(r"[^A-Za-z0-9]", "", text).upper())
    }


def detect_door_text(image_path, reader, show_preprocessing=False,
                     include_full_image=True):
    """Every confident word and phrase on the door, as a sorted list of strings."""
    cleaned, crops = prepare_regions(image_path, show_preprocessing)
    confidences = {}
    sources = ([cleaned] if include_full_image else []) + crops
    for source in sources:
        for text, confidence in read_text(reader, source).items():
            confidences[text] = max(confidence, confidences.get(text, 0.0))

    # Keep only the most confident entries (longer strings win ties) so the
    # stored list can't grow without bound.
    ranked = sorted(confidences,
                    key=lambda text: (-confidences[text], -len(text), text))
    return sorted(ranked[:MAX_TEXTS_PER_IMAGE])


def normalise_text(text):
    text = re.sub(r"[^A-Za-z0-9 ]", " ", text).upper()
    text = " ".join(text.split())
    # OCR often splits a room number: "N 220" -> "N220".
    return re.sub(r"\b([A-Z]) (\d{2,4})\b", r"\1\2", text)


def read_text(reader, image):
    """Returns {normalised phrase or word: best OCR confidence}."""
    texts = {}
    for _, text, confidence in reader.readtext(image):
        if confidence < MIN_TEXT_CONFIDENCE:
            continue
        phrase = normalise_text(text)
        # Also keep each word so a partial read of a phrase can still match.
        for entry in [phrase] + phrase.split():
            if len(entry) >= MIN_TEXT_LENGTH:
                texts[entry] = max(confidence, texts.get(entry, 0.0))
    return texts


def texts_match(query_texts, candidate_texts, ratio=TEXT_MATCH_RATIO):
    """True when any query string matches any string in the candidate's list.

    Spaces are ignored. Two strings match when they are equal, when one
    contains the other (4+ characters), or when they are at least `ratio`
    similar, which absorbs small OCR misreads.
    """
    return any(
        strings_match(query_text, candidate_text, ratio)
        for query_text in query_texts
        for candidate_text in candidate_texts
    )


def matching_texts(query_texts, candidate_texts, ratio=TEXT_MATCH_RATIO):
    """Every (query string, candidate string) pair that texts_match accepts."""
    return [
        (query_text, candidate_text)
        for query_text in query_texts
        for candidate_text in candidate_texts
        if strings_match(query_text, candidate_text, ratio)
    ]


def strings_match(query_text, candidate_text, ratio=TEXT_MATCH_RATIO):
    query = query_text.replace(" ", "")
    candidate = candidate_text.replace(" ", "")
    if query == candidate:
        return True
    if min(len(query), len(candidate)) < MIN_PARTIAL_LENGTH:
        return False
    if query in candidate or candidate in query:
        return True
    return SequenceMatcher(None, query, candidate).ratio() >= ratio


def main():
    reader = load_reader()
    tags = detect_door_tags(IMAGE_PATH, reader, show_preprocessing=True)
    print(f"{IMAGE_PATH.name} tags: {', '.join(sorted(tags)) or 'none'}")
    texts = detect_door_text(IMAGE_PATH, reader)
    print(f"{IMAGE_PATH.name} text: {texts or 'none'}")
    plt.show()


if __name__ == "__main__":
    main()
