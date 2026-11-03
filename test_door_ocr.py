from collections import defaultdict
from pathlib import Path
from door_ocr import load_reader, detect_door_text

DATASET_PATH = Path("C:/Users/Riley/OneDrive - The University of Sydney (Students)/data_train")


def main():
    reader = load_reader()
    image_paths = sorted(DATASET_PATH.rglob("*.jpg"))

    # folder name -> [detected count, total count]
    per_folder = defaultdict(lambda: [0, 0])
    for image_path in image_paths:
        texts = detect_door_text(image_path, reader)
        folder = image_path.parent.name
        per_folder[folder][1] += 1
        if texts:
            per_folder[folder][0] += 1
            print(f"{folder}/{image_path.name}: {texts}", flush=True)

    print("\nDetected per folder:")
    for folder, (detected, total) in per_folder.items():
        print(f"  {folder}: {detected}/{total}")

    detected = sum(d for d, _ in per_folder.values())
    total = len(image_paths)
    percent = 100 * detected / total if total else 0
    print(f"\nOCR detected text in {detected} out of {total} images "
          f"({percent:.1f}%)")


if __name__ == "__main__":
    main()
