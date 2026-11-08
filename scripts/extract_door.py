from pathlib import Path
import pickle
from navigation.door_ocr import load_reader, detect_door_text

REPO_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_DIR / "data_train"
OUTPUT_PATH = REPO_DIR / "aero-doors.pkl"


def main():
    reader = load_reader()

    db = {}
    count = 0
    for image_path in sorted(DATASET_PATH.rglob("*.jpg")):
        filename = image_path.name
        # Every word/phrase OCR found on the door; [] when nothing was read.
        db[filename] = detect_door_text(image_path, reader)
        print(f"{count} {image_path.parent.name}/{filename}: "
              f"{db[filename]}", flush=True)
        count += 1

    detected = sum(1 for texts in db.values() if texts)
    print(f"{len(db)} images, {detected} with text detected")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
