# AMME5710 Major Project

This folder contains an indoor visual navigation system. It takes a photo of the user's surroundings and a destination string containing a room and location, then identifies the user's current location and plans a route to the destination.

## Outputs

The system outputs:

- The estimated current room and location.
- Detected text from the query image.
- The best-matching reference image.
- The matching method used, including OCR, CNN, and ORB results.
- ORB feature-match and RANSAC inlier counts.
- The selected destination room and location.
- The shortest route between the current location and destination.
- A map showing the planned route.
- Directional images for each stage of the route.
- Batch query results saved to a CSV file.

## Project layout

```text
navigation/  Main query pipeline, OCR, features, and route modules
scripts/     Dataset, batch-query, map, and report utilities
tests/       Tests
```

## Local data setup

No global or OneDrive paths are required. Put the reference images and query photos in these repository folders:

```text
data_train/
└── n203_door/
    └── n203_door_im0_120.jpg
query/
└── query.jpg
```

- `data_train/` contains reference images grouped into `<room>_<location>` folders, such as `n203_door` or `corridor_n233`.
- `query/` contains the photo to locate. The default filename is `query.jpg`; other files can be selected with `--query`.
- Keep `roomloc_coords.txt` in the repository root; the map is stored at `assets/usyd_map.png`.

Before the first query, generate the local feature databases:

```bash
python -m navigation.extract_features
python scripts/extract_door.py
```

Run the main query pipeline from the repository root:

```bash
python -m navigation.query_main --query query.jpg --destination stairwell bottom
```

Run batch queries with:

```bash
python scripts/batch_query.py
```

Generated CSV files, figures, and logs are saved in `outputs/`.

## Environment setup

Requires Python 3.11. From the repository root, create and activate a virtual environment:

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the main dependencies:

```bash
python -c "import cv2, torch, torchvision, easyocr, networkx; print('Environment ready')"
```

## Dataset paths

Expected dataset layout:

```text
Aachen-Day-Night/
├── db/
└── query/
    └── day/
        └── nexus4/
```
