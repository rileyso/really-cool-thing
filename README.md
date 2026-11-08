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
app/         NiceGUI application
tests/       Tests
```

Run the main query pipeline from the repository root:

```bash
python -m navigation.query_main
```

Run batch queries with:

```bash
python scripts/batch_query.py
```

## Environment

- Python 3.11

From Bash, activate the existing virtual environment:

```bash
source ../.amme5710/Scripts/activate
```

Install the requirements:

```bash
python -m pip install -r requirements.txt
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
