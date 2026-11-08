import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps

REPO_DIR = Path(__file__).resolve().parent.parent
QUERY_DIR = REPO_DIR / "query"
DATASET_DIR = REPO_DIR / "data_train"


def image_path(filename):
    folder = "_".join(filename.split("_")[:2])
    return DATASET_DIR / folder / filename


def read_matches(csv_path, query_name=None):
    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if query_name:
        rows = [row for row in rows if row["query_filename"] == query_name]
    rows = [row for row in rows if row["matched_filename"] != "ERROR"]
    if not rows:
        raise SystemExit("No matching rows found in the CSV log.")
    return rows


def plot_pair(row, output_path):
    query_name = row["query_filename"]
    match_name = row["matched_filename"]
    query = ImageOps.exif_transpose(Image.open(QUERY_DIR / query_name))
    match = ImageOps.exif_transpose(Image.open(image_path(match_name)))

    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (query, match),
        (f"Query\n{query_name}", f"Matched reference\n{match_name}"),
    ):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    figure.text(
        0.5, 0.02,
        f"{row['method']}  |  ORB matches: {row['orb_matches']}  |  "
        f"RANSAC inliers: {row['ransac_inliers']}",
        ha="center", fontsize=9,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def plot_all(rows, output_path):
    # Compact contact sheet: five rows of four query-to-match pairs.
    columns = 4
    pair_rows = (len(rows) + columns - 1) // columns
    figure, axes = plt.subplots(
        pair_rows, columns * 2,
        figsize=(16, max(3, 2.8 * pair_rows)),
        squeeze=False,
    )
    for index, row in enumerate(rows):
        row_index, column_index = divmod(index, columns)
        query = ImageOps.exif_transpose(Image.open(QUERY_DIR / row["query_filename"]))
        match = ImageOps.exif_transpose(Image.open(image_path(row["matched_filename"])))
        query_axis = axes[row_index, column_index * 2]
        match_axis = axes[row_index, column_index * 2 + 1]
        query_axis.imshow(query)
        match_axis.imshow(match)
        query_axis.set_title(row["query_filename"], fontsize=8)
        match_axis.set_title(row["matched_filename"], fontsize=7)
        query_axis.text(1.04, 0.5, "→", transform=query_axis.transAxes,
                        ha="center", va="center", fontsize=14, color="red")
        query_axis.axis("off")
        match_axis.axis("off")

    for index in range(len(rows), pair_rows * columns):
        row_index, column_index = divmod(index, columns)
        axes[row_index, column_index * 2].axis("off")
        axes[row_index, column_index * 2 + 1].axis("off")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create query-to-matched-reference figures from query_results.csv."
    )
    parser.add_argument("--csv", default="outputs/query_results.csv")
    parser.add_argument("--query", help="one query filename, e.g. 20.png")
    parser.add_argument("--output", default="outputs/report_query_match.png")
    parser.add_argument(
        "--all", action="store_true",
        help="plot every successful query match instead of one query",
    )
    args = parser.parse_args()

    rows = read_matches(args.csv, None if args.all else args.query)
    if not args.all and len(rows) != 1:
        raise SystemExit("Use --query filename or --all.")
    if args.all:
        plot_all(rows, args.output)
    else:
        plot_pair(rows[0], args.output)


if __name__ == "__main__":
    main()
