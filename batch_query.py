import argparse
import csv
import re
from query_sift import (QUERY_DIR, find_best_match, load_databases,
                        load_resnet18)
from door_ocr import load_reader

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
COLUMNS = ["query_file", "final_file", "final_location", "method",
           "matched_text", "query_text", "orb_matches", "ransac_inliers",
           "status"]


def natural_key(path):
    # 2.jpg before 10.jpeg.
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.stem)]


def format_matches(matched_texts):
    # "N233" when both sides are identical, "ORISED ~ AUTHORISED" otherwise.
    seen = []
    for query_text, candidate_text in matched_texts:
        entry = (query_text if query_text == candidate_text
                 else f"{query_text} ~ {candidate_text}")
        if entry not in seen:
            seen.append(entry)
    return "; ".join(seen)


def short_matches(matched_texts, limit=2):
    # Console summary: exact matches first, then the longest strings.
    ranked = sorted(
        {(query_text, candidate_text) for query_text, candidate_text
         in matched_texts},
        key=lambda pair: (pair[0] != pair[1], -len(pair[0]), pair[0]),
    )
    shown = format_matches(ranked[:limit])
    extra = len(ranked) - limit
    if not ranked:
        return "none"
    return shown + (f" (+{extra})" if extra > 0 else "")


def main(out_path):
    query_paths = sorted(
        (path for path in QUERY_DIR.iterdir()
         if path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )

    ocr_reader = load_reader()
    db, door_db = load_databases()
    model, preprocess = load_resnet18()

    rows = []
    for query_path in query_paths:
        row = dict.fromkeys(COLUMNS, "")
        row["query_file"] = query_path.name
        try:
            result = find_best_match(query_path, ocr_reader, model,
                                     preprocess, db, door_db)
        except Exception as error:
            row["status"] = f"error: {error}"
            print(f"{query_path.name:<9} ERROR no text match in top "
                  f"candidates", flush=True)
            rows.append(row)
            continue

        final_path = result["best_candidate_path"]
        # Same room_in-room naming the route graph uses, e.g. corridor_n233.
        final_location = "_".join(final_path.name.split("_")[:2])
        row.update({
            "final_file": final_path.name,
            "final_location": final_location,
            "method": result["method"],
            "matched_text": format_matches(result["matched_texts"]),
            "query_text": "; ".join(result["query_texts"]),
            "orb_matches": result["orb_matches"],
            "ransac_inliers": result["ransac_inliers"],
            "status": "ok",
        })
        rows.append(row)
        # Text matches when there were any, otherwise how it fell back.
        detail = (short_matches(result["matched_texts"])
                  if result["method"] == "text+orb"
                  else f"[{result['method']}]")
        print(f"{query_path.name:<9} {final_location:<18} "
              f"{final_path.name:<28} {detail}", flush=True)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["status"].startswith("ok"))
    print(f"\n{ok} of {len(rows)} queries located. Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run query_sift's pipeline over every image in the "
                    "query folder and save the results to a CSV.")
    parser.add_argument("--out", default="query_results.csv",
                        help="CSV output path (default: query_results.csv)")
    main(parser.parse_args().out)
