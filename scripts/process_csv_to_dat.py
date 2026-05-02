#!/usr/bin/env python3
"""
Process per-image CSV files into binned .dat files for pgfplots.

Usage:
  python process_csv_to_dat.py \
      --csv_dir ../../docs/report/figures \
      --output_dir ../../docs/report/figures

Reads:  per_image_baseline.csv, per_image_richprompt.csv,
        per_image_yolo.csv, per_image_yolo_rp.csv
Writes: mae_vs_density.dat, mae_vs_size.dat
"""

import argparse
import csv
import os
import sys


CONFIGS = {
    "baseline":   {"csv": "per_image_baseline.csv",   "label": "VA-Count"},
    "richprompt": {"csv": "per_image_richprompt.csv",  "label": "+Rich Prompt"},
    "yolo":       {"csv": "per_image_yolo.csv",        "label": "+YOLO-World"},
    "yolo_rp":    {"csv": "per_image_yolo_rp.csv",     "label": "+YOLO+RP"},
}

DENSITY_BINS = [
    (0, 20),
    (20, 50),
    (50, 100),
    (100, 200),
    (200, 500),
    (500, 5000),
]

SIZE_BINS = [
    (0, 500),
    (500, 1500),
    (1500, 3000),
    (3000, 6000),
    (6000, 30000),
]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def bin_mae(rows, key, bins):
    """Compute mean MAE per bin."""
    out = []
    for lo, hi in bins:
        items = [float(r["mae"]) for r in rows if lo <= float(r[key]) < hi]
        if items:
            mean = sum(items) / len(items)
            out.append((lo, hi, round(mean, 2), len(items)))
        else:
            out.append((lo, hi, 0, 0))
    return out


def write_dat(path, header, rows):
    with open(path, "w") as f:
        f.write("  ".join(header) + "\n")
        for row in rows:
            f.write("  ".join(str(v) for v in row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default="../../docs/report/figures")
    parser.add_argument("--output_dir", default="../../docs/report/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # === MAE vs Density ===
    density_header = ["bin_mid"]
    density_rows_dict = {}

    for cfg_key, cfg in CONFIGS.items():
        csv_path = os.path.join(args.csv_dir, cfg["csv"])
        if not os.path.exists(csv_path):
            print(f"SKIP {csv_path} (not found)")
            continue
        rows = read_csv(csv_path)
        binned = bin_mae(rows, "density", DENSITY_BINS)
        density_header.append(cfg_key)

        for lo, hi, mean_mae, n in binned:
            mid = (lo + hi) / 2
            if mid not in density_rows_dict:
                density_rows_dict[mid] = [mid]
            density_rows_dict[mid].append(mean_mae)

    if len(density_header) > 1:
        density_path = os.path.join(args.output_dir, "mae_vs_density.dat")
        sorted_rows = [density_rows_dict[k] for k in sorted(density_rows_dict)]
        write_dat(density_path, density_header, sorted_rows)
        print(f"Wrote {density_path}")

    # === MAE vs Size ===
    size_header = ["bin_mid"]
    size_rows_dict = {}

    for cfg_key, cfg in CONFIGS.items():
        csv_path = os.path.join(args.csv_dir, cfg["csv"])
        if not os.path.exists(csv_path):
            continue
        rows = read_csv(csv_path)
        binned = bin_mae(rows, "avg_obj_size", SIZE_BINS)
        size_header.append(cfg_key)

        for lo, hi, mean_mae, n in binned:
            mid = (lo + hi) / 2
            if mid not in size_rows_dict:
                size_rows_dict[mid] = [mid]
            size_rows_dict[mid].append(mean_mae)

    if len(size_header) > 1:
        size_path = os.path.join(args.output_dir, "mae_vs_size.dat")
        sorted_rows = [size_rows_dict[k] for k in sorted(size_rows_dict)]
        write_dat(size_path, size_header, sorted_rows)
        print(f"Wrote {size_path}")

    print("\nDone! Data files ready for pgfplots.")


if __name__ == "__main__":
    main()
