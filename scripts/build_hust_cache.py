"""Convert official HUST pickles to compact first-H curve caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from batt_rul.data.hust import CURVE_COLUMNS, build_hust_cell_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    paths = sorted(args.raw_root.glob("*.pkl"))
    if args.limit is not None:
        paths = paths[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, path in enumerate(paths, start=1):
        metadata, arrays = build_hust_cell_cache(
            path, max_history=args.max_history, points=args.points
        )
        output = args.output_root / f"HUST_{metadata['cell_id']}.npz"
        np.savez_compressed(
            output,
            **arrays,
            curve_columns=np.asarray(CURVE_COLUMNS),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        manifest.append({**metadata, "cache_file": output.name})
        print(f"[{index}/{len(paths)}] {metadata['cell_id']} -> {output.name}", flush=True)

    frame = pd.DataFrame(manifest).sort_values(["split", "cell_id"])
    frame.to_csv(args.output_root / "manifest.csv", index=False)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cells": len(frame),
                "train": int((frame["split"] == "train").sum()),
                "test": int((frame["split"] == "test").sum()),
            }
        )
    )


if __name__ == "__main__":
    main()

