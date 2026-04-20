from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fire_es.research.resource_prediction import compare_rank_vs_resource_modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank-target-column", default="rank_tz_vector")
    parser.add_argument("--resource-target-column", default="equipment_count")
    parser.add_argument("--split-protocol", default="group_shuffle")
    parser.add_argument("--features", nargs="+", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = compare_rank_vs_resource_modes(
        df,
        feature_columns=args.features,
        rank_target_column=args.rank_target_column,
        resource_target_column=args.resource_target_column,
        split_protocol=args.split_protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
