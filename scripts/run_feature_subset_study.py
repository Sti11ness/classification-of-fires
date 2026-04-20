from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fire_es.research.feature_subset_study import run_feature_subset_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-column", default="rank_tz_vector")
    parser.add_argument("--split-protocol", default="group_shuffle")
    parser.add_argument("--features", nargs="+", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = run_feature_subset_study(
        df,
        feature_pool=args.features,
        semantic_target_column=args.target_column,
        split_protocol=args.split_protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
