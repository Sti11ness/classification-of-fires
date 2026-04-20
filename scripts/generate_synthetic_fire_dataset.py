from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fire_es.simulation.digital_twin import generate_synthetic_fire_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    synthetic, profile = generate_synthetic_fire_dataset(
        df,
        n_rows=args.rows,
        missing_rate=args.missing_rate,
        noise_scale=args.noise_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(args.output, index=False, encoding="utf-8-sig")
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
