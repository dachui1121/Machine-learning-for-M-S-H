"""Apply the four trusted frozen models to cleaned 3/7/28 d input rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from frozen_common import SOURCE_FEATURES
from verify_archive import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Clean CSV with the 13 source-feature columns")
    parser.add_argument("--output", type=Path, required=True, help="New prediction CSV path")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    missing = sorted(set(SOURCE_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if not data["curing_age"].isin([3, 7, 28]).all():
        raise ValueError("The frozen target feature mapping is validated only for 3, 7, and 28 d")
    output = data.copy()
    for label in ("M2", "M3", "M4", "M5"):
        output[f"{label}_predicted_MPa"] = predict(label, data)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(args.output)


if __name__ == "__main__":
    main()
