"""Aggregate row-level evaluation outputs by country."""

import argparse
from pathlib import Path

import pandas as pd


def normalize_answer(value):
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text


def evaluate_results(input_file, output_file=None):
    df = pd.read_csv(input_file)
    required = {"country", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df["answer"] = df["answer"].apply(normalize_answer)
    df = df[df["answer"].isin(["yes", "no"])].copy()
    if df.empty:
        raise ValueError("No valid yes/no answers found after normalization.")

    country_counts = df.groupby("country")["answer"].value_counts().unstack(fill_value=0)
    for col in ["yes", "no"]:
        if col not in country_counts.columns:
            country_counts[col] = 0
    denom = country_counts["yes"] + country_counts["no"]
    country_counts["yes_percentage"] = (country_counts["yes"] / denom.where(denom != 0, 1)) * 100

    result = country_counts[["yes", "no", "yes_percentage"]].sort_index()
    print(result)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path)
        print(f"Aggregated results saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, help="Row-level evaluation CSV.")
    parser.add_argument("-o", "--output_file", default=None, help="Optional CSV path for aggregated results.")
    args = parser.parse_args()
    evaluate_results(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
