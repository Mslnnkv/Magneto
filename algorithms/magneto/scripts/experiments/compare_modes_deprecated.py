from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from magneto import Magneto

SOURCE_PATH = ROOT / "miller2_vertical_70_ec_av_source.csv"
TARGET_PATH = ROOT / "miller2_vertical_70_ec_av_target.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def pretty_print_matches(title, matches, limit=50):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)

    for i, (pair, score) in enumerate(sorted_matches[:limit], start=1):
        print(f"{i:02d}. {pair} -> {score:.6f}")


def run_mode(source_df, target_df, mode_name):
    matcher = Magneto(
        embedding_model="mpnet",
        encoding_mode=mode_name,
        sampling_mode="mixed",
        sampling_size=5,
        topk=10,
        use_bp_reranker=False,
    )
    return matcher.get_matches(source_df, target_df)


def main():
    source_df = pd.read_csv(SOURCE_PATH)
    target_df = pd.read_csv(TARGET_PATH)

    source_df = normalize_columns(source_df)
    target_df = normalize_columns(target_df)

    print("SOURCE shape:", source_df.shape)
    print("TARGET shape:", target_df.shape)

    print("\nSOURCE columns:")
    for col in source_df.columns:
        print("-", col)

    print("\nTARGET columns:")
    for col in target_df.columns:
        print("-", col)

    legacy_matches = run_mode(source_df, target_df, "header_values_verbose")
    contextual_matches = run_mode(source_df, target_df, "table_context_target_values")

    pretty_print_matches("LEGACY MODE: header_values_verbose", legacy_matches, limit=50)
    pretty_print_matches("CONTEXTUAL MODE: table_context_target_values", contextual_matches, limit=50)


if __name__ == "__main__":
    main()