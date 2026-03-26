import pandas as pd
from magneto import Magneto

SOURCE_PATH = "miller2_vertical_70_ec_av_source.csv"
TARGET_PATH = "miller2_vertical_70_ec_av_target.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def pretty_print_matches(title, matches, limit=50):
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")

    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    for i, (pair, score) in enumerate(sorted_matches[:limit], start=1):
        print(f"{i:02d}. {pair} -> {score:.6f}")


def run_mode(source_df, target_df, mode_name, max_context_columns=None):
    matcher = Magneto(
        embedding_model="mpnet",
        encoding_mode=mode_name,
        sampling_mode="mixed",
        sampling_size=5,
        topk=10,
        use_bp_reranker=False,
        max_context_columns=max_context_columns,
    )
    return matcher.get_matches(source_df, target_df)


def main():
    source_df = pd.read_csv(SOURCE_PATH)
    target_df = pd.read_csv(TARGET_PATH)

    source_df = normalize_columns(source_df)
    target_df = normalize_columns(target_df)

    print("SOURCE shape:", source_df.shape)
    print("TARGET shape:", target_df.shape)

    legacy_matches = run_mode(source_df, target_df, "header_values_verbose")
    contextual_v1_matches = run_mode(source_df, target_df, "table_context_target_values")
    contextual_v2_matches = run_mode(
        source_df,
        target_df,
        "table_context_window_target_values",
        max_context_columns=7,
    )

    pretty_print_matches("LEGACY MODE: header_values_verbose", legacy_matches, limit=50)
    pretty_print_matches("CONTEXTUAL V1: table_context_target_values", contextual_v1_matches, limit=50)
    pretty_print_matches("CONTEXTUAL V2: table_context_window_target_values", contextual_v2_matches, limit=50)


if __name__ == "__main__":
    main()