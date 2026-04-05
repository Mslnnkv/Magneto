from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from magneto import Magneto

SOURCE_PATH = "synthetic_benchmark_version_3/version_3_source.csv"
TARGET_PATH = "synthetic_benchmark_version_3/version_3_target.csv"


def run(model_name):
    source_df = pd.read_csv(SOURCE_PATH)
    target_df = pd.read_csv(TARGET_PATH)

    matcher = Magneto(
        embedding_model=model_name,
        encoding_mode="table_context_window_span",
        sampling_mode="mixed",
        sampling_size=5,
        topk=5,
        use_bp_reranker=False,
        max_context_columns=7,
    )
    matches = matcher.get_matches(source_df, target_df)

    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== MODEL: {model_name} ===")
    for pair, score in sorted_matches[:30]:
        print(pair, score)


if __name__ == "__main__":
    run("mpnet")
    run("finetuned_context_window_span_mpnet.pth")
