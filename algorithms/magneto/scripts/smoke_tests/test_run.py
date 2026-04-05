from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from magneto import Magneto

source_df = pd.read_csv(ROOT / "source.csv")
target_df = pd.read_csv(ROOT / "target.csv")

matcher = Magneto(
    embedding_model="mpnet",
    encoding_mode="header_values_verbose",
    topk=5,
    use_bp_reranker=False
)

matches = matcher.get_matches(source_df, target_df)
print(matches)