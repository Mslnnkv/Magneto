from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from magneto.contextual_column_encoder import ContextualColumnEncoder
from magneto.embedding_matcher import DEFAULT_MODELS
from transformers import AutoTokenizer

source_df = pd.read_csv(ROOT / "source.csv")
source_df.columns = source_df.columns.str.strip()

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["mpnet"])

encoder = ContextualColumnEncoder(
    tokenizer=tokenizer,
    encoding_mode="table_context_target_values",
    sampling_mode="mixed",
    n_samples=3,
)

col = source_df.columns[0]
text = encoder.encode(source_df, col)

print("COLUMN:", col)
print()
print(text)