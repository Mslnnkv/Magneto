from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from magneto.span_embedding_matcher import SpanEmbeddingMatcher
from magneto.span_contextual_encoder import SpanContextualEncoder
from magneto.embedding_matcher import DEFAULT_MODELS
from transformers import AutoTokenizer

SOURCE_PATH = ROOT / "miller2_vertical_70_ec_av_source.csv"

source_df = pd.read_csv(SOURCE_PATH)
source_df.columns = source_df.columns.str.strip()

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["mpnet"], use_fast=True)

encoder = SpanContextualEncoder(
    tokenizer=tokenizer,
    encoding_mode="table_context_window_span",
    sampling_mode="mixed",
    n_samples=5,
    max_context_columns=7,
    include_target_type=True,
    include_target_values=True,
    target_start_marker="<<TARGET_START>>",
    target_end_marker="<<TARGET_END>>",
)

target_col = source_df.columns[0]
encoded = encoder.encode(source_df, target_col)

print("TARGET COLUMN:", target_col)
print()
print("TEXT:")
print(encoded["text"])
print()
print("TARGET CHAR START:", encoded["target_char_start"])
print("TARGET CHAR END:", encoded["target_char_end"])
print()
print("TARGET SUBSTRING:")
print(encoded["text"][encoded["target_char_start"]:encoded["target_char_end"]])

matcher = SpanEmbeddingMatcher(
    params={
        "topk": 10,
        "embedding_threshold": 0.1,
        "embedding_model": "mpnet",
        "encoding_mode": "table_context_window_span",
        "sampling_mode": "mixed",
        "sampling_size": 5,
        "max_context_columns": 7,
        "include_target_type": True,
        "include_target_values": True,
        "target_start_marker": "<<TARGET_START>>",
        "target_end_marker": "<<TARGET_END>>",
        "span_max_length": 512,
    }
)

embedding = matcher._extract_span_embedding(encoded)
print()
print("EMBEDDING SHAPE:", embedding.shape)