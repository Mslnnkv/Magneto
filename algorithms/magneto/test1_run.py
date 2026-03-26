import pandas as pd
from magneto import Magneto

source_df = pd.read_csv("source.csv")
target_df = pd.read_csv("target.csv")

source_df.columns = source_df.columns.str.strip()
target_df.columns = target_df.columns.str.strip()

matcher = Magneto(
    embedding_model="mpnet",
    encoding_mode="table_context_target_values",
    sampling_mode="mixed",
    sampling_size=5,
    topk=10,
    use_bp_reranker=False,
)

matches = matcher.get_matches(source_df, target_df)
print(matches)