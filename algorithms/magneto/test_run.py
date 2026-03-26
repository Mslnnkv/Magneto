import pandas as pd
from magneto import Magneto

source_df = pd.read_csv("source.csv")
target_df = pd.read_csv("target.csv")

matcher = Magneto(
    embedding_model="mpnet",
    encoding_mode="header_values_verbose",
    topk=5,
    use_bp_reranker=False
)

matches = matcher.get_matches(source_df, target_df)
print(matches)