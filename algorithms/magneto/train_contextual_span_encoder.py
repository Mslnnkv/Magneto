import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from magneto.embedding_matcher import DEFAULT_MODELS
from magneto.span_contextual_encoder import SpanContextualEncoder


BASE_MODEL = "mpnet"   # можно заменить на путь или другой ключ
MODEL_SAVE_PATH = "finetuned_context_window_span_mpnet.pth"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MARGIN = 0.2

TRIPLET_FILES = [
    "synthetic_context_needed_benchmark/context_needed_triplets.csv",
    "synthetic_hard_context_benchmark/hard_triplets.csv",
]

# если хочешь добавить реальные пары, просто положи triplets csv в этот список


class TripletTableDataset(Dataset):
    def __init__(self, triplet_csvs):
        dfs = []
        for csv_path in triplet_csvs:
            dfs.append(pd.read_csv(csv_path))
        self.triplets = pd.concat(dfs, ignore_index=True)

        # кеш таблиц
        self.table_cache = {}

    def _load_table(self, path):
        if path not in self.table_cache:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            self.table_cache[path] = df
        return self.table_cache[path]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]

        source_df = self._load_table(row["source_csv"])
        target_df = self._load_table(row["target_csv"])

        return {
            "source_df": source_df,
            "target_df": target_df,
            "source_col": row["source_col"],
            "positive_target_col": row["positive_target_col"],
            "negative_target_col": row["negative_target_col"],
        }


def collate_fn(batch):
    return batch


class SpanEncoderTrainer(nn.Module):
    def __init__(self, sentence_model, tokenizer, device, max_context_columns=7, sampling_size=5):
        super().__init__()
        self.sentence_model = sentence_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_columns = max_context_columns
        self.sampling_size = sampling_size

        self.encoder = SpanContextualEncoder(
            tokenizer=self.tokenizer,
            encoding_mode="table_context_window_span",
            sampling_mode="mixed",
            n_samples=self.sampling_size,
            max_context_columns=self.max_context_columns,
            include_target_type=True,
            include_target_values=True,
            target_start_marker="<<TARGET_START>>",
            target_end_marker="<<TARGET_END>>",
        )

    def get_transformer_module(self):
        for module in self.sentence_model._modules.values():
            if hasattr(module, "auto_model"):
                return module.auto_model
        raise ValueError("Could not access underlying transformer model.")

    def encode_batch_columns(self, df_col_pairs):
        encoded_items = []
        for df, col in df_col_pairs:
            encoded_items.append(self.encoder.encode(df, col))

        texts = [item["text"] for item in encoded_items]

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        offset_mapping = tokenized["offset_mapping"]
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        transformer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        transformer = self.get_transformer_module()
        outputs = transformer(**transformer_inputs)
        last_hidden_state = outputs.last_hidden_state  # [B, L, H]

        embeddings = []

        for i, item in enumerate(encoded_items):
            char_start = item["target_char_start"]
            char_end = item["target_char_end"]
            offsets = offset_mapping[i].tolist()

            span_token_indices = []
            for idx, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                if end <= char_start:
                    continue
                if start >= char_end:
                    continue
                span_token_indices.append(idx)

            if not span_token_indices:
                # fallback to CLS-like first token if span not found
                span_hidden = last_hidden_state[i, 0:1, :]
            else:
                span_hidden = last_hidden_state[i, span_token_indices, :]

            emb = span_hidden.mean(dim=0)
            emb = torch.nn.functional.normalize(emb, p=2, dim=0)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = DEFAULT_MODELS[BASE_MODEL]

    sentence_model = SentenceTransformer(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    trainer_model = SpanEncoderTrainer(
        sentence_model=sentence_model,
        tokenizer=tokenizer,
        device=device,
        max_context_columns=7,
        sampling_size=5,
    ).to(device)

    dataset = TripletTableDataset(TRIPLET_FILES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(trainer_model.parameters(), lr=LR)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    trainer_model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            anchor_pairs = [(item["source_df"], item["source_col"]) for item in batch]
            positive_pairs = [(item["target_df"], item["positive_target_col"]) for item in batch]
            negative_pairs = [(item["target_df"], item["negative_target_col"]) for item in batch]

            anchor_emb = trainer_model.encode_batch_columns(anchor_pairs)
            positive_emb = trainer_model.encode_batch_columns(positive_pairs)
            negative_emb = trainer_model.encode_batch_columns(negative_pairs)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{EPOCHS} | loss = {avg_loss:.6f}")

    torch.save(sentence_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved fine-tuned weights to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()