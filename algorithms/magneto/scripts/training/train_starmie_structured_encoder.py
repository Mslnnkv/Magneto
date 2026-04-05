from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from magneto.embedding_matcher import DEFAULT_MODELS
from magneto.starmie_contextual_encoder import StarmieContextualEncoder


BASE_MODEL = "mpnet"
MODEL_SAVE_PATH = str(ROOT / "finetuned_context_window_starmie_structured_mpnet.pth")
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MARGIN = 0.2
COL_MARKER = "<<COL>>"

TRIPLET_FILES = [
    str(ROOT / "synthetic_benchmark_version_3" / "version_3_triplets.csv"),
    str(ROOT / "synthetic_benchmark_version_4" / "version_4_triplets.csv"),
]

random.seed(42)


class TripletTableDataset(Dataset):
    def __init__(self, triplet_csvs):
        self.triplets = pd.concat([pd.read_csv(csv_path) for csv_path in triplet_csvs], ignore_index=True)
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
        return {
            "source_df": self._load_table(row["source_csv"]),
            "target_df": self._load_table(row["target_csv"]),
            "source_col": row["source_col"],
            "positive_target_col": row["positive_target_col"],
            "negative_target_col": row["negative_target_col"],
        }


def collate_fn(batch):
    return batch


def apply_starmie_augmentation(df, target_col):
    augmented = df.copy()

    if len(augmented) > 12:
        sample_size = max(12, int(len(augmented) * random.uniform(0.55, 0.9)))
        sampled_indices = sorted(random.sample(range(len(augmented)), k=min(sample_size, len(augmented))))
        augmented = augmented.iloc[sampled_indices].reset_index(drop=True)

    if len(augmented.columns) > 3 and random.random() < 0.6:
        removable_columns = [column for column in augmented.columns if column != target_col]
        column_to_drop = random.choice(removable_columns)
        augmented = augmented.drop(columns=[column_to_drop])

    if random.random() < 0.7:
        cell_drop_probability = random.uniform(0.05, 0.15)
        for column in augmented.columns:
            mask = [random.random() < cell_drop_probability for _ in range(len(augmented))]
            if any(mask):
                augmented.loc[mask, column] = None

    if random.random() < 0.35:
        augmented = augmented.sample(frac=1.0, random_state=random.randint(0, 10_000)).reset_index(drop=True)

    if target_col not in augmented.columns:
        augmented[target_col] = df[target_col]

    return augmented


class StarmieStructuredTrainer(nn.Module):
    def __init__(self, sentence_model, tokenizer, device, max_context_columns=7, sampling_size=5):
        super().__init__()
        self.sentence_model = sentence_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_columns = max_context_columns
        self.sampling_size = sampling_size

        self.encoder = StarmieContextualEncoder(
            tokenizer=self.tokenizer,
            encoding_mode="table_context_window_starmie_structured",
            sampling_mode="mixed",
            n_samples=self.sampling_size,
            max_context_columns=self.max_context_columns,
            col_marker=COL_MARKER,
            include_header=True,
            include_values=True,
            include_target_type=True,
        )
        self.col_marker_id = self.tokenizer.convert_tokens_to_ids(COL_MARKER)

    def get_transformer_module(self):
        for module in self.sentence_model._modules.values():
            if hasattr(module, "auto_model"):
                return module.auto_model
        raise ValueError("Could not access underlying transformer model.")

    def encode_batch_columns(self, df_col_pairs, apply_augmentation=False):
        encoded_items = []
        for df, col in df_col_pairs:
            working_df = apply_starmie_augmentation(df, col) if apply_augmentation else df
            if col not in working_df.columns:
                working_df[col] = df[col]
            encoded_items.append(self.encoder.encode(working_df, col))

        texts = [item["text"] for item in encoded_items]
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        transformer = self.get_transformer_module()
        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        embeddings = []
        for row_idx, encoded_item in enumerate(encoded_items):
            marker_positions = (input_ids[row_idx] == self.col_marker_id).nonzero(as_tuple=False).flatten().tolist()
            target_marker_ordinal = encoded_item["target_marker_ordinal"]

            if target_marker_ordinal >= len(marker_positions):
                target_position = 0
            else:
                target_position = marker_positions[target_marker_ordinal]

            emb = last_hidden_state[row_idx, target_position, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=0)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)


def ensure_marker_token(sentence_model, tokenizer):
    vocab = tokenizer.get_vocab()
    if COL_MARKER not in vocab:
        tokenizer.add_special_tokens({"additional_special_tokens": [COL_MARKER]})
        for module in sentence_model._modules.values():
            if hasattr(module, "auto_model"):
                module.auto_model.resize_token_embeddings(len(tokenizer))
                break


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = DEFAULT_MODELS[BASE_MODEL]

    sentence_model = SentenceTransformer(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    ensure_marker_token(sentence_model, tokenizer)

    trainer_model = StarmieStructuredTrainer(
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

            anchor_emb = trainer_model.encode_batch_columns(anchor_pairs, apply_augmentation=True)
            positive_emb = trainer_model.encode_batch_columns(positive_pairs, apply_augmentation=True)
            negative_emb = trainer_model.encode_batch_columns(negative_pairs, apply_augmentation=True)

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
