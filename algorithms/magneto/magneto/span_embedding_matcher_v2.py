import os

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from magneto.embedding_matcher import DEFAULT_MODELS
from magneto.span_contextual_encoder_v2 import SpanContextualEncoderV2
from magneto.utils.embedding_utils import compute_cosine_similarity_simple


class SpanEmbeddingMatcherV2:
    def __init__(self, params):
        self.params = params
        self.topk = params["topk"]
        self.embedding_threshold = params["embedding_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = params["embedding_model"]
        self.max_length = params.get("span_max_length", 512)

        self.target_block_start_marker = params.get("target_block_start_marker", "<<TARGET_BLOCK_START>>")
        self.target_block_end_marker = params.get("target_block_end_marker", "<<TARGET_BLOCK_END>>")

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        if self.model_name in DEFAULT_MODELS:
            model_path = DEFAULT_MODELS[self.model_name]
            self.model = SentenceTransformer(model_path, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            print(f"Loaded default model '{self.model_name}' on {self.device}")
        elif "/" in self.model_name and not self.model_name.endswith((".pth", ".pt", ".bin", ".ckpt")):
            try:
                print(f"Attempting to load HuggingFace model '{self.model_name}'...")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
                print(f"Successfully loaded HuggingFace model '{self.model_name}' on {self.device}")
            except Exception as e:
                print(f"Failed to load HuggingFace model '{self.model_name}': {e}")
                print("Falling back to default 'mpnet' model")
                model_path = DEFAULT_MODELS["mpnet"]
                self.model = SentenceTransformer(model_path, device=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        elif os.path.exists(self.model_name) and os.path.isfile(self.model_name):
            base_key = next((key for key in DEFAULT_MODELS if key in self.model_name), "mpnet")
            if base_key not in DEFAULT_MODELS:
                base_key = "mpnet"

            base_model_path = DEFAULT_MODELS[base_key]
            self.model = SentenceTransformer(base_model_path, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
            print(f"Loaded base model '{base_key}' on {self.device}")
            print(f"Loading fine-tuned weights from {self.model_name}")

            state_dict = torch.load(self.model_name, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval().to(self.device)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        self.model.eval()

    def _build_encoder(self):
        return SpanContextualEncoderV2(
            tokenizer=self.tokenizer,
            encoding_mode=self.params["encoding_mode"],
            sampling_mode=self.params["sampling_mode"],
            n_samples=self.params["sampling_size"],
            max_context_columns=self.params.get("max_context_columns", 7),
            include_target_type=self.params.get("include_target_type", True),
            include_target_values=self.params.get("include_target_values", True),
            target_block_start_marker=self.params.get("target_block_start_marker", "<<TARGET_BLOCK_START>>"),
            target_block_end_marker=self.params.get("target_block_end_marker", "<<TARGET_BLOCK_END>>"),
        )

    def _get_transformer_module(self):
        for module in self.model._modules.values():
            if hasattr(module, "auto_model"):
                return module.auto_model
        raise ValueError("Could not access underlying transformer model.")

    def _mean_pool(self, hidden_states):
        return hidden_states.mean(dim=0)

    def _extract_span_embedding(self, encoded_item):
        text = encoded_item["text"]
        target_char_start = encoded_item["target_char_start"]
        target_char_end = encoded_item["target_char_end"]

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        offset_mapping = tokenized["offset_mapping"][0].tolist()
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        transformer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        transformer = self._get_transformer_module()

        with torch.no_grad():
            outputs = transformer(**transformer_inputs)
            last_hidden_state = outputs.last_hidden_state[0]

        span_token_indices = []
        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                continue
            if end <= target_char_start:
                continue
            if start >= target_char_end:
                continue
            span_token_indices.append(idx)

        if not span_token_indices:
            raise ValueError(
                f"Could not find span tokens for target column '{encoded_item['target_col']}'."
            )

        span_hidden = last_hidden_state[span_token_indices]
        span_embedding = self._mean_pool(span_hidden)
        span_embedding = torch.nn.functional.normalize(span_embedding, p=2, dim=0)

        return span_embedding.cpu()

    def _encode_dataframe_columns(self, df, encoder):
        encoded_pairs = []
        for col in df.columns:
            encoded_item = encoder.encode(df, col)
            embedding = self._extract_span_embedding(encoded_item)
            encoded_pairs.append((col, embedding, encoded_item))
        return encoded_pairs

    def get_embedding_similarity_candidates(self, source_df, target_df):
        encoder = self._build_encoder()

        source_encoded = self._encode_dataframe_columns(source_df, encoder)
        target_encoded = self._encode_dataframe_columns(target_df, encoder)

        if not source_encoded or not target_encoded:
            return {}

        source_cols = [col for col, _, _ in source_encoded]
        target_cols = [col for col, _, _ in target_encoded]

        source_embeddings = torch.stack([emb for _, emb, _ in source_encoded])
        target_embeddings = torch.stack([emb for _, emb, _ in target_encoded])

        top_k = min(self.topk, len(target_cols))
        similarities, indices = compute_cosine_similarity_simple(
            source_embeddings, target_embeddings, top_k
        )

        candidates = {}
        for i, source_col in enumerate(source_cols):
            for j in range(top_k):
                target_idx = indices[i, j]
                similarity = similarities[i, j].item()

                if similarity >= self.embedding_threshold:
                    target_col = target_cols[target_idx]
                    candidates[(source_col, target_col)] = similarity

        return candidates