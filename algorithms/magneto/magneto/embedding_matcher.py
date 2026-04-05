import os
from pathlib import Path

import torch
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer

from magneto.column_encoder import ColumnEncoder
from magneto.contextual_column_encoder import ContextualColumnEncoder
from magneto.encoding_modes import CONTEXTUAL_ENCODING_MODES, LEGACY_ENCODING_MODES
from magneto.utils.embedding_utils import compute_cosine_similarity_simple
from magneto.utils.utils import detect_column_type, get_samples

DEFAULT_MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "roberta": "sentence-transformers/all-roberta-large-v1",
    "e5": "intfloat/e5-base",
    "arctic": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2"
}


class EmbeddingMatcher:
    def __init__(self, params):
        self.params = params
        self.topk = params["topk"]
        self.embedding_threshold = params["embedding_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = params["embedding_model"]
        self.use_prompt_query = True if "arctic" in self.model_name else False

        # Load model and tokenizer
        if self.model_name in DEFAULT_MODELS:
            model_path = DEFAULT_MODELS[self.model_name]
            self.model = self._load_sentence_transformer(model_path)
            self.tokenizer = self._load_tokenizer(model_path)
            print(f"Loaded default model '{self.model_name}' on {self.device}")
        elif "/" in self.model_name and not self.model_name.endswith((".pth", ".pt", ".bin", ".ckpt")):
            try:
                print(f"Attempting to load HuggingFace model '{self.model_name}'...")
                self.model = self._load_sentence_transformer(self.model_name)
                self.tokenizer = self._load_tokenizer(self.model_name)
                print(f"Successfully loaded HuggingFace model '{self.model_name}' on {self.device}")
            except Exception as e:
                print(f"Failed to load HuggingFace model '{self.model_name}': {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to default 'mpnet' model")
                model_path = DEFAULT_MODELS["mpnet"]
                self.model = self._load_sentence_transformer(model_path)
                self.tokenizer = self._load_tokenizer(model_path)
        else:
            resolved_local_model_path = self._resolve_local_model_path()

            if not resolved_local_model_path:
                raise ValueError(f"Invalid model name: {self.model_name}")

            base_key = next((key for key in DEFAULT_MODELS if key in self.model_name), "mpnet")
            if base_key not in DEFAULT_MODELS:
                print(f"Warning: No base model detected in {self.model_name}, defaulting to 'mpnet'")
                base_key = "mpnet"
            base_model_path = DEFAULT_MODELS[base_key]
            self.model = self._load_sentence_transformer(base_model_path)
            self.tokenizer = self._load_tokenizer(base_model_path)
            print(f"Loaded base model '{base_key}' on {self.device}")
            print(f"Loading fine-tuned weights from {resolved_local_model_path}")
            state_dict = torch.load(resolved_local_model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval().to(self.device)

    def _resolve_local_model_path(self):
        candidate = Path(self.model_name)
        if candidate.exists() and candidate.is_file():
            return str(candidate)

        package_root_candidate = Path(__file__).resolve().parents[1] / self.model_name
        if package_root_candidate.exists() and package_root_candidate.is_file():
            return str(package_root_candidate)

        return None

    def _load_sentence_transformer(self, model_path):
        try:
            return SentenceTransformer(model_path, device=self.device, local_files_only=True)
        except Exception:
            return SentenceTransformer(model_path, device=self.device)

    def _load_tokenizer(self, model_path):
        try:
            return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(model_path)

    def _get_embeddings(self, texts, use_prompt_query=False, batch_size=32):
        """Get embeddings using Sentence Transformer's encode method"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                if use_prompt_query:
                    print("Using prompt query")
                    embeds = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=self.device,
                        prompt_name="query"
                    )
                else:
                    embeds = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=self.device
                    )
            embeddings.append(embeds)
        return torch.cat(embeddings)

    def _build_encoder(self):
        encoding_mode = self.params["encoding_mode"]

        if encoding_mode in LEGACY_ENCODING_MODES:
            return ColumnEncoder(
                self.tokenizer,
                encoding_mode=encoding_mode,
                sampling_mode=self.params["sampling_mode"],
                n_samples=self.params["sampling_size"],
            )

        if encoding_mode in CONTEXTUAL_ENCODING_MODES:
            return ContextualColumnEncoder(
                self.tokenizer,
                encoding_mode=encoding_mode,
                sampling_mode=self.params["sampling_mode"],
                n_samples=self.params["sampling_size"],
                max_context_columns=self.params.get("max_context_columns"),
                include_target_type=self.params.get("include_target_type", True),
                include_target_values=self.params.get("include_target_values", True),
                target_marker_start=self.params.get("target_marker_start", "<TARGET>"),
                target_marker_end=self.params.get("target_marker_end", "</TARGET>"),
            )

        raise ValueError(f"Unsupported encoding mode: {encoding_mode}")

    def _encode_columns(self, df, encoder):
        encoded_pairs = []
        for col in df.columns:
            encoded_text = encoder.encode(df, col)
            encoded_pairs.append((encoded_text, col))
        return encoded_pairs

    def get_embedding_similarity_candidates(self, source_df, target_df):
        encoder = self._build_encoder()

        input_encoded_pairs = self._encode_columns(source_df, encoder)
        target_encoded_pairs = self._encode_columns(target_df, encoder)

        cleaned_input_cols = [text for text, _ in input_encoded_pairs]
        cleaned_target_cols = [text for text, _ in target_encoded_pairs]

        if len(cleaned_input_cols) == 0 or len(cleaned_target_cols) == 0:
            return {}

        input_embeddings = self._get_embeddings(cleaned_input_cols, self.use_prompt_query)
        target_embeddings = self._get_embeddings(cleaned_target_cols)

        top_k = min(self.topk, len(cleaned_target_cols))
        similarities, indices = compute_cosine_similarity_simple(
            input_embeddings, target_embeddings, top_k
        )

        candidates = {}
        for i, (_, original_input) in enumerate(input_encoded_pairs):
            for j in range(top_k):
                target_idx = indices[i, j]
                similarity = similarities[i, j].item()
                if similarity >= self.embedding_threshold:
                    _, original_target = target_encoded_pairs[target_idx]
                    candidates[(original_input, original_target)] = similarity

        return candidates
