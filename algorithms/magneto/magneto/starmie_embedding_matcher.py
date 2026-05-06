import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from magneto.embedding_matcher import DEFAULT_MODELS
from magneto.starmie_contextual_encoder import StarmieContextualEncoder
from magneto.utils.embedding_utils import compute_cosine_similarity_simple


class StarmieEmbeddingMatcher:
    #
    # 1) строим контекстное текстовое представление столбца,
    # 2) прогоняет его через transformer,
    # 3) берёт hidden state marker token <<COL>> как embedding столбца,
    # 4) считает cosine similarity между source- и target-столбцами.
    def __init__(self, params):
        # параметры retrieval-этапа
        self.params = params
        self.topk = params["topk"]
        self.embedding_threshold = params["embedding_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = params["embedding_model"]
        self.max_length = params.get("span_max_length", 512)

        # Специальный маркер, который ставится перед каждой колонкой
        self.col_marker = params.get("col_marker", "<<COL>>")

        self._load_model_and_tokenizer()
        self._ensure_marker_token()

    def _resolve_local_model_path(self):
        #прямой путь к локальному checkpoint
        candidate = Path(self.model_name)
        if candidate.exists() and candidate.is_file():
            return str(candidate)

        #ну выбираем
        package_root_candidate = Path(__file__).resolve().parents[1] / self.model_name
        if package_root_candidate.exists() and package_root_candidate.is_file():
            return str(package_root_candidate)

        return None

    def _load_sentence_transformer(self, model_path):
        # В офлайн-сценарии сначала пытаемся загрузить модель только из локального кэша
        try:
            return SentenceTransformer(model_path, device=self.device, local_files_only=True)
        except Exception:
            return SentenceTransformer(model_path, device=self.device)

    def _load_tokenizer(self, model_path):
        # Аналогично загружаем tokenizer: сначала локально, затем обычным способом
        try:
            return AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def _load_model_and_tokenizer(self):
        # Вариант 1: model_name — это чтото из DEFAULT_MODELS, например "mpnet"
        if self.model_name in DEFAULT_MODELS:
            model_path = DEFAULT_MODELS[self.model_name]
            self.model = self._load_sentence_transformer(model_path)
            self.tokenizer = self._load_tokenizer(model_path)
            print(f"Loaded default model '{self.model_name}' on {self.device}")
        # Вариант 2: model_name — это Hugging Face model id
        elif "/" in self.model_name and not self.model_name.endswith((".pth", ".pt", ".bin", ".ckpt")):
            try:
                print(f"Attempting to load HuggingFace model '{self.model_name}'...")
                self.model = self._load_sentence_transformer(self.model_name)
                self.tokenizer = self._load_tokenizer(self.model_name)
                print(f"Successfully loaded HuggingFace model '{self.model_name}' on {self.device}")
            except Exception as e:
                print(f"Failed to load HuggingFace model '{self.model_name}': {e}")
                print("Falling back to default 'mpnet' model")
                model_path = DEFAULT_MODELS["mpnet"]
                self.model = self._load_sentence_transformer(model_path)
                self.tokenizer = self._load_tokenizer(model_path)
        # Вариант 3: model_name — это локальный .pth checkpoint с fine-tuned весами
        else:
            resolved_local_model_path = self._resolve_local_model_path()
            if not resolved_local_model_path:
                raise ValueError(f"Invalid model name: {self.model_name}")

            # Пытаемся определить, какой базовый backbone использовался при обучении
            base_key = next((key for key in DEFAULT_MODELS if key in self.model_name), "mpnet")
            if base_key not in DEFAULT_MODELS:
                base_key = "mpnet"

            # Сначала загружаем базовую модель и tokenizer, затем накладываем fine-tuned веса
            base_model_path = DEFAULT_MODELS[base_key]
            self.model = self._load_sentence_transformer(base_model_path)
            self.tokenizer = self._load_tokenizer(base_model_path)
            print(f"Loaded base model '{base_key}' on {self.device}")
            print(f"Loading fine-tuned weights from {resolved_local_model_path}")

            state_dict = torch.load(resolved_local_model_path, map_location=self.device, weights_only=True)
            # Если checkpoint обучался с дополнительным <<COL>>, согласуем размер словаря
            self._align_model_with_checkpoint(state_dict)
            self.model.load_state_dict(state_dict)
            self.model.eval().to(self.device)

        self.model.eval()

    def _align_model_with_checkpoint(self, state_dict):
        # Ищем матрицу токеновых эмбеддингов внутри checkpoint-а
        embedding_key = next(
            (
                key
                for key in state_dict.keys()
                if key.endswith("embeddings.word_embeddings.weight")
            ),
            None,
        )
        if embedding_key is None:
            return

        saved_vocab_size = state_dict[embedding_key].shape[0]
        current_vocab_size = self._get_transformer_module().get_input_embeddings().weight.shape[0]

        # Если размеры уже совпадают, дополнительных действий не нужно
        if saved_vocab_size == current_vocab_size:
            return

        # если checkpoint был обучен с одним добавленным special token <<COL>>
        if saved_vocab_size == current_vocab_size + 1 and self.col_marker not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.col_marker]})
            self._get_transformer_module().resize_token_embeddings(len(self.tokenizer))
            return

        raise RuntimeError(
            f"Checkpoint vocab size ({saved_vocab_size}) does not match current model vocab size "
            f"({current_vocab_size}), and it could not be reconciled automatically."
        )

    def _ensure_marker_token(self):
        # Гарантируем, что tokenizer знает специальный marker token <<COL>>
        vocab = self.tokenizer.get_vocab()
        if self.col_marker not in vocab:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.col_marker]})
            transformer = self._get_transformer_module()
            transformer.resize_token_embeddings(len(self.tokenizer))

        # Сохраняем id этого маркера, чтобы потом искать его позиции в input_ids
        self.col_marker_id = self.tokenizer.convert_tokens_to_ids(self.col_marker)
        if self.col_marker_id is None:
            raise ValueError(f"Could not get token id for marker {self.col_marker}")

    def _build_encoder(self):
        # Создаём encoder, который превращает target-столбец в Starmie-like сериализацию
        return StarmieContextualEncoder(
            tokenizer=self.tokenizer,
            encoding_mode=self.params["encoding_mode"],
            sampling_mode=self.params["sampling_mode"],
            n_samples=self.params["sampling_size"],
            max_context_columns=self.params.get("max_context_columns", 7),
            col_marker=self.col_marker,
            include_header=self.params.get("include_header", True),
            include_values=self.params.get("include_values", True),
            include_target_type=self.params.get("include_target_type", True),
        )

    def _get_transformer_module(self):
        # SentenceTransformer — это обёртка; здесь мы достаём внутренний HF transformer
        for module in self.model._modules.values():
            if hasattr(module, "auto_model"):
                return module.auto_model
        raise ValueError("Could not access underlying transformer model.")

    def _extract_marker_embedding(self, encoded_item):
        # encoded_item приходит из StarmieContextualEncoder и содержит:
        # text — сериализацию окна колонок,
        # target_marker_ordinal — порядковый номер target-столбца внутри окна
        text = encoded_item["text"]
        target_marker_ordinal = encoded_item["target_marker_ordinal"]

        # Токенизируем сериализованный текст окна
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # Находим позиции всех маркеров <<COL>> в токенизированной последовательности
        marker_positions = (input_ids[0] == self.col_marker_id).nonzero(as_tuple=False).flatten().tolist()

        if target_marker_ordinal >= len(marker_positions):
            raise ValueError(
                f"Marker ordinal {target_marker_ordinal} out of range. "
                f"Found only {len(marker_positions)} marker tokens."
            )

        # Выбираем именно тот marker token, который соответствует target-столбцу
        target_marker_position = marker_positions[target_marker_ordinal]

        transformer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        transformer = self._get_transformer_module()

        with torch.no_grad():
            outputs = transformer(**transformer_inputs)
            last_hidden_state = outputs.last_hidden_state[0]

        # Ключевой шаг: вектор столбца = hidden state target marker token
        marker_embedding = last_hidden_state[target_marker_position]
        # Нормализуем embedding перед cosine similarity
        marker_embedding = torch.nn.functional.normalize(marker_embedding, p=2, dim=0)

        return marker_embedding.cpu()

    def _encode_dataframe_columns(self, df, encoder):
        # Здесь строятся эмбеддинги для всех столбцов одной таблицы
        encoded_pairs = []
        for col in df.columns:
            # 1) получаем сериализацию target-столбца с контекстом
            encoded_item = encoder.encode(df, col)
            # 2) превращаем её в marker-based embedding
            embedding = self._extract_marker_embedding(encoded_item)
            encoded_pairs.append((col, embedding, encoded_item))
        return encoded_pairs

    def get_embedding_similarity_candidates(self, source_df, target_df):
        # Создаём contextual encoder и кодируем все source/target-колонки
        encoder = self._build_encoder()

        source_encoded = self._encode_dataframe_columns(source_df, encoder)
        target_encoded = self._encode_dataframe_columns(target_df, encoder)

        if not source_encoded or not target_encoded:
            return {}

        source_cols = [col for col, _, _ in source_encoded]
        target_cols = [col for col, _, _ in target_encoded]

        source_embeddings = torch.stack([emb for _, emb, _ in source_encoded])
        target_embeddings = torch.stack([emb for _, emb, _ in target_encoded])

        # Считаем cosine similarity и берём top-k target-кандидатов для каждого source-столбца
        top_k = min(self.topk, len(target_cols))
        similarities, indices = compute_cosine_similarity_simple(
            source_embeddings, target_embeddings, top_k
        )

        candidates = {}
        for i, source_col in enumerate(source_cols):
            for j in range(top_k):
                target_idx = indices[i, j]
                similarity = similarities[i, j].item()

                # Возвращаем только те пары, которые проходят по порогу similarity
                if similarity >= self.embedding_threshold:
                    target_col = target_cols[target_idx]
                    candidates[(source_col, target_col)] = similarity

        return candidates
