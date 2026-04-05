from magneto.utils.utils import detect_column_type, get_samples


class StarmieContextualEncoder:
    def __init__(
        self,
        tokenizer,
        encoding_mode="table_context_window_starmie_marker",
        sampling_mode="mixed",
        n_samples=10,
        max_context_columns=7,
        col_marker="<<COL>>",
        include_header=True,
        include_values=True,
        include_target_type=True,
    ):
        self._tokenizer = tokenizer
        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.n_samples = n_samples
        self.max_context_columns = max_context_columns
        self.col_marker = col_marker
        self.include_header = include_header
        self.include_values = include_values
        self.include_target_type = include_target_type

        supported_modes = {
            "table_context_window_starmie_marker",
            "table_context_window_starmie_structured",
        }
        if self.encoding_mode not in supported_modes:
            raise ValueError(
                f"Unsupported starmie encoding mode: {self.encoding_mode}. "
                f"Supported modes are: {list(supported_modes)}"
            )

    def encode(self, df, target_col):
        columns = list(df.columns)
        window_columns = self._limit_columns_around_target(
            columns, target_col, self.max_context_columns
        )

        if target_col not in window_columns:
            raise ValueError(f"Target column '{target_col}' not found in window columns.")

        target_marker_ordinal = window_columns.index(target_col)
        if self.encoding_mode == "table_context_window_starmie_marker":
            text = self._build_legacy_window_serialization(df, window_columns)
        else:
            text = self._build_structured_window_serialization(df, window_columns)

        return {
            "text": text,
            "target_col": target_col,
            "target_marker_ordinal": target_marker_ordinal,
            "window_columns": window_columns,
        }

    def _build_legacy_window_serialization(self, df, window_columns):
        parts = ["Window Table Representation."]

        for col in window_columns:
            parts.append(self._build_legacy_column_block(df, col))

        return "\n".join(parts)

    def _build_structured_window_serialization(self, df, window_columns):
        parts = [
            "Column Ordered Table Serialization.",
            "Each marker denotes one contextualized column representation.",
        ]

        for col in window_columns:
            parts.append(self._build_structured_column_block(df, col))

        return "\n".join(parts)

    def _build_legacy_column_block(self, df, col):
        block_parts = [self.col_marker]

        if self.include_header:
            block_parts.append(str(col))

        if self.include_values:
            tokens = get_samples(df[col], n=self.n_samples, mode=self.sampling_mode)
            tokens = self._clean_token_list(tokens)
            if tokens:
                block_parts.extend(tokens)

        return " | ".join(block_parts)

    def _build_structured_column_block(self, df, col):
        block_parts = [self.col_marker]

        if self.include_header:
            block_parts.append(f"header = {col}")

        if self.include_target_type:
            block_parts.append(f"type = {detect_column_type(df[col])}")

        if self.include_values:
            tokens = get_samples(df[col], n=self.n_samples, mode=self.sampling_mode)
            tokens = self._clean_token_list(tokens)
            values_text = " ; ".join(tokens) if tokens else "NONE"
            block_parts.append(f"values = {values_text}")

        return " ; ".join(block_parts)

    def _limit_columns_around_target(self, columns, target_col, max_columns):
        if target_col not in columns or max_columns is None or max_columns >= len(columns):
            return columns

        if max_columns <= 1:
            return [target_col]

        target_idx = columns.index(target_col)
        left_slots = (max_columns - 1) // 2
        right_slots = max_columns - 1 - left_slots

        start = max(0, target_idx - left_slots)
        end = min(len(columns), target_idx + right_slots + 1)

        current_len = end - start
        if current_len < max_columns:
            missing = max_columns - current_len

            new_start = max(0, start - missing)
            gained_left = start - new_start
            start = new_start
            missing -= gained_left

            end = min(len(columns), end + missing)

        return columns[start:end]

    def _clean_token_list(self, tokens):
        cleaned = []
        for token in tokens:
            if token is None:
                continue
            token = str(token).strip()
            if not token:
                continue
            cleaned.append(token)
        return cleaned
