from magneto.utils.utils import detect_column_type, get_samples


class ContextualColumnEncoder:
    def __init__(
        self,
        tokenizer,
        encoding_mode="table_context_target_values",
        sampling_mode="mixed",
        n_samples=10,
        max_context_columns=None,
        include_target_type=True,
        include_target_values=True,
        target_marker_start="<TARGET>",
        target_marker_end="</TARGET>",
    ):
        self._tokenizer = tokenizer
        self.cls_token = getattr(tokenizer, "cls_token", "") or ""
        self.sep_token = getattr(tokenizer, "sep_token", "") or ""
        self.eos_token = getattr(tokenizer, "eos_token", "") or ""

        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.n_samples = n_samples
        self.max_context_columns = max_context_columns
        self.include_target_type = include_target_type
        self.include_target_values = include_target_values
        self.target_marker_start = target_marker_start
        self.target_marker_end = target_marker_end

        self._serialization_methods = {
            "table_context_target_values": self._serialize_table_context_target_values,
            "table_context_window_target_values": self._serialize_table_context_window_target_values,
        }

        if encoding_mode not in self._serialization_methods:
            raise ValueError(
                f"Unsupported contextual encoding mode: {encoding_mode}. "
                f"Supported modes are: {list(self._serialization_methods.keys())}"
            )

    def encode(self, df, col):
        return self._serialization_methods[self.encoding_mode](df, col)

    def _serialize_table_context_target_values(self, df, target_col):
        schema_block = self._build_full_table_schema_context(df, target_col)
        target_block = self._build_target_column_block(df, target_col)

        parts = [
            self.cls_token,
            "Contextual Column Representation",
            schema_block,
            target_block,
            self.eos_token,
        ]
        return self._join_non_empty(parts)

    def _serialize_table_context_window_target_values(self, df, target_col):
        window_block = self._build_window_context(df, target_col)
        target_block = self._build_target_column_block(df, target_col, repeat_header=True)

        parts = [
            self.cls_token,
            "Windowed Contextual Column Representation",
            window_block,
            target_block,
            self.eos_token,
        ]
        return self._join_non_empty(parts)

    def _build_full_table_schema_context(self, df, target_col):
        columns = list(df.columns)
        formatted_columns = self._format_schema_columns(columns, target_col)
        return f"Table Columns: {formatted_columns}"

    def _build_window_context(self, df, target_col):
        columns = list(df.columns)

        if target_col not in columns:
            return f"Target Column Position Context: {target_col}"

        window_size = self.max_context_columns if self.max_context_columns is not None else 7
        window_columns = self._limit_columns_around_target(columns, target_col, window_size)

        target_idx_in_window = window_columns.index(target_col)
        left_columns = window_columns[:target_idx_in_window]
        right_columns = window_columns[target_idx_in_window + 1:]

        left_text = " | ".join(str(col) for col in left_columns) if left_columns else "NONE"
        right_text = " | ".join(str(col) for col in right_columns) if right_columns else "NONE"

        return (
            f"Left Context Columns: {left_text}. "
            f"Target Marker: {self.target_marker_start} {target_col} {self.target_marker_end}. "
            f"Right Context Columns: {right_text}."
        )

    def _build_target_column_block(self, df, target_col, repeat_header=False):
        parts = []

        parts.append(f"Target Column Name: {target_col}")

        if repeat_header:
            parts.append(f"Target Column Focus: {target_col}")

        if self.include_target_type:
            data_type = detect_column_type(df[target_col])
            parts.append(f"Target Column Type: {data_type}")

        if self.include_target_values:
            tokens = get_samples(df[target_col], n=self.n_samples, mode=self.sampling_mode)
            tokens = self._clean_token_list(tokens)
            sample_text = " | ".join(tokens) if tokens else "NONE"
            parts.append(f"Target Sample Values: {sample_text}")

        return self._join_non_empty(parts)

    def _format_schema_columns(self, columns, target_col):
        formatted = []
        for col in columns:
            if col == target_col:
                formatted.append(
                    f"{self.target_marker_start} {col} {self.target_marker_end}"
                )
            else:
                formatted.append(str(col))
        return " | ".join(formatted)

    def _limit_columns_around_target(self, columns, target_col, max_columns):
        if target_col not in columns or max_columns >= len(columns):
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

    def _join_non_empty(self, parts):
        parts = [str(part).strip() for part in parts if part is not None and str(part).strip()]

        if not parts:
            return ""

        text = " [SEP] ".join(parts)

        if self.cls_token:
            text = text.replace(f"{self.cls_token} [SEP] ", f"{self.cls_token} ")

        if self.eos_token:
            text = text.replace(f" [SEP] {self.eos_token}", f" {self.eos_token}")

        return text