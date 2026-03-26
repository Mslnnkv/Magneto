from magneto.utils.utils import detect_column_type, get_samples


class SpanContextualEncoderV2:
    def __init__(
        self,
        tokenizer,
        encoding_mode="table_context_window_span_target_block",
        sampling_mode="mixed",
        n_samples=10,
        max_context_columns=7,
        include_target_type=True,
        include_target_values=True,
        target_block_start_marker="<<TARGET_BLOCK_START>>",
        target_block_end_marker="<<TARGET_BLOCK_END>>",
    ):
        self._tokenizer = tokenizer
        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.n_samples = n_samples
        self.max_context_columns = max_context_columns
        self.include_target_type = include_target_type
        self.include_target_values = include_target_values
        self.target_block_start_marker = target_block_start_marker
        self.target_block_end_marker = target_block_end_marker

        supported_modes = {"table_context_window_span_target_block"}
        if self.encoding_mode not in supported_modes:
            raise ValueError(
                f"Unsupported span v2 encoding mode: {self.encoding_mode}. "
                f"Supported modes are: {list(supported_modes)}"
            )

    def encode(self, df, target_col):
        text = self._build_window_serialization(df, target_col)

        start_marker = self.target_block_start_marker
        end_marker = self.target_block_end_marker

        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)

        if start_idx == -1 or end_idx == -1:
            raise ValueError("Target block markers not found in serialized text.")

        target_char_start = start_idx + len(start_marker)
        while target_char_start < len(text) and text[target_char_start] == " ":
            target_char_start += 1

        target_char_end = end_idx
        while target_char_end > 0 and text[target_char_end - 1] == " ":
            target_char_end -= 1

        return {
            "text": text,
            "target_col": target_col,
            "target_char_start": target_char_start,
            "target_char_end": target_char_end,
            "target_block_start_marker": start_marker,
            "target_block_end_marker": end_marker,
        }

    def _build_window_serialization(self, df, target_col):
        columns = list(df.columns)
        window_columns = self._limit_columns_around_target(
            columns, target_col, self.max_context_columns
        )

        target_idx_in_window = window_columns.index(target_col)
        left_columns = window_columns[:target_idx_in_window]
        right_columns = window_columns[target_idx_in_window + 1:]

        left_text = " | ".join(str(col) for col in left_columns) if left_columns else "NONE"
        right_text = " | ".join(str(col) for col in right_columns) if right_columns else "NONE"

        target_block_parts = [
            f"Target Column Header: {target_col}.",
        ]

        if self.include_target_type:
            data_type = detect_column_type(df[target_col])
            target_block_parts.append(f"Target Column Type: {data_type}.")

        if self.include_target_values:
            tokens = get_samples(df[target_col], n=self.n_samples, mode=self.sampling_mode)
            tokens = self._clean_token_list(tokens)
            values_text = " | ".join(tokens) if tokens else "NONE"
            target_block_parts.append(f"Target Sample Values: {values_text}.")

        target_block_text = " ".join(target_block_parts)

        parts = [
            "Window Context.",
            f"Left Columns: {left_text}.",
            self.target_block_start_marker,
            target_block_text,
            self.target_block_end_marker + ".",
            f"Right Columns: {right_text}.",
        ]

        return " ".join(parts)

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