LEGACY_ENCODING_MODES = {
    "header_values_default",
    "header_values_prefix",
    "header_values_repeat",
    "header_values_verbose",
    "header_only",
    "header_values_verbose_notype",
    "header_values_columnvaluepair_notype",
    "header_header_values_repeat_notype",
    "header_values_default_notype",
}

SIMPLE_CONTEXTUAL_ENCODING_MODES = {
    "table_context_target_values",
    "table_context_window_target_values",
}

SPAN_CONTEXTUAL_ENCODING_MODES = {
    "table_context_window_span",
    "table_context_window_span_target_block",
    "table_context_window_starmie_marker",
    "table_context_window_starmie_structured",
}

CONTEXTUAL_ENCODING_MODES = (
    SIMPLE_CONTEXTUAL_ENCODING_MODES | SPAN_CONTEXTUAL_ENCODING_MODES
)

ALL_ENCODING_MODES = LEGACY_ENCODING_MODES | CONTEXTUAL_ENCODING_MODES
