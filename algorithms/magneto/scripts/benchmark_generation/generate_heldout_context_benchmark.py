import random
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


random.seed(42)

OUTPUT_DIR = ROOT / "synthetic_heldout_context_benchmark"
OUTPUT_DIR.mkdir(exist_ok=True)

PERSON_NAMES = ["Alex", "Sam", "Chris", "Jordan", "Taylor", "Morgan", "Casey", "Jamie"]
SHARED_CODES = ["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8"]
SHARED_STATUS = ["active", "inactive", "pending", "closed", "open"]
SHARED_DATES = [f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(800)]
SHARED_IDS = [f"R{100000 + i}" for i in range(1200)]

ENTITY_SPECS = {
    "clinical_trials": {
        "anchor_left": ["respiratory", "cardiology", "neurology", "immunology", "oncology"],
        "anchor_right": ["ward-a", "ward-b", "ward-c", "lab-west", "lab-east"],
    },
    "cargo_shipments": {
        "anchor_left": ["crossdock", "linehaul", "customs", "fulfillment", "returns"],
        "anchor_right": ["hub-north", "hub-south", "hub-east", "hub-west", "port-central"],
    },
    "course_enrollment": {
        "anchor_left": ["foundation", "applied", "research", "seminar", "capstone"],
        "anchor_right": ["faculty-math", "faculty-law", "faculty-bio", "faculty-cs", "faculty-arts"],
    },
    "hotel_bookings": {
        "anchor_left": ["breakfast", "half-board", "full-board", "late-checkin", "business-stay"],
        "anchor_right": ["city-center", "airport", "old-town", "waterfront", "expo-district"],
    },
    "mobile_subscriptions": {
        "anchor_left": ["prepaid", "family-plan", "business-plan", "roaming", "data-plus"],
        "anchor_right": ["4g-core", "5g-core", "fiber-mix", "iot-edge", "metro-backbone"],
    },
    "utility_meters": {
        "anchor_left": ["residential", "commercial", "industrial", "peak-load", "green-tariff"],
        "anchor_right": ["sector-north", "sector-south", "sector-east", "sector-west", "grid-central"],
    },
}

SOURCE_ENTITY_ORDER = [
    "clinical_trials",
    "cargo_shipments",
    "course_enrollment",
    "hotel_bookings",
    "mobile_subscriptions",
    "utility_meters",
]

TARGET_ENTITY_ORDER = [
    "mobile_subscriptions",
    "clinical_trials",
    "utility_meters",
    "course_enrollment",
    "hotel_bookings",
    "cargo_shipments",
]


def noisy_text(text, p=0.04):
    text = str(text)
    if random.random() > p or len(text) < 5:
        return text
    idx = random.randint(0, len(text) - 1)
    repl = random.choice("abcdefghijklmnopqrstuvwxyz")
    return text[:idx] + repl + text[idx + 1:]


def apply_noise(df, p=0.04):
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].astype(str).map(lambda value: noisy_text(value, p=p))
    return out


def build_generic_columns(n_rows, date_offset):
    return {
        "record_id": [random.choice(SHARED_IDS) for _ in range(n_rows)],
        "name_like": [f"{random.choice(PERSON_NAMES)} {i % 97}" for i in range(n_rows)],
        "code_like": [random.choice(SHARED_CODES) for _ in range(n_rows)],
        "event_date": [SHARED_DATES[(date_offset + i) % len(SHARED_DATES)] for i in range(n_rows)],
        "state_like": [random.choice(SHARED_STATUS) for _ in range(n_rows)],
    }


def build_entity_block(entity_name, n_rows, date_offset):
    spec = ENTITY_SPECS[entity_name]
    generic = build_generic_columns(n_rows=n_rows, date_offset=date_offset)

    return pd.DataFrame(
        {
            "anchor_left": [random.choice(spec["anchor_left"]) for _ in range(n_rows)],
            "record_id": generic["record_id"],
            "name_like": generic["name_like"],
            "code_like": generic["code_like"],
            "anchor_right": [random.choice(spec["anchor_right"]) for _ in range(n_rows)],
            "event_date": generic["event_date"],
            "state_like": generic["state_like"],
        }
    )


def rename_with_opaque_headers(df, prefix):
    rename_map = {column: f"{prefix}_{idx}" for idx, column in enumerate(df.columns, start=1)}
    inverse_roles = {opaque: original for original, opaque in rename_map.items()}
    return df.rename(columns=rename_map), inverse_roles


def build_source_target(n_rows=280):
    source_blocks = []
    target_blocks = []
    ground_truth_rows = []
    target_roles_by_entity = {}

    source_offsets = {
        entity: idx * 17 for idx, entity in enumerate(SOURCE_ENTITY_ORDER, start=1)
    }
    target_offsets = {
        entity: idx * 29 for idx, entity in enumerate(TARGET_ENTITY_ORDER, start=1)
    }

    for idx, entity in enumerate(SOURCE_ENTITY_ORDER, start=1):
        block = build_entity_block(entity_name=entity, n_rows=n_rows, date_offset=source_offsets[entity])
        block = apply_noise(block, p=0.04)
        block, inverse_roles = rename_with_opaque_headers(block, prefix=f"s{idx}")
        source_blocks.append((entity, block, inverse_roles))

    for idx, entity in enumerate(TARGET_ENTITY_ORDER, start=1):
        block = build_entity_block(entity_name=entity, n_rows=n_rows, date_offset=target_offsets[entity])
        block = apply_noise(block, p=0.04)
        block, inverse_roles = rename_with_opaque_headers(block, prefix=f"t{idx}")
        target_blocks.append((entity, block, inverse_roles))
        target_roles_by_entity[entity] = inverse_roles

    source_df = pd.concat([block for _, block, _ in source_blocks], axis=1)
    target_df = pd.concat([block for _, block, _ in target_blocks], axis=1)

    generic_roles = ["record_id", "name_like", "code_like", "event_date", "state_like"]

    for entity, _, source_inverse_roles in source_blocks:
        target_inverse_roles = target_roles_by_entity[entity]
        source_by_role = {role: column for column, role in source_inverse_roles.items()}
        target_by_role = {role: column for column, role in target_inverse_roles.items()}

        for role in generic_roles:
            ground_truth_rows.append(
                {
                    "entity": entity,
                    "role": role,
                    "source_column": source_by_role[role],
                    "target_column": target_by_role[role],
                }
            )

    ground_truth_df = pd.DataFrame(ground_truth_rows)
    return source_df, target_df, ground_truth_df


def main():
    source_df, target_df, ground_truth_df = build_source_target(n_rows=280)

    source_df.to_csv(OUTPUT_DIR / "heldout_source.csv", index=False)
    target_df.to_csv(OUTPUT_DIR / "heldout_target.csv", index=False)
    ground_truth_df.to_csv(OUTPUT_DIR / "heldout_ground_truth.csv", index=False)

    print("Saved held-out benchmark to:", OUTPUT_DIR.resolve())
    print("Source shape:", source_df.shape)
    print("Target shape:", target_df.shape)
    print("Ground truth size:", len(ground_truth_df))
    print()
    print("First source columns:", list(source_df.columns[:14]))
    print("First target columns:", list(target_df.columns[:14]))


if __name__ == "__main__":
    main()
