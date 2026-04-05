import random
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


random.seed(42)

OUTPUT_DIR = ROOT / "synthetic_benchmark_version_6"
OUTPUT_DIR.mkdir(exist_ok=True)

PERSON_NAMES = ["Alex", "Sam", "Chris", "Jordan", "Taylor", "Morgan", "Casey", "Jamie"]
SHARED_CODES = ["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8"]
SHARED_STATUS = ["active", "inactive", "pending", "closed", "open"]
SHARED_DATES = [f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(900)]
SHARED_IDS = [f"X{100000 + i}" for i in range(1500)]

ENTITY_SPECS = {
    "medical_registry": {
        "anchor_left": ["cardiology", "oncology", "neurology", "immunology", "pediatrics"],
        "anchor_mid": ["ward-a", "ward-b", "ward-c", "lab-west", "lab-east"],
    },
    "cargo_registry": {
        "anchor_left": ["linehaul", "fulfillment", "returns", "customs", "crossdock"],
        "anchor_mid": ["hub-north", "hub-south", "hub-east", "hub-west", "port-central"],
    },
    "education_registry": {
        "anchor_left": ["foundation", "seminar", "research", "applied", "capstone"],
        "anchor_mid": ["faculty-math", "faculty-law", "faculty-bio", "faculty-cs", "faculty-arts"],
    },
    "hospitality_registry": {
        "anchor_left": ["breakfast", "half-board", "full-board", "late-checkin", "business-stay"],
        "anchor_mid": ["airport", "city-center", "old-town", "waterfront", "expo-district"],
    },
    "telecom_registry": {
        "anchor_left": ["prepaid", "family-plan", "business-plan", "roaming", "data-plus"],
        "anchor_mid": ["4g-core", "5g-core", "fiber-mix", "iot-edge", "metro-backbone"],
    },
    "energy_registry": {
        "anchor_left": ["residential", "commercial", "industrial", "peak-load", "green-tariff"],
        "anchor_mid": ["sector-north", "sector-south", "sector-east", "sector-west", "grid-central"],
    },
}

SOURCE_ORDER = [
    "medical_registry",
    "cargo_registry",
    "education_registry",
    "hospitality_registry",
    "telecom_registry",
    "energy_registry",
]

TARGET_ORDER = [
    "telecom_registry",
    "medical_registry",
    "energy_registry",
    "education_registry",
    "hospitality_registry",
    "cargo_registry",
]


def noisy_text(text, p=0.035):
    text = str(text)
    if random.random() > p or len(text) < 5:
        return text
    idx = random.randint(0, len(text) - 1)
    repl = random.choice("abcdefghijklmnopqrstuvwxyz")
    return text[:idx] + repl + text[idx + 1:]


def apply_noise(df, p=0.035):
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].astype(str).map(lambda value: noisy_text(value, p=p))
    return out


def build_generic_columns(n_rows, date_offset):
    return {
        "record_id": [random.choice(SHARED_IDS) for _ in range(n_rows)],
        "display_name": [f"{random.choice(PERSON_NAMES)} {i % 83}" for i in range(n_rows)],
        "reference_code": [random.choice(SHARED_CODES) for _ in range(n_rows)],
        "event_status": [random.choice(SHARED_STATUS) for _ in range(n_rows)],
        "event_date": [SHARED_DATES[(date_offset + i) % len(SHARED_DATES)] for i in range(n_rows)],
    }


def build_entity_block(entity_name, n_rows, date_offset):
    spec = ENTITY_SPECS[entity_name]
    generic = build_generic_columns(n_rows=n_rows, date_offset=date_offset)

    return pd.DataFrame(
        {
            "anchor_left": [random.choice(spec["anchor_left"]) for _ in range(n_rows)],
            "record_id": generic["record_id"],
            "display_name": generic["display_name"],
            "reference_code": generic["reference_code"],
            "anchor_mid": [random.choice(spec["anchor_mid"]) for _ in range(n_rows)],
            "event_status": generic["event_status"],
            "event_date": generic["event_date"],
        }
    )


def rename_with_opaque_headers(df, prefix):
    rename_map = {column: f"{prefix}_{idx}" for idx, column in enumerate(df.columns, start=1)}
    inverse_roles = {opaque: original for original, opaque in rename_map.items()}
    return df.rename(columns=rename_map), inverse_roles


def build_source_target(n_rows=300):
    source_blocks = []
    target_blocks = []
    gt_rows = []
    target_role_map = {}

    source_offsets = {entity: idx * 13 for idx, entity in enumerate(SOURCE_ORDER, start=1)}
    target_offsets = {entity: idx * 19 for idx, entity in enumerate(TARGET_ORDER, start=1)}

    for idx, entity in enumerate(SOURCE_ORDER, start=1):
        block = build_entity_block(entity_name=entity, n_rows=n_rows, date_offset=source_offsets[entity])
        block = apply_noise(block, p=0.035)
        block, inverse_roles = rename_with_opaque_headers(block, prefix=f"s{idx}")
        source_blocks.append((entity, block, inverse_roles))

    for idx, entity in enumerate(TARGET_ORDER, start=1):
        block = build_entity_block(entity_name=entity, n_rows=n_rows, date_offset=target_offsets[entity])
        block = apply_noise(block, p=0.035)
        block, inverse_roles = rename_with_opaque_headers(block, prefix=f"t{idx}")
        target_blocks.append((entity, block, inverse_roles))
        target_role_map[entity] = inverse_roles

    source_df = pd.concat([block for _, block, _ in source_blocks], axis=1)
    target_df = pd.concat([block for _, block, _ in target_blocks], axis=1)

    roles_for_evaluation = ["record_id", "display_name", "reference_code", "event_status", "event_date"]

    for entity, _, inverse_roles in source_blocks:
        target_inverse_roles = target_role_map[entity]
        src_by_role = {role: col for col, role in inverse_roles.items()}
        tgt_by_role = {role: col for col, role in target_inverse_roles.items()}

        for role in roles_for_evaluation:
            gt_rows.append(
                {
                    "entity": entity,
                    "role": role,
                    "source_column": src_by_role[role],
                    "target_column": tgt_by_role[role],
                }
            )

    return source_df, target_df, pd.DataFrame(gt_rows)


def main():
    source_df, target_df, gt_df = build_source_target(n_rows=300)

    source_df.to_csv(OUTPUT_DIR / "version_6_source.csv", index=False)
    target_df.to_csv(OUTPUT_DIR / "version_6_target.csv", index=False)
    gt_df.to_csv(OUTPUT_DIR / "version_6_ground_truth.csv", index=False)

if __name__ == "__main__":
    main()
