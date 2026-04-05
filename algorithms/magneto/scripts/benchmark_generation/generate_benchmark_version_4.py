from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import pandas as pd


random.seed(42)

OUTPUT_DIR = ROOT / "synthetic_benchmark_version_4"
OUTPUT_DIR.mkdir(exist_ok=True)

PERSON_NAMES = ["Alex", "Sam", "Chris", "Jordan", "Taylor", "Morgan", "Casey", "Jamie"]
OBJECT_NAMES = ["Alpha", "Delta", "Nova", "Prime", "Atlas", "Echo", "Orbit", "Vertex"]

GENERIC_CODES = ["A1", "B2", "C3", "D4", "E5", "F6", "G7"]
GENERIC_STATUS = ["active", "inactive", "pending", "closed", "open"]
GENERIC_DATES_1 = [f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(500)]
GENERIC_DATES_2 = [f"2022-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(500)]


def noisy_text(text, p=0.07):
    text = str(text)
    if random.random() > p or len(text) < 4:
        return text
    i = random.randint(0, len(text) - 1)
    repl = random.choice("abcdefghijklmnopqrstuvwxyz")
    return text[:i] + repl + text[i + 1:]


def make_block(n_rows, entity):
    if entity == "customer":
        return pd.DataFrame({
            "id": [f"X{1000+i}" for i in range(n_rows)],
            "name": [f"{random.choice(PERSON_NAMES)} {i}" for i in range(n_rows)],
            "code": [random.choice(GENERIC_CODES) for _ in range(n_rows)],
            "date": [GENERIC_DATES_1[i] for i in range(n_rows)],
            "status": [random.choice(GENERIC_STATUS) for _ in range(n_rows)],
        })
    if entity == "employee":
        return pd.DataFrame({
            "id": [f"X{2000+i}" for i in range(n_rows)],
            "name": [f"{random.choice(PERSON_NAMES)} {i}" for i in range(n_rows)],
            "code": [random.choice(GENERIC_CODES) for _ in range(n_rows)],
            "date": [GENERIC_DATES_2[i] for i in range(n_rows)],
            "status": [random.choice(GENERIC_STATUS) for _ in range(n_rows)],
        })
    if entity == "product":
        return pd.DataFrame({
            "id": [f"X{3000+i}" for i in range(n_rows)],
            "name": [f"{random.choice(OBJECT_NAMES)} {i}" for i in range(n_rows)],
            "code": [random.choice(GENERIC_CODES) for _ in range(n_rows)],
            "date": [GENERIC_DATES_1[(i * 3) % len(GENERIC_DATES_1)] for i in range(n_rows)],
            "status": [random.choice(GENERIC_STATUS) for _ in range(n_rows)],
        })
    if entity == "order":
        return pd.DataFrame({
            "id": [f"X{4000+i}" for i in range(n_rows)],
            "name": [f"{random.choice(OBJECT_NAMES)} {i}" for i in range(n_rows)],
            "code": [random.choice(GENERIC_CODES) for _ in range(n_rows)],
            "date": [GENERIC_DATES_2[(i * 5) % len(GENERIC_DATES_2)] for i in range(n_rows)],
            "status": [random.choice(GENERIC_STATUS) for _ in range(n_rows)],
        })
    if entity == "supplier":
        return pd.DataFrame({
            "id": [f"X{5000+i}" for i in range(n_rows)],
            "name": [f"{random.choice(OBJECT_NAMES)} {i}" for i in range(n_rows)],
            "code": [random.choice(GENERIC_CODES) for _ in range(n_rows)],
            "date": [GENERIC_DATES_1[(i * 7) % len(GENERIC_DATES_1)] for i in range(n_rows)],
            "status": [random.choice(GENERIC_STATUS) for _ in range(n_rows)],
        })
    raise ValueError(entity)


def apply_value_noise(df, p=0.05):
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(str).map(lambda x: noisy_text(x, p=p))
    return out


def build_source_target(n_rows=250):
    entities_source = ["customer", "product", "employee", "order", "supplier"]
    entities_target = ["employee", "customer", "supplier", "product", "order"]

    source_blocks = []
    target_blocks = []
    gt_rows = []

    for idx, entity in enumerate(entities_source, start=1):
        block = apply_value_noise(make_block(n_rows, entity), p=0.05)
        renamed = {c: f"{c}_{idx}" for c in block.columns}
        block = block.rename(columns=renamed)
        source_blocks.append(block)

    # semantic index map for target blocks
    target_index_by_entity = {}
    for idx, entity in enumerate(entities_target, start=1):
        block = apply_value_noise(make_block(n_rows, entity), p=0.05)
        renamed = {c: f"{c}_{idx}" for c in block.columns}
        block = block.rename(columns=renamed)
        target_blocks.append(block)
        target_index_by_entity[entity] = idx

    source_df = pd.concat(source_blocks, axis=1)
    target_df = pd.concat(target_blocks, axis=1)

    # Build GT by semantic entity + local column role
    for s_idx, entity in enumerate(entities_source, start=1):
        t_idx = target_index_by_entity[entity]
        for role in ["id", "name", "code", "date", "status"]:
            gt_rows.append({
                "entity": entity,
                "role": role,
                "source_column": f"{role}_{s_idx}",
                "target_column": f"{role}_{t_idx}",
            })

    gt_df = pd.DataFrame(gt_rows)
    return source_df, target_df, gt_df


def main():
    source_df, target_df, gt_df = build_source_target(n_rows=250)

    source_df.to_csv(OUTPUT_DIR / "version_4_source.csv", index=False)
    target_df.to_csv(OUTPUT_DIR / "version_4_target.csv", index=False)
    gt_df.to_csv(OUTPUT_DIR / "version_4_ground_truth.csv", index=False)



if __name__ == "__main__":
    main()