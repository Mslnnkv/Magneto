import random
from pathlib import Path

import pandas as pd


random.seed(42)

OUTPUT_DIR = Path("synthetic_context_needed_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)

COUNTRIES = ["DE", "FR", "IT", "ES", "NL"]
CITIES = ["Berlin", "Paris", "Rome", "Madrid", "Amsterdam"]

CURRENCIES = ["EUR", "USD", "GBP", "CHF"]
AMOUNTS = [10.0, 25.5, 100.0, 250.75, 999.99]

DEPARTMENTS = ["HR", "ENG", "FIN", "MKT", "OPS"]
MANAGERS = ["Miller", "Smith", "Taylor", "Brown", "Wilson"]

CATEGORIES = ["Electronics", "Home", "Office", "Gaming", "Garden"]
BRANDS = ["Alpha", "Delta", "Nova", "Prime", "Echo"]

GENERIC_CODES = ["A1", "B2", "C3", "D4", "E5"]
GENERIC_STATUS = ["active", "inactive", "pending", "closed"]
GENERIC_IDS = [f"X{i}" for i in range(1000, 2000)]


def noisy(val, p=0.05):
    val = str(val)
    if random.random() > p or len(val) < 4:
        return val
    i = random.randint(0, len(val) - 1)
    c = random.choice("abcdefghijklmnopqrstuvwxyz")
    return val[:i] + c + val[i + 1:]


def make_geo_block(n):
    return pd.DataFrame({
        "id": [random.choice(GENERIC_IDS) for _ in range(n)],
        "code": [random.choice(GENERIC_CODES) for _ in range(n)],
        "status": [random.choice(GENERIC_STATUS) for _ in range(n)],
        "country": [random.choice(COUNTRIES) for _ in range(n)],
        "city": [random.choice(CITIES) for _ in range(n)],
    })


def make_finance_block(n):
    return pd.DataFrame({
        "id": [random.choice(GENERIC_IDS) for _ in range(n)],
        "code": [random.choice(GENERIC_CODES) for _ in range(n)],
        "status": [random.choice(GENERIC_STATUS) for _ in range(n)],
        "currency": [random.choice(CURRENCIES) for _ in range(n)],
        "amount": [random.choice(AMOUNTS) for _ in range(n)],
    })


def make_org_block(n):
    return pd.DataFrame({
        "id": [random.choice(GENERIC_IDS) for _ in range(n)],
        "code": [random.choice(GENERIC_CODES) for _ in range(n)],
        "status": [random.choice(GENERIC_STATUS) for _ in range(n)],
        "department": [random.choice(DEPARTMENTS) for _ in range(n)],
        "manager": [random.choice(MANAGERS) for _ in range(n)],
    })


def make_product_block(n):
    return pd.DataFrame({
        "id": [random.choice(GENERIC_IDS) for _ in range(n)],
        "code": [random.choice(GENERIC_CODES) for _ in range(n)],
        "status": [random.choice(GENERIC_STATUS) for _ in range(n)],
        "category": [random.choice(CATEGORIES) for _ in range(n)],
        "brand": [random.choice(BRANDS) for _ in range(n)],
    })


def apply_noise(df, p=0.05):
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(str).map(lambda x: noisy(x, p=p))
    return out


def build_tables(n_rows=250):
    source_blocks = [
        ("geo", make_geo_block(n_rows)),
        ("finance", make_finance_block(n_rows)),
        ("org", make_org_block(n_rows)),
        ("product", make_product_block(n_rows)),
    ]

    target_blocks = [
        ("product", make_product_block(n_rows)),
        ("geo", make_geo_block(n_rows)),
        ("finance", make_finance_block(n_rows)),
        ("org", make_org_block(n_rows)),
    ]

    source_dfs = []
    target_dfs = []
    gt_rows = []

    # source names made deliberately uninformative
    for i, (block_name, df) in enumerate(source_blocks, start=1):
        df = apply_noise(df, p=0.04)
        rename_map = {
            col: f"c{i}_{j}"
            for j, col in enumerate(df.columns, start=1)
        }
        inverse_roles = {f"c{i}_{j}": col for j, col in enumerate(df.columns, start=1)}
        df = df.rename(columns=rename_map)
        source_dfs.append((block_name, df, inverse_roles))

    target_role_map = {}
    for i, (block_name, df) in enumerate(target_blocks, start=1):
        df = apply_noise(df, p=0.04)
        rename_map = {
            col: f"t{i}_{j}"
            for j, col in enumerate(df.columns, start=1)
        }
        inverse_roles = {f"t{i}_{j}": col for j, col in enumerate(df.columns, start=1)}
        df = df.rename(columns=rename_map)
        target_dfs.append((block_name, df, inverse_roles))
        target_role_map[block_name] = inverse_roles

    source_df = pd.concat([x[1] for x in source_dfs], axis=1)
    target_df = pd.concat([x[1] for x in target_dfs], axis=1)

    # Ground truth by semantic block + role
    for block_name, df, inverse_roles in source_dfs:
        tgt_roles = target_role_map[block_name]
        src_by_role = {role: col for col, role in inverse_roles.items()}
        tgt_by_role = {role: col for col, role in tgt_roles.items()}

        for role in ["id", "code", "status", "country", "city", "currency", "amount",
                     "department", "manager", "category", "brand"]:
            if role in src_by_role and role in tgt_by_role:
                gt_rows.append({
                    "block": block_name,
                    "role": role,
                    "source_column": src_by_role[role],
                    "target_column": tgt_by_role[role],
                })

    gt_df = pd.DataFrame(gt_rows)
    return source_df, target_df, gt_df


def main():
    source_df, target_df, gt_df = build_tables(n_rows=250)

    source_df.to_csv(OUTPUT_DIR / "context_needed_source.csv", index=False)
    target_df.to_csv(OUTPUT_DIR / "context_needed_target.csv", index=False)
    gt_df.to_csv(OUTPUT_DIR / "context_needed_ground_truth.csv", index=False)

    print("Saved to:", OUTPUT_DIR.resolve())
    print("Source shape:", source_df.shape)
    print("Target shape:", target_df.shape)
    print("Ground truth size:", len(gt_df))
    print("\nSource columns:", list(source_df.columns))
    print("\nTarget columns:", list(target_df.columns))


if __name__ == "__main__":
    main()