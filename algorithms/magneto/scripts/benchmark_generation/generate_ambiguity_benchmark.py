from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import pandas as pd


random.seed(42)

OUTPUT_DIR = ROOT / "synthetic_ambiguity_benchmark"
OUTPUT_DIR.mkdir(exist_ok=True)


FIRST_NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Hugo"]
PRODUCT_NAMES = ["Phone", "Laptop", "Tablet", "Monitor", "Keyboard", "Mouse"]
DEPARTMENTS = ["HR", "Finance", "Engineering", "Marketing", "Sales"]
SUPPLIERS = ["NordTrade", "BlueSupply", "DeltaGoods", "PrimeSource", "UrbanParts"]

COUNTRY_CODES = ["DE", "FR", "IT", "ES", "NL"]
CATEGORY_CODES = ["ELEC", "HOME", "OFFC", "GAME"]
DEPT_CODES = ["HR", "FIN", "ENG", "MKT", "SLS"]
ORDER_STATUS = ["paid", "shipped", "returned", "cancelled"]
PERSON_STATUS = ["active", "inactive", "pending"]
PRODUCT_STATUS = ["available", "archived", "draft"]


def make_dates(n, year):
    return [f"{year}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]


def build_customer_block(n):
    return pd.DataFrame({
        "customer_id": [f"CUST{1000+i}" for i in range(n)],
        "customer_name": [f"{random.choice(FIRST_NAMES)} {i}" for i in range(n)],
        "customer_code": [random.choice(COUNTRY_CODES) for _ in range(n)],
        "customer_date": make_dates(n, 2023),
        "customer_status": [random.choice(PERSON_STATUS) for _ in range(n)],
    })


def build_product_block(n):
    return pd.DataFrame({
        "product_id": [f"PROD{2000+i}" for i in range(n)],
        "product_name": [f"{random.choice(PRODUCT_NAMES)} {i}" for i in range(n)],
        "product_code": [random.choice(CATEGORY_CODES) for _ in range(n)],
        "product_date": make_dates(n, 2022),
        "product_status": [random.choice(PRODUCT_STATUS) for _ in range(n)],
    })


def build_order_block(n):
    return pd.DataFrame({
        "order_id": [f"ORD{3000+i}" for i in range(n)],
        "order_name": [f"Order {i}" for i in range(n)],
        "order_code": [random.choice(COUNTRY_CODES) for _ in range(n)],
        "order_date": make_dates(n, 2024),
        "order_status": [random.choice(ORDER_STATUS) for _ in range(n)],
    })


def build_employee_block(n):
    return pd.DataFrame({
        "employee_id": [f"EMP{4000+i}" for i in range(n)],
        "employee_name": [f"{random.choice(FIRST_NAMES)} {i}" for i in range(n)],
        "employee_code": [random.choice(DEPT_CODES) for _ in range(n)],
        "employee_date": make_dates(n, 2021),
        "employee_status": [random.choice(PERSON_STATUS) for _ in range(n)],
    })


def build_supplier_block(n):
    return pd.DataFrame({
        "supplier_id": [f"SUP{5000+i}" for i in range(n)],
        "supplier_name": [f"{random.choice(SUPPLIERS)} {i}" for i in range(n)],
        "supplier_code": [random.choice(COUNTRY_CODES) for _ in range(n)],
        "supplier_date": make_dates(n, 2020),
        "supplier_status": [random.choice(PERSON_STATUS) for _ in range(n)],
    })


def add_light_noise(series, p=0.08):
    out = []
    for v in series.astype(str):
        if random.random() < p and len(v) > 4:
            i = random.randint(0, len(v) - 1)
            repl = random.choice("abcdefghijklmnopqrstuvwxyz")
            v = v[:i] + repl + v[i + 1:]
        out.append(v)
    return pd.Series(out)


def make_source_table(block_df, prefix):
    """
    Source: partially shortened noisy headers, but still hint at entity.
    """
    mapping = {
        f"{prefix}_id": "id",
        f"{prefix}_name": "name",
        f"{prefix}_code": "code",
        f"{prefix}_date": "date",
        f"{prefix}_status": "status",
    }

    df = block_df.rename(columns=mapping).copy()

    cols = list(df.columns)
    df = df[cols]

    # Add noise to values
    for c in df.columns:
        df[c] = add_light_noise(df[c], p=0.05)

    return df


def make_target_table(block_df, prefix):
    """
    Target: same ambiguous headers.
    """
    mapping = {
        f"{prefix}_id": "id",
        f"{prefix}_name": "name",
        f"{prefix}_code": "code",
        f"{prefix}_date": "date",
        f"{prefix}_status": "status",
    }
    return block_df.rename(columns=mapping).copy()


def build_wide_source_target(n_rows=200):
    """
    Build one pair of wide tables with repeated ambiguous column groups.
    Context is needed because many columns share identical local names.
    """
    customer = build_customer_block(n_rows)
    product = build_product_block(n_rows)
    order = build_order_block(n_rows)
    employee = build_employee_block(n_rows)
    supplier = build_supplier_block(n_rows)

    source_blocks = []
    target_blocks = []
    gt_rows = []

    block_specs = [
        ("customer", customer),
        ("product", product),
        ("order", order),
        ("employee", employee),
        ("supplier", supplier),
    ]

    for idx, (prefix, block_df) in enumerate(block_specs, start=1):
        source_block = make_source_table(block_df, prefix).copy()
        target_block = make_target_table(block_df, prefix).copy()

        source_cols = {
            "id": f"id_{idx}",
            "name": f"name_{idx}",
            "code": f"code_{idx}",
            "date": f"date_{idx}",
            "status": f"status_{idx}",
        }
        target_cols = {
            "id": f"id_{idx}",
            "name": f"name_{idx}",
            "code": f"code_{idx}",
            "date": f"date_{idx}",
            "status": f"status_{idx}",
        }

        source_block = source_block.rename(columns=source_cols)
        target_block = target_block.rename(columns=target_cols)

        # Ground truth
        for base_col in ["id", "name", "code", "date", "status"]:
            gt_rows.append({
                "entity": prefix,
                "semantic_column": f"{prefix}_{base_col}",
                "source_column": f"{base_col}_{idx}",
                "target_column": f"{base_col}_{idx}",
            })

        source_blocks.append(source_block)
        target_blocks.append(target_block)

    # Interleave columns to make context matter more
    source_df = pd.concat(source_blocks, axis=1)
    target_df = pd.concat(target_blocks, axis=1)

    # Reorder in a way that keeps entity-local neighborhoods but not perfect block order
    source_order = [
        "id_1", "name_1", "code_1", "date_1", "status_1",
        "id_3", "name_3", "code_3", "date_3", "status_3",
        "id_2", "name_2", "code_2", "date_2", "status_2",
        "id_5", "name_5", "code_5", "date_5", "status_5",
        "id_4", "name_4", "code_4", "date_4", "status_4",
    ]

    target_order = [
        "id_2", "name_2", "code_2", "date_2", "status_2",
        "id_1", "name_1", "code_1", "date_1", "status_1",
        "id_4", "name_4", "code_4", "date_4", "status_4",
        "id_3", "name_3", "code_3", "date_3", "status_3",
        "id_5", "name_5", "code_5", "date_5", "status_5",
    ]

    source_df = source_df[source_order]
    target_df = target_df[target_order]

    gt_df = pd.DataFrame(gt_rows)
    return source_df, target_df, gt_df


def main():
    source_df, target_df, gt_df = build_wide_source_target(n_rows=250)

    source_path = OUTPUT_DIR / "ambiguity_source.csv"
    target_path = OUTPUT_DIR / "ambiguity_target.csv"
    gt_path = OUTPUT_DIR / "ambiguity_ground_truth.csv"

    source_df.to_csv(source_path, index=False)
    target_df.to_csv(target_path, index=False)
    gt_df.to_csv(gt_path, index=False)

    print("Saved:")
    print(source_path.resolve())
    print(target_path.resolve())
    print(gt_path.resolve())
    print()
    print("SOURCE columns:")
    print(list(source_df.columns))
    print()
    print("TARGET columns:")
    print(list(target_df.columns))
    print()
    print("Ground truth size:", len(gt_df))


if __name__ == "__main__":
    main()