from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import string
import pandas as pd


random.seed(42)


OUTPUT_DIR = ROOT / "synthetic_context_benchmark"
OUTPUT_DIR.mkdir(exist_ok=True)


DOMAIN_SCHEMAS = {
    "customers": {
        "customer_id": lambda n: [f"CUST{1000+i}" for i in range(n)],
        "customer_name": lambda n: [random.choice(["Alice", "Bob", "Carol", "David", "Emma", "Frank"]) + f" {i}" for i in range(n)],
        "signup_date": lambda n: [f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)],
        "country_code": lambda n: [random.choice(["DE", "FR", "IT", "ES", "NL"]) for _ in range(n)],
        "status": lambda n: [random.choice(["active", "inactive", "pending"]) for _ in range(n)],
    },
    "products": {
        "product_id": lambda n: [f"PROD{2000+i}" for i in range(n)],
        "product_name": lambda n: [random.choice(["Phone", "Laptop", "Tablet", "Monitor", "Keyboard"]) + f" {i}" for i in range(n)],
        "release_date": lambda n: [f"2022-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)],
        "category_code": lambda n: [random.choice(["ELEC", "HOME", "OFFC", "GAME"]) for _ in range(n)],
        "status": lambda n: [random.choice(["available", "archived", "draft"]) for _ in range(n)],
    },
    "orders": {
        "order_id": lambda n: [f"ORD{3000+i}" for i in range(n)],
        "customer_id": lambda n: [f"CUST{1000+(i % 50)}" for i in range(n)],
        "order_date": lambda n: [f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)],
        "country_code": lambda n: [random.choice(["DE", "FR", "IT", "ES", "NL"]) for _ in range(n)],
        "amount": lambda n: [round(random.uniform(10, 1000), 2) for _ in range(n)],
        "status": lambda n: [random.choice(["paid", "shipped", "returned", "cancelled"]) for _ in range(n)],
    },
    "employees": {
        "employee_id": lambda n: [f"EMP{4000+i}" for i in range(n)],
        "employee_name": lambda n: [random.choice(["Mia", "Noah", "Liam", "Olivia", "Ava", "Sofia"]) + f" {i}" for i in range(n)],
        "hire_date": lambda n: [f"2021-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)],
        "department_code": lambda n: [random.choice(["HR", "ENG", "FIN", "MKT"]) for _ in range(n)],
        "status": lambda n: [random.choice(["active", "leave", "terminated"]) for _ in range(n)],
    },
}


TARGET_RENAMES = {
    "customer_id": "id",
    "customer_name": "name",
    "signup_date": "date",
    "country_code": "code",
    "product_id": "id",
    "product_name": "name",
    "release_date": "date",
    "category_code": "code",
    "order_id": "id",
    "order_date": "date",
    "employee_id": "id",
    "employee_name": "name",
    "hire_date": "date",
    "department_code": "code",
    "amount": "value",
    "status": "status",
}


SOURCE_RENAMES = {
    "customer_id": "cust_id",
    "customer_name": "cust_nm",
    "signup_date": "sgn_dt",
    "country_code": "cntry_cd",
    "product_id": "prod_id",
    "product_name": "prod_nm",
    "release_date": "rel_dt",
    "category_code": "cat_cd",
    "order_id": "ord_id",
    "order_date": "ord_dt",
    "employee_id": "emp_id",
    "employee_name": "emp_nm",
    "hire_date": "hire_dt",
    "department_code": "dept_cd",
    "amount": "amt",
    "status": "sts",
}


def typo(text: str, p: float = 0.25) -> str:
    if random.random() > p or len(text) < 4:
        return text
    i = random.randint(0, len(text) - 1)
    c = random.choice(string.ascii_lowercase)
    return text[:i] + c + text[i + 1:]


def add_noise_to_values(series: pd.Series, p: float = 0.1) -> pd.Series:
    out = []
    for val in series.astype(str):
        if random.random() < p and len(val) > 3:
            out.append(typo(val, p=1.0))
        else:
            out.append(val)
    return pd.Series(out)


def build_base_table(domain: str, n_rows: int = 200) -> pd.DataFrame:
    schema = DOMAIN_SCHEMAS[domain]
    data = {col: gen(n_rows) for col, gen in schema.items()}
    return pd.DataFrame(data)


def make_source_table(df: pd.DataFrame) -> pd.DataFrame:
    source_df = df.copy()

    renamed = {}
    for col in source_df.columns:
        new_name = SOURCE_RENAMES.get(col, col)
        new_name = typo(new_name, p=0.2)
        renamed[col] = new_name
    source_df = source_df.rename(columns=renamed)

    for col in source_df.columns:
        source_df[col] = add_noise_to_values(source_df[col], p=0.08)

    shuffled_cols = list(source_df.columns)
    random.shuffle(shuffled_cols)
    source_df = source_df[shuffled_cols]

    return source_df


def make_target_table(df: pd.DataFrame) -> pd.DataFrame:
    target_df = df.copy()

    renamed = {}
    for col in target_df.columns:
        renamed[col] = TARGET_RENAMES.get(col, col)
    target_df = target_df.rename(columns=renamed)

    shuffled_cols = list(target_df.columns)
    random.shuffle(shuffled_cols)
    target_df = target_df[shuffled_cols]

    return target_df


def build_ground_truth(source_df: pd.DataFrame, target_df: pd.DataFrame, original_df: pd.DataFrame):
    source_map = {}
    for original_col in original_df.columns:
        noisy_source = SOURCE_RENAMES.get(original_col, original_col)
        source_map[original_col] = None
        for actual_col in source_df.columns:
            base = actual_col.replace(" ", "")
            if noisy_source[:3] in base or noisy_source in actual_col:
                source_map[original_col] = actual_col

    target_map = {}
    for original_col in original_df.columns:
        target_name = TARGET_RENAMES.get(original_col, original_col)
        for actual_col in target_df.columns:
            if actual_col == target_name:
                target_map[original_col] = actual_col

    gt_rows = []
    for original_col in original_df.columns:
        src_candidates = []
        for c in source_df.columns:
            src_candidates.append(c)

        source_match = None
        for c in source_df.columns:
            if SOURCE_RENAMES.get(original_col, original_col).split("_")[0] in c:
                source_match = c
                break

        if source_match is None:
            # fallback by position via original names approximation
            for c in source_df.columns:
                if c.startswith(original_col[:3]):
                    source_match = c
                    break

        target_match = TARGET_RENAMES.get(original_col, original_col)
        if source_match is not None and target_match in target_df.columns:
            gt_rows.append(
                {
                    "original_column": original_col,
                    "source_column": source_match,
                    "target_column": target_match,
                }
            )

    return pd.DataFrame(gt_rows)


def generate_pair(domain: str, n_rows: int = 200):
    base_df = build_base_table(domain, n_rows=n_rows)
    source_df = make_source_table(base_df)
    target_df = make_target_table(base_df)

    gt_rows = []
    for original_col in base_df.columns:
        source_base = SOURCE_RENAMES.get(original_col, original_col)
        source_match = None
        for col in source_df.columns:
            if source_base[:3] in col or source_base in col:
                source_match = col
                break
        if source_match is None:
            source_match = source_df.columns[base_df.columns.get_loc(original_col)]

        target_match = TARGET_RENAMES.get(original_col, original_col)
        gt_rows.append(
            {
                "original_column": original_col,
                "source_column": source_match,
                "target_column": target_match,
            }
        )

    gt_df = pd.DataFrame(gt_rows)
    return source_df, target_df, gt_df


def main():
    summary_rows = []

    for domain in DOMAIN_SCHEMAS:
        source_df, target_df, gt_df = generate_pair(domain, n_rows=250)

        source_path = OUTPUT_DIR / f"{domain}_source.csv"
        target_path = OUTPUT_DIR / f"{domain}_target.csv"
        gt_path = OUTPUT_DIR / f"{domain}_ground_truth.csv"

        source_df.to_csv(source_path, index=False)
        target_df.to_csv(target_path, index=False)
        gt_df.to_csv(gt_path, index=False)

        summary_rows.append(
            {
                "domain": domain,
                "source_file": source_path.name,
                "target_file": target_path.name,
                "ground_truth_file": gt_path.name,
                "n_source_cols": source_df.shape[1],
                "n_target_cols": target_df.shape[1],
                "n_rows": source_df.shape[0],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)

    print("Generated benchmark files in:", OUTPUT_DIR.resolve())
    print(summary_df)


if __name__ == "__main__":
    main()