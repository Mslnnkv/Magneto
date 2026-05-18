# How To Run

This is a short command checklist for reproducing the diploma experiments.

## 1. Prepare the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r algorithms\magneto\requirements.txt
```

## 2. Generate benchmark versions

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_1.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_2.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_3.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_4.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_5.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_6.py
```

## 3. Build triplets for fine-tuning

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\build_training_triplets.py
```

Triplets are built from `version_3` and `version_4`.

## 4. Train the contextual models

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\train_contextual_span_encoder.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\train_starmie_structured_encoder.py
```

The scripts save:

- `algorithms\magneto\finetuned_context_window_span_mpnet.pth`
- `algorithms\magneto\finetuned_context_window_starmie_structured_mpnet.pth`

## 5. Run evaluation

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_heldout_context_benchmark.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_starmie_context_benchmark.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_all_synthetic_benchmarks.py
```

## 6. Main outputs

- `algorithms\magneto\evaluation_outputs\all_synthetic_benchmarks_summary.csv`
- `algorithms\magneto\evaluation_outputs\all_synthetic_benchmarks_summary.png`
- `algorithms\magneto\evaluation_outputs\all_synthetic_benchmarks_summary_bw.png`

For the final thesis comparison, use held-out benchmarks `version_5` and `version_6`.
