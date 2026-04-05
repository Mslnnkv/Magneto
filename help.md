# How To Run

## 1. Сгенерировать benchmark

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_ambiguity_benchmark.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_context_needed_benchmark.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_hard_context_benchmark.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_heldout_context_benchmark.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_starmie_context_benchmark.py
```

## 2. Построить triplets для fine-tuning

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\build_training_triplets.py
```

Triplets строятся только из:
- `context_needed`
- `hard_context`

## 3. Обучить модели

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\train_contextual_span_encoder.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\train_starmie_structured_encoder.py
```

Будут сохранены веса:
- `algorithms\magneto\finetuned_context_window_span_mpnet.pth`
- `algorithms\magneto\finetuned_context_window_starmie_structured_mpnet.pth`

## 4. Основные evaluation-запуски

Held-out benchmark:

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_heldout_context_benchmark.py
```

Starmie-style benchmark:

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_starmie_context_benchmark.py
```

Общий synthetic запуск:

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_all_synthetic_benchmarks.py
```

## 5. Результаты

- `algorithms\magneto\evaluation_outputs\all_synthetic_benchmarks_summary.csv`
- `algorithms\magneto\evaluation_outputs\all_synthetic_benchmarks_summary.png`
