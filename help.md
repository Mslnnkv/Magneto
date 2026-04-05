# How To Run

## 1. Сгенерировать benchmark-версии

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_1.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_2.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_3.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_4.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_5.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_6.py
```

## 2. Построить triplets для fine-tuning

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\build_training_triplets.py
```

Triplets строятся только из:
- `version_3`
- `version_4`

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
