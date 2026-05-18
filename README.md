# Contextual Magneto for Diploma Experiments

This repository is a diploma research branch based on the original
**Magneto** project for schema matching. The branch extends the original
embedding-based retrieval pipeline with contextual column encoders,
Starmie-inspired structured encoding, synthetic benchmarks, fine-tuning, and
evaluation scripts for context-dependent column matching.

The work here should be read as a continuation and adaptation of Magneto, not
as a separate implementation from scratch.

## What This Branch Adds

- Baseline Magneto comparison with `header_values_verbose`.
- Contextual span-based column encoding.
- Starmie-inspired structured contextual column encoding.
- Triplet-based fine-tuning for contextual encoders.
- Synthetic benchmark generation: `version_1` ... `version_6`.
- Held-out evaluation scripts and presentation-ready plots.

The main final comparison used in the diploma experiments is:

- `Magneto`
- `Contextual Magneto`
- `Contextual Magneto (fine-tuned)`

## Repository Structure

Top-level files and folders:

- [help.md](help.md) - short runbook with the main commands.
- [algorithms/magneto](algorithms/magneto) - adapted Magneto implementation.
- [experiments](experiments) - upstream experiment code that is not the main entrypoint for this branch.

Important paths inside `algorithms/magneto`:

- [scripts/benchmark_generation](algorithms/magneto/scripts/benchmark_generation) - synthetic benchmark generators.
- [scripts/training](algorithms/magneto/scripts/training) - triplet generation and fine-tuning scripts.
- [scripts/evaluation](algorithms/magneto/scripts/evaluation) - evaluation entrypoints and plot generation.
- [magneto](algorithms/magneto/magneto) - core matching code and encoders.
- [evaluation_outputs](algorithms/magneto/evaluation_outputs) - summary tables and plots.

## Environment Setup

Create and activate a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
pip install -r algorithms\magneto\requirements.txt
```

Notes:

- Embedding-based runs use `sentence-transformers/all-mpnet-base-v2` as the base model.
- If the environment is offline, this model must already exist in the local Hugging Face cache.
- Fine-tuned weights are stored locally in:
  - [finetuned_context_window_span_mpnet.pth](algorithms/magneto/finetuned_context_window_span_mpnet.pth)
  - [finetuned_context_window_starmie_structured_mpnet.pth](algorithms/magneto/finetuned_context_window_starmie_structured_mpnet.pth)

## Main Workflow

### 1. Generate synthetic benchmarks

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_1.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_2.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_3.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_4.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_5.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\benchmark_generation\generate_benchmark_version_6.py
```

### 2. Build training triplets

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\build_training_triplets.py
```

Triplets are built only from:

- `version_3`
- `version_4`

The final held-out evaluation should focus on:

- `version_5`
- `version_6`

### 3. Fine-tune contextual models

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\train_contextual_span_encoder.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\train_starmie_structured_encoder.py
```

### 4. Run evaluation

Held-out contextual benchmark:

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_heldout_context_benchmark.py
```

Starmie-style held-out benchmark:

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_starmie_context_benchmark.py
```

Full synthetic comparison:

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\evaluation\evaluate_all_synthetic_benchmarks.py
```

## Main Result Files

The main outputs used in the diploma branch are:

- [all_synthetic_benchmarks_summary.csv](algorithms/magneto/evaluation_outputs/all_synthetic_benchmarks_summary.csv)
- [all_synthetic_benchmarks_errors.csv](algorithms/magneto/evaluation_outputs/all_synthetic_benchmarks_errors.csv)
- [all_synthetic_benchmarks_summary.png](algorithms/magneto/evaluation_outputs/all_synthetic_benchmarks_summary.png)
- [all_synthetic_benchmarks_summary_bw.png](algorithms/magneto/evaluation_outputs/all_synthetic_benchmarks_summary_bw.png)

Benchmark-specific outputs:

- [version_5_evaluation_results.csv](algorithms/magneto/synthetic_benchmark_version_5/version_5_evaluation_results.csv)
- [version_5_evaluation_errors.csv](algorithms/magneto/synthetic_benchmark_version_5/version_5_evaluation_errors.csv)
- [version_6_evaluation_results.csv](algorithms/magneto/synthetic_benchmark_version_6/version_6_evaluation_results.csv)
- [version_6_evaluation_errors.csv](algorithms/magneto/synthetic_benchmark_version_6/version_6_evaluation_errors.csv)

## Deprecated Files

Files ending with `_deprecated.py` are kept only for historical reference. They
are not the main entrypoints for the current diploma workflow.

Typical examples include old ad-hoc comparison scripts, debug scripts,
smoke-test scripts, and superseded evaluation entrypoints.

The recommended workflow is documented above through:

- `scripts/benchmark_generation`
- `scripts/training`
- `scripts/evaluation`

## License and Attribution

This repository is based on the original Magneto project:
https://github.com/VIDA-NYU/magneto-matcher

The original Magneto framework is licensed under Apache-2.0. The original
license file is preserved in this repository:

- [LICENSE](LICENSE)

This diploma branch extends the original codebase with contextual column
encoders, Starmie-inspired structured encoding, synthetic benchmarks,
fine-tuning scripts, and evaluation utilities for context-dependent schema
matching experiments.
