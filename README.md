# Contextual Magneto for Diploma Experiments

This repository contains a working research branch based on the original **Magneto** project for schema matching. The current version was adapted for diploma experiments on **context-dependent column matching**, **contextual encoding**, **fine-tuning**, and **synthetic benchmark evaluation**.

Important context:
- the repository **continues** the upstream Magneto codebase rather than replacing it;
- the core Magneto retriever and part of the project structure are inherited from the original authors;
- the contextual encoders, synthetic benchmarks, training scripts, evaluation scripts, and presentation-oriented outputs were extended in this branch for the diploma work.

## What Is Implemented in This Branch

The current project focuses on the embedding-based retrieval part of Magneto and includes:

- baseline non-contextual matching (`header_values_verbose`);
- contextual span-based encoding;
- Starmie-inspired structured contextual encoding;
- fine-tuning for contextual encoders;
- synthetic benchmark generation (`version_1` ... `version_6`);
- held-out evaluation and comparison plots for diploma experiments.

The main final comparison in this branch is:
- `Magneto`
- `Contextual Magneto`
- `Contextual Magneto (с дообучением)`

## Repository Structure

Top-level files and folders:

- [`help.md`](C:/Users/AnnaM/Magneto/help.md) — short runbook with commands.
- [`algorithms/magneto`](C:/Users/AnnaM/Magneto/algorithms/magneto) — main adapted Magneto codebase.
- [`experiments`](C:/Users/AnnaM/Magneto/experiments) — upstream benchmark code that is not the main entrypoint for the diploma branch.

Important paths inside `algorithms/magneto`:

- [`scripts/benchmark_generation`](C:/Users/AnnaM/Magneto/algorithms/magneto/scripts/benchmark_generation) — synthetic benchmark generators.
- [`scripts/training`](C:/Users/AnnaM/Magneto/algorithms/magneto/scripts/training) — triplet generation and fine-tuning scripts.
- [`scripts/evaluation`](C:/Users/AnnaM/Magneto/algorithms/magneto/scripts/evaluation) — evaluation entrypoints and plot generation.
- [`magneto`](C:/Users/AnnaM/Magneto/algorithms/magneto/magneto) — core matching code and encoders.
- [`evaluation_outputs`](C:/Users/AnnaM/Magneto/algorithms/magneto/evaluation_outputs) — summary tables and plots.

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
- embedding-based runs use `sentence-transformers/all-mpnet-base-v2` as the base model;
- if the environment is offline, this model must already exist in the local Hugging Face cache;
- fine-tuned weights are stored locally in:
  - [`finetuned_context_window_span_mpnet.pth`](C:/Users/AnnaM/Magneto/algorithms/magneto/finetuned_context_window_span_mpnet.pth)
  - [`finetuned_context_window_starmie_structured_mpnet.pth`](C:/Users/AnnaM/Magneto/algorithms/magneto/finetuned_context_window_starmie_structured_mpnet.pth)

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

This means the final held-out evaluation should focus on:
- `version_5`
- `version_6`

### 3. Fine-tune contextual models

```powershell
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\train_contextual_span_encoder.py
.\.venv\Scripts\python.exe algorithms\magneto\scripts\training\train_starmie_structured_encoder.py
```

### 4. Run evaluation

Held-out benchmark:

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

The most important outputs currently used in the diploma branch are:

- [`all_synthetic_benchmarks_summary.csv`](C:/Users/AnnaM/Magneto/algorithms/magneto/evaluation_outputs/all_synthetic_benchmarks_summary.csv)
- [`all_synthetic_benchmarks_errors.csv`](C:/Users/AnnaM/Magneto/algorithms/magneto/evaluation_outputs/all_synthetic_benchmarks_errors.csv)
- [`all_synthetic_benchmarks_summary.png`](C:/Users/AnnaM/Magneto/algorithms/magneto/evaluation_outputs/all_synthetic_benchmarks_summary.png)
- [`all_synthetic_benchmarks_summary_bw.png`](C:/Users/AnnaM/Magneto/algorithms/magneto/evaluation_outputs/all_synthetic_benchmarks_summary_bw.png)

Benchmark-specific outputs:

- [`synthetic_benchmark_version_5/version_5_evaluation_results.csv`](C:/Users/AnnaM/Magneto/algorithms/magneto/synthetic_benchmark_version_5/version_5_evaluation_results.csv)
- [`synthetic_benchmark_version_5/version_5_evaluation_errors.csv`](C:/Users/AnnaM/Magneto/algorithms/magneto/synthetic_benchmark_version_5/version_5_evaluation_errors.csv)
- [`synthetic_benchmark_version_6/version_6_evaluation_results.csv`](C:/Users/AnnaM/Magneto/algorithms/magneto/synthetic_benchmark_version_6/version_6_evaluation_results.csv)
- [`synthetic_benchmark_version_6/version_6_evaluation_errors.csv`](C:/Users/AnnaM/Magneto/algorithms/magneto/synthetic_benchmark_version_6/version_6_evaluation_errors.csv)

## Naming Convention for Deprecated Files

Files with the suffix `_deprecated.py` are kept only for historical reference. They are not the main entrypoints for the current diploma workflow.

Typical examples:
- old ad-hoc comparison scripts;
- old debug scripts;
- old smoke-test scripts;
- superseded evaluation entrypoints.

The recommended workflow is the one documented above through:
- `scripts/benchmark_generation`
- `scripts/training`
- `scripts/evaluation`

## License and Attribution

This repository is based on the original Magneto project:
https://github.com/VIDA-NYU/magneto-matcher

The original Magneto framework is licensed under Apache-2.0.
The original license file is preserved in this repository.

This diploma branch extends the original codebase with contextual column encoders,
Starmie-inspired structured encoding, synthetic benchmarks, fine-tuning scripts,
and evaluation utilities for context-dependent schema matching experiments.

