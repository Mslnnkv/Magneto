# Magneto Package in the Diploma Branch

This directory contains the **adapted Magneto implementation** used in the diploma experiments.

The code is based on the original Magneto project, but in this branch it is used primarily for:

- baseline embedding-based schema matching;
- contextual column encoding;
- Starmie-inspired structured contextual encoding;
- fine-tuning with triplet loss;
- synthetic benchmark generation and evaluation.

## What Is the Main Entry Point

For the current branch, the recommended entrypoints are not the old upstream benchmark scripts, but:

- [`scripts/benchmark_generation`](C:/Users/AnnaM/Magneto/algorithms/magneto/scripts/benchmark_generation)
- [`scripts/training`](C:/Users/AnnaM/Magneto/algorithms/magneto/scripts/training)
- [`scripts/evaluation`](C:/Users/AnnaM/Magneto/algorithms/magneto/scripts/evaluation)

For full setup and reproduction steps, see the repository root README:

- [`README.md`](C:/Users/AnnaM/Magneto/README.md)

## Notes

- Files ending with `_deprecated.py` are preserved only as historical or debugging artifacts.
- The active fine-tuned checkpoints are:
  - [`finetuned_context_window_span_mpnet.pth`](C:/Users/AnnaM/Magneto/algorithms/magneto/finetuned_context_window_span_mpnet.pth)
  - [`finetuned_context_window_starmie_structured_mpnet.pth`](C:/Users/AnnaM/Magneto/algorithms/magneto/finetuned_context_window_starmie_structured_mpnet.pth)
- The active summary outputs are stored in:
  - [`evaluation_outputs`](C:/Users/AnnaM/Magneto/algorithms/magneto/evaluation_outputs)

## Attribution

The upstream Magneto framework belongs to its original authors. This package directory contains a continued and adapted working branch for the diploma project rather than a separate independent reimplementation.
