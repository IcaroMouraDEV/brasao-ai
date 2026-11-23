# Evaluation Reports

Generated comparative benchmark artifacts (`comparison_<timestamp>.json` and `comparison_<timestamp>.md`) are written here by `evals/runner.py` and ignored by Git by default.

Run a smoke test with:

```bash
make benchmark-dry-run
```

Run a full live comparison (Remote Gemini vs. Local Ollama) with:

```bash
make benchmark
```
