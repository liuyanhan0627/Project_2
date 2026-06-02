# Experiment Results

Generated runtime directories stay under `outputs/` and remain ignored by Git.
Result export bundles under `experiments/result_exports/` are intentionally tracked and can be pushed to GitHub.

For GitHub transfer, run:

```bash
python3 scripts/collect_results.py --outputs outputs --registry experiments/registry.csv
python3 scripts/export_results_for_github.py \
  --outputs outputs \
  --registry experiments/registry.csv \
  --export-root experiments/result_exports
```

Then commit the exported bundle:

```bash
git add experiments/result_exports/<bundle_name>
git commit -m "Add experiment results"
git push origin <results-branch>
```

The export copies configs, metadata, logs, summaries, metrics, lightweight experiment state files, and JSONL generations, while excluding model-weight checkpoint files.
