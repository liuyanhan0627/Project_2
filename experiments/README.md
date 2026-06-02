# Experiment Results

Generated runtime directories stay under `outputs/` and remain ignored by Git.

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

The export copies configs, metadata, logs, summaries, metrics, and JSONL generations, while excluding checkpoint/model-weight files.
