# NSCC Runbook

This project was originally run on AutoDL. On NSCC, keep the same code entry
points, but let PBS allocate GPUs first. Do not run training directly on the
login node.

## Resource Meaning

The guide from the server uses commands like:

```bash
qsub -I \
  -P personal-e1547010 \
  -q normal \
  -l select=1:ngpus=2:ncpus=32:mem=110gb \
  -l walltime=01:00:00
```

For this project:

- `select=1` means one PBS chunk/node for one job. Usually leave it as `1`.
- `ngpus=2` requests two GPUs. Group A/C use `cuda:0` for the big model and
  `cuda:1` for the small model, so two GPUs is the recommended default.
- `ncpus=32` and `mem=110gb` match the two-GPU request from the shared guide.
- `walltime=01:00:00` is good for smoke tests. Use `02:00:00` or `03:00:00`
  for short sweeps if the queue allows it.

Requesting four GPUs does not automatically make the current sweep scripts
four times faster, because each config currently runs its jobs sequentially.
Use four GPUs only after changing the configs/scripts to use them, or submit
separate two-GPU jobs instead.

## First-Time Setup On NSCC

SSH to the login node:

```bash
ssh e1547010@aspire2a.nus.edu.sg
```

Clone or upload the repo, then enter it:

```bash
git clone <your-repo-url> ASPS-workspace
cd ASPS-workspace
```

Create or reuse a virtual environment. The scripts default to `~/collmenv`,
matching the shared guide. The recommended NSCC path is to use the system
PyTorch module and install the lighter Python dependencies into that virtualenv:

```bash
qsub -P personal-e1547010 \
  -q normal \
  -l select=1:ncpus=4:mem=16gb \
  -l walltime=00:30:00 \
  scripts/nscc_setup_venv.pbs
```

The setup script loads `pytorch/2.6.0-py3-cu11.8`, creates `~/collmenv`, and
installs the standard Python dependencies adapted from the AutoDL
`ASPS/envs/gsm8k_strategyqa.yml` environment. By default it installs regular
Hugging Face `transformers`, which is enough for Group A/C and baseline runs.

If you only want to install into your user site-packages instead of a
virtualenv, use:

```bash
qsub -P personal-e1547010 \
  -q normal \
  -l select=1:ncpus=4:mem=16gb \
  -l walltime=00:30:00 \
  scripts/nscc_install_deps.pbs
```

Group D/CNTP needs the patched ASPS `transformers` package in
`ASPS/custom_transformers_packages/gsm8k_strategyqa`. On the current NSCC
Python 3.13 module, that patched package requires an old `tokenizers` version
that may need a separate Python <= 3.12 environment. Treat Group D as a
separate setup step unless `INSTALL_CNTP=1` succeeds on your environment.

Create a private environment file on NSCC:

```bash
cat > ~/.asps_nscc_env <<'EOF'
export HF_TOKEN=hf_your_token_here
export NSCC_MODULES="pytorch/2.6.0-py3-cu11.8"
export VENV_PATH=$HOME/collmenv
export HF_HOME=$HOME/scratch/hf_cache
export TRANSFORMERS_CACHE=$HOME/scratch/hf_cache
export BIG_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export SMALL_MODEL=meta-llama/Llama-3.2-1B-Instruct
EOF
chmod 600 ~/.asps_nscc_env
```

Do not commit this file.

## Interactive Smoke Test

Use an interactive allocation first, because it is easier to debug missing
packages or model-cache problems:

```bash
PROJECT=personal-e1547010 WALLTIME=01:00:00 bash scripts/nscc_interactive.sh
```

After the prompt moves onto the compute node:

```bash
source ~/.asps_nscc_env
source ~/collmenv/bin/activate
python3 train.py --config configs/smoke.yaml --dry-run
python3 train.py --config configs/smoke.yaml
```

If smoke succeeds, exit the interactive session.

## Batch Submission

Submit the default short daytime sweep:

```bash
PROJECT=personal-e1547010 \
WALLTIME=02:00:00 \
RUN_SCRIPT=scripts/run_group_ac_daytime_k2c_sweep.sh \
bash scripts/nscc_submit.sh
```

Other useful scripts:

```bash
RUN_SCRIPT=scripts/run_group_ac_k2_refine_sweep.sh bash scripts/nscc_submit.sh
RUN_SCRIPT=scripts/run_group_ac_k2_overnight_mix_sweep.sh bash scripts/nscc_submit.sh
RUN_SCRIPT=scripts/run_group_ac_ruler_small_overnight.sh bash scripts/nscc_submit.sh
```

You can narrow a run by exporting `RUN_CONFIGS` before submitting. Because
`scripts/nscc_submit.sh` uses `qsub -V`, PBS receives the variable:

```bash
PROJECT=personal-e1547010 \
RUN_CONFIGS="a_k2_h15_d16 c_k2_h14_a002_d16_margin005" \
RUN_SCRIPT=scripts/run_group_ac_daytime_k2c_sweep.sh \
bash scripts/nscc_submit.sh
```

## Monitoring And Canceling

Check queue status:

```bash
qstat -u "$USER"
```

Watch the PBS wrapper log:

```bash
tail -f nscc_logs/*.log
```

Watch the experiment-level logs:

```bash
tail -f outputs/*/logs/train.log
tail -f outputs/*/logs/*.log
```

Cancel a job if it waits too long or the resource request is wrong:

```bash
qdel <job_id>
```

## If A Job Hits Walltime

The current runner writes independent output directories and lightweight
checkpoint metadata, but it does not resume from the middle of a config. If a
sweep is killed by walltime, submit the remaining config names with
`RUN_CONFIGS`.

Use shorter two-GPU jobs during busy periods. One-GPU jobs usually queue faster,
but this repo's Group A/C configs expect two visible GPUs. One GPU is mainly
for baseline or Group B runs unless you create a dedicated single-GPU config.
