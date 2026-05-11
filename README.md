# interoception

Exploratory experiment: inject simulated wallclock time into an LLM's context between turns and look at what it does. No training yet — just an observation harness.

## Setup

```
# 1. Have hardware-proprioception cloned at ~/hardware-proprioception
git clone https://github.com/singhh5050/hardware-proprioception.git ~/hardware-proprioception

# 2. Install (on a GPU box — vLLM needs CUDA)
pip install -e ~/hardware-proprioception
pip install -e .
```

## Run one rollout

On Hopper (H100/GH200) with vLLM 0.20.x, set `VLLM_USE_DEEP_GEMM=0` — the
warmup probes the optional `deep_gemm` FP8 kernels even for bf16 models and
will error out if the lib isn't installed.

```
VLLM_USE_DEEP_GEMM=0 python scripts/run_one.py \
    --hf-model Qwen/Qwen2.5-7B-Instruct \
    --sim-model Qwen2.5-7B \
    --hardware H100_SXM \
    --target-s 30 \
    --problem 0 \
    --out runs/q0_T30.json
```

`--hf-model` is the checkpoint vLLM loads. `--sim-model` is the name in the `hwprop` catalog (the simulator does not distinguish base / instruct variants — both Qwen2.5-7B and Qwen2.5-7B-Instruct map to `Qwen2.5-7B`).

## What the script does

For each rollout:

1. Builds a system prompt that tells the model it has `T` seconds and that the user will inject `[X seconds elapsed]` messages between its turns.
2. Generates up to `chunk_tokens` tokens (default 256).
3. Asks the `hwprop` roofline simulator how long that chunk would take on the target hardware. Adds to a running `elapsed_s`.
4. If the model emitted `<answer>...</answer>`, stops with a success.
5. If `elapsed_s >= T`, stops with a timeout.
6. Otherwise appends `[Xs elapsed]` as a fresh user message and goes back to step 2.

The transcript is printed and (with `--out`) dumped as JSON for later inspection.

## Building the dataset

The original `PROBLEMS` list in `tasks.py` has 4 hardcoded Countdown puzzles — fine for base-model exploration, useless for training. Build a stratified dataset from Jiayi-Pan/Countdown-Tasks-3to4:

```
python scripts/build_dataset.py --out-dir data --train-size 10000 --eval-size 500
```

This downloads the parquet (~3 MB, cached at `~/.cache/interoception/`), filters to 4-number problems, runs our Countdown solver on each, drops unsolvable + trivial (>50 solutions), buckets by `solution_count` (rare / med / common), and samples uniformly across buckets. Writes `train.jsonl` + `eval.jsonl`. Takes ~4 min for the default 60K-solve cap.

Load in code via `interoception.tasks.load_problems("data/train.jsonl")`.

## Prompt styles

The system prompt has three variants (set via `--prompt-style`). See the docstring on `build_system_prompt` in `src/interoception/rollout.py` for the exact text.

- **default** — naive baseline. Mentions the budget and the `[Xs elapsed]` tag, says "pace yourself," no specific rules.
- **medium** — same setup as default but adds "take these signals seriously and adjust your strategy accordingly." Tests whether emphasizing the signal as important is enough, without specifying a policy.
- **strong** — full prescriptive policy: state remaining budget every turn, switch strategies if repeating, commit at 70%, commit immediately if correct. Tests the ceiling of prompt-only elicitation.

## Sweep over budgets and problems

```
VLLM_USE_DEEP_GEMM=0 python scripts/run_sweep.py \
    --hf-model Qwen/Qwen2.5-7B-Instruct \
    --sim-model Qwen2.5-7B \
    --hardware H100_SXM \
    --target-s 15 30 60 120 \
    --problems 0 1 2 3 \
    --seeds 0 1 \
    --out-dir runs/sweep_qwen7b_h100
```

Loads the model once, runs every (problem, T, seed) combo, writes one JSON per rollout plus `summary.csv`, then prints a per-T table: emission rate, timeout rate, correctness, avg elapsed, avg output tokens.

## Calibrated catalog

Models the sim has been calibrated against (use these for `--sim-model`):

```
Falcon3-7B, Gemma-3-1B, LLaMA-3.2-{1B,3B}, Phi-4-{14B,mini-3.8B},
Qwen2.5-{1.5B,3B,7B,14B,72B}, SmolLM2-1.7B
```

Hardware:

```
A100_{40GB,80GB}, A40, B200, B300, GH200, H100_SXM, H200, L40S, ...
```

(see `hwprop.specs.get_hardware_specs()` for the full list)
