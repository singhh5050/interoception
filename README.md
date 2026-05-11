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

```
python scripts/run_one.py \
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

## Sweep over budgets and problems

```
python scripts/run_sweep.py \
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
