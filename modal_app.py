"""Modal app — interoception RL sweep on prime-rl.

prime-rl is PrimeIntellect's production RL framework (verifiers-rl is labeled
"educational, not maintained" — see their own README). It uses verifiers envs
under the hood; orchestrates vllm + trainer headlessly (no tmux required).

Architecture notes:
- prime-rl has private submodules we can't access (research-environments,
  configs/private). We init only the public ones (verifiers, renderers).
- `uv sync` validates every workspace member exists, so we patch the pyproject
  with `scripts/dev/patch_prime_rl_pyproject.py` to drop missing entries.
- prime-rl writes the resolved config (`rl.toml`) to output_dir per run, which
  solves the supervisor's "save defaults" concern natively.

Smoke run:    modal run modal_app.py::smoke
Phase 1 (4x): modal run --detach modal_app.py::phase1
Sweep (12x):  modal run --detach modal_app.py::sweep    # Phase 1 + 8 Phase 2 cells
"""
import modal

APP_NAME = "interoception-rl"

volume = modal.Volume.from_name("interoception-cache", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")

# Pin prime-rl to a specific commit so the resolved config (and defaults) are
# reproducible. b22e768 is what we read locally — pre-supervisor's "save
# defaults" note, that's our reference.
PRIME_RL_SHA = "b22e768fc419a1e8664729fd3fdfde98d1c13766"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "curl", "ca-certificates", "openssh-client")
    # Install uv. tomli_w (used by the patcher) is installed in the next step via pip.
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
    # Stage the patcher in the image (it'll run at build time below)
    .add_local_file(
        "scripts/dev/patch_prime_rl_pyproject.py",
        remote_path="/root/patch_pyproject.py",
        copy=True,
    )
    # Clone prime-rl, init only public submodules, patch pyproject, uv sync.
    .run_commands(
        # Add github.com to known_hosts (silences interactive prompts)
        "mkdir -p /root/.ssh && ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null || true",
        # Rewrite SSH URLs to HTTPS so public submodules clone without keys
        'git config --global url."https://github.com/".insteadOf "git@github.com:"',
        f"cd /root && git clone https://github.com/PrimeIntellect-ai/prime-rl.git",
        f"cd /root/prime-rl && git checkout {PRIME_RL_SHA}",
        # Init public submodules only. The private ones (research-environments,
        # configs/private) require auth we don't have — `|| true` tolerates failure.
        "cd /root/prime-rl && (git submodule update --init -- deps/verifiers || true)",
        "cd /root/prime-rl && (git submodule update --init -- deps/renderers || true)",
        # Patch pyproject.toml: drop workspace members + sources for the
        # private dirs we couldn't clone. Uses tomllib (proper parsing) not regex.
        "python3 -m pip install tomli_w && python3 /root/patch_pyproject.py /root/prime-rl/pyproject.toml",
        # Sync the venv. prime-rl source unconditionally imports flash_attn (via
        # ring_flash_attn), so the flash-attn optional extra is required — without
        # it, `rl` fails on `ModuleNotFoundError: No module named 'flash_attn'`.
        # We skip --all-extras because flash-attn-cute references unreleased
        # flash_attn_4 from git, and our patched pyproject already dropped `envs`.
        "cd /root/prime-rl && /root/.local/bin/uv sync --extra flash-attn",
    )
    # env_pkg uses copy=True so we can pip install it as a subsequent build step.
    .add_local_dir(
        "environments/interoception_countdown",
        remote_path="/root/env_pkg",
        copy=True,
    )
    .run_commands(
        # hwprop is required when env's timing_source="sim" (the training path).
        # Public repo, install into prime-rl's venv via uv pip.
        "cd /root && git clone https://github.com/singhh5050/hardware-proprioception.git",
        "cd /root/prime-rl && /root/.local/bin/uv pip install -e /root/hardware-proprioception",
        # Now install our env package.
        "cd /root/prime-rl && /root/.local/bin/uv pip install -e /root/env_pkg",
    )
    # data + configs read at runtime, no install step after — non-copy mounts.
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App(APP_NAME)


@app.function(
    # 2 GPUs: prime-rl runs vllm on one set, trainer on another. Matches the
    # gsm8k example's orchestrator gpu allocation pattern.
    gpu="A100-80GB:2",
    image=image,
    volumes={"/cache": volume},
    secrets=[wandb_secret],
    timeout=6 * 3600,
)
def train_run(toml_name: str, run_name: str, wandb_project: str = "interoception",
              extra_args: list[str] | None = None) -> dict:
    """Run a prime-rl training job. prime-rl orchestrates vllm + trainer internally.

    extra_args: optional CLI overrides appended to `rl @ <toml>` (e.g. for smokes:
    ['--max-steps', '4', '--ckpt.interval', '2', '--output-dir', '/cache/runs/x_smoke'])."""
    import os
    import subprocess
    import time

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_NAME"] = run_name

    cfg_path = f"/root/configs/{toml_name}"
    if not os.path.exists(cfg_path):
        return {"ok": False, "error": f"missing config: {cfg_path}"}

    cmd = [
        "/root/.local/bin/uv", "run", "rl",
        "@", cfg_path,
        "--wandb.project", wandb_project,
        "--wandb.name", run_name,
        *(extra_args or []),
    ]
    print(f"[prime-rl] launching: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    # NO capture_output: let stdout/stderr stream to container fd's so Modal
    # logs them live. We give up the stdout_tail in the result dict, but the
    # volume's logs/ dir + wandb have everything needed for forensic auditing.
    proc = subprocess.run(cmd, cwd="/root/prime-rl", text=True)
    dur = time.time() - t0

    volume.commit()
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_s": round(dur, 1),
    }


@app.local_entrypoint()
def smoke():
    print("Launching prime-rl smoke...")
    result = train_run.remote("rl/smoke.toml", "smoke-001")
    print("\n=== smoke result ===")
    for k, v in result.items():
        if isinstance(v, str) and len(v) > 200:
            print(f"--- {k} ---\n{v}\n")
        else:
            print(f"{k}: {v!r}")


@app.local_entrypoint()
def phase1():
    """Phase 1 only — Qwen2.5-3B x {hyp, exp} x {s0, s1}. Kept for backward compat;
    prefer `sweep` which launches Phase 1 + Phase 2 (10 cross-model cells) together."""
    cfgs = [
        ("rl/phase1_qwen25_3b_hyp_s0.toml", "qwen25-3b-hyp-s0"),
        ("rl/phase1_qwen25_3b_hyp_s1.toml", "qwen25-3b-hyp-s1"),
        ("rl/phase1_qwen25_3b_exp_s0.toml", "qwen25-3b-exp-s0"),
        ("rl/phase1_qwen25_3b_exp_s1.toml", "qwen25-3b-exp-s1"),
    ]
    print(f"Launching Phase 1: {len(cfgs)} runs in parallel")
    # Spawn-and-collect: one failure doesn't kill the rest (see sweep() for details).
    calls = [(cfg, train_run.spawn(*cfg)) for cfg in cfgs]
    results = []
    for cfg, call in calls:
        try:
            r = call.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        results.append(r)
        print(f"  {cfg[1]}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def single_qwen3_4b_hyp_s1():
    """Backfill — relaunches just qwen3-4b-hyp-s1 to fill the missing seed in the
    Qwen3-4B 2x2. The cell crashed during sweep_003 due to a stale evicted.txt
    from sweep_002 (bash-timeout-killed). Stale file has since been cleaned."""
    cfgs = [("rl/phase2_qwen3_4b_hyp_s1.toml", "qwen3-4b-hyp-s1")]
    calls = [(cfg, train_run.spawn(*cfg)) for cfg in cfgs]
    for cfg, call in calls:
        try:
            r = call.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        print(f"  {cfg[1]}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def v2():
    """v2 minimal single-cell run (Kanishk's spec): Qwen3-4B, hyperbolic c*min(1,T/t),
    no bonus, no 5T cutoff, 128 tok/chunk, T~U(15,130), G=8. See configs/rl/v2_qwen3_4b.toml."""
    cfgs = [("rl/v2_qwen3_4b.toml", "v2-qwen3-4b-hyp")]
    calls = [(cfg, train_run.spawn(*cfg)) for cfg in cfgs]
    for cfg, call in calls:
        try:
            r = call.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        print(f"  {cfg[1]}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def controls_smoke():
    """Short smoke of the REAL ctrlA/ctrlB configs (4 steps, ckpt every 2) to validate
    config validation + the new env flags + B's single-turn path BEFORE the full run.
    Writes to *_smoke output dirs and *-smoke wandb names so it doesn't touch the real
    runs. Verify after: returncode 0, weights/step_{2,4} written w/ adapter, B commits
    (not all timeout), f_term logged."""
    # warmup/decay overridden so the schedule fits 4 steps (decay_steps>=2 avoids the
    # known LinearLR ZeroDivisionError at decay_steps=1).
    common = ["--max-steps", "4", "--ckpt.interval", "2",
              "--trainer.scheduler.warmup-steps", "1", "--trainer.scheduler.decay-steps", "2"]
    jobs = [
        ("rl/ctrlA_qwen3_4b.toml", "ctrlA-smoke", common + ["--output-dir", "/cache/runs/ctrlA_smoke"]),
        ("rl/ctrlB_qwen3_4b.toml", "ctrlB-smoke", common + ["--output-dir", "/cache/runs/ctrlB_smoke"]),
    ]
    print(f"Launching control smokes: {len(jobs)} runs (4 steps each)")
    calls = [(j, train_run.spawn(j[0], j[1], extra_args=j[2])) for j in jobs]
    for (toml, name, _), call in calls:
        try:
            r = call.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        print(f"  {name}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def controls():
    """Clean 3-way for the f(t,T) ablation (Kanishk's 2026-05-24 thread). All Qwen3-4B,
    identical fixed env + eval (uniform T, temp 1.0, seed 777), checkpoint every 50.
      0 (ctrl0): TREATMENT — reward c*f(t,T), multi-turn, [Xs elapsed] injected.
      A (ctrlA): no time reward (c only), multi-turn, [Xs elapsed] still injected.
      B (ctrlB): no time signal — c only, single turn (max_turns=1), no injection.
    Treatment is re-run (not re-eval of old v2) so all three share fixed training code."""
    cfgs = [
        ("rl/ctrl0_qwen3_4b.toml", "ctrl0-qwen3-4b-treatment"),
        ("rl/ctrlA_qwen3_4b.toml", "ctrlA-qwen3-4b-noTimeReward"),
        ("rl/ctrlB_qwen3_4b.toml", "ctrlB-qwen3-4b-noTimeSignal"),
    ]
    print(f"Launching controls: {len(cfgs)} runs in parallel")
    calls = [(cfg, train_run.spawn(*cfg)) for cfg in cfgs]
    for cfg, call in calls:
        try:
            r = call.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        print(f"  {cfg[1]}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def sweep():
    """Scope-C sweep: 3 models x {hyp, exp} x {s0, s1} = 12 runs in parallel.

    Models:
      - Qwen2.5-3B (Phase 1: weakest baseline ~0%, original target)
      - Qwen3-4B    (Phase 2: mid baseline ~20%)
      - gemma-4-E4B (Phase 2: strongest baseline 25-35%)
    See scripts/dev/render_sweep_tomls.py for the rendering matrix.
    """
    cfgs = [
        # Phase 1 — Qwen2.5-3B
        ("rl/phase1_qwen25_3b_hyp_s0.toml", "qwen25-3b-hyp-s0"),
        ("rl/phase1_qwen25_3b_hyp_s1.toml", "qwen25-3b-hyp-s1"),
        ("rl/phase1_qwen25_3b_exp_s0.toml", "qwen25-3b-exp-s0"),
        ("rl/phase1_qwen25_3b_exp_s1.toml", "qwen25-3b-exp-s1"),
        # Phase 2a — Qwen3-4B
        ("rl/phase2_qwen3_4b_hyp_s0.toml", "qwen3-4b-hyp-s0"),
        ("rl/phase2_qwen3_4b_hyp_s1.toml", "qwen3-4b-hyp-s1"),
        ("rl/phase2_qwen3_4b_exp_s0.toml", "qwen3-4b-exp-s0"),
        ("rl/phase2_qwen3_4b_exp_s1.toml", "qwen3-4b-exp-s1"),
        # Phase 2b — gemma-4-E4B
        ("rl/phase2_gemma4_e4b_hyp_s0.toml", "gemma4-e4b-hyp-s0"),
        ("rl/phase2_gemma4_e4b_hyp_s1.toml", "gemma4-e4b-hyp-s1"),
        ("rl/phase2_gemma4_e4b_exp_s0.toml", "gemma4-e4b-exp-s0"),
        ("rl/phase2_gemma4_e4b_exp_s1.toml", "gemma4-e4b-exp-s1"),
    ]
    print(f"Launching sweep: {len(cfgs)} runs in parallel")
    # Spawn-and-collect pattern (vs starmap): one cell crashing/raising doesn't
    # take down the rest. Modal's `starmap` raises RemoteError on any child
    # failure, which kills the entire local entrypoint and stops the app —
    # that's how we lost an entire sweep when one Qwen2.5-3B cell crashed at
    # step 8. Here we collect results per-cell, catching exceptions so the
    # other 11 cells keep training.
    calls = [(cfg, train_run.spawn(*cfg)) for cfg in cfgs]
    results = []
    for cfg, call in calls:
        try:
            r = call.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        results.append(r)
        print(f"  {cfg[1]}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s  err={r.get('error', '')}")
