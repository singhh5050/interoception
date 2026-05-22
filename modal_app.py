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
def train_run(toml_name: str, run_name: str, wandb_project: str = "interoception") -> dict:
    """Run a prime-rl training job. prime-rl orchestrates vllm + trainer internally."""
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
    cfgs = [
        ("rl/phase1_hyp_s0.toml", "qwen25-3b-hyp-s0"),
        ("rl/phase1_hyp_s1.toml", "qwen25-3b-hyp-s1"),
        ("rl/phase1_exp_s0.toml", "qwen25-3b-exp-s0"),
        ("rl/phase1_exp_s1.toml", "qwen25-3b-exp-s1"),
    ]
    print(f"Launching Phase 1: {len(cfgs)} runs in parallel")
    results = list(train_run.starmap(cfgs))
    for cfg, r in zip(cfgs, results):
        print(f"  {cfg[1]}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s")
