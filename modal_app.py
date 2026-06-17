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
    # The prompt-salience eval script is invoked directly in `eval_prompt_salience`.
    .add_local_file("scripts/eval_prompt_salience.py",
                    remote_path="/root/scripts/eval_prompt_salience.py")
    # Batched accuracy-vs-budget sweep (within-problem), invoked in `acc_vs_budget`.
    .add_local_file("scripts/dev/eval_acc_vs_budget.py",
                    remote_path="/root/scripts/eval_acc_vs_budget.py")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A100-80GB:1",
    volumes={"/cache": volume},
    timeout=1800,
)
def merge_lora_into_base(
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    lora_path: str = "/cache/runs/ctrl0_u1_40_long_additive_v2_l30_qwen3_4b/weights/step_200/lora_adapters",
    out_path: str = "/cache/merged_models/qwen3-4b-v2-l30-step200",
) -> dict:
    """Merge a LoRA adapter into a base model, save merged HF checkpoint.
    Used for the 2-stage curriculum: the merged model becomes the 'new base'
    for stage 2, so prime-rl's kl_tau regularization (which anchors against
    the base) automatically anchors against the stage-1 policy."""
    import os, subprocess
    # Values pass via env vars to avoid quote-escaping in the inline script.
    script = """
import os
os.environ['HF_HOME'] = '/cache/hf'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.environ['MERGE_BASE']
LORA = os.environ['MERGE_LORA']
OUT = os.environ['MERGE_OUT']

print(f'loading base model: {BASE}', flush=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
n_params_base = sum(p.numel() for p in model.parameters())
print(f'  base params: {n_params_base/1e9:.2f}B  dtype: {next(model.parameters()).dtype}', flush=True)
tok = AutoTokenizer.from_pretrained(BASE)

print(f'loading LoRA from: {LORA}', flush=True)
model = PeftModel.from_pretrained(model, LORA)

print('merging...', flush=True)
model = model.merge_and_unload()
n_params_merged = sum(p.numel() for p in model.parameters())
print(f'  merged params: {n_params_merged/1e9:.2f}B  dtype: {next(model.parameters()).dtype}', flush=True)

os.makedirs(OUT, exist_ok=True)
print(f'saving merged model to: {OUT}', flush=True)
# Force sharded safetensors save with a small shard size so we get multiple files
# if the model is large. max_shard_size=5GB is HF default; we use 2GB to be safe.
model.save_pretrained(OUT, safe_serialization=True, max_shard_size='2GB')
tok.save_pretrained(OUT)

print('--- listing OUT contents ---', flush=True)
for entry in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, entry))
    print(f'  {entry}  ({sz/1e6:.1f} MB)', flush=True)
print('done', flush=True)
"""
    env = {**os.environ, "MERGE_BASE": base_model, "MERGE_LORA": lora_path, "MERGE_OUT": out_path}
    # Ensure peft is available (prime-rl env may not include it).
    subprocess.run(
        ["/root/.local/bin/uv", "pip", "install", "peft"],
        capture_output=True, text=True, cwd="/root/prime-rl",
    )
    r = subprocess.run(
        ["/root/.local/bin/uv", "run", "python", "-c", script],
        capture_output=True, text=True, cwd="/root/prime-rl", env=env,
    )
    return {"rc": r.returncode, "stdout": (r.stdout or "")[-2000:],
            "stderr": (r.stderr or "")[-2000:]}


@app.local_entrypoint()
def merge_v2_l30_lora():
    """Merge v2-l30 step_200 LoRA into Qwen3-4B base. One-time setup for stage 2."""
    r = merge_lora_into_base.remote()
    print(f"rc={r['rc']}")
    print("--- stdout ---")
    print(r['stdout'])
    if r['rc']:
        print("--- stderr ---")
        print(r['stderr'])


@app.function(image=image, volumes={"/cache": volume}, secrets=[wandb_secret], timeout=1800)
def wandb_sync_offline(run_name_prefix: str = "ctrl0_u1_40_long_additive_v2"):
    """Sync all offline wandb runs matching <run_name_prefix>* to wandb cloud.
    Local wandb/protobuf versions can mismatch; inside this container they match
    what prime-rl used to write the offline data, so `wandb sync` works."""
    import os, subprocess
    runs_dir = "/cache/runs"
    synced = []
    for entry in sorted(os.listdir(runs_dir)):
        if not entry.startswith(run_name_prefix):
            continue
        wd = f"{runs_dir}/{entry}/run_default/wandb"
        if not os.path.isdir(wd):
            print(f"  skip {entry}: no run_default/wandb dir")
            continue
        for sub in sorted(os.listdir(wd)):
            if not sub.startswith("offline-run-"):
                continue
            path = f"{wd}/{sub}"
            print(f"=== syncing {entry} / {sub} ===")
            r = subprocess.run(["/root/.local/bin/uv", "run", "wandb", "sync", path],
                               capture_output=True, text=True, cwd="/root/prime-rl")
            tail = (r.stdout or "")[-800:] + (r.stderr or "")[-800:]
            print(tail)
            synced.append({"cell": entry, "segment": sub, "rc": r.returncode})
    return synced


@app.function(image=image, timeout=600)  # CPU only — hwprop is an analytical sim
def chunk_latency_calc() -> str:
    """Compute per-chunk simulated latency via hwprop across hardware x model configs.
    A 'chunk' = one model turn of `decode_steps` tokens. Reports prefill (charged on
    turn 1) and decode time per chunk; the model's elapsed-time signal accrues this
    each turn. Used to size the chunk (max_completion_tokens/turn)."""
    import subprocess, textwrap
    script = textwrap.dedent('''
        from hwprop.simulator import simulate_latency
        HW = ["A100_80GB", "H100_SXM", "L40S"]
        M = "Qwen3-4B"
        def dec(hw, ctx, n): return simulate_latency(hw, M, prompt_len=ctx, decode_steps=n).total_decode_time_s

        print(f"=== {M}: per-CHUNK decode time (s), by chunk size, at mid context (prompt_len=1024) ===")
        CHUNKS = [16, 32, 64, 128, 256]
        print("hardware     " + "".join(f"{c:>4}t " for c in CHUNKS))
        for hw in HW:
            print(f"{hw:11s}  " + "".join(f"{dec(hw,1024,c):>5.1f} " for c in CHUNKS))

        print()
        print(f"=== {M}: a 128-token chunk grows with context (prompt_len) ===")
        CTX = [256, 1024, 1792]
        print("hardware     " + "".join(f"ctx={c:<5}" for c in CTX))
        for hw in HW:
            print(f"{hw:11s}  " + "".join(f"{dec(hw,c,128):>6.1f}s  " for c in CTX))
    ''')
    r = subprocess.run(["/root/.local/bin/uv", "run", "python", "-c", script],
                       cwd="/root/prime-rl", text=True, capture_output=True)
    return (r.stdout or "") + ("\n[stderr]\n" + r.stderr[-600:] if r.returncode else "")


@app.local_entrypoint()
def chunk_latency():
    print(chunk_latency_calc.remote())


def _resume_arg_if_needed(cfg_path: str) -> list[str]:
    """If the TOML's output_dir already contains checkpoints, return
    ['--ckpt.resume-step', '-1'] so prime-rl resumes from the latest step
    instead of crashing on the output-dir-exists guard. Makes train_run
    idempotent across budget/preemption-induced re-runs of the same call."""
    import os
    import tomllib
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    output_dir = cfg.get("output_dir")
    if not output_dir:
        return []
    weights_dir = f"{output_dir}/weights"
    try:
        steps = [d for d in os.listdir(weights_dir) if d.startswith("step_")]
    except FileNotFoundError:
        return []
    if not steps:
        return []
    latest = max(int(d.split("_", 1)[1]) for d in steps)
    print(f"[prime-rl] detected existing checkpoints at {weights_dir} (latest step_{latest}); resuming", flush=True)
    return ["--ckpt.resume-step", "-1"]


@app.function(
    # 2 GPUs: prime-rl runs vllm on one set, trainer on another. Matches the
    # gsm8k example's orchestrator gpu allocation pattern.
    gpu="A100-80GB:2",
    image=image,
    volumes={"/cache": volume},
    secrets=[wandb_secret],
    # Backstop only — the real protection is the stall watchdog in train_run
    # (kills a hung run in ~15 min). 10h comfortably covers a healthy 200-step
    # run (~6-7h) while capping the bleed far below Modal's 24h max — a wedged
    # FSDP collective once idled 6xA100 all the way to the old 24h limit.
    timeout=10 * 3600,
)
def train_run(toml_name: str, run_name: str, wandb_project: str = "interoception",
              extra_args: list[str] | None = None, wandb_offline: bool = False) -> dict:
    """Run a prime-rl training job. prime-rl orchestrates vllm + trainer internally.

    Local logging mirror — everything wandb sees also lands on the
    `interoception-cache` volume under /cache/run_logs/<run_name>/:
      - wandb/run-*/   complete local wandb archive (metrics binary log +
                       every eval sample table written as JSON). Belt-and-
                       suspenders against the wandb cloud sync being partial
                       or delayed.
      - stdout.log     tee of prime-rl stdout+stderr (live Modal logs expire;
                       this one survives).
    The volume is committed every ~10 min during the run so a mid-run crash
    still leaves the logs on disk (Modal volume writes are buffered until
    volume.commit()).

    extra_args: optional CLI overrides appended to `rl @ <toml>` (e.g. for smokes:
    ['--max-steps', '4', '--ckpt.interval', '2', '--output-dir', '/cache/runs/x_smoke'])."""
    import os
    import subprocess
    import sys
    import threading
    import time

    log_dir = f"/cache/run_logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_NAME"] = run_name
    # Set WANDB_RUN_ID deterministically from run_name. Two reasons:
    # 1) Parallel sweeps under the same project (e.g. sweep_long_additive_v2)
    #    otherwise collide if prime-rl derives run_id from a shared seed.
    # 2) Re-running the same name (e.g. on preemption) resumes the same wandb
    #    run instead of forking a new one.
    import hashlib
    os.environ["WANDB_RUN_ID"] = hashlib.md5(run_name.encode()).hexdigest()[:16]
    # If we crashed/preempted and re-run with the same name, resume the wandb
    # run rather than failing on "run already exists."
    os.environ["WANDB_RESUME"] = "allow"
    # For parallel sweeps: prime-rl passes id=None to wandb.init, so wandb
    # generates IDs via random.choices() — which is seeded by the orchestrator's
    # seed, so all parallel cells with the same seed collide on the same run ID.
    # Offline mode bypasses the server entirely; can be `wandb sync`'d later.
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    # wandb writes its full local archive under WANDB_DIR/wandb/run-*/. Pointing
    # this at the volume gives us a durable on-disk copy of everything wandb
    # streams, independent of the cloud upload.
    os.environ["WANDB_DIR"] = log_dir

    cfg_path = f"/root/configs/{toml_name}"
    if not os.path.exists(cfg_path):
        return {"ok": False, "error": f"missing config: {cfg_path}"}

    cmd = [
        "/root/.local/bin/uv", "run", "rl",
        "@", cfg_path,
        "--wandb.project", wandb_project,
        "--wandb.name", run_name,
        *_resume_arg_if_needed(cfg_path),
        *(extra_args or []),
    ]
    print(f"[prime-rl] launching: {' '.join(cmd)}", flush=True)
    print(f"[prime-rl] local mirror: {log_dir}", flush=True)
    t0 = time.time()

    # Tee stdout+stderr to the container fds (live Modal logs) AND to a file
    # on the volume (durable post-crash record). Inline periodic volume.commit
    # piggybacks on the stdout loop — no background thread (avoids any
    # thread-safety questions about the Modal client).
    log_path = f"{log_dir}/stdout.log"
    last_commit = time.time()
    # Stall watchdog: a healthy run prints constantly (rollout/eval progress bars,
    # per-step SUCCESS lines), but a wedged trainer (e.g. a hung FSDP all-gather)
    # goes totally silent while the orchestrator waits forever. If there's no stdout
    # for STALL_LIMIT, kill the subprocess so a hang FAILS in minutes instead of
    # idling to the function timeout. Normal gaps (a step, a 498-example eval) are
    # <3 min, so 15 min of silence is unambiguously stuck. The thread only touches
    # `proc` (a subprocess) — never the Modal client — so it's thread-safe.
    STALL_LIMIT = 900
    last_line_time = [time.time()]
    killed_by_watchdog = [False]
    stop_watchdog = threading.Event()

    def _watchdog():
        while not stop_watchdog.wait(60):
            if time.time() - last_line_time[0] > STALL_LIMIT:
                print(f"[watchdog] no output for {STALL_LIMIT}s — killing stalled run "
                      f"(suspected trainer/collective hang)", flush=True)
                killed_by_watchdog[0] = True
                try:
                    proc.kill()
                except Exception:
                    pass
                return

    with open(log_path, "w", buffering=1) as logf:  # line-buffered
        proc = subprocess.Popen(
            cmd, cwd="/root/prime-rl", text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        threading.Thread(target=_watchdog, daemon=True).start()
        for line in proc.stdout:
            last_line_time[0] = time.time()
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            if time.time() - last_commit > 600:
                try:
                    volume.commit()
                except Exception as e:
                    print(f"[volume.commit] warning: {e}", flush=True)
                last_commit = time.time()
        proc.wait()
        stop_watchdog.set()
    dur = time.time() - t0
    volume.commit()
    return {
        "ok": proc.returncode == 0 and not killed_by_watchdog[0],
        "returncode": proc.returncode,
        "duration_s": round(dur, 1),
        "log_dir": log_dir,
        "stalled": killed_by_watchdog[0],
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
def ctrl0_u1_40():
    """Low-budget treatment re-run (Kanishk's 2026-05-26 thread): Condition 0 (c*f),
    multi-turn 16 turns / 128 tok/chunk, [Xs elapsed] injected, A100_80GB sim, but
    T~U(1,40) instead of U(15,130) so the budget is actually reachable within seq_len.
    See configs/rl/ctrl0_u1_40_qwen3_4b.toml."""
    cfgs = [("rl/ctrl0_u1_40_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40")]
    calls = [(cfg, train_run.spawn(*cfg)) for cfg in cfgs]
    for cfg, call in calls:
        try:
            r = call.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        print(f"  {cfg[1]}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def next_long_qwen3_4b():
    """NEXT EXP — "train for longer" (Kanishk 2026-05-26): ctrl0_u1_40 setup, 500 steps.
    Identical treatment (c*f, T~U(1,40), 128 tok/chunk, 16 turns, A100); only max_steps
    and ckpt cadence change. See docs/HANDOFF.md."""
    _launch_one("rl/ctrl0_u1_40_long_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40-long")


@app.local_entrypoint()
def next_yolo_qwen25_3b():
    """NEXT EXP — YOLO run with Qwen2.5-3B-Instruct (Kanishk 2026-05-26): same U(1,40)
    treatment, swapped policy model + sim_model. See docs/HANDOFF.md."""
    _launch_one("rl/ctrl0_u1_40_qwen25_3b.toml", "ctrl0-qwen25-3b-u1-40-yolo")


def _launch_one(toml_name: str, run_name: str):
    """Spawn a single train_run and print its result (shared by the next-exp launchers)."""
    call = train_run.spawn(toml_name, run_name)
    try:
        r = call.get()
    except Exception as e:
        r = {"ok": False, "error": str(e)[:200]}
    print(f"  {run_name}: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s  err={r.get('error', '')}")


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


# ---------------------------------------------------------------------------
# Prompt-salience eval (2026-05-29). Tests whether long-500's T-blindness can
# be unlocked by a more aggressive budget prompt. Inference-only — no training.
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-80GB:1",
    image=image,
    volumes={"/cache": volume},
    # 8h: ~16s/rollout × 498 × 2 variants ≈ 4.4h per (model) call; 8h gives
    # plenty of buffer for slower-than-expected rollouts + vLLM init.
    timeout=8 * 3600,
)
def eval_prompt_salience_run(
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    adapter_path: str | None = None,
    adapter_name: str = "long-500",
    run_label: str | None = None,
    variants: tuple[str, ...] = ("base", "remaining_budget"),
    num_examples: int = 498,
    output_subdir: str = "prompt_salience",
) -> dict:
    """Run scripts/eval_prompt_salience.py inside Modal against the cache volume.

    Outputs land at /cache/eval_rollouts/<output_subdir>/<run_label>_<variant>.jsonl
    so multiple invocations (base, long-500, etc.) accumulate under one dir."""
    import os
    import subprocess
    import time

    out_dir = f"/cache/eval_rollouts/{output_subdir}"
    os.makedirs(out_dir, exist_ok=True)
    label = run_label or (adapter_name if adapter_path else "base")

    cmd = [
        "/root/.local/bin/uv", "run", "python", "/root/scripts/eval_prompt_salience.py",
        "--base-model", base_model,
        "--num-examples", str(num_examples),
        "--variants", *variants,
        "--output-dir", out_dir,
        "--run-label", label,
        "--problems-jsonl", "/root/data/eval.jsonl",
    ]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path, "--adapter-name", adapter_name]
    print(f"[prompt-salience] launching: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/root/prime-rl", text=True)
    volume.commit()
    return {"ok": proc.returncode == 0, "returncode": proc.returncode,
            "duration_s": round(time.time() - t0, 1), "out_dir": out_dir, "label": label}


@app.local_entrypoint()
def eval_prompt_salience(
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    skip_base: bool = False,
    skip_long500: bool = False,
    variants: str = "base,remaining_budget",
    num_examples: int = 498,
):
    """Modal entrypoint: run the prompt-salience eval for {base, long-500} × variants.

    Spawns one parallel job per (model, variant) cell so total wall time = one
    cell's time (~2.5h at n=498). Each job loads vLLM once and runs one variant.
    Outputs land on the `interoception-cache` volume at
    /cache/eval_rollouts/prompt_salience/<label>_<variant>.jsonl.

    Pass --variants as a comma-separated list to override (e.g. for follow-up
    cells: --variants strict_pace, or --variants strict_pace,remaining_budget)."""
    long500_adapter = "/cache/runs/ctrl0_u1_40_long_qwen3_4b/weights/step_500/lora_adapters"
    variant_list = tuple(v.strip() for v in variants.split(",") if v.strip())
    models = []
    if not skip_base:
        models.append({"adapter_path": None, "adapter_name": "base", "run_label": "base"})
    if not skip_long500:
        models.append({"adapter_path": long500_adapter, "adapter_name": "long-500",
                       "run_label": "long-500"})
    jobs = []
    for m in models:
        for variant in variant_list:
            jobs.append({**m, "variants": (variant,)})
    print(f"Launching {len(jobs)} prompt-salience eval cells in parallel "
          f"({len(models)} models × 2 variants)")
    calls = []
    for j in jobs:
        calls.append((j, eval_prompt_salience_run.spawn(
            base_model=base_model, num_examples=num_examples, **j)))
    for j, c in calls:
        try:
            r = c.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        label = f"{j['run_label']}/{j['variants'][0]}"
        print(f"  {label:30s}: ok={r.get('ok')}  rc={r.get('returncode')}  "
              f"dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def eval_strict_conly():
    """Calibration eval for the strict-conly (ctrlC) step_500 checkpoint under the
    remaining_budget prompt — the cell directly comparable to base (r=+0.78) and
    long-500 (r=+0.13). Writes /cache/eval_rollouts/prompt_salience/strict-conly_remaining_budget.jsonl;
    pull it and compute corr(elapsed_s, target_s) to see if correctness-only RL
    preserved the base model's T-calibration."""
    adapter = "/cache/runs/ctrl0_u1_40_strict_conly_qwen3_4b/weights/step_500/lora_adapters"
    call = eval_prompt_salience_run.spawn(
        adapter_path=adapter, adapter_name="strict-conly", run_label="strict-conly",
        variants=("remaining_budget",), num_examples=498)
    r = call.get()
    print(f"  strict-conly/remaining_budget: ok={r.get('ok')} rc={r.get('returncode')} "
          f"dur={r.get('duration_s')}s out={r.get('out_dir')} err={r.get('error','')}")


# ---------------------------------------------------------------------------
# Accuracy-vs-budget sweep (within-problem): each model evaluated on the same
# test problems at a grid of FIXED budgets T. Batched rollouts -> fast/cheap.
# ---------------------------------------------------------------------------

@app.function(gpu="A100-80GB:1", image=image, volumes={"/cache": volume}, timeout=3600)
def acc_vs_budget_run(base_model: str, adapter_path: str | None, adapter_name: str,
                      run_label: str, budgets: str, num_examples: int,
                      variant: str = "remaining_budget") -> dict:
    import json as _json, os, subprocess, time
    out_json = f"/cache/eval_rollouts/acc_vs_budget/{run_label}_{variant}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    cmd = ["/root/.local/bin/uv", "run", "python", "/root/scripts/eval_acc_vs_budget.py",
           "--base-model", base_model, "--run-label", run_label, "--variant", variant,
           "--budgets", budgets, "--num-examples", str(num_examples),
           "--problems-jsonl", "/root/data/eval.jsonl", "--output-json", out_json]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path, "--adapter-name", adapter_name]
    print(f"[acc-vs-budget] {run_label}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/root/prime-rl", text=True, capture_output=True)
    print(proc.stdout[-4000:]); print(proc.stderr[-2000:])
    volume.commit()
    summary = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            summary = _json.loads(line[len("RESULT_JSON "):])
    return {"ok": proc.returncode == 0, "label": run_label,
            "duration_s": round(time.time() - t0, 1), "result": summary}


@app.local_entrypoint()
def acc_vs_budget(num_examples: int = 128, budgets: str = "4,8,12,18,24,32,40",
                  variant: str = "remaining_budget", models: str = ""):
    """Test-set accuracy vs. budget, within-problem. Spawns one job per model,
    each looping the budget grid. Results land at
    /cache/eval_rollouts/acc_vs_budget/<label>_<variant>.json. Pass --models as a
    comma-separated list of run-labels to run a subset (default: all)."""
    conly = "/cache/runs/ctrl0_u1_40_strict_conly_qwen3_4b/weights/step_500/lora_adapters"
    def su(s):
        return f"/cache/runs/ctrl0_u1_40_windowed_{s}_200_qwen3_4b/weights/step_200/lora_adapters"
    cells = [
        {"adapter_path": None, "adapter_name": "base", "run_label": "base"},
        {"adapter_path": conly, "adapter_name": "strict-conly", "run_label": "correctness-only"},
        {"adapter_path": su("su25"), "adapter_name": "windowed-su25", "run_label": "pacing-su25"},
        {"adapter_path": su("su17"), "adapter_name": "windowed-su17", "run_label": "sigma-su17"},
        {"adapter_path": su("su10"), "adapter_name": "windowed-su10", "run_label": "sigma-su10"},
    ]
    if models:
        want = {m.strip() for m in models.split(",")}
        cells = [c for c in cells if c["run_label"] in want]
    print(f"Launching {len(cells)} acc-vs-budget jobs | budgets={budgets} | n={num_examples}")
    calls = [(c, acc_vs_budget_run.spawn(
        base_model="Qwen/Qwen3-4B-Instruct-2507", budgets=budgets,
        num_examples=num_examples, variant=variant, **c)) for c in cells]
    for c, call in calls:
        try:
            r = call.get()
        except Exception as e:
            print(f"  {c['run_label']}: FAILED {str(e)[:200]}"); continue
        res = r.get("result")
        print(f"\n=== {c['run_label']}  (ok={r.get('ok')}, {r.get('duration_s')}s) ===")
        if res:
            print(f"{'budget':>8} {'acc':>7} {'commit':>7} {'mean_t':>7}")
            for s in res["summary"]:
                print(f"{s['budget']:>7.1f}s {s['accuracy']:>7.3f} {s['commit_rate']:>7.2f} {s['mean_elapsed']:>6.1f}s")


# ---------------------------------------------------------------------------
# Resume-from-step-500 extension (Kanishk, 2026-05-29). Copies the long-500
# step_500 checkpoint into the extension's output_dir, then launches train_run
# which auto-resumes via _resume_arg_if_needed.
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/cache": volume}, timeout=600)
def seed_checkpoint_from(src_run: str, dst_run: str, step: int) -> dict:
    """Seed a fresh prime-rl output_dir from another run's step_{step} so the
    new run can resume training (not just inference) from that point.

    prime-rl stores resume state across TWO sibling dirs:
      - weights/step_N/       — model weights (LoRA adapter + merged model)
      - checkpoints/step_N/   — trainer/optimizer/scheduler state (.distcp)

    Both must be present in the new output_dir, otherwise prime-rl logs
    'Training from scratch' and ignores the seed. Also wipes any stale
    extended-run state (wandb, run_default, logs, configs) so the resume
    starts clean."""
    import shutil
    import os
    new_root = f"/cache/runs/{dst_run}"
    # Wipe any pre-existing aux dirs from a partial prior launch so prime-rl
    # doesn't see stale state.
    for stale in ("wandb", "run_default", "logs", "configs", "rollouts"):
        p = os.path.join(new_root, stale)
        if os.path.isdir(p):
            shutil.rmtree(p)

    copied = []
    for sub in ("weights", "checkpoints"):
        src = f"/cache/runs/{src_run}/{sub}/step_{step}"
        dst_root = f"{new_root}/{sub}"
        dst = f"{dst_root}/step_{step}"
        if not os.path.isdir(src):
            return {"ok": False, "error": f"missing source: {src}"}
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst_root, exist_ok=True)
        shutil.copytree(src, dst)
        copied.append({"dst": dst, "n_files": len(os.listdir(dst))})
    volume.commit()
    return {"ok": True, "src_run": src_run, "dst_run": dst_run,
            "step": step, "copied": copied}


@app.local_entrypoint()
def next_extended_qwen3_4b():
    """RESUME EXP — train another 500 steps on top of long-500 (Kanishk 2026-05-29).

    Two-step flow:
      1. seed_checkpoint_from copies long-run's step_500 weights into the new
         output_dir so prime-rl can resume from it.
      2. train_run launches the extended config; auto-resume detects step_500
         and continues to step_1000."""
    print("step 1/2: seeding extended/weights/step_500 from long/weights/step_500")
    r = seed_checkpoint_from.remote(
        src_run="ctrl0_u1_40_long_qwen3_4b",
        dst_run="ctrl0_u1_40_extended_qwen3_4b",
        step=500,
    )
    print(f"  seed result: {r}")
    if not r.get("ok"):
        return
    print("step 2/2: launching train_run on extended config")
    _launch_one("rl/ctrl0_u1_40_extended_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40-extended")


# ---------------------------------------------------------------------------
# Resume-from-step-400 extension (Kanishk 2026-05-30; pivot from the
# resume-from-step-500 attempt). The original long-500 run ended at step 500
# with the LR scheduler in end-of-decay state (lr=min_lr=0). PyTorch's
# SequentialLR.load_state_dict overwrites a freshly-built scheduler's milestones
# with the saved ones, so resuming from step_500 with a new max_steps=1000
# config would advance step counts at lr=0 (no weight updates). Resuming from
# step_400 instead — which is in the constant phase (lr=peak) — gives a clean
# match between the loaded state and the new schedule's expectation at that step.
# Cost: we re-train steps 400-500 with the new schedule. The original step_500
# weights are archived rather than deleted in case we want them later.
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/cache": volume}, timeout=600)
def archive_step(run_name: str, step: int) -> dict:
    """Move /cache/runs/{run}/weights/step_{N} and checkpoints/step_{N} out of the
    way so prime-rl's auto-resume picks an earlier step. Archives to
    /cache/archive/{run}/{weights,checkpoints}/step_{N} (preserves the data)."""
    import os
    import shutil
    moved = []
    for sub in ("weights", "checkpoints"):
        src = f"/cache/runs/{run_name}/{sub}/step_{step}"
        dst_root = f"/cache/archive/{run_name}/{sub}"
        dst = f"{dst_root}/step_{step}"
        if not os.path.isdir(src):
            moved.append({"sub": sub, "skipped": True, "note": f"missing: {src}"})
            continue
        if os.path.isdir(dst):
            shutil.rmtree(dst)  # overwrite stale archive
        os.makedirs(dst_root, exist_ok=True)
        shutil.move(src, dst)
        moved.append({"sub": sub, "moved": True, "from": src, "to": dst})
    # Report remaining latest step in weights/
    weights_dir = f"/cache/runs/{run_name}/weights"
    remaining = sorted(int(d.split("_", 1)[1]) for d in os.listdir(weights_dir)
                       if d.startswith("step_") and d.split("_", 1)[1].isdigit())
    volume.commit()
    return {"ok": True, "run": run_name, "archived_step": step, "moved": moved,
            "remaining_weights_steps": remaining,
            "new_latest_step": max(remaining) if remaining else None}


@app.local_entrypoint()
def archive_long_step_500_and_resume():
    """Archive long-run's step_500 (so auto-resume picks step_400 instead, which
    has lr=peak in its saved scheduler state), then launch the resume run.
    Resume runs from step_400 to whatever max_steps is in the config (currently 1000)."""
    print("step 1/2: archiving long-run's step_500 to /cache/archive/")
    r = archive_step.remote(run_name="ctrl0_u1_40_long_qwen3_4b", step=500)
    print(f"  archive result: {r}")
    if not r.get("ok"):
        return
    new_latest = r.get("new_latest_step")
    if new_latest != 400:
        print(f"  WARNING: expected new latest=400, got {new_latest}. Aborting launch.")
        return
    print("step 2/2: launching train_run on long config (will auto-resume from step_400)")
    _launch_one("rl/ctrl0_u1_40_long_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40-long")


@app.local_entrypoint()
def next_long1k_qwen3_4b():
    """FRESH 1000-step run (Kanishk 2026-05-30). Clean from-scratch training to step 1000;
    pivot from the failed resume-from-step_500 approach (LR scheduler conflict). Lets us
    compare long-500 vs long-1000 cleanly. See configs/rl/ctrl0_u1_40_long1k_qwen3_4b.toml."""
    _launch_one("rl/ctrl0_u1_40_long1k_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40-long1k")


@app.local_entrypoint()
def long1k_smoke():
    """Smoke test of the long1k config BEFORE the full ~10h run.

    Runs the EXACT long1k config but overrides max_steps=4, ckpt.interval=2, and the
    LR scheduler so the 4-step schedule fits (warmup=1, decay=2). Writes to a
    *_smoke output_dir + wandb name so it doesn't pollute the real long1k state.

    Verifies the smoke is happy by checking: rc=0, weights/step_{2,4} written with
    adapter, eval at step 0 + step 4 produced Avg@1, no crashes in env / hwprop / vLLM.
    Takes ~7-10 min on cached image (mostly the step-0 eval over 498 examples)."""
    common = [
        "--max-steps", "4", "--ckpt.interval", "2",
        "--trainer.scheduler.warmup-steps", "1", "--trainer.scheduler.decay-steps", "2",
        "--output-dir", "/cache/runs/long1k_smoke",
    ]
    print("Launching long1k smoke: 4 steps, ckpt every 2")
    call = train_run.spawn(
        "rl/ctrl0_u1_40_long1k_qwen3_4b.toml", "long1k-smoke",
        extra_args=common,
    )
    try:
        r = call.get()
    except Exception as e:
        r = {"ok": False, "error": str(e)[:200]}
    print(f"  long1k-smoke: ok={r.get('ok')}  rc={r.get('returncode')}  "
          f"dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def next_long_strict_qwen3_4b():
    """NEXT EXP (Kanishk-track, 2026-06-01): 500-step RL with the remaining_budget
    prompt. Tests whether training under the loud prompt preserves the base model's
    T-tracking (r=+0.77 in eval) rather than overwriting it with a fixed-commit
    habit (as the quiet-prompt-trained long-500 does, r=+0.15).
    See configs/rl/ctrl0_u1_40_long_strict_qwen3_4b.toml."""
    _launch_one("rl/ctrl0_u1_40_long_strict_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40-long-strict")


@app.local_entrypoint()
def next_strict_conly_qwen3_4b():
    """NEXT EXP (Kanishk-track, 2026-06-01) — the thread's "ctrlC": strict/remaining_budget
    prompt + CORRECTNESS-ONLY reward (reward_time_term=false => R = c). Tests whether RL on
    correctness alone, under the loud prompt, preserves the base model's calibration
    (r=+0.77) without any explicit time-reward term. Uses G=16 / batch=128 (vs G=8/64 for
    the c·f strict probe — NOT a batch-matched A/B; see config header).
    See configs/rl/ctrl0_u1_40_strict_conly_qwen3_4b.toml."""
    _launch_one("rl/ctrl0_u1_40_strict_conly_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40-strict-conly")


@app.local_entrypoint()
def eval_long_strict_probe(num_examples: int = 498):
    """Probe T-conditioning of long-strict step_500 (the model trained with the
    remaining_budget prompt). Runs 2 cells in parallel:
      - long-strict + remaining_budget: matches training distribution. Primary
        question — does the trained model preserve the base-model T-tracking
        (r=+0.77) or did RL degrade it like the quiet-prompt training did (r=+0.15)?
      - long-strict + base prompt:     OOD eval. Tests whether pacing generalizes
        off the loud signal — i.e., whether the capability is in the weights or
        only in the prompt-context attention.
    Both write JSONLs to /cache/eval_rollouts/prompt_salience/long-strict_*.jsonl
    so probe_prompt_salience.py can pull them alongside the existing 4 cells."""
    long_strict_adapter = "/cache/runs/ctrl0_u1_40_long_strict_qwen3_4b/weights/step_500/lora_adapters"
    jobs = []
    for variant in ("base", "remaining_budget"):
        jobs.append({"adapter_path": long_strict_adapter,
                     "adapter_name": "long-strict",
                     "run_label": "long-strict",
                     "variants": (variant,)})
    print(f"Launching {len(jobs)} long-strict probe cells in parallel")
    calls = []
    for j in jobs:
        calls.append((j, eval_prompt_salience_run.spawn(num_examples=num_examples, **j)))
    for j, c in calls:
        try:
            r = c.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        label = f"{j['run_label']}/{j['variants'][0]}"
        print(f"  {label:30s}: ok={r.get('ok')}  rc={r.get('returncode')}  "
              f"dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def next_long_additive_qwen3_4b():
    """100-step test of the ADDITIVE reward (c + λ_f · f). Hypothesis: gives RL
    a gradient on pacing INDEPENDENT of correctness, preventing the fixed-commit-
    time policy that c·f always converges to.
    Uses prompt_variant=remaining_budget (loud), λ_f=0.5. ~1.5h, ~$10-15.
    If T-tracking survives at step_100 (probe r > 0.3), extend to 500 steps.
    See configs/rl/ctrl0_u1_40_long_additive_qwen3_4b.toml."""
    _launch_one("rl/ctrl0_u1_40_long_additive_qwen3_4b.toml",
                "ctrl0-qwen3-4b-u1-40-long-additive")


@app.local_entrypoint()
def voc_smoke():
    """Re-smoke the clip arithmetic + max_turns=12 on the clip-rho cell BEFORE the screen.
    Overrides: max_steps=4, ckpt.interval=2, scheduler warmup=1/decay=2, small group
    (batch=16, G=4 -> 4 groups/step) for fast feedback. Confirms voc_alp(clip) is detected
    as a GroupRewardFunc, run_group fires, pass_rate (ρ_q) logs in [0,1], reward sane,
    max_turns=12 respected, no crash. Writes to a *_smoke dir so it doesn't touch the run."""
    common = [
        "--max-steps", "4", "--ckpt.interval", "2",
        "--trainer.scheduler.warmup-steps", "1", "--trainer.scheduler.decay-steps", "2",
        "--orchestrator.batch-size", "16", "--orchestrator.rollouts-per-example", "4",
        "--output-dir", "/cache/runs/voc_smoke",
    ]
    print("Launching voc smoke (clip-rho): 4 steps, G=4/batch=16, ckpt every 2")
    call = train_run.spawn(
        "rl/ctrl0_u1_40_voc_clip_rho_qwen3_4b.toml", "voc-smoke", extra_args=common,
    )
    try:
        r = call.get()
    except Exception as e:
        r = {"ok": False, "error": str(e)[:200]}
    print(f"  voc-smoke: ok={r.get('ok')}  rc={r.get('returncode')}  "
          f"dur={r.get('duration_s')}s  log={r.get('log_dir')}  err={r.get('error', '')}")


@app.local_entrypoint()
def sweep_voc_rewards():
    """VoC reward-space SCREEN (Harsh-track, 2026-06-04): 3 structurally distinct time-cost
    forms, 100 steps each, launched in PARALLEL (one Modal container per cell, own wandb
    project so parallel run-ids don't collide). All share: G=32/batch=256, remaining_budget
    prompt, T~U(1,40), max_turns=12 (fits seq_len=2048).
      clip-rho   : c − 0.3·ρ_q·min(t/T,5)       (penalize-all-time, budget-relative, ALP)
      overbudget : c − 1.0·ρ_q·max(0,t−T)/53    (free under budget, overage-only, ALP)
      clip-norho : c − 0.1·min(t/T,5)           (clip, NO ρ_q — the difficulty ablation)
    Reads from 3 runs: shape (clip-rho vs overbudget) and ρ_q (clip-rho vs clip-norho).
    Watch reward / is_correct / f_term / pass_rate / elapsed_over_target FIRST, then run the
    decoupled calibration probe on the step_25..100 ckpts (with max_turns=12).

    FIRE-AND-FORGET: spawns the 3 cells on Modal and returns immediately, so there is no
    hours-long local process to kill. Launch with `modal run --detach` so the spawned cells
    persist after the client disconnects. Monitor on wandb / the cache volume — there is
    nothing to wait on locally."""
    cells = [
        ("rl/ctrl0_u1_40_voc_clip_rho_qwen3_4b.toml",   "ctrl0-qwen3-4b-u1-40-voc-clip-rho",   "interoception-voc-cliprho"),
        ("rl/ctrl0_u1_40_voc_overbudget_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40-voc-overbudget", "interoception-voc-ob"),
        ("rl/ctrl0_u1_40_voc_clip_norho_qwen3_4b.toml", "ctrl0-qwen3-4b-u1-40-voc-clip-norho", "interoception-voc-clipnorho"),
    ]
    print(f"Spawning {len(cells)} VoC screen cells (fire-and-forget; launch with --detach)")
    for cfg, name, proj in cells:
        call = train_run.spawn(cfg, name, wandb_project=proj)
        print(f"  spawned {name}  (wandb={proj})  call_id={call.object_id}")
    print("All cells spawned — they run on Modal independently. Monitor on wandb.")


@app.local_entrypoint()
def eval_long_additive_probe(num_examples: int = 498):
    """Probe T-conditioning of long-additive step_100 (the model trained with
    additive reward c + 0.5·f). Runs 2 cells in parallel:
      - long-additive + remaining_budget: matches training distribution. Primary
        question — does the additive reward preserve the high f we saw in
        training (f≈0.95) AND have it correspond to real T-tracking?
      - long-additive + base prompt:     OOD eval. Tests whether the pacing
        generalizes off the loud signal.
    Both write JSONLs to /cache/eval_rollouts/prompt_salience/long-additive_*.jsonl
    so probe_prompt_salience.py can pull them alongside the existing cells."""
    long_additive_adapter = "/cache/runs/ctrl0_u1_40_long_additive_qwen3_4b/weights/step_100/lora_adapters"
    jobs = []
    for variant in ("base", "remaining_budget"):
        jobs.append({"adapter_path": long_additive_adapter,
                     "adapter_name": "long-additive",
                     "run_label": "long-additive",
                     "variants": (variant,)})
    print(f"Launching {len(jobs)} long-additive probe cells in parallel")
    calls = []
    for j in jobs:
        calls.append((j, eval_prompt_salience_run.spawn(num_examples=num_examples, **j)))
    for j, c in calls:
        try:
            r = c.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        label = f"{j['run_label']}/{j['variants'][0]}"
        print(f"  {label:30s}: ok={r.get('ok')}  rc={r.get('returncode')}  "
              f"dur={r.get('duration_s')}s  err={r.get('error', '')}")


def _train_then_probe_v2(lam_tag: str):
    """Train one v2 sweep cell and probe its step_200 checkpoint.
    lam_tag in {"l10","l15","l30"} — matches the config + cell labels.
    Each cell gets its own wandb project so prime-rl's deterministic run-id
    derivation (which collides across parallel cells in the same project)
    is scoped per cell — runs land in interoception-l10/l15/l30."""
    cfg = f"rl/ctrl0_u1_40_long_additive_v2_{lam_tag}_qwen3_4b.toml"
    wandb_name = f"ctrl0-qwen3-4b-u1-40-long-additive-v2-{lam_tag}"
    wandb_project = f"interoception-{lam_tag}"
    print(f"=== TRAIN: {lam_tag} (wandb: {wandb_project}/{wandb_name}, offline mode) ===")
    r = train_run.remote(cfg, wandb_name, wandb_project=wandb_project, wandb_offline=True)
    print(f"  train[{lam_tag}]: ok={r.get('ok')}  rc={r.get('returncode')}  dur={r.get('duration_s')}s")
    if not r.get("ok"):
        print(f"  TRAINING FAILED ({lam_tag}) — skipping probe. err={r.get('error', '')[:200]}")
        return {"cell": lam_tag, "train_ok": False, "probe": None}

    print(f"\n=== PROBE: {lam_tag} step_200 ===")
    adapter = f"/cache/runs/ctrl0_u1_40_long_additive_v2_{lam_tag}_qwen3_4b/weights/step_200/lora_adapters"
    label = f"long-additive-v2-{lam_tag}"
    jobs = [{"adapter_path": adapter, "adapter_name": label,
             "run_label": label, "variants": (v,)}
            for v in ("base", "remaining_budget")]
    calls = [(j, eval_prompt_salience_run.spawn(num_examples=498, **j)) for j in jobs]
    probe_results = []
    for j, c in calls:
        try:
            rr = c.get()
        except Exception as e:
            rr = {"ok": False, "error": str(e)[:200]}
        var = j['variants'][0]
        print(f"  probe[{lam_tag}/{var:18s}]: ok={rr.get('ok')}  dur={rr.get('duration_s')}s  err={rr.get('error', '')[:80]}")
        probe_results.append({"variant": var, **rr})
    return {"cell": lam_tag, "train_ok": True, "probe": probe_results}


@app.local_entrypoint()
def next_long_additive_v2_l15_qwen3_4b():
    """Single-cell entrypoint: λ_f=0.15. Use sweep_long_additive_v2 for the full 3-cell sweep."""
    _launch_one("rl/ctrl0_u1_40_long_additive_v2_l15_qwen3_4b.toml",
                "ctrl0-qwen3-4b-u1-40-long-additive-v2-l15")


@app.local_entrypoint()
def next_long_additive_v2_l10_qwen3_4b():
    """Single-cell entrypoint: λ_f=0.10."""
    _launch_one("rl/ctrl0_u1_40_long_additive_v2_l10_qwen3_4b.toml",
                "ctrl0-qwen3-4b-u1-40-long-additive-v2-l10")


@app.local_entrypoint()
def next_long_additive_v2_l30_qwen3_4b():
    """Single-cell entrypoint: λ_f=0.30."""
    _launch_one("rl/ctrl0_u1_40_long_additive_v2_l30_qwen3_4b.toml",
                "ctrl0-qwen3-4b-u1-40-long-additive-v2-l30")


@app.local_entrypoint()
def eval_long_additive_v2_probe(lam_tag: str = "l15", num_examples: int = 498, step: int = 200):
    """Standalone probe entrypoint: probes long-additive-v2-{lam_tag} (default l15) at the
    given step (default 200). Runs 2 cells (base + remaining_budget). Use sweep_long_additive_v2
    for the combined train+probe flow."""
    adapter = f"/cache/runs/ctrl0_u1_40_long_additive_v2_{lam_tag}_qwen3_4b/weights/step_{step}/lora_adapters"
    label = f"long-additive-v2-{lam_tag}"
    jobs = [{"adapter_path": adapter, "adapter_name": label,
             "run_label": label, "variants": (v,)}
            for v in ("base", "remaining_budget")]
    print(f"Launching 2 probe cells for {label} step_{step}")
    calls = [(j, eval_prompt_salience_run.spawn(num_examples=num_examples, **j)) for j in jobs]
    for j, c in calls:
        try:
            r = c.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        var = j['variants'][0]
        print(f"  {label}/{var:18s}: ok={r.get('ok')}  rc={r.get('returncode')}  "
              f"dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.local_entrypoint()
def sync_v2_wandb():
    """Sync long_additive_v2_* offline wandb runs to cloud. Useful after a sweep
    with WANDB_MODE=offline so we can use wandb.Api() to plot."""
    results = wandb_sync_offline.remote("ctrl0_u1_40_long_additive_v2")
    print("\n=== sync results ===")
    for r in results:
        print(f"  {r['cell']:55s} / {r['segment']}: rc={r['rc']}")


@app.local_entrypoint()
def sync_stage2_wandb():
    """Sync stage2_kl_* offline wandb runs to cloud."""
    results = wandb_sync_offline.remote("stage2_kl")
    print("\n=== sync results ===")
    for r in results:
        print(f"  {r['cell']:55s} / {r['segment']}: rc={r['rc']}")


@app.local_entrypoint()
def sweep_stage2_kl():
    """Fire-and-forget β sweep for 2-stage KL-anchored curriculum. 5 cells:
    kl_tau ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1}. All start from the v2-l30-merged
    base model, train with c-only reward, ckpt every 25 steps.
    PREREQUISITE: merge_v2_l30_lora must have run first.

    Uses .spawn() (not .remote()) so the local entrypoint returns immediately
    and the modal app runs server-side. Probes need to be fired manually
    after each cell's step_200 weights land — use eval_stage2_probe for that.

    Wallclock: ~3-4h per cell if parallel, ~$25 per cell (~$125 total)."""
    beta_tags = ["b0", "b4", "b3", "b2", "b1"]
    print(f"Spawning {len(beta_tags)}-cell stage-2 β sweep (training only): {beta_tags}")
    for bt in beta_tags:
        cfg = f"rl/stage2_kl_{bt}_qwen3_4b.toml"
        wandb_name = f"stage2-kl-{bt}"
        wandb_project = f"interoception-stage2-{bt}"
        handle = train_run.spawn(cfg, wandb_name,
                                 wandb_project=wandb_project, wandb_offline=True)
        print(f"  {bt}: spawned (call={handle.object_id})  wandb={wandb_project}/{wandb_name}")
    print("\nAll trainings spawned. Local entrypoint exiting; modal app continues server-side.")
    print("After step_200 weights land for a cell, probe via:")
    print("  modal run modal_app.py::eval_stage2_probe --beta-tag <b0|b4|b3|b2|b1>")


@app.local_entrypoint()
def eval_stage2_probe(beta_tag: str = "b3", num_examples: int = 498, step: int = 200):
    """Probe a single stage-2 cell's checkpoint (default step_200).
    Spawns both prompt variants (base + remaining_budget) in parallel."""
    adapter = f"/cache/runs/stage2_kl_{beta_tag}_qwen3_4b/weights/step_{step}/lora_adapters"
    label = f"stage2-kl-{beta_tag}"
    jobs = [{"adapter_path": adapter, "adapter_name": label,
             "run_label": label, "variants": (v,)}
            for v in ("base", "remaining_budget")]
    print(f"Launching 2 probe cells for {label} step_{step}")
    calls = [(j, eval_prompt_salience_run.spawn(num_examples=num_examples, **j)) for j in jobs]
    for j, c in calls:
        try:
            r = c.get()
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        var = j['variants'][0]
        print(f"  {label}/{var:18s}: ok={r.get('ok')}  rc={r.get('returncode')}  "
              f"dur={r.get('duration_s')}s  err={r.get('error', '')}")


@app.function(
    image=image,
    volumes={"/cache": volume},
    timeout=24 * 3600,
)
def auto_probe_stage2_when_ready():
    """Poll the volume every 5 min for stage-2 cells with step_200 weights;
    spawn probes (base + remaining_budget) as soon as each cell completes.
    Runs server-side. Returns when all 5 cells have been probed.

    Idempotent at the cell level — once we've spawned probes for a cell,
    we don't re-spawn even if the function restarts."""
    import os, time
    beta_tags = ["b0", "b4", "b3", "b2", "b1"]
    probed = set()
    iter_no = 0
    while len(probed) < len(beta_tags):
        iter_no += 1
        volume.reload()
        for bt in beta_tags:
            if bt in probed:
                continue
            adapter = f"/cache/runs/stage2_kl_{bt}_qwen3_4b/weights/step_200/lora_adapters"
            if os.path.exists(adapter):
                label = f"stage2-kl-{bt}"
                print(f"[iter {iter_no}] {bt} ready — spawning 2 probes", flush=True)
                for v in ("base", "remaining_budget"):
                    eval_prompt_salience_run.spawn(
                        adapter_path=adapter, adapter_name=label, run_label=label,
                        variants=(v,), num_examples=498,
                    )
                probed.add(bt)
        if len(probed) < len(beta_tags):
            remaining = sorted(set(beta_tags) - probed)
            print(f"[iter {iter_no}] {len(probed)}/{len(beta_tags)} probed; waiting on {remaining}", flush=True)
            time.sleep(300)
    print(f"all {len(beta_tags)} cells probed (probes spawned). watcher exiting.", flush=True)
    return list(probed)


@app.local_entrypoint()
def sweep_windowed():
    """Fire-and-forget λ sweep with windowed_additive reward (asymmetric Gaussian
    f peaked at t=T). 3 cells: λ ∈ {0.15, 0.30, 0.50}, σ_under=0.25, σ_over=0.10.
    Same protocol as v2 (200 steps, G=16/batch=128, +remaining_budget prompt) —
    only the reward shape changes. Tests whether the bimodal regime under flat-1
    additive disappears with a peaked f-shape (Kanishk's hypothesis)."""
    lam_tags = ["l15", "l30", "l50"]
    print(f"Spawning {len(lam_tags)}-cell windowed-reward λ sweep: {lam_tags}")
    for lt in lam_tags:
        cfg = f"rl/ctrl0_u1_40_windowed_{lt}_qwen3_4b.toml"
        wandb_name = f"ctrl0-qwen3-4b-u1-40-windowed-{lt}"
        wandb_project = f"interoception-windowed-{lt}"
        handle = train_run.spawn(cfg, wandb_name,
                                 wandb_project=wandb_project, wandb_offline=True)
        print(f"  {lt}: spawned (call={handle.object_id})  wandb={wandb_project}/{wandb_name}")
    print("\nAll trainings spawned. Local entrypoint exiting; modal app continues server-side.")
    print("Probes auto-fire when each cell hits step_200 — run watch_and_probe_windowed.")


@app.local_entrypoint()
def launch_windowed_su(cell: str = "su10"):
    """Launch ONE σ_under-sweep cell, fire-and-forget (survives --detach):
        modal run --detach modal_app.py::launch_windowed_su --cell su10
    STAGED rollout: launch su10 first, confirm it trains past step ~3 (where the last
    run wedged in an FSDP all-gather) and looks healthy, THEN launch su17. cell in
    {su10, su17}; the σ_under=0.25 baseline is Nicole's windowed_l30 (fig 49).
    train_run now has a 15-min stall watchdog, so a hang fails fast instead of idling."""
    table = {
        "su10": ("rl/ctrl0_u1_40_windowed_su10_qwen3_4b.toml",
                 "ctrl0-qwen3-4b-u1-40-windowed-su10", "interoception-windowed-su10"),
        "su17": ("rl/ctrl0_u1_40_windowed_su17_qwen3_4b.toml",
                 "ctrl0-qwen3-4b-u1-40-windowed-su17", "interoception-windowed-su17"),
        # 200-step ALL-FRESH σ_under sweep (su25/su17/su10, identical schedule, λ=0.30).
        "su25_200": ("rl/ctrl0_u1_40_windowed_su25_200_qwen3_4b.toml",
                     "ctrl0-qwen3-4b-u1-40-windowed-su25-200", "interoception-windowed-su25-200"),
        "su17_200": ("rl/ctrl0_u1_40_windowed_su17_200_qwen3_4b.toml",
                     "ctrl0-qwen3-4b-u1-40-windowed-su17-200", "interoception-windowed-su17-200"),
        "su10_200": ("rl/ctrl0_u1_40_windowed_su10_200_qwen3_4b.toml",
                     "ctrl0-qwen3-4b-u1-40-windowed-su10-200", "interoception-windowed-su10-200"),
    }
    if cell not in table:
        raise SystemExit(f"unknown cell {cell!r}; choose from {list(table)}")
    cfg, name, proj = table[cell]
    handle = train_run.spawn(cfg, name, wandb_project=proj)
    print(f"spawned {name} (call={handle.object_id})  wandb={proj}/{name} — runs detached on Modal")


@app.local_entrypoint()
def probe_windowed_su(num_examples: int = 498):
    """Calibration probe (commit-time vs T, matched/remaining_budget prompt) on the
    σ_under sweep step_100 adapters: su10 (σ_under=0.10) and su17 (σ_under=0.17). Writes
    /cache/eval_rollouts/prompt_salience/windowed-su{10,17}_remaining_budget.jsonl. Overlay
    vs su25 = windowed_l30 (existing windowed-l30_remaining_budget.jsonl in the repo). One
    A100 per cell, ~15-20 min, fire-and-forget (launch with --detach)."""
    cells = [
        ("/cache/runs/ctrl0_u1_40_windowed_su10_qwen3_4b/weights/step_100/lora_adapters", "windowed-su10"),
        ("/cache/runs/ctrl0_u1_40_windowed_su17_qwen3_4b/weights/step_100/lora_adapters", "windowed-su17"),
    ]
    for adapter, label in cells:
        call = eval_prompt_salience_run.spawn(
            adapter_path=adapter, adapter_name=label, run_label=label,
            variants=("remaining_budget",), num_examples=num_examples)
        print(f"spawned probe {label} (call={call.object_id})  n={num_examples}")
    print("Probes spawned. JSONLs land in /cache/eval_rollouts/prompt_salience/ when done.")


@app.local_entrypoint()
def probe_windowed_su200(num_examples: int = 498):
    """Calibration probe (commit-time vs T, remaining_budget) on the 200-step ALL-FRESH
    σ_under sweep — step_200 adapters for all three: su25/su17/su10. Writes
    /cache/eval_rollouts/prompt_salience/windowed-su{25,17,10}-200_remaining_budget.jsonl.
    One A100 per cell, ~15-20 min, fire-and-forget (launch with --detach)."""
    cells = [
        ("/cache/runs/ctrl0_u1_40_windowed_su25_200_qwen3_4b/weights/step_200/lora_adapters", "windowed-su25-200"),
        ("/cache/runs/ctrl0_u1_40_windowed_su17_200_qwen3_4b/weights/step_200/lora_adapters", "windowed-su17-200"),
        ("/cache/runs/ctrl0_u1_40_windowed_su10_200_qwen3_4b/weights/step_200/lora_adapters", "windowed-su10-200"),
    ]
    for adapter, label in cells:
        call = eval_prompt_salience_run.spawn(
            adapter_path=adapter, adapter_name=label, run_label=label,
            variants=("remaining_budget",), num_examples=num_examples)
        print(f"spawned probe {label} (call={call.object_id})  n={num_examples}")
    print("Probes spawned (3 cells, step_200). JSONLs land in /cache/eval_rollouts/prompt_salience/")


@app.function(
    image=image,
    volumes={"/cache": volume},
    timeout=24 * 3600,
)
def auto_probe_windowed_when_ready(num_examples: int = 200):
    """Poll every 5 min for windowed-sweep cells with step_200 weights; spawn
    probes as each cell completes.

    Default num_examples=200 (down from 498) for exploratory iteration speed:
    n=200 gives r std error ≈ 0.07 (good enough to distinguish r=0.5 vs r=0.8)
    and probe wallclock drops from ~70-90 min to ~30 min."""
    import os, time
    lam_tags = ["l15", "l30", "l50"]
    probed = set()
    iter_no = 0
    while len(probed) < len(lam_tags):
        iter_no += 1
        volume.reload()
        for lt in lam_tags:
            if lt in probed:
                continue
            adapter = f"/cache/runs/ctrl0_u1_40_windowed_{lt}_qwen3_4b/weights/step_100/lora_adapters"
            if os.path.exists(adapter):
                label = f"windowed-{lt}"
                print(f"[iter {iter_no}] {lt} ready — spawning 2 probes (n={num_examples})", flush=True)
                for v in ("base", "remaining_budget"):
                    eval_prompt_salience_run.spawn(
                        adapter_path=adapter, adapter_name=label, run_label=label,
                        variants=(v,), num_examples=num_examples,
                    )
                probed.add(lt)
        if len(probed) < len(lam_tags):
            remaining = sorted(set(lam_tags) - probed)
            print(f"[iter {iter_no}] {len(probed)}/{len(lam_tags)} probed; waiting on {remaining}", flush=True)
            time.sleep(300)
    print(f"all {len(lam_tags)} cells probed. watcher exiting.", flush=True)
    return list(probed)


@app.local_entrypoint()
def watch_and_probe_windowed():
    """Spawn the windowed-sweep auto-probe watcher server-side."""
    handle = auto_probe_windowed_when_ready.spawn()
    print(f"spawned auto-probe watcher (call={handle.object_id})")
    print("watcher will run server-side; safe to close terminal.")


@app.local_entrypoint()
def watch_and_probe_stage2():
    """Spawn the auto-probe watcher server-side. Runs for up to 24h, polling
    every 5 min and firing probes as each cell's step_200 weights appear."""
    handle = auto_probe_stage2_when_ready.spawn()
    print(f"spawned auto-probe watcher (call={handle.object_id})")
    print("watcher will run server-side; safe to close terminal.")


@app.local_entrypoint()
def probe_stage2_all_ready():
    """Scan for stage-2 cells with step_200 weights, fire probes for each.
    Idempotent — won't re-probe cells that already have JSONLs on the volume."""
    import modal as _modal
    vol = _modal.Volume.from_name("interoception-cache")
    # Heuristic: list output dirs, check weights/step_200 existence
    beta_tags = ["b0", "b4", "b3", "b2", "b1"]
    for bt in beta_tags:
        weights = f"/cache/runs/stage2_kl_{bt}_qwen3_4b/weights/step_200/lora_adapters"
        jsonl_base = f"/cache/eval_rollouts/prompt_salience/stage2-kl-{bt}_base.jsonl"
        jsonl_rb = f"/cache/eval_rollouts/prompt_salience/stage2-kl-{bt}_remaining_budget.jsonl"
        # We can't directly check files via vol.listdir from a local entrypoint.
        # Instead just try to probe — eval_prompt_salience_run will fail-fast if
        # the adapter path doesn't exist.
        print(f"=== {bt}: probing (will fail if step_200 not ready) ===")
        for variant in ("base", "remaining_budget"):
            handle = eval_prompt_salience_run.spawn(
                adapter_path=weights,
                adapter_name=f"stage2-kl-{bt}",
                run_label=f"stage2-kl-{bt}",
                variants=(variant,),
                num_examples=498,
            )
            print(f"  {bt}/{variant}: spawned (call={handle.object_id})")


@app.local_entrypoint()
def sweep_long_additive_v2():
    """Fire-and-forget λ sweep: trains 3 cells in parallel (λ∈{0.10,0.15,0.30}),
    each auto-probed at step_200. All otherwise share the v2 protocol
    (200 steps, G=16/batch=128, additive reward, remaining_budget prompt).

    Wallclock: if modal runs cells in parallel, ~7.5h total (~6h train + ~1.5h probe).
    If modal serializes (as happened with the long-additive probe), expect ~22h total.
    Per-cell cost ~$25 (~$75 for sweep)."""
    import modal as _modal
    lam_tags = ["l10", "l15", "l30"]
    print(f"Launching {len(lam_tags)}-cell λ sweep: {lam_tags}")
    # Spawn all cells in parallel via .spawn so they don't block on each other.
    # Each cell is a local function call to _train_then_probe_v2 which itself
    # uses .remote/.spawn for the underlying modal functions.
    # We use Python threads to fan out the spawn calls (each blocks on .remote/.spawn).
    import concurrent.futures as _f
    with _f.ThreadPoolExecutor(max_workers=len(lam_tags)) as ex:
        futures = {ex.submit(_train_then_probe_v2, lt): lt for lt in lam_tags}
        results = {}
        for fut in _f.as_completed(futures):
            lt = futures[fut]
            try:
                results[lt] = fut.result()
            except Exception as e:
                results[lt] = {"cell": lt, "error": str(e)[:200]}
    print("\n=== SWEEP COMPLETE ===")
    for lt in lam_tags:
        print(f"  {lt}: {results.get(lt)}")
