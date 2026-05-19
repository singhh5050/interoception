#!/bin/bash
# Track A trajectory sweep runner.
# 3 models x 2 timing modes x 10 problems x T in {30,60} = 120 rollouts.
# Sim timing uses model-matched hwprop calibration (Gemma-3-4B sub for gemma-4).
# Real timing uses time.perf_counter around llm.generate.

set +e
export HF_HOME=/workspace/hf-cache
cd /workspace/interoception
mkdir -p runs

declare -A SIM_MODEL
SIM_MODEL["Qwen/Qwen2.5-3B-Instruct"]="Qwen2.5-3B"
SIM_MODEL["Qwen/Qwen3-4B-Instruct-2507"]="Qwen3-4B"
SIM_MODEL["google/gemma-4-E4B-it"]="Gemma-3-4B"

declare -A SLUG
SLUG["Qwen/Qwen2.5-3B-Instruct"]="qwen25_3b"
SLUG["Qwen/Qwen3-4B-Instruct-2507"]="qwen3_4b"
SLUG["google/gemma-4-E4B-it"]="gemma4_e4b"

for HF in "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen3-4B-Instruct-2507" "google/gemma-4-E4B-it"; do
  S=${SLUG[$HF]}
  M=${SIM_MODEL[$HF]}

  echo ""
  echo "######################################################"
  echo "### $HF  SIM  (sim_model=$M, hardware=A100_80GB)"
  echo "######################################################"
  date
  python scripts/run_sweep.py \
    --hf-model "$HF" \
    --sim-model "$M" \
    --hardware A100_80GB \
    --target-s 30 60 \
    --problems-file data/eval.jsonl \
    --num-problems 10 \
    --out-dir "runs/traj_sim_${S}"
  echo "exit=$?  done with sim ${S}"

  echo ""
  echo "######################################################"
  echo "### $HF  REAL  (perf_counter)"
  echo "######################################################"
  date
  python scripts/run_sweep.py \
    --hf-model "$HF" \
    --realtime \
    --target-s 30 60 \
    --problems-file data/eval.jsonl \
    --num-problems 10 \
    --out-dir "runs/traj_real_${S}"
  echo "exit=$?  done with real ${S}"
done

echo ""
echo "ALL_SWEEPS_DONE  $(date)"
