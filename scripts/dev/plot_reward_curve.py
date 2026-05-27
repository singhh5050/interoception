"""Desmos-style plot of the reward function itself: reward = c * f(t, T), f = min(1, T/t).

X = elapsed sim-time t (seconds), Y = reward (shown for a CORRECT answer, c=1; an
incorrect answer is 0 everywhere). One curve per budget T. Overlays the chunk-time grid
(128 tok ~ 4.5 s on A100_80GB) and the band where the model actually operates, to show
geometrically why U(1,40) crushes f: the model lives at ~40-60 s, far right of every kink.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHUNK_S = 4.5          # 128-token chunk on A100_80GB (hwprop)
BUDGETS = [5, 10, 20, 40]   # representative of T ~ U(1,40)
T_MAX = 80
t = np.linspace(0.01, T_MAX, 2000)

fig, ax = plt.subplots(figsize=(9, 5.5))

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(BUDGETS)))
for T, c in zip(BUDGETS, colors):
    f = np.minimum(1.0, T / t)
    ax.plot(t, f, color=c, lw=2.4, label=f"T = {T}s")
    ax.plot([T], [1.0], "o", color=c, ms=6)            # the kink: on-budget -> over-budget
    ax.annotate(f"T={T}", (T, 1.0), textcoords="offset points", xytext=(3, 6),
                fontsize=9, color=c, fontweight="bold")

# chunk-time grid: achievable elapsed is ~ n_turns * 4.5s
for n in range(1, int(T_MAX / CHUNK_S) + 1):
    ax.axvline(n * CHUNK_S, color="0.85", lw=0.6, zorder=0)

# where the model actually lives (mean_completion_tokens ~1200-1700 tok => ~42-60s)
ax.axvspan(42, 60, color="crimson", alpha=0.10, zorder=0)
ax.annotate("model's actual\nelapsed (~42–60s)", (51, 0.86), ha="center", fontsize=9,
            color="crimson", fontweight="bold")
# show the reward it actually collects there, per budget
for T, c in zip(BUDGETS, colors):
    f_here = min(1.0, T / 51)
    ax.plot([51], [f_here], "s", color=c, ms=7, mec="k", mew=0.6, zorder=5)
    ax.annotate(f"{f_here:.2f}", (51, f_here), textcoords="offset points", xytext=(7, -3),
                fontsize=8.5, color=c)

ax.set_xlim(0, T_MAX)
ax.set_ylim(0, 1.08)
ax.set_xlabel("elapsed simulated time  t  (seconds)", fontsize=11)
ax.set_ylabel("reward  =  f(t, T)  =  min(1, T/t)   [for a correct answer]", fontsize=11)
ax.set_title("Reward as a function of elapsed time, per time-budget T\n"
             "flat (=1) while on budget, then 1/t decay once t > T", fontsize=12)
ax.legend(title="time budget", loc="upper right", framealpha=0.95)
ax.grid(axis="y", alpha=0.25)
ax.text(0.5, 0.02, "grey lines = chunk boundaries (128 tok ≈ 4.5 s each);  incorrect answer ⇒ reward = 0 everywhere",
        transform=ax.transAxes, ha="center", fontsize=8, color="0.4")

fig.tight_layout()
out = "analysis/figures/24_reward_vs_elapsed.png"
fig.savefig(out, dpi=140)
print("wrote", out)
