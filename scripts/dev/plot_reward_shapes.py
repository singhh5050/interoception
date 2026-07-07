"""Reward-formulation gallery: one pane per reward family, all swept values drawn.

Paper-style panel figure. Solid lines: R(t) for a correct answer. Dashed: wrong
answer. Sweeps shown are the ones actually trained: additive λ ∈ {.10,.15,.30,.50},
window λ ∈ {.15,.30,.50}, window σ_under ∈ {.10,.17,.25}, multiplicative
f ∈ {hyperbolic, exponential}, VoC cost ∈ {overbudget, clipped}. The KL
curriculum (β sweep over R=c) is a training scheme, not a shape — caption it.
"""
from __future__ import annotations
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = pathlib.Path("analysis/figures/66_reward_shapes.png")

x = np.linspace(0.01, 3.0, 600)  # t/T


def hyperbolic(x):
    return np.minimum(1.0, 1.0 / x)


def exponential(x):
    return np.where(x <= 1.0, 1.0, np.exp(-(x - 1.0)))


def gauss(x, su, so):
    sig = np.where(x <= 1.0, su, so)
    return np.exp(-(((x - 1.0) / sig) ** 2))


# (panel letter, name, equation, fixed-param note, [(label, color, f(c)->R)])
PANES = [
    ("a", "Multiplicative gate", r"$R = c \cdot f(t,T)$", "",
     [(r"$f=\min(1,\,T/t)$", "#e08214", lambda c: c * hyperbolic(x)),
      (r"$f=e^{-(t-T)/T}$", "#8c510a", lambda c: c * exponential(x))]),
    ("b", "Accuracy only / KL curriculum",
     r"$R = c \; - \; \beta\, D_{\mathrm{KL}}\!\left(\pi \,\Vert\, \pi_{\mathrm{paced}}\right)$",
     "$\\beta \\in \\{0,\\, 10^{-4},\\, 10^{-3},\\, 10^{-2},\\, 10^{-1}\\}$\n"
     "$\\beta{=}0$: negative control",
     [("", "#b2182b", lambda c: np.full_like(x, float(c)))]),
    ("c", "Flat-top additive", r"$R = c + \lambda \cdot \min(1,\,T/t)$", "",
     [(f"$\\lambda={lam:.2f}$", col, (lambda lam: lambda c: c + lam * hyperbolic(x))(lam))
      for lam, col in [(0.10, "#a6dba0"), (0.15, "#5aae61"),
                       (0.30, "#1b7837"), (0.50, "#00441b")]]),
    ("d", "Gaussian window, λ sweep",
     r"$R = c + \lambda\, e^{-((t-T)/(\sigma T))^2}$",
     r"$\sigma_u{=}0.25,\ \sigma_o{=}0.10$",
     [(f"$\\lambda={lam:.2f}$", col, (lambda lam: lambda c: c + lam * gauss(x, 0.25, 0.10))(lam))
      for lam, col in [(0.15, "#92c5de"), (0.30, "#4393c3"), (0.50, "#2166ac")]]),
    ("e", "Gaussian window, σ$_u$ sweep",
     r"$R = c + \lambda\, e^{-((t-T)/(\sigma T))^2}$",
     r"$\lambda{=}0.30,\ \sigma_o{=}0.10$",
     [(f"$\\sigma_u={su:.2f}$", col, (lambda su: lambda c: c + 0.30 * gauss(x, su, 0.10))(su))
      for su, col in [(0.10, "#c2a5cf"), (0.17, "#9970ab"), (0.25, "#762a83")]]),
    ("f", "Value of computation", r"$R = c - \gamma \cdot \mathrm{cost}(t,T)$",
     r"$\gamma{=}0.5$",
     [(r"$\mathrm{cost}=\max(0,\,t-T)/T$", "#8c564b",
       lambda c: c - 0.5 * np.maximum(0.0, x - 1.0)),
      (r"$\mathrm{cost}=\min(t/T,\,k)$", "#bf812d",
       lambda c: c - 0.5 * np.minimum(x, 2.0))]),
]


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12.5, "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "figure.dpi": 160,
    })
    # 12.8 x 5.25 in ~ the 2.45:1 body area of a 16:9 slide under its title
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 5.25), sharex=True, sharey=True)

    for ax, (letter, name, eq, note, curves) in zip(axes.flat, PANES):
        ax.axhline(0.0, color="#000", lw=0.6, alpha=0.35, zorder=1)
        ax.axvline(1.0, color="#000", lw=0.6, ls=(0, (2, 3)), alpha=0.35, zorder=1)
        for label, color, fn in curves:
            ax.plot(x, fn(1), color=color, lw=1.9, label=label or None, zorder=3)
            ax.plot(x, fn(0), color=color, lw=1.2, ls="--", alpha=0.5, zorder=2)
        ax.set_title(f"$\\bf{{{letter}}}$   {name}\n{eq}", fontsize=12.5,
                     loc="left", pad=8)
        if note:
            ax.text(0.03, 0.05, note, transform=ax.transAxes, fontsize=10.5,
                    color="#666")
        if any(lbl for lbl, _, _ in curves):
            loc = "center right" if letter == "c" else "upper right"
            ax.legend(frameon=False, fontsize=10, loc=loc,
                      handlelength=1.2, borderaxespad=0.0, labelspacing=0.2)
        ax.set_ylim(-0.7, 1.65)
        ax.set_xlim(0, 3)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0", "$T$", "$2T$", "$3T$"])
        ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])

    for ax in axes[1]:
        ax.set_xlabel("commit time $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("reward $R$")

    handles = [Line2D([0], [0], color="#333", lw=1.9),
               Line2D([0], [0], color="#333", lw=1.2, ls="--", alpha=0.5)]
    fig.legend(handles, ["correct ($c=1$)", "wrong ($c=0$)"],
               loc="lower center", ncol=2, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.035))

    fig.tight_layout(h_pad=2.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
