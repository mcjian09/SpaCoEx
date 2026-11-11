from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter

OUTPUT_FOLDER = Path.cwd()
FONT_SIZE = 22

colors = simulation["group_colors"].copy()
colors.update({"Strongly varying": "#D55E00", "Medium varying": "#0072B2"})
handles = [
    Line2D([0], [0], color=colors[group], linewidth=3, label=group)
    for group in simulation["group_labels"]
]

with plt.rc_context({
    "font.family": "DejaVu Sans",
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "pdf.fonttype": 42,
}):
    fig = plt.figure(figsize=(23, 17), layout="constrained")
    grid = fig.add_gridspec(4, 3, height_ratios=[0.23, 1, 0.12, 1])

    for penalty, header_row, plot_row in (
        ("k", 0, 1), ("k_kminus1", 2, 3)
    ):
        result = results[(penalty, "logeuclidean")]
        weights = np.asarray(result["weights"])
        lambdas = np.asarray(result["lambda"])
        x = lambdas if penalty == "k" else np.sqrt(lambdas)
        xlabel = r"$\lambda$" if penalty == "k" else r"$\sqrt{\lambda}$"

        header = fig.add_subplot(grid[header_row, :])
        header.set_axis_off()
        if penalty == "k":
            header.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0, 1.0),
                ncol=2,
                frameon=False,
                fontsize=FONT_SIZE,
                columnspacing=1.2,
                borderaxespad=0,
            )
            header.text(0.5, 0.18, r"Gene-level",
                        ha="center", va="center")
        else:
            header.text(0.5, 0.5, r"Pairwise",
                        ha="center", va="center")

        axes = [fig.add_subplot(grid[plot_row, column]) for column in range(3)]
        rng = np.random.default_rng(JITTER_SEED)
        offsets = rng.uniform(-1, 1, weights.shape[1]) * X_JITTER * np.ptp(x)

        # Gene weights: jitter changes only the displayed x positions.
        for index, group in enumerate(simulation["gene_group"]):
            display_x = np.clip(x + offsets[index], x[0], x[-1])
            display_x[0], display_x[-1] = x[0], x[-1]
            axes[0].step(display_x, weights[:, index], where="post",
                         color=colors[group], linewidth=2.3, alpha=0.85)
        axes[0].set(xlabel=xlabel, ylabel="weight",
                    xlim=(x[0], x[-1]), ylim=(-0.03, 1.03))
        axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1])

        # Active-gene count, not the sum of fractional weights.
        axes[1].step(x, result["selected_k"], where="post",
                     color="#0072B2", linewidth=2.8)
        axes[1].set(xlabel=xlabel, ylabel="number selected",
                    xlim=(x[0], x[-1]), ylim=(0, weights.shape[1] + 1))
        axes[1].yaxis.set_major_locator(MultipleLocator(5))

        order = np.argsort(result["k"], kind="stable")
        axes[2].plot(
            np.asarray(result["k"])[order],
            np.asarray(result["percent_retained"])[order],
            marker="o", markersize=6.5, linewidth=2,
            color="#0072B2", clip_on=False,
        )
        axes[2].set(xlabel="number genes", ylabel="variance explained",
                    xlim=(0, weights.shape[1]), ylim=(0, 100))
        axes[2].yaxis.set_major_formatter(PercentFormatter(100))
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

        for ax in axes:
            ax.xaxis.labelpad = ax.yaxis.labelpad = 12
            ax.tick_params(labelsize=FONT_SIZE, width=1.4, length=6, pad=7)
            ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
            ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)
            ax.grid(color="#DDDDDD", linewidth=0.8, alpha=0.65)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.3)

    for extension in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FOLDER / f"publication_both_penalties.{extension}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.show()
    plt.close(fig)

print(f"Saved publication_both_penalties.png/.pdf to:\n{OUTPUT_FOLDER}")