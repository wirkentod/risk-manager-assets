"""Visualization helpers for correlation matrices."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(corr, out_path=None, figsize=(10, 8), cmap="vlag"):
    """Plot or save a correlation heatmap.

    If `out_path` is None the plot will be shown interactively.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0)
    plt.tight_layout()
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
        plt.close()
    else:
        plt.show()
