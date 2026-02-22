from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, out_path: str, title: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()