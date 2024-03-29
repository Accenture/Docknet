import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(axe: plt.Axes, X0: np.ndarray, X1: np.ndarray, color,
                 x0_range, x1_range, title: str):
    axe.scatter(X0, X1, c=color, s=2)
    aspect = (x0_range[1] - x0_range[0]) / (x1_range[1] - x1_range[0])
    axe.set_aspect(aspect)
    axe.set_xlim(x0_range)
    axe.set_ylim(x1_range)
    axe.set_title(title)
    axe.set_xlabel('x0')
    axe.set_ylabel('x1')
