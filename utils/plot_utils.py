import matplotlib
import matplotlib.pyplot as plt
import numpy as np
def plot_helper(x,y,**kwargs):
    MACOSKO_COLORS = {
        "Amacrine cells": "#A5C93D",
        "Astrocytes": "#8B006B",
        "Bipolar cells": "#2000D7",
        "Cones": "#538CBA",
        "Fibroblasts": "#8B006B",
        "Horizontal cells": "#B33B19",
        "Microglia": "#8B006B",
        "Muller glia": "#8B006B",
        "Pericytes": "#8B006B",
        "Retinal ganglion cells": "#C38A1F",
        "Rods": "#538CBA",
        "Vascular endothelium": "#8B006B",
    }
    fig, ax = plt.subplots(figsize=(8, 8))

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}
    classes = np.unique(y)

    if 'use_macosko' in kwargs and kwargs['use_macosko']:
        colors = {k: v for k, v in MACOSKO_COLORS.items() if k in classes}
    else:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
    
    point_colors = list(map(colors.get, y))
    ax.scatter(x[:,0],x[:,1],c=point_colors, rasterized=True, **plot_params)
    legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
    legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
    
    ax.legend(handles=legend_handles, **legend_kwargs_)
    fig.savefig("tsne.png") if not 'title' in kwargs else fig.savefig(kwargs['title']+".png")

