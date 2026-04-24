"""
gif.py — Animated GIF generation for parameter evolution over time.

gradient_gif renders one GIF per parameter: each frame is a colour-coded network
map where link colour encodes the parameter value at that timestep.
"""

import os

import numpy as np
import pandas as pd
import imageio
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from config import DL, links, LOCAL_FIGURE, LOCAL_GIF, NODATA_COLOR
from network.draw import sublink, polyg


def gradient_gif(param, param_name, fps):
    """
    Create animated GIFs showing temporal evolution of one or more parameters.

    One GIF and one PNG-per-frame folder are produced per parameter. Frames are
    coloured with the "coolwarm" colormap; no-data cells (NaN) are rendered in
    NODATA_COLOR so viewers don't confuse them with valid zero-speed segments.

    Parameters
    ----------
    param      : list of ndarray, each shape (1, T, N)
    param_name : list of str – used for folder names and plot titles
    fps        : float – frames per second in the output GIF
    """
    for pn in param_name:
        os.makedirs(os.path.join(LOCAL_FIGURE, str(pn)), exist_ok=True)

    for i, p in enumerate(param):
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(DL.node_coordinates[:, 0], DL.node_coordinates[:, 1],
                   s=10, c="black", alpha=0.5, zorder=-2)
        ax.set_aspect("equal")
        ax.set_title(param_name[i])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

        norm = mcolors.Normalize(np.nanmin(p[0, :]), np.nanmax(p[0, :]))
        cmap = cm.get_cmap("coolwarm")

        os.makedirs(LOCAL_GIF, exist_ok=True)
        gif_path = os.path.join(LOCAL_GIF, f"{param_name[i]}.gif")

        polyg(ax, "black", 1, 1)

        # Pre-build line artists so we can update colours in-place each frame
        link_collections = []
        for j, row in links.iterrows():
            x, y  = sublink(row)
            color = NODATA_COLOR if pd.isna(p[0, 0, j]) else cmap(norm(p[0, 0, j]))
            coll  = ax.plot(x, y, c=color, linewidth=1)[0]
            link_collections.append(coll)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=str(param_name[i]), location="right")

        with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
            for t in range(p.shape[1]):
                ax.set_title(f"{param_name[i]} @ {fps} fps : t = {t}", fontsize=10)

                for j, coll in enumerate(link_collections):
                    color = NODATA_COLOR if pd.isna(p[0, t, j]) else cmap(norm(p[0, t, j]))
                    coll.set_color(color)

                fig.canvas.draw()
                w, h  = fig.canvas.get_width_height()
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                image = image.reshape(h, w, 4)[:, :, :3]

                os.makedirs(f"figure/{param_name[i]}", exist_ok=True)
                fig.savefig(f"figure/{param_name[i]}/graph{t}.png")

                writer.append_data(image)
                print(f"{param_name[i]} t={t}")

        plt.close(fig)
