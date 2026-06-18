"""
network/draw.py — Primitive drawing functions for the road network.

All functions receive a matplotlib Axes and operate on the global `links` and `DL`
objects from config. They produce no figures on their own — callers create figures,
call these helpers, then save/close.
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DL, links, NODATA_COLOR


def sublink(row):
    """
    Return the (x, y) endpoint arrays for a single link row, oriented for clean rendering.

    The direction swap is governed by `out_ang` so that gradient coloring always runs
    from-node → to-node regardless of how the link was digitised.

    Usage
    -----
    for _, row in links.iterrows():
        x, y = sublink(row)
    """
    if row["out_ang"] > np.pi / 2 or (row["out_ang"] < 0 and row["out_ang"] >= -np.pi / 2):
        x = np.array([row["to_x"],   row["from_x"]])
        y = np.array([row["from_y"], row["to_y"]])
    else:
        x = np.array([row["from_x"], row["to_x"]])
        y = np.array([row["from_y"], row["to_y"]])
    return x, y


def link(ax, grad=False, color="red", zorder=2, norm=None, p=None, t=None, cmap=None):
    """
    Draw all road links on `ax`.

    Parameters
    ----------
    ax     : matplotlib Axes
    grad   : bool  – colour each link by the value of `p` at timestep `t`
    color  : str   – flat colour used when grad=False
    zorder : int   – drawing layer
    norm   : mcolors.Normalize – required when grad=True
    p      : ndarray, shape (1, T, N) – required when grad=True
    t      : int   – timestep index, required when grad=True
    cmap   : colormap – required when grad=True
    """
    if not grad:
        for _, row in links.iterrows():
            x, y = sublink(row)
            ax.plot(x, y, c=color, linewidth=1, zorder=zorder)
        return

    if norm is None or p is None or t is None:
        raise ValueError("norm, p, and t are required when grad=True.")

    for j, row in links.iterrows():
        x = np.array([row["from_x"], row["to_x"]])
        y = np.array([row["from_y"], row["to_y"]])
        if pd.isna(p[0, t, j]):
            z = NODATA_COLOR
        else:
            if cmap is None:
                raise ValueError("cmap is required when grad=True.")
            z = cmap(norm(p[0, t, j]))
        ax.plot(x, y, c=z, linewidth=1)


def polyg(ax, color="blue", alpha=1, zorder=3):
    """Draw intersection polygons on `ax`."""
    for _, data in DL.intersection_polygon.items():
        poly = data["polygon"]
        x = np.array([p[0] for p in poly] + [poly[0][0]])
        y = np.array([p[1] for p in poly] + [poly[0][1]])
        ax.plot(x, y, c=color, alpha=alpha, zorder=zorder)


def graph():
    """Render the baseline network map (nodes + links + intersection polygons) to graph.png."""
    fig, ax = plt.subplots(dpi=250)
    ax.scatter(DL.node_coordinates[:, 0], DL.node_coordinates[:, 1],
               s=10, c="black", alpha=1, zorder=1)
    link(ax)
    polyg(ax)
    ax.set_aspect("equal")
    ax.set_title("Nodes + Links + Intersection Polygons", fontsize=10)
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]",  fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    fig.savefig("graph.png")
    plt.close(fig)


def numerical_sort(file):
    """
    Sort key for filenames in natural numerical order.

    Transforms [graph1.png, graph10.png, ..., graph2.png] into [graph1.png, graph2.png, ...].
    Returns -1 for filenames with no embedded digits (sorts them first).
    """
    numbers = re.findall(r"\d+", file)
    return int(numbers[0]) if numbers else -1



def graph_base(
    node_color="#0F766E",
    node_size=12,
    node_alpha=0.4,
    link_color="#374151",
    link_width=1.0,
    poly_face="#F4A261",
    poly_edge="#E76F51",
    poly_alpha=0.75,
):
    """Render the styled baseline network map used for visual inspection."""

    fig, ax = plt.subplots(figsize=(10, 8), dpi=250)

    # Intersection polygons
    for _, data in DL.intersection_polygon.items():
        poly = np.asarray(data["polygon"])

        ax.fill(
            poly[:, 0],
            poly[:, 1],
            facecolor=poly_face,
            edgecolor=poly_edge,
            linewidth=0.8,
            alpha=poly_alpha,
            zorder=1,
        )

    # Links
    for _, row in links.iterrows():
        x, y = sublink(row)
        ax.plot(
            x,
            y,
            color=link_color,
            linewidth=link_width,
            zorder=2,
        )

    # Nodes
    ax.scatter(
        DL.node_coordinates[:, 0],
        DL.node_coordinates[:, 1],
        s=node_size,
        c=node_color,
        alpha=node_alpha,
        zorder=3,
    )

    ax.set_aspect("equal")
    ax.set_title("Network", fontsize=10)
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)

    fig.savefig("graph.png", bbox_inches="tight")
    plt.close(fig)