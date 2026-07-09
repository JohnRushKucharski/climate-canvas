'''Plotting utilities for climate impact data.'''
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.collections import QuadMesh

from climate_canvas.data_utilities import evenly_space, check_threshold, contour_levels

def plot_response_surface(xs, ys, zs,
                          interpolate: bool = False,
                          labels: tuple[str, str, str] = ('x', 'y', 'z'),
                          title: str = 'Response Surface',
                          save_path: Path | None = None,
                          show: bool = True,
                          threshold: float | None = None,
                          color_map: str = 'RdBu',
                          color_map_ticks: list[float] | None = None
                          ) -> None:
    '''Plot response surface from climate impact data.

    threshold: optional z-value that becomes the colormap's center color
    (defaults to the midpoint of the z-value range). Contour levels are
    generated as 5 evenly-spaced levels below and 5 above the threshold,
    with the threshold's own contour line drawn bolder.
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    # original (possibly irregular) knot values, before interpolation resamples them
    xticks, yticks = xs, ys
    if interpolate: # and
        xs, ys, zs = evenly_space(xs, ys, zs, None, (25, 25))
    z_range = (float(np.nanmin(zs)), float(np.nanmax(zs)))
    threshold = check_threshold(threshold, z_range)
    norm = TwoSlopeNorm(vmin=z_range[0], vcenter=threshold, vmax=z_range[1])
    levels, widths = contour_levels(z_range, threshold)
    # pcolormesh/contour take xs/ys directly, honoring irregular spacing
    # (imshow+extent would stretch xs/ys as if they were evenly spaced).
    im = ax.pcolormesh(xs, ys, zs, shading='nearest', cmap=color_map, norm=norm)
    ax.contour(xs, ys, zs, levels=levels, linewidths=widths, colors='black')
    ax.set_aspect('auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(labels[2])
    if color_map_ticks:
        cbar.set_ticks(color_map_ticks)
    # autoscale to the mesh's full cell extent (nearest-shading pads edge cells
    # symmetrically beyond xs/ys min/max) instead of clipping at xs/ys min/max,
    # which would cut edge-cell pixels in half (quarter at corners).
    ax.autoscale(enable=True, tight=True)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(which='major', linestyle='--', color='gray', alpha=0.6)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_response_surfaces(paths: list[str], axis_labels: tuple[str, str, str] = ('x', 'y', 'z'),
                           subplot_labels: list[str] | None = None, title: str = '',
                           interpolate: bool = False, color_map: str | list[str] = 'RdYlBu',
                           threshold: float | None = None) -> None:
    '''Plot multiple response surfaces.'''
    groups = group_paths(paths)
    keys_sorted = sorted(groups.keys())
    cmap = (color_map if isinstance(color_map, str)
            else LinearSegmentedColormap.from_list('', color_map))
    fig, axs = plt.subplots(1, ncols=len(groups), figsize=(10*len(groups), 10))
    im: QuadMesh | None = None
    for i, k in enumerate(keys_sorted):
        label = subplot_labels[i] if subplot_labels else ''
        fig, axs[i], im = surface_subplot(fig, axs[i], read_data(groups[k]),
                                  axis_labels, label,
                                  interpolate, cmap, threshold)
    if im is None:
        raise ValueError('paths must contain at least one file.')
    fig.subplots_adjust(right=0.98)
    cbar_ax = fig.add_axes((0.99, 0.11, 0.02, 0.77))
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(title)
    plt.show()

def surface_subplot(fig, ax, data: tuple[np.ndarray, np.ndarray, np.ndarray],
                    axis_labels: tuple[str, str, str], subplot_label: str,
                    interpolate: bool, color_map: str | LinearSegmentedColormap,
                    threshold: float | None) -> tuple[plt.Figure, plt.Axes, QuadMesh]:
    '''Plot a single response surface subplot.'''
    x, y, z = data
    if interpolate:
        x, y, z = evenly_space(x, y, z, None, (25, 25))
    z_range = (float(np.nanmin(z)), float(np.nanmax(z)))
    vcenter = check_threshold(threshold, z_range)
    norm = TwoSlopeNorm(vmin=z_range[0], vcenter=vcenter, vmax=z_range[1])
    im = ax.pcolormesh(x, y, z, shading='nearest', cmap=color_map, norm=norm)
    ax.contour(x, y, z, colors='black')
    ax.set_aspect('auto')
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.title.set_text(subplot_label)
    return fig, ax, im

def group_paths(paths: list[str]) -> dict[str, Path]:
    '''Validate path.'''
    groups: dict[str, Path] = {}
    for str_path in paths:
        path = Path(str_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f'{path} does not exist.')
        if (group_id:=path.stem.split('_')[-1]) not in groups:
            groups[group_id] = Path(str_path)
        else:
            raise ValueError(f'{path} has duplicate id.')
    return groups

def read_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Read data from csv file.'''
    df = pd.read_csv(path)
    z = df.iloc[:, 1:].to_numpy(dtype=float)
    y = np.array(df.iloc[:, 0], dtype=float)
    x = np.array(df.columns.values[1:], dtype=float)
    return x, y, z
