'''Plotting utilities for climate impact data.'''
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from climate_canvas.data_utilities import evenly_space

def plot_response_surface(xs, ys, zs,
                          interpolate: bool = False,
                          labels: tuple[str, str, str] = ('x', 'y', 'z'),
                          title: str = 'Response Surface',
                          save_path: Path | None = None,
                          show: bool = True
                          ) -> None:
    '''Plot response surface from climate impact data.'''
    fig, ax = plt.subplots(figsize=(10, 10))
    if interpolate: # and
        xs, ys, zs = evenly_space(xs, ys, zs, None, (25, 25))
    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    im = ax.imshow(zs, extent=extent, aspect='auto', cmap='RdYlBu', origin='lower')
    # todo: contour lines are rotated, this has happened to others.
    ax.contour(zs, extent=extent, origin='lower', colors='black')
    fig.colorbar(im, ax=ax)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_response_surfaces(paths: list[str], axis_labels: tuple[str, str, str] = ('x', 'y', 'z'),
                           subplot_labels: None|list[str] = None, title: str = '',
                           interpolate: bool = False, color_map: str|list['str'] ='RdYlBu',
                           threshold: float = None|float) -> None:
    '''Plot multiple response surfaces.'''
    groups = group_paths(paths)
    keys_sorted = sorted(groups.keys())
    fig, axs = plt.subplots(1, ncols=len(groups), figsize=(10*len(groups), 10))
    for i, k in enumerate(keys_sorted):
        fig, axs[i] = surface_subplot(fig, axs[i], read_data(groups[k]),
                                  axis_labels, subplot_labels[i],
                                  interpolate, color_map, threshold = None)
    fig.subplots_adjust(right=0.98)
    cbar_ax = fig.add_axes([0.99, 0.11, 0.02, 0.77])
    fig.colorbar(axs[0].images[0], cax=cbar_ax)
    plt.show()

def surface_subplot(fig, ax, data: tuple[np.ndarray, np.ndarray, np.ndarray],
                    axis_labels: tuple[str, str, str], subplot_label: str,
                    interpolate: bool, color_map: str, threshold: float) -> tuple[plt.Figure, plt.Axes]:
    '''Plot a single response surface subplot.'''
    x, y, z = data
    if interpolate:
        x, y, z = evenly_space(x, y, z, None, (25, 25))
    extent = (x.min(), x.max(), y.min(), y.max())
    ax.imshow(z, extent=extent, aspect='auto', cmap=color_map,
              vmin=-0.5, vmax=1.5)
    ax.contour(z, extent=extent, origin='image', colors='black')
    ax.title.set_text(subplot_label)
    return fig, ax

def group_paths(paths: list[str]) -> dict[str, Path]:
    '''Validate path.'''
    groups = dict()
    for str_path in paths:
        path = Path(str_path)
        if not path.exists or not path.is_file():
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


    #     im = ax.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()))
    #     ax.contour(z, colors='black')
    #     fig.colorbar(im, ax=ax)
    #     plt.show()
    # else:
    #     Z = zs
    #     X, Y = np.meshgrid(xs, ys)
    #     #fig, ax = plt.subplots()
    #     im = ax.imshow(Z)
    #     ax.contour(Z, colors='black')
    #     fig.colorbar(im, ax=ax)
    #     plt.show()

    # print(X.shape)
    # print(Z.shape)

    # print(ys)
    # print(xs)
    # print(zs)
    # for i, row in enumerate(Y): # i.e., temp
    #     for j, col in enumerate(X): # i.e., precip
    #         Z[i, j] = find_z((col[i], row[j]), (xs, ys), zs)
    #         print(f'x/p:{col[i]}, y/t:{row[j]}, z:{Z[i, j]}')
    # print(Z.shape)
    # print(Z)

    # print(X.shape)
    # print(Y.shape)
    # print(Z.shape)
    # print(Z)
    # print(X.shape)
    # print(Y.shape)
    # Z1 = np.exp(-X**2 - Y**2)
    # Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    # Z = (Z1 - Z2) * 2
    # print(Z1.shape)
    # print(Z2.shape)
    # print(Z.shape)
    # print(Z)
