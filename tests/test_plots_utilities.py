'''Tests for climate_canvas.plots_utilities.'''
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; avoids blocking/opening windows in tests.

import numpy as np
from matplotlib import pyplot as plt

from climate_canvas.plots_utilities import plot_response_surface


def test_plot_response_surface_writes_file_when_save_path_given(tmp_path):
    '''save_path is provided -> a plot file exists at that path after the call.'''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[2.0, 1.9, 1.0], [5.0, 4.5, 4.0]])
    save_path = tmp_path / 'plot.png'

    plot_response_surface(xs, ys, zs, save_path=save_path, show=False)

    assert save_path.exists()


def test_plot_response_surface_does_not_show_when_show_false(tmp_path, monkeypatch):
    '''show=False -> plt.show() is not invoked (no blocking interactive window).'''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[2.0, 1.9, 1.0], [5.0, 4.5, 4.0]])
    calls = []
    monkeypatch.setattr('climate_canvas.plots_utilities.plt.show', lambda: calls.append(True))

    plot_response_surface(xs, ys, zs, save_path=tmp_path / 'plot.png', show=False)

    assert calls == []


def test_plot_response_surface_shows_by_default(monkeypatch):
    '''Default show=True preserves existing interactive behavior.'''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[2.0, 1.9, 1.0], [5.0, 4.5, 4.0]])
    calls = []
    monkeypatch.setattr('climate_canvas.plots_utilities.plt.show', lambda: calls.append(True))

    plot_response_surface(xs, ys, zs)

    assert calls == [True]


def test_plot_response_surface_closes_figure_after_call():
    '''Figure is closed after the call so repeated calls (e.g. per-component) don't leak.'''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[2.0, 1.9, 1.0], [5.0, 4.5, 4.0]])

    plot_response_surface(xs, ys, zs, show=False)

    assert plt.get_fignums() == []


def test_plot_response_surface_orients_rows_with_ys_ascending_bottom_to_top(monkeypatch):
    '''zs[0] (smallest y) must be plotted at the bottom of the image, matching the
    y-axis (extent bottom = ys.min()). imshow defaults to origin='upper', which would
    place zs[0] at the top instead -- a y-axis flip bug. origin='lower' is required.
    '''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[2.0, 1.9, 1.0], [5.0, 4.5, 4.0]])
    captured = {}
    original_imshow = plt.Axes.imshow

    def capturing_imshow(self, *args, **kwargs):
        captured['origin'] = kwargs.get('origin')
        return original_imshow(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, 'imshow', capturing_imshow)

    plot_response_surface(xs, ys, zs, show=False)

    assert captured['origin'] == 'lower'


def test_plot_response_surface_labels_colorbar_with_zlabel(monkeypatch):
    '''labels[2] (z label) is rendered as the colorbar's label.'''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[2.0, 1.9, 1.0], [5.0, 4.5, 4.0]])
    captured = {}
    original_colorbar = plt.Figure.colorbar

    def capturing_colorbar(self, *args, **kwargs):
        cbar = original_colorbar(self, *args, **kwargs)
        captured['cbar'] = cbar
        return cbar

    monkeypatch.setattr(plt.Figure, 'colorbar', capturing_colorbar)

    plot_response_surface(xs, ys, zs, labels=('x', 'y', 'portion'), show=False)

    assert captured['cbar'].ax.get_ylabel() == 'portion'
