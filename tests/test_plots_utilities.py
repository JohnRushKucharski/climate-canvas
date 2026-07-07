'''Tests for climate_canvas.plots_utilities.'''
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; avoids blocking/opening windows in tests.

import numpy as np  # noqa: E402 pylint: disable=wrong-import-position
from matplotlib import pyplot as plt  # pylint: disable=wrong-import-position

from climate_canvas.plots_utilities import plot_response_surface  # pylint: disable=wrong-import-position


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

    assert not calls


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
    '''zs[0] (smallest y) must be plotted at ys[0] and zs[-1] at ys[-1], matching the
    actual y coordinates supplied (not a normalized/evenly-spaced index). pcolormesh
    is passed the real xs/ys arrays, so row order is preserved without needing an
    origin flip like imshow required.
    '''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[2.0, 1.9, 1.0], [5.0, 4.5, 4.0]])
    captured = {}
    original_pcolormesh = plt.Axes.pcolormesh

    def capturing_pcolormesh(self, *args, **kwargs):
        captured.setdefault('args', args)  # keep only the first call (the surface itself)
        return original_pcolormesh(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, 'pcolormesh', capturing_pcolormesh)

    plot_response_surface(xs, ys, zs, show=False)

    captured_xs, captured_ys, captured_zs = captured['args']
    assert np.array_equal(captured_xs, xs)
    assert np.array_equal(captured_ys, ys)
    assert np.array_equal(captured_zs, zs)


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


def test_plot_response_surface_threshold_centers_norm_at_threshold(monkeypatch):
    '''threshold sets pcolormesh's TwoSlopeNorm vcenter (colormap midpoint value).'''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[0.0, 0.1, 0.5], [0.8, 0.9, 1.0]])
    captured = {}
    original_pcolormesh = plt.Axes.pcolormesh

    def capturing_pcolormesh(self, *args, **kwargs):
        captured.setdefault('norm', kwargs.get('norm'))
        return original_pcolormesh(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, 'pcolormesh', capturing_pcolormesh)

    plot_response_surface(xs, ys, zs, show=False, threshold=0.2)

    assert captured['norm'].vcenter == 0.2
    assert captured['norm'].vmin == 0.0
    assert captured['norm'].vmax == 1.0


def test_plot_response_surface_threshold_defaults_to_midpoint(monkeypatch):
    '''threshold=None defaults the norm's vcenter to the midpoint of the z-range.'''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[0.0, 0.1, 0.5], [0.8, 0.9, 1.0]])
    captured = {}
    original_pcolormesh = plt.Axes.pcolormesh

    def capturing_pcolormesh(self, *args, **kwargs):
        captured.setdefault('norm', kwargs.get('norm'))
        return original_pcolormesh(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, 'pcolormesh', capturing_pcolormesh)

    plot_response_surface(xs, ys, zs, show=False)

    assert captured['norm'].vcenter == 0.5


def test_plot_response_surface_color_map_ticks_sets_colorbar_ticks(monkeypatch):
    '''color_map_ticks, when provided, are applied to the colorbar.'''
    xs = np.array([0.0, 0.5, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([[0.0, 0.1, 0.5], [0.8, 0.9, 1.0]])
    captured = {}
    original_colorbar = plt.Figure.colorbar

    def capturing_colorbar(self, *args, **kwargs):
        cbar = original_colorbar(self, *args, **kwargs)
        captured['cbar'] = cbar
        return cbar

    monkeypatch.setattr(plt.Figure, 'colorbar', capturing_colorbar)

    plot_response_surface(xs, ys, zs, show=False, color_map_ticks=[0.0, 0.5, 1.0])

    assert list(captured['cbar'].get_ticks()) == [0.0, 0.5, 1.0]
