# Add save_path and show params to plot_response_surface

**Status**: accepted

hydropattern's `run --plot` needs to save one response-surface plot per component to its
output directory, save-only by default (no blocking interactive windows when producing
multiple plots), with an opt-in to also display them. We considered leaving
`plot_response_surface` untouched and having hydropattern manipulate shared `pyplot`
state (`plt.savefig()`/`plt.close()`) directly after calling it, but that leaks
matplotlib figure-lifecycle responsibility across the package boundary and is fragile
if the internal figure-creation in `plot_response_surface` ever changes. Instead we
extended the function's signature with optional `save_path: Path | None = None` and
`show: bool = True` params, keeping figure lifecycle ownership inside climate-canvas.
This is a public API change for a dependency consumed by hydropattern, so worth
recording why we chose to extend the signature rather than work around it externally.
