"""Figure style and the recurring plot types used in the paper.

One place for the journal-ready defaults (column widths, font sizes,
colour-blind-safe diverging maps) so that every figure script produces a
consistent set. Maps use Cartopy when it is installed and fall back to a
plain rectangular projection when it is not, so the workflow never dies
for want of an optional dependency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

MM = 1 / 25.4  # millimetres -> inches

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:  # pragma: no cover - optional
    HAS_CARTOPY = False


def set_style(font_size: int = 8) -> None:
    mpl.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # editable text in the final PDF
        "ps.fonttype": 42,
    })


def figure(width_mm: float = 183, height_mm: float = 100, **kw):
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM), **kw)


def _projection(name: str = "Robinson"):
    if not HAS_CARTOPY:
        return None
    return getattr(ccrs, name)(central_longitude=0)


def map_axes(ax=None, projection: str = "Robinson", coastlines: bool = True):
    """Create (or decorate) a global map axis."""
    if ax is None:
        proj = _projection(projection)
        fig = plt.figure(figsize=(183 * MM, 90 * MM))
        ax = fig.add_subplot(1, 1, 1, projection=proj) if proj else fig.add_subplot(1, 1, 1)
    if HAS_CARTOPY and hasattr(ax, "coastlines"):
        if coastlines:
            ax.coastlines(linewidth=0.3, color="0.35")
        ax.add_feature(cfeature.BORDERS, linewidth=0.15, edgecolor="0.6")
        ax.set_global()
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_aspect("equal")
    return ax


def plot_map(da: xr.DataArray, ax=None, vmin=None, vmax=None,
             cmap: str = "BrBG", stipple: xr.DataArray | None = None,
             hatch: str = "///", title: str = "", cbar_label: str = "",
             projection: str = "Robinson", add_colorbar: bool = True):
    """Filled global map with optional significance stippling."""
    ax = map_axes(ax, projection)
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(da.values), 98)) or 1.0
    if vmin is None:
        vmin = -vmax
    kw = {"transform": ccrs.PlateCarree()} if (HAS_CARTOPY and hasattr(ax, "projection")) else {}
    mesh = ax.pcolormesh(da["lon"], da["lat"], da.values, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="auto",
                         rasterized=True, **kw)
    if stipple is not None:
        st = stipple.astype(float).where(stipple)
        ax.contourf(st["lon"], st["lat"], st.fillna(0).values,
                    levels=[0.5, 1.5], colors="none", hatches=[hatch], **kw)
    if title:
        ax.set_title(title, loc="left")
    if add_colorbar:
        cb = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.03,
                          shrink=0.62, aspect=32, extend="both")
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.set_label(cbar_label or da.attrs.get("units", ""))
        cb.outline.set_linewidth(0.4)
    return ax, mesh


def plot_lag_profile(comp: xr.DataArray, ax=None, ci: tuple = None,
                     color: str = "C0", label: str = "",
                     highlight_seasons: bool = True):
    """Composite anomaly as a function of lag from the event peak."""
    if ax is None:
        _, ax = figure(89, 60)
    lags = comp["lag"].values
    ax.plot(lags, comp.values, color=color, lw=1.2, label=label)
    if ci is not None:
        ax.fill_between(lags, ci[0], ci[1], color=color, alpha=0.2, lw=0)
    ax.axhline(0, color="0.4", lw=0.5)
    ax.axvline(0, color="0.4", lw=0.5, ls=":")
    if highlight_seasons:
        ax.axvspan(-6, -1, color="0.9", zorder=0)   # developing
        ax.axvspan(0, 2, color="0.8", zorder=0)     # mature DJF
    ax.set_xlabel("lag from event peak (months)")
    return ax


def save_figure(fig, path: str | Path, formats=("png", "pdf"), dpi: int = 300):
    """Write a figure in every requested format; returns the paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi)
        out.append(p)
    plt.close(fig)
    return out


def panel_label(ax, text: str, x: float = -0.02, y: float = 1.06):
    ax.text(x, y, text, transform=ax.transAxes, fontweight="bold",
            va="bottom", ha="left")
