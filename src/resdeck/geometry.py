"""Where wells and points are on the grid, column by column.

A column is the vertical stack of cells that share one (i, j). Placement
works on columns and on 2D maps, ``(nx, ny)`` arrays with one value per
column. Code that needs to know where a well is should ask ``well_cells``
and not read ``well.i`` and ``well.j``, so that wells which are not
vertical can be added later.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .model import Model
from .wells import Well

# A point is on the grid when it lies within this many cell widths of the
# nearest column centre, measured along each grid direction separately. A cell
# reaches 0.5 from its centre; the margin allows for uneven column spacing.
_REACH = 0.75
# Past an outermost column there is no next cell to be in, so outwards from it
# the limit is the cell's own reach: the edge of the grid.
_EDGE = 0.5


def well_cells(well: Well) -> list[tuple[int, int, int]]:
    """The cells ``well`` is completed in, as 1-based (i, j, k), in the order
    of its completions."""
    return [(well.i, well.j, k) for c in well.completions for k in range(c.k_top, c.k_bottom + 1)]


def active_columns(model: Model) -> np.ndarray:
    """``(nx, ny)`` bool map: True where the column has an active cell."""
    return model.active.any(axis=2)


def column_xy(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """``(nx, ny)`` maps of the x and y of every column: the mean cell centre
    of its active cells, or of all its cells where none is active."""
    weight = model.active.astype(float)
    weight[~active_columns(model)] = 1.0
    total = weight.sum(axis=2)
    return (model.x * weight).sum(axis=2) / total, (model.y * weight).sum(axis=2) / total


def nearest_columns(model: Model, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The column nearest to each point (``x``, ``y``).

    Returns ``(i, j, inside)`` with the shape of ``x``: 1-based column
    indices, and False in ``inside`` where the point is off the grid. The
    edge of the grid is taken as half a column spacing beyond the centres of
    the outermost columns, which is exact where those columns are evenly
    spaced. The column may be inactive; check it against ``active_columns``.
    """
    cx, cy = column_xy(model)
    if min(cx.shape) < 2:
        raise ValueError("the grid needs at least 2 columns in each direction")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))

    tree = cKDTree(np.column_stack([cx.ravel(), cy.ravel()]))
    _, nearest = tree.query(np.column_stack([x.ravel(), y.ravel()]))
    i, j = np.unravel_index(nearest, cx.shape)

    # Express each point's offset from its column centre in cell widths along
    # the grid's own I and J directions, so long thin or rotated cells are
    # measured correctly. (ax, ay) is one step in I, (bx, by) one step in J,
    # and the offset is u steps in I plus v steps in J.
    ax, bx = (g[i, j] for g in np.gradient(cx))
    ay, by = (g[i, j] for g in np.gradient(cy))
    dx, dy = x.ravel() - cx[i, j], y.ravel() - cy[i, j]
    area = ax * by - bx * ay
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (dx * by - dy * bx) / area
        v = (dy * ax - dx * ay) / area
    nx, ny = cx.shape
    inside = (
        (u >= -np.where(i == 0, _EDGE, _REACH))
        & (u <= np.where(i == nx - 1, _EDGE, _REACH))
        & (v >= -np.where(j == 0, _EDGE, _REACH))
        & (v <= np.where(j == ny - 1, _EDGE, _REACH))
    )
    return (i + 1).reshape(x.shape), (j + 1).reshape(x.shape), inside.reshape(x.shape)
