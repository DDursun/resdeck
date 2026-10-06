"""Well patterns: a central well, the anchor, and the wells arranged around it.

``build_pattern`` turns an anchor column and a radius into the ``Drill``
events of one pattern. A lone well is the smallest pattern. The orientation
is fixed by the pattern; a pattern with any well in a bad place is rejected
as a whole.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from datetime import date
from numbers import Real

import numpy as np

from .geometry import active_columns, column_xy, nearest_columns, well_cells
from .model import Model
from .scenario import Drill
from .wells import Completion, InjectorControl, ProducerControl, Well


def _ring(count: int, first: float) -> tuple[tuple[float, float], ...]:
    """``count`` points evenly spaced on a circle of radius 1, the first at
    ``first`` degrees."""
    angles = [math.radians(first + 360.0 * n / count) for n in range(count)]
    return tuple((math.cos(a), math.sin(a)) for a in angles)


# Pattern -> where its outer wells are, as steps from the anchor. The square
# patterns have their corners one step away in both directions; the triangle
# and the hexagon have their wells one step from the anchor.
PATTERNS: dict[str, tuple[tuple[float, float], ...]] = {
    "single": (),
    "four_spot": _ring(3, 90.0),  # triangle
    "five_spot": ((1, 1), (-1, 1), (-1, -1), (1, -1)),  # square corners
    "seven_spot": _ring(6, 0.0),  # hexagon
    "nine_spot": ((1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)),
}


def build_pattern(
    model: Model,
    pattern: str,
    anchor: tuple[int, int],
    *,
    radius: float | None = None,
    radius_cells: float | None = None,
    producer: ProducerControl | None = None,
    injector: InjectorControl | None = None,
    inverted: bool = False,
    diameter: float,
    at: date | None = None,
    existing: Iterable[Well] = (),
    producer_prefix: str = "P",
    injector_prefix: str = "I",
    group: str = "G1",
) -> tuple[Drill, ...]:
    """The ``Drill`` events of one pattern, the anchor's first.

    ``pattern`` is a name in ``PATTERNS`` and ``anchor`` the 1-based column
    (i, j) of its central well. The central well is a producer and the outer
    wells are injectors; ``inverted`` swaps the two. ``producer`` and
    ``injector`` are the controls the wells get; one may be left out when the
    pattern has no well of that kind.

    A pattern with outer wells needs its size, given by exactly one of the
    two below; ``single`` has no outer wells and ignores both.

    - ``radius``: the distance from the anchor to the farthest outer well, in
      the units of the grid. The pattern is laid out along the map's x and y
      and each outer well is moved to the nearest column, so distances are
      approximate.
    - ``radius_cells``: a number of columns. The corners of the square
      patterns are that many columns away in both I and J; the wells of the
      triangle and the hexagon are that many columns from the anchor.

    Every well is completed in all active layers of its column, is drilled at
    ``at`` (None is the start of the run) and joins ``group``. Names are a
    prefix and the next free number, counted from the names in ``existing``,
    the wells already there.

    Raises ``ValueError`` when a well would be off the grid, on an inactive
    column, on a well in ``existing`` or on another well of the pattern.
    """
    existing = list(existing)
    columns = _columns(model, pattern, anchor, radius, radius_cells)
    where = f"{pattern} at ({columns[0][0]}, {columns[0][1]})"
    taken = {(i, j): well.name for well in existing for i, j, _ in well_cells(well)}
    for column in columns:
        if column in taken:
            raise ValueError(f"{where}: column {column} already has well {taken[column]}")

    # The kind of the central well, then of the outer wells.
    kinds = (InjectorControl, ProducerControl) if inverted else (ProducerControl, InjectorControl)
    given = {ProducerControl: producer, InjectorControl: injector}
    controls = [given[kinds[0]]] + [given[kinds[1]]] * (len(columns) - 1)
    for control, kind in zip(controls[:2], kinds, strict=False):  # a lone well has one kind
        if not isinstance(control, kind):
            what = "producer" if kind is ProducerControl else "injector"
            raise ValueError(f"{where}: {what} must be a {kind.__name__}, got {control!r}")

    names = {well.name for well in existing}
    events = []
    for column, control in zip(columns, controls, strict=True):
        prefix = injector_prefix if isinstance(control, InjectorControl) else producer_prefix
        name = _next_name(prefix, names)
        names.add(name)
        completions = _completions(model, column, diameter)
        events.append(Drill(Well(name, *column, completions, control, group), at))
    return tuple(events)


def _columns(model: Model, pattern: str, anchor, radius, radius_cells) -> list[tuple[int, int]]:
    """The 1-based columns of the pattern's wells, the anchor's first."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; choose from {sorted(PATTERNS)}")
    try:
        i, j = (operator.index(n) for n in anchor)
    except (TypeError, ValueError):
        raise ValueError(f"anchor must be a column (i, j), got {anchor!r}") from None
    if not (1 <= i <= model.nx and 1 <= j <= model.ny):
        raise ValueError(f"anchor ({i}, {j}) is outside the {model.nx} x {model.ny} grid")
    where = f"{pattern} at ({i}, {j})"

    columns = [(i, j)]
    steps = np.array(PATTERNS[pattern], dtype=float).reshape(-1, 2)
    if len(steps):
        if (radius is None) == (radius_cells is None):
            raise ValueError(f"{where}: give exactly one of radius and radius_cells")
        if radius_cells is not None:
            moves = np.rint(_positive("radius_cells", radius_cells) * steps).astype(int)
            ci, cj = i + moves[:, 0], j + moves[:, 1]
            inside = (ci >= 1) & (ci <= model.nx) & (cj >= 1) & (cj <= model.ny)
        else:
            x, y = column_xy(model)
            # Scale the steps so the farthest well is ``radius`` from the anchor.
            moves = steps * _positive("radius", radius) / np.hypot(*steps.T).max()
            ci, cj, inside = nearest_columns(
                model, x[i - 1, j - 1] + moves[:, 0], y[i - 1, j - 1] + moves[:, 1]
            )
        if not inside.all():
            raise ValueError(f"{where}: {int((~inside).sum())} of its wells fall off the grid")
        columns += [(int(a), int(b)) for a, b in zip(ci, cj, strict=True)]

    active = active_columns(model)
    for n, column in enumerate(columns):
        if not active[column[0] - 1, column[1] - 1]:
            raise ValueError(f"{where}: column {column} is inactive")
        if column in columns[:n]:
            raise ValueError(f"{where}: two wells land on column {column}; use a larger radius")
    return columns


def _completions(model: Model, column: tuple[int, int], diameter: float) -> tuple[Completion, ...]:
    """One completion for every unbroken run of active layers in ``column``."""
    layers = np.flatnonzero(model.active[column[0] - 1, column[1] - 1]) + 1
    runs = np.split(layers, np.flatnonzero(np.diff(layers) > 1) + 1)
    return tuple(Completion(int(run[0]), int(run[-1]), diameter) for run in runs)


def _next_name(prefix: str, names: set[str]) -> str:
    number = 1
    while f"{prefix}{number}" in names:
        number += 1
    return f"{prefix}{number}"


def _positive(what: str, value) -> float:
    ok = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    if not ok or value <= 0:
        raise ValueError(f"{what} must be a finite number above 0, got {value!r}")
    return float(value)
