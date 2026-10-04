"""Grid and property arrays of a realization, read from EGRID and INIT files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {"INTE": ">i4", "REAL": ">f4", "DOUB": ">f8", "LOGI": ">i4", "CHAR": "S8"}
# INIT arrays that are not per-cell, whatever their length.
_NOT_CELL = {"INTEHEAD", "LOGIHEAD", "DOUBHEAD", "TABDIMS", "TAB"}


@dataclass(frozen=True, eq=False)
class Model:
    """A corner-point grid and its properties.

    All arrays are ``(nx, ny, nz)``: axis 0 is I, axis 1 is J, axis 2 is K,
    so the simulator cell (i, j, k) is ``array[i - 1, j - 1, k - 1]``.
    ``x``, ``y``, ``z`` are cell centres. ``props`` holds the per-cell INIT
    arrays (PORO, PERMX, DZ, ...); inactive cells are NaN, or 0 for
    integer arrays.
    """

    nx: int
    ny: int
    nz: int
    active: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    props: dict[str, np.ndarray]

    @classmethod
    def read(cls, egrid, init=None) -> Model:
        """Read an EGRID file and, optionally, the INIT file of the same run."""
        grid: dict[str, np.ndarray] = {}
        for name, data in read_arrays(egrid):
            if name == "ENDGRID":  # local grids follow; only the global grid is read
                break
            grid.setdefault(name, data)
        for name in ("GRIDHEAD", "COORD", "ZCORN"):
            if name not in grid:
                raise ValueError(f"{egrid} has no {name} array")
        if grid["GRIDHEAD"][0] != 1:
            raise ValueError(f"{egrid} is not a corner-point grid")

        nx, ny, nz = (int(n) for n in grid["GRIDHEAD"][1:4])
        shape = (nx, ny, nz)
        if "ACTNUM" in grid:
            active = (grid["ACTNUM"] != 0).reshape(shape, order="F")
        else:
            active = np.ones(shape, dtype=bool)
        x, y, z = _centres(grid["COORD"], grid["ZCORN"], shape)
        props = {} if init is None else _read_props(init, active)
        return cls(nx, ny, nz, active, x, y, z, props)


def read_arrays(path):
    """Yield ``(name, array)`` for every array of an unformatted Eclipse
    output file (EGRID, INIT, UNRST, ...), in file order.

    Every Fortran record is checked: its leading and trailing length markers
    must match and fit in the file, and the data blocks must add up to the
    element count in the array header. A damaged file raises ValueError."""
    buf = Path(path).read_bytes()
    if len(buf) < 24 or int.from_bytes(buf[:4], "big") != 16:
        raise ValueError(f"{path} is not an unformatted Eclipse file")

    pos = 0
    while pos < len(buf):
        # Header record: 8-char name, element count, 4-char type.
        start, size, pos = _record(buf, pos, path)
        if size != 16:
            raise ValueError(f"{path}: array header at byte {start - 4} has length {size}, not 16")
        try:
            name = buf[start : start + 8].decode("ascii").strip()
            kind = buf[start + 12 : start + 16].decode("ascii")
        except UnicodeDecodeError:
            raise ValueError(f"{path}: array header at byte {start - 4} is not text") from None
        count = int.from_bytes(buf[start + 8 : start + 12], "big", signed=True)
        if count < 0:
            raise ValueError(f"{path}: array {name} has a negative element count")
        if kind == "MESS":
            yield name, np.empty(0)
            continue
        if kind in _DTYPES:
            dtype = np.dtype(_DTYPES[kind])
        elif kind.startswith("C0") and kind[1:].isdigit() and int(kind[1:]) > 0:
            dtype = np.dtype(f"S{int(kind[1:])}")
        else:
            raise ValueError(f"{path}: array {name} has unsupported type {kind!r}")

        # The data follows in blocks, each its own Fortran record.
        blocks = []
        left = count
        while left > 0:
            start, size, pos = _record(buf, pos, path)
            n, extra = divmod(size, dtype.itemsize)
            if extra or n == 0 or n > left:
                raise ValueError(
                    f"{path}: array {name} has a data block of {size} bytes that does not "
                    f"fit its {count} elements of type {kind}"
                )
            blocks.append(np.frombuffer(buf, dtype, n, start))
            left -= n
        data = np.concatenate(blocks) if blocks else np.empty(0, dtype)

        if dtype.kind == "S":
            yield name, np.char.strip(data.astype(str))
        elif kind == "LOGI":
            yield name, data != 0
        else:
            yield name, data.astype(dtype.newbyteorder("="))


def _record(buf: bytes, pos: int, path) -> tuple[int, int, int]:
    """The Fortran record at byte ``pos``: (payload start, payload size,
    position of the next record). Both length markers must agree."""
    if pos + 4 > len(buf):
        raise ValueError(f"{path}: file ends inside a record marker at byte {pos}")
    size = int.from_bytes(buf[pos : pos + 4], "big", signed=True)
    end = pos + 4 + size
    if size < 0 or end + 4 > len(buf):
        raise ValueError(f"{path}: record at byte {pos} runs past the end of the file")
    if int.from_bytes(buf[end : end + 4], "big", signed=True) != size:
        raise ValueError(f"{path}: record at byte {pos} has mismatched length markers")
    return pos + 4, size, end + 4


def _read_props(init, active) -> dict[str, np.ndarray]:
    shape = active.shape
    mask = active.ravel(order="F")
    props: dict[str, np.ndarray] = {}
    for name, data in read_arrays(init):
        if name == "LGR":
            break
        if name == "INTEHEAD" and (tuple(data[8:11]) != shape or data[11] != mask.sum()):
            raise ValueError(f"{init} does not belong to this grid")
        if name in _NOT_CELL or name in props or data.dtype.kind not in "fib":
            continue
        if data.size == mask.size:
            full = data
        elif data.size == mask.sum():
            if data.dtype.kind == "f":
                full = np.full(mask.size, np.nan)
            else:
                full = np.zeros(mask.size, dtype=data.dtype)
            full[mask] = data
        else:
            continue
        props[name] = full.reshape(shape, order="F")
    return props


def _centres(coord, zcorn, shape):
    """Cell centres as the mean of the eight corners. A corner's depth comes
    from ZCORN and its x, y from the pillar it sits on, at that depth."""
    nx, ny, nz = shape
    pillars = coord.reshape(ny + 1, nx + 1, 6).transpose(1, 0, 2).astype(float)
    top, bottom = pillars[..., :3], pillars[..., 3:]
    zc = zcorn.reshape((2 * nx, 2 * ny, 2 * nz), order="F").astype(float)

    x, y, z = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for di in (0, 1):
        for dj in (0, 1):
            t0 = top[di : nx + di, dj : ny + dj, None, :]
            t1 = bottom[di : nx + di, dj : ny + dj, None, :]
            height = t1[..., 2] - t0[..., 2]
            for dk in (0, 1):
                zz = zc[di::2, dj::2, dk::2]
                # Fraction of the way down the pillar; 0 on a flat pillar.
                frac = np.zeros(shape)
                np.divide(zz - t0[..., 2], height, out=frac, where=height != 0)
                x += t0[..., 0] + frac * (t1[..., 0] - t0[..., 0])
                y += t0[..., 1] + frac * (t1[..., 1] - t0[..., 1])
                z += zz
    return x / 8, y / 8, z / 8
