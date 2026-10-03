import struct

import numpy as np
import pytest

from resdeck import Model, read_arrays

DTYPES = {"INTE": ">i4", "REAL": ">f4", "DOUB": ">f8", "LOGI": ">i4", "CHAR": "S8"}


def record(payload: bytes) -> bytes:
    size = struct.pack(">i", len(payload))
    return size + payload + size


def array(name: str, kind: str, values) -> bytes:
    """One array as Eclipse writes it: a header record, then blocks of data."""
    data = np.asarray(values, dtype=DTYPES[kind])
    block = 105 if kind == "CHAR" else 1000
    out = record(name.ljust(8).encode() + struct.pack(">i", data.size) + kind.encode())
    for start in range(0, data.size, block):
        out += record(data[start : start + block].tobytes())
    return out


def box_grid(nx, ny, nz, dx=100.0, dy=50.0, dz=10.0, top=1000.0):
    """COORD and ZCORN of a regular box grid with vertical pillars."""
    coord = [
        v
        for j in range(ny + 1)
        for i in range(nx + 1)
        for v in (i * dx, j * dy, top, i * dx, j * dy, top + nz * dz)
    ]
    zcorn = np.empty((2 * nx, 2 * ny, 2 * nz))
    for k in range(nz):
        zcorn[:, :, 2 * k] = top + k * dz
        zcorn[:, :, 2 * k + 1] = top + (k + 1) * dz
    return coord, zcorn.ravel(order="F")


def write_egrid(path, nx, ny, nz, actnum=None):
    coord, zcorn = box_grid(nx, ny, nz)
    data = array("GRIDHEAD", "INTE", [1, nx, ny, nz] + [0] * 96)
    data += array("COORD", "REAL", coord) + array("ZCORN", "REAL", zcorn)
    if actnum is not None:
        data += array("ACTNUM", "INTE", actnum)
    data += array("ENDGRID", "INTE", [])
    path.write_bytes(data)
    return path


def write_init(path, nx, ny, nz, n_active, arrays):
    head = [0] * 100
    head[8:12] = nx, ny, nz, n_active
    data = array("INTEHEAD", "INTE", head)
    for name, kind, values in arrays:
        data += array(name, kind, values)
    path.write_bytes(data)
    return path


def test_read_arrays_returns_names_and_values(tmp_path):
    path = tmp_path / "X.INIT"
    data = array("NUMS", "INTE", [1, 2, 3]) + array("PORO", "REAL", [0.25, 0.5])
    data += array("FLAGS", "LOGI", [1, 0]) + array("NAMES", "CHAR", [b"P1      ", b"INJ     "])
    path.write_bytes(data)
    got = dict(read_arrays(path))
    assert got["NUMS"].tolist() == [1, 2, 3]
    assert got["PORO"].tolist() == [0.25, 0.5]
    assert got["FLAGS"].tolist() == [True, False]
    assert got["NAMES"].tolist() == ["P1", "INJ"]


def test_read_arrays_joins_data_blocks(tmp_path):
    path = tmp_path / "X.INIT"
    path.write_bytes(array("BIG", "DOUB", np.arange(2500.0)))
    ((name, data),) = read_arrays(path)
    assert name == "BIG"
    assert np.array_equal(data, np.arange(2500.0))


def test_read_arrays_rejects_other_files(tmp_path):
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nDIMENS\n 10 10 3 /\n")
    with pytest.raises(ValueError, match="not an unformatted Eclipse file"):
        list(read_arrays(path))


def test_grid_dimensions_and_cell_centres(tmp_path):
    model = Model.read(write_egrid(tmp_path / "C.EGRID", 3, 2, 2))
    assert (model.nx, model.ny, model.nz) == (3, 2, 2)
    assert model.x.shape == (3, 2, 2)
    assert model.x[:, 0, 0].tolist() == [50.0, 150.0, 250.0]
    assert model.y[0, :, 0].tolist() == [25.0, 75.0]
    assert model.z[0, 0, :].tolist() == [1005.0, 1015.0]


def test_all_cells_active_without_actnum(tmp_path):
    model = Model.read(write_egrid(tmp_path / "C.EGRID", 3, 2, 2))
    assert model.active.all()
    assert model.props == {}


def test_actnum_is_indexed_i_j_k(tmp_path):
    actnum = np.ones(12, dtype=int)
    actnum[1] = 0  # second cell in natural order: i=2, j=1, k=1
    model = Model.read(write_egrid(tmp_path / "C.EGRID", 3, 2, 2, actnum))
    assert not model.active[1, 0, 0]
    assert model.active.sum() == 11


def test_init_arrays_are_spread_over_active_cells(tmp_path):
    actnum = np.ones(12, dtype=int)
    actnum[1] = 0
    egrid = write_egrid(tmp_path / "C.EGRID", 3, 2, 2, actnum)
    arrays = [
        ("PORO", "REAL", np.arange(11) / 16),
        ("SATNUM", "INTE", np.arange(1, 12)),
        ("PORV", "REAL", np.arange(12.0)),  # PORV covers all cells, active or not
        ("TAB", "DOUB", np.zeros(11)),
    ]
    model = Model.read(egrid, write_init(tmp_path / "C.INIT", 3, 2, 2, 11, arrays))
    assert sorted(model.props) == ["PORO", "PORV", "SATNUM"]
    assert np.isnan(model.props["PORO"][1, 0, 0])
    assert model.props["PORO"][2, 0, 0] == 1 / 16
    assert model.props["SATNUM"][1, 0, 0] == 0
    assert model.props["SATNUM"][2, 1, 1] == 11
    assert model.props["PORV"][1, 0, 0] == 1.0


def test_init_of_another_grid_raises(tmp_path):
    egrid = write_egrid(tmp_path / "C.EGRID", 3, 2, 2)
    init = write_init(tmp_path / "C.INIT", 4, 2, 2, 16, [])
    with pytest.raises(ValueError, match="does not belong to this grid"):
        Model.read(egrid, init)


def test_egrid_without_geometry_raises(tmp_path):
    path = tmp_path / "C.EGRID"
    path.write_bytes(array("GRIDHEAD", "INTE", [1, 3, 2, 2]))
    with pytest.raises(ValueError, match="no COORD"):
        Model.read(path)
