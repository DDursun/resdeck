import numpy as np
import pytest

from resdeck import (
    Completion,
    Model,
    ProducerControl,
    Well,
    active_columns,
    column_xy,
    nearest_columns,
    well_cells,
)


def model(nx=4, ny=3, nz=2, dx=100.0, dy=50.0, active=None) -> Model:
    """A box grid: column (i, j) is centred at ((i - 0.5) dx, (j - 0.5) dy)."""
    x = np.broadcast_to(((np.arange(nx) + 0.5) * dx)[:, None, None], (nx, ny, nz)).copy()
    y = np.broadcast_to(((np.arange(ny) + 0.5) * dy)[None, :, None], (nx, ny, nz)).copy()
    active = np.ones((nx, ny, nz), dtype=bool) if active is None else active
    return Model(nx, ny, nz, active, x, y, np.zeros((nx, ny, nz)), {})


def test_well_cells_lists_every_completed_cell():
    completions = (Completion(1, 2, 0.2), Completion(4, 4, 0.2))
    well = Well("P1", 3, 2, completions, ProducerControl("BHP", bhp=100))
    assert well_cells(well) == [(3, 2, 1), (3, 2, 2), (3, 2, 4)]


def test_active_columns_need_one_active_cell():
    active = np.ones((4, 3, 2), dtype=bool)
    active[0, 0, :] = False  # whole column inactive
    active[1, 0, 0] = False  # one cell inactive
    columns = active_columns(model(active=active))
    assert columns.shape == (4, 3)
    assert not columns[0, 0]
    assert columns[1, 0]
    assert columns.sum() == 11


def test_column_xy_is_the_column_centre():
    x, y = column_xy(model())
    assert x[:, 0].tolist() == [50.0, 150.0, 250.0, 350.0]
    assert y[0, :].tolist() == [25.0, 75.0, 125.0]


def test_column_xy_uses_active_cells_only():
    m = model()
    m.x[2, 1, 0], m.x[2, 1, 1] = 240.0, 260.0  # a leaning column
    m.active[2, 1, 1] = False
    assert column_xy(m)[0][2, 1] == 240.0


def test_inactive_column_still_has_coordinates():
    active = np.ones((4, 3, 2), dtype=bool)
    active[0, 0, :] = False
    x, y = column_xy(model(active=active))
    assert (x[0, 0], y[0, 0]) == (50.0, 25.0)


def test_point_snaps_to_the_column_it_is_in():
    i, j, inside = nearest_columns(model(), 260.0, 80.0)
    assert (int(i), int(j), bool(inside)) == (3, 2, True)


def test_points_can_be_arrays():
    i, j, inside = nearest_columns(model(), [50.0, 349.0], [25.0, 124.0])
    assert i.tolist() == [1, 4]
    assert j.tolist() == [1, 3]
    assert inside.tolist() == [True, True]


def test_point_far_from_the_grid_is_outside():
    _, _, inside = nearest_columns(model(), [-200.0, 600.0, 200.0], [75.0, 75.0, 400.0])
    assert inside.tolist() == [False, False, False]


def test_point_just_past_the_edge_is_outside():
    # The grid covers x 0 to 400 and y 0 to 150.
    _, _, inside = nearest_columns(model(), [399.0, 405.0, -5.0, 200.0], [75.0, 75.0, 75.0, 152.0])
    assert inside.tolist() == [True, False, False, False]


def test_nearest_column_may_be_inactive():
    active = np.ones((4, 3, 2), dtype=bool)
    active[1, 1, :] = False
    m = model(active=active)
    i, j, inside = nearest_columns(m, 150.0, 75.0)
    assert (int(i), int(j), bool(inside)) == (2, 2, True)
    assert not active_columns(m)[i - 1, j - 1]


def test_single_row_grid_raises():
    with pytest.raises(ValueError, match="at least 2 columns in each direction"):
        nearest_columns(model(ny=1), 50.0, 25.0)


def test_long_thin_cells_are_measured_along_each_direction():
    # Cells 100 long and 1 wide: the grid covers x 0 to 400, y 0 to 3.
    thin = model(dx=100.0, dy=1.0)
    i, j, inside = nearest_columns(thin, [90.0, 50.0], [0.5, -5.0])
    assert (i[0], j[0], inside[0]) == (1, 1, True)  # far along the cell, still in it
    assert not inside[1]  # five cell widths below the grid


def test_well_cells_follow_completion_order():
    completions = (Completion(4, 4, 0.2), Completion(1, 2, 0.2))
    well = Well("P1", 3, 2, completions, ProducerControl("BHP", bhp=100))
    assert [k for _, _, k in well_cells(well)] == [4, 1, 2]
