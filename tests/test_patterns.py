from datetime import date

import numpy as np
import pytest

from resdeck import (
    PATTERNS,
    Completion,
    InjectorControl,
    Model,
    ProducerControl,
    Well,
    build_pattern,
    column_xy,
    well_cells,
)

PRODUCER = ProducerControl("BHP", bhp=100)
INJECTOR = InjectorControl("WATER", "RATE", rate=50, bhp=400)


def model(nx=21, ny=21, nz=3, dx=10.0, dy=10.0, active=None) -> Model:
    """A box grid: column (i, j) is centred at ((i - 0.5) dx, (j - 0.5) dy)."""
    x = np.broadcast_to(((np.arange(nx) + 0.5) * dx)[:, None, None], (nx, ny, nz)).copy()
    y = np.broadcast_to(((np.arange(ny) + 0.5) * dy)[None, :, None], (nx, ny, nz)).copy()
    active = np.ones((nx, ny, nz), dtype=bool) if active is None else active
    return Model(nx, ny, nz, active, x, y, np.zeros((nx, ny, nz)), {})


def build(pattern, anchor=(11, 11), m=None, **options):
    options = {"producer": PRODUCER, "injector": INJECTOR, "diameter": 0.2, **options}
    return build_pattern(m or model(), pattern, anchor, **options)


def columns(events):
    return [well_cells(e.well)[0][:2] for e in events]


def names(events):
    return [e.well.name for e in events]


def test_single_well_is_a_producer_at_the_anchor():
    (event,) = build("single", (4, 7))
    assert event.well.name == "P1"
    assert columns([event]) == [(4, 7)]
    assert event.well.control == PRODUCER


def test_inverted_single_well_is_an_injector():
    (event,) = build("single", inverted=True)
    assert event.well.name == "I1"
    assert event.well.control == INJECTOR


def test_single_well_needs_only_its_own_control():
    (event,) = build("single", injector=None)
    assert event.well.control == PRODUCER


def test_five_spot_in_cells_puts_corners_that_many_columns_away():
    events = build("five_spot", radius_cells=3)
    assert columns(events) == [(11, 11), (14, 14), (8, 14), (8, 8), (14, 8)]


def test_five_spot_has_a_central_producer_and_four_injectors():
    events = build("five_spot", radius_cells=3)
    assert names(events) == ["P1", "I1", "I2", "I3", "I4"]
    assert [e.well.control for e in events] == [PRODUCER] + [INJECTOR] * 4


def test_inverted_five_spot_has_a_central_injector_and_four_producers():
    events = build("five_spot", radius_cells=3, inverted=True)
    assert names(events) == ["I1", "P1", "P2", "P3", "P4"]
    assert [e.well.control for e in events] == [INJECTOR] + [PRODUCER] * 4


def test_five_spot_radius_is_the_distance_to_the_corners():
    # Cells are 10 wide, so corners 3 columns away are 30 * sqrt(2) from the anchor.
    events = build("five_spot", radius=30 * np.sqrt(2))
    assert columns(events) == [(11, 11), (14, 14), (8, 14), (8, 8), (14, 8)]


def test_four_spot_is_a_triangle():
    events = build("four_spot", radius_cells=4)
    assert columns(events) == [(11, 11), (11, 15), (8, 9), (14, 9)]


def test_seven_spot_has_six_wells_about_a_radius_away():
    m = model()
    events = build("seven_spot", m=m, radius=47.0)
    x, y = column_xy(m)
    distance = [
        np.hypot(x[i - 1, j - 1] - 105.0, y[i - 1, j - 1] - 105.0) for i, j in columns(events)
    ]
    assert len(set(columns(events))) == 7
    assert distance[0] == 0.0
    assert all(abs(d - 47.0) < 10.0 for d in distance[1:])


def test_nine_spot_has_corner_and_side_wells():
    events = build("nine_spot", radius_cells=2)
    assert columns(events)[0] == (11, 11)
    assert set(columns(events)[1:]) == {
        (13, 13), (11, 13), (9, 13), (9, 11), (9, 9), (11, 9), (13, 9), (13, 11)
    }  # fmt: skip


def test_every_pattern_starts_at_the_anchor():
    for pattern in PATTERNS:
        assert columns(build(pattern, radius_cells=4))[0] == (11, 11)


def test_wells_are_completed_in_all_active_layers():
    (event,) = build("single")
    assert event.well.completions == (Completion(1, 3, 0.2),)


def test_inactive_layers_split_the_completions():
    active = np.ones((21, 21, 5), dtype=bool)
    active[10, 10, 2] = False
    (event,) = build("single", m=model(nz=5, active=active))
    assert event.well.completions == (Completion(1, 2, 0.2), Completion(4, 5, 0.2))


def test_names_continue_from_existing_wells():
    existing = [Well("P1", 1, 1, (Completion(1, 3, 0.2),), PRODUCER)]
    existing += [e.well for e in build("five_spot", (5, 5), radius_cells=2, existing=existing)]
    events = build("five_spot", (15, 15), radius_cells=2, inverted=True, existing=existing)
    assert names(events) == ["I5", "P3", "P4", "P5", "P6"]


def test_date_group_and_prefixes_reach_every_well():
    events = build(
        "five_spot",
        radius_cells=2,
        at=date(2020, 1, 1),
        group="NORTH",
        producer_prefix="PROD",
        injector_prefix="INJ",
    )
    assert {e.at for e in events} == {date(2020, 1, 1)}
    assert {e.well.group for e in events} == {"NORTH"}
    assert names(events) == ["PROD1", "INJ1", "INJ2", "INJ3", "INJ4"]


def test_well_off_the_grid_rejects_the_pattern():
    with pytest.raises(ValueError, match="fall off the grid"):
        build("five_spot", (2, 11), radius_cells=3)
    with pytest.raises(ValueError, match="fall off the grid"):
        build("five_spot", (2, 11), radius=45.0)


def test_well_just_past_the_edge_rejects_the_pattern():
    # A 50 x 50 grid: three corners land 2 past the edge, nearest to an edge column.
    with pytest.raises(ValueError, match="3 of its wells fall off the grid"):
        build("five_spot", (2, 2), m=model(nx=5, ny=5), radius=17 * np.sqrt(2))


def test_single_well_ignores_the_size():
    assert columns(build("single", radius=30.0)) == [(11, 11)]


def test_inactive_column_rejects_the_pattern():
    active = np.ones((21, 21, 3), dtype=bool)
    active[13, 13, :] = False
    with pytest.raises(ValueError, match=r"column \(14, 14\) is inactive"):
        build("five_spot", m=model(active=active), radius_cells=3)


def test_existing_well_on_a_column_rejects_the_pattern():
    existing = [Well("OLD", 8, 8, (Completion(1, 3, 0.2),), PRODUCER)]
    with pytest.raises(ValueError, match=r"column \(8, 8\) already has well OLD"):
        build("five_spot", radius_cells=3, existing=existing)


def test_too_small_radius_rejects_the_pattern():
    with pytest.raises(ValueError, match="two wells land on column"):
        build("five_spot", radius=2.0)


def test_radius_must_be_given_exactly_once():
    with pytest.raises(ValueError, match="exactly one of radius and radius_cells"):
        build("five_spot")
    with pytest.raises(ValueError, match="exactly one of radius and radius_cells"):
        build("five_spot", radius=30.0, radius_cells=3)


def test_radius_must_be_positive():
    with pytest.raises(ValueError, match="radius_cells must be a finite number above 0"):
        build("five_spot", radius_cells=0)


def test_unknown_pattern_raises():
    with pytest.raises(ValueError, match="unknown pattern 'six_spot'"):
        build("six_spot", radius_cells=3)


def test_anchor_outside_the_grid_raises():
    with pytest.raises(ValueError, match=r"anchor \(22, 11\) is outside the 21 x 21 grid"):
        build("single", (22, 11))


def test_missing_control_raises():
    with pytest.raises(ValueError, match="injector must be a InjectorControl"):
        build("five_spot", radius_cells=3, injector=None)
