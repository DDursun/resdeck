import numpy as np
import pytest

from resdeck import (
    Completion,
    Deck,
    InjectorControl,
    Model,
    ProducerControl,
    Well,
    add_well,
    format_well,
)
from resdeck.keywords import WELSPECS

TEXT = """\
RUNSPEC
DIMENS
   3 3 2 /
-- wells, connections, groups, wells per group
WELLDIMS
-- sized for the base case
   1 1 1 1 /   -- one well
GRID
SCHEDULE
RPTRST
   BASIC=2 /
WELSPECS
   'OLD' 'G1' 1 1 1* 'OIL' /
/
TSTEP
   10 /
"""


def read(tmp_path, text=TEXT) -> Deck:
    path = tmp_path / "CASE.DATA"
    path.write_text(text, newline="\n")
    return Deck.read(path)


def well(name="P1", i=2, j=3, k=(1, 2), control=None) -> Well:
    control = control or ProducerControl("ORAT", oil_rate=500, bhp=100)
    return Well(name, i, j, (Completion(k[0], k[1], diameter=0.2),), control)


def model(active=None) -> Model:
    active = np.ones((3, 3, 2), dtype=bool) if active is None else active
    zeros = np.zeros((3, 3, 2))
    return Model(3, 3, 2, active, zeros, zeros, zeros, {})


def test_format_well_text():
    assert format_well(well()) == (
        "WELSPECS\n  'P1' 'G1' 2 3 1* 'OIL' /\n/\n\n"
        "COMPDAT\n  'P1' 2 3 1 2 'OPEN' 1* 1* 0.2 1* 0.0 /\n/\n\n"
        "WCONPROD\n  'P1' 'OPEN' 'ORAT' 500.0 1* 1* 1* 1* 100.0 /\n/\n\n"
    )


def test_one_compdat_record_per_completion():
    completions = (Completion(1, 1, 0.2), Completion(2, 2, 0.2, open=False, skin=1.5))
    text = format_well(Well("P1", 1, 1, completions, ProducerControl("BHP", bhp=100)))
    assert "  'P1' 1 1 1 1 'OPEN' 1* 1* 0.2 1* 0.0 /\n" in text
    assert "  'P1' 1 1 2 2 'SHUT' 1* 1* 0.2 1* 1.5 /\n" in text


def test_only_the_well_and_welldims_change(tmp_path):
    new = add_well(read(tmp_path), well())
    expected = TEXT.replace("1 1 1 1 /", "2 2 2 2 /")
    expected = expected.replace("SCHEDULE\n", "SCHEDULE\n" + format_well(well()))
    assert new.text == expected


def test_original_deck_is_not_changed(tmp_path):
    deck = read(tmp_path)
    add_well(deck, well())
    assert deck.text == TEXT


def test_welldims_keeps_later_items(tmp_path):
    deck = read(tmp_path, TEXT.replace("1 1 1 1 /", "5 2* 3 1* 7 /"))
    assert "   6 2 1 4 1* 7 /   -- one well\n" in add_well(deck, well()).text


def test_welldims_is_added_when_missing(tmp_path):
    deck = read(tmp_path, "RUNSPEC\nDIMENS\n 3 3 2 /\nSCHEDULE\n")
    assert add_well(deck, well()).text.startswith("RUNSPEC\nWELLDIMS\n  1 2 1 1 /\n\nDIMENS\n")


def test_written_deck_reads_back(tmp_path):
    add_well(read(tmp_path), well()).write(tmp_path / "NEW.DATA")
    welspecs = Deck.read(tmp_path / "NEW.DATA").find(WELSPECS)
    assert [r.items["well"] for kw in welspecs for r in kw.records] == ["P1", "OLD"]


def test_duplicate_well_name_raises(tmp_path):
    with pytest.raises(ValueError, match="already in the deck"):
        add_well(read(tmp_path), well("OLD"))


def test_deck_without_schedule_raises(tmp_path):
    with pytest.raises(ValueError, match="no SCHEDULE section"):
        add_well(read(tmp_path, "RUNSPEC\nGRID\n"), well())


def test_well_outside_grid_raises(tmp_path):
    with pytest.raises(ValueError, match="outside the 3 x 3 grid"):
        add_well(read(tmp_path), well(i=4), model())


def test_completion_below_grid_raises(tmp_path):
    with pytest.raises(ValueError, match="below the 2 layers"):
        add_well(read(tmp_path), well(k=(1, 3)), model())


def test_well_in_inactive_cells_raises(tmp_path):
    active = np.ones((3, 3, 2), dtype=bool)
    active[1, 2, :] = False
    with pytest.raises(ValueError, match="no completed cell is active"):
        add_well(read(tmp_path), well(), model(active))


def test_long_well_name_raises():
    with pytest.raises(ValueError, match="1 to 8 characters"):
        format_well(well("PRODUCER1"))


def test_reversed_completion_interval_raises():
    with pytest.raises(ValueError, match="bad completion interval"):
        format_well(well(k=(2, 1)))


def test_control_mode_needs_its_target():
    with pytest.raises(ValueError, match="mode LRAT needs liquid_rate"):
        format_well(well(control=ProducerControl("LRAT", bhp=100)))


def test_unknown_control_mode_raises():
    with pytest.raises(ValueError, match="control mode must be one of"):
        format_well(well(control=ProducerControl("GRAT")))


SUMMARY_TEXT = """\
RUNSPEC
WELLDIMS
   1 1 1 1 /
SUMMARY
FOPR
WOPR
  'OLD'
/
WBHP
/
WGOR
  'P*' /
WWCT   -- water cut
  OLD /
SCHEDULE
WOPR
  'OLD' /
"""


def test_well_is_added_to_summary_well_lists(tmp_path):
    text = add_well(read(tmp_path, SUMMARY_TEXT), well()).text
    assert "WOPR\n  'OLD'\n 'P1' /\n" in text
    assert "WWCT   -- water cut\n  OLD  'P1' /\n" in text


def test_summary_list_for_all_wells_is_left_alone(tmp_path):
    assert "WBHP\n/\n" in add_well(read(tmp_path, SUMMARY_TEXT), well()).text


def test_summary_list_with_matching_pattern_is_left_alone(tmp_path):
    assert "WGOR\n  'P*' /\n" in add_well(read(tmp_path, SUMMARY_TEXT), well()).text


def test_only_the_summary_section_is_extended(tmp_path):
    text = add_well(read(tmp_path, SUMMARY_TEXT), well()).text
    assert text.endswith("WOPR\n  'OLD' /\n")


def test_summary_can_be_left_unchanged(tmp_path):
    text = add_well(read(tmp_path, SUMMARY_TEXT), well(), add_to_summary=False).text
    assert text.split("SCHEDULE\n")[0] == SUMMARY_TEXT.split("SCHEDULE\n")[0].replace(
        "1 1 1 1 /", "2 2 2 2 /"
    )


def injector(fluid="WATER", mode="RATE", rate=800.0, bhp=300.0) -> Well:
    control = InjectorControl(fluid, mode, rate=rate, bhp=bhp)
    return Well("I1", 1, 2, (Completion(1, 1, diameter=0.2),), control)


def test_format_injector_text():
    assert format_well(injector()) == (
        "WELSPECS\n  'I1' 'G1' 1 2 1* 'WATER' /\n/\n\n"
        "COMPDAT\n  'I1' 1 2 1 1 'OPEN' 1* 1* 0.2 1* 0.0 /\n/\n\n"
        "WCONINJE\n  'I1' 'WATER' 'OPEN' 'RATE' 800.0 1* 300.0 /\n/\n\n"
    )


def test_injector_phase_follows_the_injected_fluid():
    assert "  'I1' 'G1' 1 2 1* 'GAS' /\n" in format_well(injector("GAS"))


def test_injector_is_added_like_a_producer(tmp_path):
    new = add_well(read(tmp_path), injector())
    assert new.text.split("SCHEDULE\n")[1].startswith(format_well(injector()))
    assert "   2 1 2 2 /" in new.text


def test_unknown_injected_fluid_raises():
    with pytest.raises(ValueError, match="injected fluid must be one of"):
        format_well(injector("OIL"))


def test_injector_mode_needs_its_target():
    with pytest.raises(ValueError, match="mode RATE needs rate"):
        format_well(injector(rate=None))


def test_producer_mode_on_injector_raises():
    with pytest.raises(ValueError, match="control mode must be one of"):
        format_well(injector(mode="ORAT"))
