import json
from datetime import date, datetime

import numpy as np
import pytest

from resdeck import (
    Completion,
    Deck,
    Drill,
    InjectorControl,
    Open,
    ProducerControl,
    Scenario,
    SetControl,
    Shut,
    Well,
    apply,
    load_scenario,
    report_dates,
    save_scenario,
)
from resdeck.scenario import from_dict, to_dict

TEXT = """\
RUNSPEC
START
   1 'JAN' 2020 /
WELLDIMS
   1 1 1 1 /
SCHEDULE
WELSPECS
   'OLD' 'G1' 1 1 1* 'OIL' /
/
TSTEP
   3*10 /
"""

JAN11, JAN15, JAN21 = datetime(2020, 1, 11), datetime(2020, 1, 15), datetime(2020, 1, 21)


def read(tmp_path, text=TEXT) -> Deck:
    path = tmp_path / "CASE.DATA"
    path.write_text(text, newline="\n")
    return Deck.read(path)


def producer(name="P1") -> Well:
    control = ProducerControl("ORAT", oil_rate=500, bhp=100)
    return Well(name, 2, 3, (Completion(1, 2, diameter=0.2),), control)


def injector(name="I1") -> Well:
    control = InjectorControl("WATER", "RATE", rate=800, bhp=300)
    return Well(name, 1, 2, (Completion(1, 1, diameter=0.2),), control)


def scenario() -> Scenario:
    events = (
        Drill(injector()),
        Drill(producer(), JAN11),
        SetControl("P1", ProducerControl("ORAT", oil_rate=200, bhp=100), JAN21),
        Shut("OLD", JAN11),
        Open("OLD", JAN21),
    )
    return Scenario(events, metadata={"seed": 7, "generator": "by hand"})


def order(deck: Deck, *snippets: str) -> list[int]:
    return [deck.text.index(s) for s in snippets]


def test_apply_plays_every_event(tmp_path):
    text = apply(read(tmp_path), scenario()).text
    for snippet in ("'I1' 'G1'", "'P1' 'G1'", "'ORAT' 200.0", "'OLD' 'SHUT'", "'OLD' 'OPEN' /"):
        assert snippet in text


def test_events_land_on_their_dates(tmp_path):
    deck = apply(read(tmp_path), scenario())
    at = order(deck, "'I1' 'G1'", "TSTEP", "'P1' 'G1'", "'OLD' 'SHUT'", "'ORAT' 200.0")
    assert at == sorted(at)
    assert deck.text.count("TSTEP") == 3


def test_event_dates_become_report_steps(tmp_path):
    deck = apply(read(tmp_path), Scenario((Shut("OLD", JAN15),)))
    assert JAN15 in report_dates(deck)
    assert "TSTEP\n  10 4 /\nWELOPEN\n  'OLD' 'SHUT' /\n" in deck.text


def test_events_are_played_in_date_order(tmp_path):
    late_first = Scenario((Shut("P1", JAN21), Drill(producer(), JAN11)))
    deck = apply(read(tmp_path), late_first)
    assert deck.text.index("'P1' 'G1'") < deck.text.index("'P1' 'SHUT'")


def test_events_on_one_date_keep_scenario_order(tmp_path):
    deck = apply(read(tmp_path), Scenario((Shut("OLD", JAN11), Open("OLD", JAN11))))
    assert deck.text.index("'OLD' 'SHUT'") < deck.text.index("'OLD' 'OPEN' /")


def test_scenario_dates_extend_the_run(tmp_path):
    deck = apply(read(tmp_path), Scenario(dates=(date(2020, 3, 1), date(2020, 4, 1))))
    assert report_dates(deck)[-2:] == [datetime(2020, 3, 1), datetime(2020, 4, 1)]


def test_date_and_datetime_can_be_mixed(tmp_path):
    events = (Shut("OLD", date(2020, 1, 11)), Open("OLD", JAN21))
    assert "'OLD' 'SHUT'" in apply(read(tmp_path), Scenario(events)).text


def test_apply_does_not_change_its_inputs(tmp_path):
    deck = read(tmp_path)
    apply(deck, scenario())
    assert deck.text == TEXT


def test_event_before_its_well_is_drilled_raises(tmp_path):
    bad = Scenario((Drill(producer(), JAN21), Shut("P1", JAN11)))
    with pytest.raises(ValueError, match="well P1 is not defined by 11 Jan 2020"):
        apply(read(tmp_path), bad)


def test_empty_scenario_changes_nothing(tmp_path):
    deck = read(tmp_path)
    assert apply(deck, Scenario()).text == deck.text


def test_dict_round_trip():
    assert from_dict(to_dict(scenario())) == scenario()


def test_to_dict_is_plain_json():
    data = json.loads(json.dumps(to_dict(scenario())))
    assert data["metadata"] == {"seed": 7, "generator": "by hand"}
    assert data["events"][0]["at"] is None
    assert data["events"][1]["at"] == "2020-01-11T00:00:00"
    assert data["events"][1]["well"]["control"]["type"] == "producer"
    assert data["events"][2] == {
        "type": "set_control",
        "at": "2020-01-21T00:00:00",
        "well": "P1",
        "control": {
            "type": "producer",
            "mode": "ORAT",
            "oil_rate": 200,
            "liquid_rate": None,
            "bhp": 100,
        },
    }


def test_saved_scenario_gives_the_same_deck(tmp_path):
    save_scenario(scenario(), tmp_path / "scenario.json")
    loaded = load_scenario(tmp_path / "scenario.json")
    assert loaded == scenario()
    assert apply(read(tmp_path), loaded).text == apply(read(tmp_path), scenario()).text


def test_numpy_numbers_can_be_saved(tmp_path):
    well = Well(
        "P1", np.int64(2), np.int64(3), (Completion(1, 2, np.float64(0.2)),), producer().control
    )
    save_scenario(Scenario((Drill(well),)), tmp_path / "scenario.json")
    assert load_scenario(tmp_path / "scenario.json").events[0].well.i == 2


def test_unknown_event_type_raises():
    with pytest.raises(ValueError, match="unknown event type 'frack'"):
        from_dict({"events": [{"type": "frack", "at": None, "well": "P1"}]})


def test_incomplete_scenario_raises():
    with pytest.raises(ValueError, match="not a valid scenario"):
        from_dict({"events": [{"type": "shut", "at": None}]})


def test_start_date_and_none_are_the_same_moment(tmp_path):
    # 1 Jan 2020 is the START date: both events are at the start, in list order.
    events = (Shut("OLD", date(2020, 1, 1)), Open("OLD", None))
    deck = apply(read(tmp_path), Scenario(events))
    assert deck.text.index("'OLD' 'SHUT'") < deck.text.index("'OLD' 'OPEN' /")


def test_none_then_start_date_keeps_order_too(tmp_path):
    events = (Shut("OLD", None), Open("OLD", date(2020, 1, 1)))
    deck = apply(read(tmp_path), Scenario(events))
    assert deck.text.index("'OLD' 'SHUT'") < deck.text.index("'OLD' 'OPEN' /")
