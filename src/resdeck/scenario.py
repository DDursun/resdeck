"""Scenarios: a development plan as plain data, and playing it onto a deck.

A scenario is a list of dated well events. It holds no deck text and no
simulator syntax, so it can be saved as JSON next to the simulation results
and written by another simulator's writer later. ``apply`` is the Eclipse /
OPM Flow writer: it plays the events onto a ``Deck``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .deck import Deck
from .model import Model
from .schedule import add_dates, as_datetime, report_dates
from .wells import (
    Completion,
    InjectorControl,
    ProducerControl,
    Well,
    add_well,
    open_well,
    set_control,
    shut_well,
)


@dataclass(frozen=True)
class Drill:
    """Drill ``well`` and put it on its control at ``at``."""

    well: Well
    at: date | None = None


@dataclass(frozen=True)
class SetControl:
    """Switch the well named ``well`` to ``control`` at ``at``: a new rate or
    pressure target, or a new mode."""

    well: str
    control: ProducerControl | InjectorControl
    at: date | None


@dataclass(frozen=True)
class Shut:
    """Shut the well named ``well`` at ``at``."""

    well: str
    at: date | None


@dataclass(frozen=True)
class Open:
    """Open the well named ``well`` again at ``at``."""

    well: str
    at: date | None


Event = Drill | SetControl | Shut | Open
_EVENT_TYPES = {"drill": Drill, "set_control": SetControl, "shut": Shut, "open": Open}
_CONTROL_TYPES = {"producer": ProducerControl, "injector": InjectorControl}


@dataclass(frozen=True)
class Scenario:
    """A development plan.

    ``events`` are the well events; each has ``at``, a date or datetime, or
    None for the start of the run. ``dates`` are report dates the run must
    have besides the event dates, for example a regular reporting interval
    or the end of the run. ``metadata`` is free: seed, generator, config,
    realization, anything JSON can hold.
    """

    events: tuple[Event, ...] = ()
    dates: tuple[date, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def apply(
    deck: Deck, scenario: Scenario, model: Model | None = None, *, add_to_summary: bool = True
) -> Deck:
    """A new deck with ``scenario`` played onto ``deck``.

    Every date in the scenario becomes a report step first, so events need
    not fall on the report steps the deck already has. The events are then
    played in date order; events on the same date keep their order in the
    scenario. ``model`` and ``add_to_summary`` are passed on to ``add_well``.
    """
    wanted = [*scenario.dates, *(e.at for e in scenario.events if e.at is not None)]
    start = None
    if wanted:
        deck = add_dates(deck, wanted)
        start = report_dates(deck)[0]

    def when(event):
        # None and the START date itself are the same moment: both sort as the
        # start of the run, so such events keep their order in the scenario.
        at = None if event.at is None else as_datetime(event.at)
        return datetime.min if at is None or at == start else at

    for event in sorted(scenario.events, key=when):
        if isinstance(event, Drill):
            deck = add_well(deck, event.well, model, at=event.at, add_to_summary=add_to_summary)
        elif isinstance(event, SetControl):
            deck = set_control(deck, event.well, event.control, event.at)
        elif isinstance(event, Shut):
            deck = shut_well(deck, event.well, event.at)
        elif isinstance(event, Open):
            deck = open_well(deck, event.well, event.at)
        else:
            raise ValueError(f"unknown event {event!r}")
    return deck


def save_scenario(scenario: Scenario, path) -> None:
    """Write ``scenario`` to ``path`` as JSON."""
    text = json.dumps(to_dict(scenario), indent=2, default=_plain)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load_scenario(path) -> Scenario:
    """Read a scenario written by ``save_scenario``. Dates come back as
    datetimes."""
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def to_dict(scenario: Scenario) -> dict:
    """``scenario`` as dicts, lists, strings and numbers; dates in ISO format."""
    return {
        "metadata": scenario.metadata,
        "dates": [_iso(d) for d in scenario.dates],
        "events": [_event_dict(e) for e in scenario.events],
    }


def from_dict(data: dict) -> Scenario:
    """The scenario described by ``data``, the inverse of ``to_dict``."""
    try:
        return Scenario(
            events=tuple(_event(e) for e in data["events"]),
            dates=tuple(datetime.fromisoformat(d) for d in data.get("dates", ())),
            metadata=dict(data.get("metadata", {})),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"not a valid scenario: {error!r}") from None


def _iso(at: date | None) -> str | None:
    return None if at is None else as_datetime(at).isoformat()


def _plain(value):
    """JSON fallback for numpy numbers, which generators will produce."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot save {value!r} in a scenario")


def _control_dict(control: ProducerControl | InjectorControl) -> dict:
    kind = "injector" if isinstance(control, InjectorControl) else "producer"
    return {"type": kind, **dataclasses.asdict(control)}


def _event_dict(event: Event) -> dict:
    kind = next((k for k, cls in _EVENT_TYPES.items() if isinstance(event, cls)), None)
    if kind is None:
        raise ValueError(f"unknown event {event!r}")
    out: dict[str, Any] = {"type": kind, "at": _iso(event.at)}
    if isinstance(event, Drill):
        out["well"] = {
            **dataclasses.asdict(event.well),
            "control": _control_dict(event.well.control),
        }
    else:
        out["well"] = event.well
    if isinstance(event, SetControl):
        out["control"] = _control_dict(event.control)
    return out


def _control(data: dict) -> ProducerControl | InjectorControl:
    fields = dict(data)
    kind = fields.pop("type")
    if kind not in _CONTROL_TYPES:
        raise ValueError(f"not a valid scenario: unknown control type {kind!r}")
    return _CONTROL_TYPES[kind](**fields)


def _event(data: dict) -> Event:
    kind = data["type"]
    if kind not in _EVENT_TYPES:
        raise ValueError(f"not a valid scenario: unknown event type {kind!r}")
    at = None if data["at"] is None else datetime.fromisoformat(data["at"])
    if kind == "drill":
        fields = dict(data["well"])
        fields["completions"] = tuple(Completion(**c) for c in fields["completions"])
        fields["control"] = _control(fields["control"])
        return Drill(Well(**fields), at)
    if kind == "set_control":
        return SetControl(data["well"], _control(data["control"]), at)
    return _EVENT_TYPES[kind](data["well"], at)
