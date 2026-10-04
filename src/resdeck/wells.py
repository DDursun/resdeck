"""Wells, and the well events of a schedule: drilling a well, changing its
control, shutting and opening it. Events are tied to a report date."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from fnmatch import fnmatchcase
from numbers import Integral, Real

from .deck import Deck, render
from .keywords import (
    COMPDAT,
    RUNSPEC,
    SECTIONS,
    SUMMARY,
    WCONINJE,
    WCONPROD,
    WELL_SUMMARY,
    WELLDIMS,
    WELOPEN,
    WELSPECS,
)
from .model import Model
from .schedule import insertion_point

# Control mode -> the field of the control that holds its target.
_PRODUCER_TARGETS = {"ORAT": "oil_rate", "LRAT": "liquid_rate", "BHP": "bhp"}
_INJECTOR_TARGETS = {"RATE": "rate", "BHP": "bhp"}
_INJECTED = ("WATER", "GAS")
_PHASES = ("OIL", "WATER", "GAS", "LIQ")
# Well and group names: no characters that break quoting or act as patterns.
_NAME = re.compile(r"[^\s'\"/*?]{1,8}")


@dataclass(frozen=True)
class Completion:
    """An interval of layers the well is connected to, ``k_top`` to
    ``k_bottom`` inclusive, 1-based."""

    k_top: int
    k_bottom: int
    diameter: float
    open: bool = True
    skin: float = 0.0


@dataclass(frozen=True)
class ProducerControl:
    """How a producer is operated. ``mode`` is ORAT, LRAT or BHP and its
    target must be set; the other values act as limits."""

    mode: str
    oil_rate: float | None = None
    liquid_rate: float | None = None
    bhp: float | None = None


@dataclass(frozen=True)
class InjectorControl:
    """How an injector is operated. ``fluid`` is WATER or GAS; ``mode`` is
    RATE (surface rate) or BHP and its target must be set; the other value
    acts as a limit."""

    fluid: str
    mode: str
    rate: float | None = None
    bhp: float | None = None


@dataclass(frozen=True)
class Well:
    """A vertical well in column (``i``, ``j``), 1-based. The control makes it
    a producer or an injector. ``phase`` is the preferred phase written to
    WELSPECS; by default OIL for a producer and the injected fluid for an
    injector."""

    name: str
    i: int
    j: int
    completions: tuple[Completion, ...]
    control: ProducerControl | InjectorControl
    group: str = "G1"
    phase: str | None = None


def format_well(well: Well) -> str:
    """Deck text that defines ``well``: WELSPECS, COMPDAT, and WCONPROD for a
    producer or WCONINJE for an injector."""
    _check_well(well)
    ctrl = well.control
    phase = well.phase or (ctrl.fluid if isinstance(ctrl, InjectorControl) else "OIL")
    welspecs = {
        "well": well.name,
        "group": well.group,
        "i": well.i,
        "j": well.j,
        "phase": phase,
    }
    compdat = [
        {
            "well": well.name,
            "i": well.i,
            "j": well.j,
            "k_top": c.k_top,
            "k_bottom": c.k_bottom,
            "status": "OPEN" if c.open else "SHUT",
            "diameter": float(c.diameter),
            "skin": float(c.skin),
        }
        for c in well.completions
    ]
    return (
        render(WELSPECS, [welspecs]) + render(COMPDAT, compdat) + _format_control(well.name, ctrl)
    )


def add_well(
    deck: Deck,
    well: Well,
    model: Model | None = None,
    *,
    at=None,
    add_to_summary: bool = True,
) -> Deck:
    """A new deck with ``well`` defined at report date ``at`` (a date from
    ``report_dates``), or at the start of the run when ``at`` is None.
    WELLDIMS is enlarged to make room for it. With ``model``, the well is
    checked against the grid.

    With ``add_to_summary``, the well is added to every well summary vector
    the SUMMARY section asks for by well name (WOPR, WBHP, ...), so it is
    reported like the wells already there.

    Wells defined in INCLUDE files are not seen by the duplicate-name check,
    and summary vectors in INCLUDE files are not extended.
    """
    text = format_well(well)
    if well.name in _well_names(deck):
        raise ValueError(f"well {well.name} is already in the deck")
    if model is not None:
        _check_on_grid(well, model)
    deck = _fit_welldims(_insert_at(deck, text, at))
    return _add_to_summary(deck, well.name) if add_to_summary else deck


def set_control(deck: Deck, name: str, control: ProducerControl | InjectorControl, at) -> Deck:
    """A new deck where well ``name`` switches to ``control`` at report date
    ``at``: a new rate or pressure target, or a new mode. Writing a control
    also opens the well if it was shut."""
    _check_control(name, control)
    return _insert_at(deck, _format_control(name, control), at, well=name)


def shut_well(deck: Deck, name: str, at) -> Deck:
    """A new deck where well ``name`` is shut at report date ``at``."""
    return _insert_at(deck, render(WELOPEN, [{"well": name, "status": "SHUT"}]), at, well=name)


def open_well(deck: Deck, name: str, at) -> Deck:
    """A new deck where well ``name`` is opened again at report date ``at``,
    with the control it had before it was shut."""
    return _insert_at(deck, render(WELOPEN, [{"well": name, "status": "OPEN"}]), at, well=name)


def _insert_at(deck: Deck, text: str, at, well: str | None = None) -> Deck:
    """Insert ``text`` at report date ``at``. With ``well``, that well must
    already be defined there: by a WELSPECS earlier in the schedule, which
    includes earlier events on the same date."""
    deck, offset = insertion_point(deck, at)
    if well is not None and well not in _well_names(deck, before=offset):
        when = "the start of the run" if at is None else f"{at:%d %b %Y}"
        raise ValueError(f"well {well} is not defined by {when}")
    newline = "" if offset == 0 or deck.text[offset - 1] == "\n" else "\n"
    return deck.insert(offset, newline + text)


def _format_control(name: str, ctrl: ProducerControl | InjectorControl) -> str:
    injector = isinstance(ctrl, InjectorControl)
    if injector:
        record = {"well": name, "fluid": ctrl.fluid, "status": "OPEN", "mode": ctrl.mode}
        targets = _INJECTOR_TARGETS
    else:
        record = {"well": name, "status": "OPEN", "mode": ctrl.mode}
        targets = _PRODUCER_TARGETS
    for field in targets.values():
        value = getattr(ctrl, field)
        record[field] = None if value is None else float(value)
    return render(WCONINJE if injector else WCONPROD, [record])


def _well_names(deck: Deck, before: int | None = None) -> set[str]:
    """Wells defined by WELSPECS, only those above offset ``before`` if given."""
    return {
        record.items["well"]
        for kw in deck.find(WELSPECS)
        if before is None or kw.start < before
        for record in kw.records
    }


def _check_well(well: Well) -> None:
    _check_name("well", well.name)
    _check_name("group", well.group)
    name = well.name
    if well.phase is not None and well.phase not in _PHASES:
        raise ValueError(f"well {name}: phase must be one of {list(_PHASES)}")
    _check_index(name, "i", well.i)
    _check_index(name, "j", well.j)
    if not well.completions:
        raise ValueError(f"well {name} has no completions")
    for c in well.completions:
        _check_index(name, "k_top", c.k_top)
        _check_index(name, "k_bottom", c.k_bottom)
        if c.k_top > c.k_bottom:
            raise ValueError(f"well {name}: bad completion interval {c.k_top}-{c.k_bottom}")
        _check_number(name, "diameter", c.diameter, positive=True)
        _check_number(name, "skin", c.skin)
    _check_control(name, well.control)


def _check_control(name: str, ctrl: ProducerControl | InjectorControl) -> None:
    if isinstance(ctrl, InjectorControl):
        if ctrl.fluid not in _INJECTED:
            raise ValueError(f"well {name}: injected fluid must be one of {list(_INJECTED)}")
        targets = _INJECTOR_TARGETS
    else:
        targets = _PRODUCER_TARGETS
    if ctrl.mode not in targets:
        raise ValueError(f"well {name}: control mode must be one of {sorted(targets)}")
    if getattr(ctrl, targets[ctrl.mode]) is None:
        raise ValueError(f"well {name}: mode {ctrl.mode} needs {targets[ctrl.mode]}")
    for field in targets.values():
        value = getattr(ctrl, field)
        if value is not None:
            # Rates may be zero; a pressure target or limit must be above zero.
            _check_number(name, field, value, positive=field == "bhp", minimum=0.0)


def _check_name(kind: str, name) -> None:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(
            f"{kind} name {name!r} must be 1 to 8 characters, without spaces, quotes, /, * or ?"
        )


def _check_index(name: str, what: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"well {name}: {what} must be a whole number of 1 or more, got {value!r}")


def _check_number(name: str, what: str, value, *, positive: bool = False, minimum=None) -> None:
    """``value`` must be a finite number; above 0 if ``positive``, at least
    ``minimum`` if given."""
    ok = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    number = float(value) if ok else 0.0
    if positive:
        ok, limit = ok and number > 0, " above 0"
    elif minimum is not None:
        ok, limit = ok and number >= minimum, f" of {minimum:g} or more"
    else:
        limit = ""
    if not ok:
        raise ValueError(f"well {name}: {what} must be a finite number{limit}, got {value!r}")


def _check_on_grid(well: Well, model: Model) -> None:
    if not (1 <= well.i <= model.nx and 1 <= well.j <= model.ny):
        raise ValueError(
            f"well {well.name}: ({well.i}, {well.j}) is outside the {model.nx} x {model.ny} grid"
        )
    column = model.active[well.i - 1, well.j - 1]
    connected = False
    for c in well.completions:
        if c.k_bottom > model.nz:
            raise ValueError(f"well {well.name}: layer {c.k_bottom} is below the {model.nz} layers")
        connected |= bool(column[c.k_top - 1 : c.k_bottom].any())
    if not connected:
        raise ValueError(f"well {well.name}: no completed cell is active")


def _add_to_summary(deck: Deck, name: str) -> Deck:
    """Append ``name`` to the well list of each well summary vector in the
    SUMMARY section. Lists that already cover it (empty list = all wells, or
    a matching pattern such as 'P*') are left alone."""
    summary = deck.find(SUMMARY)
    if not summary:
        return deck
    start = summary[0].end
    end = min((kw.start for kw in deck.find(*SECTIONS) if kw.start >= start), default=None)
    end = len(deck.text) if end is None else end
    records = [kw.records[0] for kw in deck.find(*WELL_SUMMARY) if start <= kw.start < end]
    # Last to first, so the offsets of earlier records stay valid.
    for record in reversed(records):
        wells = record.items["wells"]
        if wells and not any(fnmatchcase(name, pattern) for pattern in wells):
            deck = deck.insert(record.end, f" '{name}' ")
    return deck


def _fit_welldims(deck: Deck) -> Deck:
    """Make WELLDIMS large enough for the wells the deck defines: number of
    wells, most connections of one well, number of groups (FIELD not
    counted) and most wells in one group. Larger existing values are kept."""
    group_of = {
        record.items["well"]: record.items["group"]
        for kw in deck.find(WELSPECS)
        for record in kw.records
    }
    connections: Counter[str] = Counter()
    for kw in deck.find(COMPDAT):
        for record in kw.records:
            k_top, k_bottom = record.items["k_top"], record.items["k_bottom"]
            if k_top is not None and k_bottom is not None:
                connections[record.items["well"]] += k_bottom - k_top + 1
    wells_in = Counter(group for group in group_of.values() if group)
    required = {
        "max_wells": len(group_of),
        "max_connections": max(connections.values(), default=0),
        "max_groups": len(wells_in),
        "max_wells_per_group": max(wells_in.values(), default=0),
    }

    found = deck.find(WELLDIMS)
    if not found:
        runspec = deck.find(RUNSPEC)
        if not runspec:
            raise ValueError("deck has no RUNSPEC section")
        return deck.insert(runspec[0].end, render(WELLDIMS, [required]))
    record = found[0].records[0]
    new = {name: max(value, record.items[name] or 0) for name, value in required.items()}
    if all(record.items[name] == value for name, value in new.items()):
        return deck
    return deck.update(record, **new)
