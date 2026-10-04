"""Wells, and the well events of a schedule: drilling a well, changing its
control, shutting and opening it. Events are tied to a report date."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

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
    deck = _grow_welldims(_insert_at(deck, text, at), well)
    return _add_to_summary(deck, well.name) if add_to_summary else deck


def set_control(deck: Deck, name: str, control: ProducerControl | InjectorControl, at) -> Deck:
    """A new deck where well ``name`` switches to ``control`` at report date
    ``at``: a new rate or pressure target, or a new mode. Writing a control
    also opens the well if it was shut."""
    _check_control(name, control)
    _check_defined(deck, name)
    return _insert_at(deck, _format_control(name, control), at)


def shut_well(deck: Deck, name: str, at) -> Deck:
    """A new deck where well ``name`` is shut at report date ``at``."""
    _check_defined(deck, name)
    return _insert_at(deck, render(WELOPEN, [{"well": name, "status": "SHUT"}]), at)


def open_well(deck: Deck, name: str, at) -> Deck:
    """A new deck where well ``name`` is opened again at report date ``at``,
    with the control it had before it was shut."""
    _check_defined(deck, name)
    return _insert_at(deck, render(WELOPEN, [{"well": name, "status": "OPEN"}]), at)


def _insert_at(deck: Deck, text: str, at) -> Deck:
    deck, offset = insertion_point(deck, at)
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


def _well_names(deck: Deck) -> set[str]:
    return {record.items["well"] for kw in deck.find(WELSPECS) for record in kw.records}


def _check_defined(deck: Deck, name: str) -> None:
    if name not in _well_names(deck):
        raise ValueError(f"well {name} is not defined in the deck")


def _check_well(well: Well) -> None:
    if not 1 <= len(well.name) <= 8:
        raise ValueError(f"well name {well.name!r} must be 1 to 8 characters")
    if not well.completions:
        raise ValueError(f"well {well.name} has no completions")
    for c in well.completions:
        if not 1 <= c.k_top <= c.k_bottom:
            raise ValueError(f"well {well.name}: bad completion interval {c.k_top}-{c.k_bottom}")
    _check_control(well.name, well.control)


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


def _grow_welldims(deck: Deck, well: Well) -> Deck:
    """Make room in WELLDIMS for one more well. The group counts are raised
    by one whether or not the group is new; too large is harmless."""
    found = deck.find(WELLDIMS)
    old = found[0].records[0].items if found else {}
    layers = sum(c.k_bottom - c.k_top + 1 for c in well.completions)
    new = {
        "max_wells": (old.get("max_wells") or 0) + 1,
        "max_connections": max(old.get("max_connections") or 0, layers),
        "max_groups": (old.get("max_groups") or 0) + 1,
        "max_wells_per_group": (old.get("max_wells_per_group") or 0) + 1,
    }
    if found:
        return deck.update(found[0].records[0], **new)
    runspec = deck.find(RUNSPEC)
    if not runspec:
        raise ValueError("deck has no RUNSPEC section")
    return deck.insert(runspec[0].end, render(WELLDIMS, [new]))
