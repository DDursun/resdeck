"""Report steps of the SCHEDULE section.

The run starts at the START date; every DATES record and every TSTEP value
ends one report step. ``report_dates`` lists the start and those step ends:
the dates events can be tied to. ``add_dates`` adds report steps, and
``insertion_point`` gives the place in the text for keywords that take
effect at one of the dates.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .deck import Deck, Keyword, render
from .keywords import DATES, END, INCLUDE, SCHEDULE, START, TSTEP

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_MONTH_NUMBER = {name: n for n, name in enumerate(MONTHS, 1)} | {"JLY": 7}


def report_dates(deck: Deck) -> list[datetime]:
    """The START date, then the end of every report step, in order."""
    start = _start(deck)
    return [start] + [when for _, _, when in _steps(deck, start)]


def add_dates(deck: Deck, dates) -> Deck:
    """A new deck with report steps ending at ``dates``. A date inside an
    existing step splits it in two; dates after the last step are appended
    as one DATES keyword. Dates that are already report dates are skipped."""
    start = _start(deck)
    wanted = sorted({_as_datetime(d) for d in dates})
    if wanted and wanted[0] < start:
        raise ValueError(f"{wanted[0]:%d %b %Y} is before the START date {start:%d %b %Y}")

    after_end = []
    for when in wanted:
        steps = _steps(deck, start)
        times = [start] + [t for _, _, t in steps]
        if when in times:
            continue
        if when > times[-1]:
            after_end.append(when)
            continue
        # The step that ends at times[n + 1] contains ``when``: split it.
        n = next(i for i, t in enumerate(times) if t > when) - 1
        kw, index, _ = steps[n]
        values = _values(kw)
        if kw.spec is TSTEP:
            values[index : index + 1] = [_days(when - times[n]), _days(times[n + 1] - when)]
        else:
            values.insert(index, _date_items(when))
        deck = deck.splice(kw.start, kw.end, _render(kw.spec, values))

    if after_end:
        end = _section(deck)[1]
        deck = deck.splice(end, end, _newline(deck, end) + _render(DATES, after_end))
    return deck


def insertion_point(deck: Deck, at=None) -> tuple[Deck, int]:
    """Where keywords that take effect at report date ``at`` go, None for the
    start of the run: after anything already there, before the next
    DATES or TSTEP. A DATES or TSTEP keyword that ``at`` falls inside is
    split in two, so the deck returned may differ from ``deck``."""
    time_keywords = _time_keywords(deck)
    if at is None or (time_keywords and _as_datetime(at) == _start(deck)):
        if time_keywords:
            return deck, time_keywords[0].start
        return deck, _section(deck)[1]

    at = _as_datetime(at)
    start = _start(deck)
    steps = _steps(deck, start)
    n = next((i for i, (_, _, when) in enumerate(steps) if when == at), None)
    if n is None:
        raise ValueError(f"no report step at {at:%d %b %Y %H:%M}; add it with add_dates")
    if n == len(steps) - 1:
        raise ValueError(f"{at:%d %b %Y} is the end of the run; nothing is simulated after it")

    kw, index, _ = steps[n]
    values = _values(kw)
    if index < len(values) - 1:
        text = _render(kw.spec, values[: index + 1]) + _render(kw.spec, values[index + 1 :])
        deck = deck.splice(kw.start, kw.end, text)
        steps = _steps(deck, start)
    return deck, steps[n + 1][0].start


def _start(deck: Deck) -> datetime:
    found = deck.find(START)
    if not found:
        raise ValueError("deck has no START date")
    return _date(found[0].records[0].items, "START")


def _section(deck: Deck) -> tuple[int, int]:
    """Text offsets of the SCHEDULE section: after its keyword, to END or the
    end. A SCHEDULE that includes another file is rejected, since its report
    steps and wells would be invisible here."""
    schedule = deck.find(SCHEDULE)
    if not schedule:
        raise ValueError("deck has no SCHEDULE section")
    begin = schedule[0].end
    ends = [kw.start for kw in deck.find(END) if kw.start >= begin]
    end = ends[0] if ends else len(deck.text)
    for kw in deck.find(INCLUDE):
        if begin <= kw.start < end:
            line = deck.text.count("\n", 0, kw.start) + 1
            raise ValueError(
                f"SCHEDULE includes another file at line {line}; schedules in INCLUDE "
                "files are not supported, copy its contents into the deck"
            )
    return begin, end


def _time_keywords(deck: Deck) -> list[Keyword]:
    begin, end = _section(deck)
    return [kw for kw in deck.find(DATES, TSTEP) if begin <= kw.start < end]


def _steps(deck: Deck, start: datetime) -> list[tuple[Keyword, int, datetime]]:
    """Every report step as (keyword, position in that keyword, end date)."""
    steps = []
    when = start
    for kw in _time_keywords(deck):
        for index, value in enumerate(_values(kw)):
            if kw.spec is TSTEP:
                if value <= 0:
                    raise ValueError(f"TSTEP has a step of {value} days")
                new = when + timedelta(days=value)
            else:
                new = _date(value, "DATES")
                if new <= when:
                    raise ValueError(f"DATES {new:%d %b %Y} is not after {when:%d %b %Y}")
            when = new
            steps.append((kw, index, when))
    return steps


def _values(kw: Keyword) -> list:
    """Step lengths of a TSTEP, or record items of a DATES keyword."""
    if kw.spec is TSTEP:
        return list(kw.records[0].items["steps"]) if kw.records else []
    return [record.items for record in kw.records]


def _render(spec, values) -> str:
    """A DATES or TSTEP keyword, without the blank line ``render`` adds, so
    that splitting a keyword keeps the spacing around it."""
    if spec is TSTEP:
        steps = tuple(int(v) if float(v).is_integer() else v for v in values)
        text = render(TSTEP, [{"steps": steps}])
    else:
        text = render(DATES, [v if isinstance(v, dict) else _date_items(v) for v in values])
    return text[:-1]


def _date(items: dict, keyword: str) -> datetime:
    month = _MONTH_NUMBER.get(str(items["month"]).upper())
    if items["day"] is None or items["year"] is None or month is None:
        raise ValueError(f"{keyword} has an incomplete date: {items}")
    when = datetime(items["year"], month, items["day"])
    if items["time"]:
        hours, minutes, seconds = (items["time"].split(":") + ["0", "0"])[:3]
        when += timedelta(hours=int(hours), minutes=int(minutes), seconds=float(seconds))
    return when


def _date_items(when: datetime) -> dict:
    time = when.strftime("%H:%M:%S") if when.time() != datetime.min.time() else None
    return {"day": when.day, "month": MONTHS[when.month - 1], "year": when.year, "time": time}


def _days(delta: timedelta) -> int | float:
    days = delta.total_seconds() / 86400
    return int(days) if days.is_integer() else days


def _as_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    raise ValueError(f"expected a date or datetime, got {value!r}")


def _newline(deck: Deck, offset: int) -> str:
    """A line break if ``offset`` is not at the start of a line."""
    return "" if offset == 0 or deck.text[offset - 1] == "\n" else "\n"
