"""Read, edit and write Eclipse-style simulation decks (.DATA files).

A deck is kept as its text. Only keywords with a ``KeywordSpec`` (see
``keywords.py``) are looked for and parsed; everything else is left exactly
as written, and edits replace or insert text in place. INCLUDE files are not
opened.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .keywords import INCLUDE, LIST, KeywordSpec

_REPEAT = re.compile(r"(\d+)\*(.*)")
# latin-1 maps every byte to a character, so any deck round-trips unchanged.
_ENCODING = "latin-1"


@dataclass(frozen=True)
class Record:
    """One record of a keyword. ``items`` maps every item name of the spec to
    its value, None where the deck leaves it defaulted. ``start`` and ``end``
    are the offsets in the deck text of the first token and of the closing
    ``/``."""

    items: dict[str, Any]
    start: int
    end: int


@dataclass(frozen=True)
class Keyword:
    """One occurrence of a keyword. ``start`` is the offset of its line in
    the deck text and ``end`` the offset of the line after its data."""

    spec: KeywordSpec
    records: tuple[Record, ...]
    start: int
    end: int


@dataclass(frozen=True)
class Deck:
    """The text of one deck file.

    ``insert``, ``splice`` and ``update`` return a new deck; offsets of keywords found
    in the old one are no longer valid in it, so call ``find`` again.
    """

    path: Path
    text: str

    @classmethod
    def read(cls, path) -> Deck:
        path = Path(path)
        return cls(path, path.read_text(encoding=_ENCODING))

    def write(self, path) -> None:
        """Write the deck to ``path``. Relative INCLUDE paths are rewritten
        so they still point at the original files from the new location."""
        path = Path(path)
        src, dst = self.path.parent.resolve(), path.parent.resolve()
        deck = self
        if src != dst:
            # Last to first, so the offsets of earlier keywords stay valid.
            for kw in reversed(self.find(INCLUDE)):
                old = kw.records[0].items["path"]
                if old and "$" not in old and not os.path.isabs(old):
                    deck = deck.update(kw.records[0], path=_relative(src / old, dst))
        path.write_text(deck.text, encoding=_ENCODING, newline="\n")

    def find(self, *specs: KeywordSpec) -> tuple[Keyword, ...]:
        """Every occurrence of the keywords ``specs``, in file order."""
        by_name = {spec.name: spec for spec in specs}
        names = "|".join(by_name)
        pattern = re.compile(rf"^[ \t]*({names})[ \t]*(?:--.*)?$", re.MULTILINE)
        return tuple(
            _keyword(self.text, by_name[m.group(1)], m) for m in pattern.finditer(self.text)
        )

    def insert(self, offset: int, text: str) -> Deck:
        """A new deck with ``text`` inserted at ``offset``."""
        return self.splice(offset, offset, text)

    def splice(self, start: int, end: int, text: str) -> Deck:
        """A new deck with the text between offsets ``start`` and ``end``
        replaced by ``text``."""
        return dataclasses.replace(self, text=self.text[:start] + text + self.text[end:])

    def update(self, record: Record, **items) -> Deck:
        """A new deck with the named items of ``record`` set to new values.
        Only the tokens of that record change."""
        names = list(record.items)
        tokens = _expand(
            [raw for _, raw in _scan(self.text, record.start, record.end) if raw is not None]
        )
        tokens += [None] * (len(names) - len(tokens))
        for name, value in items.items():
            if name not in names:
                raise ValueError(f"record has no item {name!r}")
            tokens[names.index(name)] = _format(value)
        text = _join(tokens)
        text = text + " " if text else text
        return dataclasses.replace(
            self, text=self.text[: record.start] + text + self.text[record.end :]
        )


def render(spec: KeywordSpec, records=()) -> str:
    """Deck text of a new keyword. Each record is a dict of item name to
    value; items left out or None are defaulted. Strings are quoted."""
    if spec.size is not LIST and len(records) != spec.size:
        raise ValueError(f"{spec.name} takes {spec.size} record(s), got {len(records)}")
    lines = [spec.name]
    for record in records:
        unknown = set(record) - set(spec.items)
        if unknown:
            raise ValueError(f"{spec.name} has no item {sorted(unknown)[0]!r}")
        values = [record.get(name) for name in spec.items]
        if spec.variadic:
            values = values[:-1] + list(values[-1] or ())
        lines.append(f"  {_join([_format(v) for v in values])} /")
    if spec.size is LIST:
        lines.append("/")
    return "\n".join(lines) + "\n\n"


def _format(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return repr(float(value))
    raise ValueError(f"cannot write {value!r} to a deck")


def _join(tokens, width: int = 78) -> str:
    """Tokens as record text; None is a defaulted item, dropped at the end.
    Long records are wrapped, since deck lines are limited to 132 characters."""
    tokens = list(tokens)
    while tokens and tokens[-1] is None:
        tokens.pop()
    lines = [""]
    for token in ("1*" if t is None else t for t in tokens):
        if lines[-1] and len(lines[-1]) + 1 + len(token) > width:
            lines.append(token)
        else:
            lines[-1] = f"{lines[-1]} {token}" if lines[-1] else token
    return "\n  ".join(lines)


def _expand(tokens) -> list[str | None]:
    """Expand repeat counts: ``2*5`` is two ``5`` tokens, ``2*`` two defaults."""
    out: list[str | None] = []
    for token in tokens:
        m = _REPEAT.fullmatch(token)
        if m:
            out += [m.group(2) or None] * int(m.group(1))
        else:
            out.append(token)
    return out


def _relative(target: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(target, start)).as_posix()
    except ValueError:  # different drive on Windows
        return target.as_posix()


def _tokens(line: str):
    """Yield ``(position, token)`` for one line. Quoted tokens keep their
    quotes; None stands for the ``/`` that ends a record. Comments and
    anything after the ``/`` are dropped."""
    n = len(line)

    def ends_record(p):
        return line[p] == "/" and (p + 1 == n or line[p + 1].isspace() or line[p + 1] == "-")

    p = 0
    while p < n:
        c = line[p]
        if c.isspace():
            p += 1
        elif c in "'\"":
            end = line.find(c, p + 1)
            end = n - 1 if end < 0 else end
            yield p, line[p : end + 1]
            p = end + 1
        elif line.startswith("--", p):
            return
        elif ends_record(p):
            yield p, None
            return
        else:
            # A "/" inside a token (an unquoted path) is part of the token.
            end = p + 1
            while end < n and not line[end].isspace() and line[end] not in "'\"":
                if ends_record(end):
                    break
                end += 1
            yield p, line[p:end]
            p = end


def _scan(text: str, pos: int, stop: int | None = None):
    """Yield ``(offset, token)`` from offset ``pos`` on, line by line."""
    stop = len(text) if stop is None else stop
    while pos < stop:
        eol = text.find("\n", pos, stop)
        eol = stop if eol < 0 else eol
        for p, token in _tokens(text[pos:eol]):
            yield pos + p, token
        pos = eol + 1


def _keyword(text: str, spec: KeywordSpec, match: re.Match) -> Keyword:
    data = min(match.end() + 1, len(text))
    if spec.size == 0:
        return Keyword(spec, (), match.start(), data)

    line = text.count("\n", 0, match.start()) + 1
    records: list[Record] = []
    tokens: list[str] = []
    first = None
    for at, token in _scan(text, data):
        if token is not None:
            first = at if first is None else first
            tokens.append(token)
            continue
        if spec.size is LIST and not tokens:
            break  # the lone "/" that closes the list
        records.append(Record(_items(spec, tokens, line), at if first is None else first, at))
        tokens, first = [], None
        if len(records) == spec.size:
            break
    else:
        raise ValueError(f"{spec.name} at line {line} is not closed by '/'")

    eol = text.find("\n", at)
    return Keyword(spec, tuple(records), match.start(), len(text) if eol < 0 else eol + 1)


def _items(spec: KeywordSpec, tokens, line: int) -> dict[str, Any]:
    tokens = _expand(tokens)
    tokens += [None] * (len(spec.items) - len(tokens))
    items: dict[str, Any] = {}
    for n, name in enumerate(spec.items):
        if spec.variadic and n == len(spec.items) - 1:
            rest = tokens[n:]
            items[name] = tuple(_value(spec, name, t, line) for t in rest if t is not None)
        else:
            items[name] = _value(spec, name, tokens[n], line)
    return items


def _value(spec: KeywordSpec, name: str, token: str | None, line: int):
    if token is None:
        return None
    bare = token.strip(token[0]) if token[0] in "'\"" else token
    try:
        return spec.items[name](bare)
    except ValueError:
        raise ValueError(f"{spec.name} at line {line}: bad {name} value {token!r}") from None
