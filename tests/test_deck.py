import pytest

from resdeck import Deck, render
from resdeck.keywords import (
    DIMENS,
    INCLUDE,
    PROPS,
    SCHEDULE,
    START,
    WELL_SUMMARY,
    WELLDIMS,
    WELSPECS,
)

TEXT = """\
-- a small deck
RUNSPEC
TITLE
   My case 1

DIMENS
   2 2 1 /

START
   1 'JAN' 2020 /

WELLDIMS
-- wells, then connections
   5 2* 3 /   -- room for five

GRID
INCLUDE
   'grid/case.grdecl' /   -- the realization

SUMMARY
WBHP
  PROD
  INJ
/

SCHEDULE   -- wells from here
WELSPECS
   'P1' 'G1' 1 1 1* 'OIL' /   -- first well
   INJ  G1  2 2 8335.5 WATER/
/
TSTEP
   3*10 /
"""


def read(tmp_path, text=TEXT) -> Deck:
    path = tmp_path / "CASE.DATA"
    path.write_text(text, newline="\n")
    return Deck.read(path)


def test_single_record_keyword_has_named_typed_items(tmp_path):
    (dimens,) = read(tmp_path).find(DIMENS)
    assert dimens.records[0].items == {"nx": 2, "ny": 2, "nz": 1}


def test_missing_trailing_items_are_none(tmp_path):
    (start,) = read(tmp_path).find(START)
    assert start.records[0].items == {"day": 1, "month": "JAN", "year": 2020, "time": None}


def test_repeat_counts_are_expanded(tmp_path):
    (welldims,) = read(tmp_path).find(WELLDIMS)
    items = welldims.records[0].items
    assert [items[n] for n in list(items)[:4]] == [5, None, None, 3]


def test_list_keyword_records(tmp_path):
    (welspecs,) = read(tmp_path).find(WELSPECS)
    first, second = (r.items for r in welspecs.records)
    assert (first["well"], first["group"], first["i"], first["ref_depth"]) == ("P1", "G1", 1, None)
    assert (second["well"], second["ref_depth"], second["phase"]) == ("INJ", 8335.5, "WATER")


def test_keyword_line_may_carry_a_comment(tmp_path):
    assert len(read(tmp_path).find(SCHEDULE)) == 1


def test_keyword_span_covers_its_data(tmp_path):
    deck = read(tmp_path)
    (welspecs,) = deck.find(WELSPECS)
    assert deck.text[welspecs.start :].startswith("WELSPECS\n")
    assert deck.text[welspecs.end :].startswith("TSTEP\n")


def test_keyword_not_in_the_deck(tmp_path):
    assert read(tmp_path).find(PROPS) == ()


def test_slash_inside_a_path_does_not_end_the_record(tmp_path):
    (include,) = read(tmp_path, "INCLUDE\n  ../grid/case.inc /\n").find(INCLUDE)
    assert include.records[0].items["path"] == "../grid/case.inc"


def test_bad_value_raises_with_keyword_and_line(tmp_path):
    with pytest.raises(ValueError, match="DIMENS at line 1: bad ny value 'x'"):
        read(tmp_path, "DIMENS\n 2 x 1 /\n").find(DIMENS)


def test_unclosed_keyword_raises(tmp_path):
    with pytest.raises(ValueError, match="DIMENS at line 1 is not closed"):
        read(tmp_path, "DIMENS\n 2 2 1\n").find(DIMENS)


def test_write_reproduces_the_file(tmp_path):
    read(tmp_path).write(tmp_path / "COPY.DATA")
    assert (tmp_path / "COPY.DATA").read_text() == TEXT


def test_write_elsewhere_keeps_include_pointing_at_the_original(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "runs" / "a").mkdir(parents=True)
    out = tmp_path / "runs" / "a" / "CASE.DATA"
    read(tmp_path / "base").write(out)
    expected = TEXT.replace("'grid/case.grdecl'", "'../../base/grid/case.grdecl'")
    assert out.read_text() == expected


def test_update_changes_only_the_record_tokens(tmp_path):
    deck = read(tmp_path)
    record = deck.find(WELLDIMS)[0].records[0]
    new = deck.update(record, max_wells=6, max_connections=4)
    assert new.text == TEXT.replace("5 2* 3 /", "6 4 1* 3 /")
    assert deck.text == TEXT


def test_update_fills_an_all_default_record(tmp_path):
    deck = read(tmp_path, "WELLDIMS\n/\n")
    new = deck.update(deck.find(WELLDIMS)[0].records[0], max_groups=2)
    assert new.text == "WELLDIMS\n1* 1* 2 /\n"


def test_update_unknown_item_raises(tmp_path):
    deck = read(tmp_path)
    with pytest.raises(ValueError, match="no item 'wells'"):
        deck.update(deck.find(WELLDIMS)[0].records[0], wells=6)


def test_insert_after_a_keyword(tmp_path):
    deck = read(tmp_path)
    new = deck.insert(deck.find(SCHEDULE)[0].end, "RPTRST\n  BASIC=2 /\n")
    assert "SCHEDULE   -- wells from here\nRPTRST\n  BASIC=2 /\nWELSPECS\n" in new.text


def test_render_quotes_strings_and_defaults_gaps():
    text = render(WELSPECS, [{"well": "P1", "group": "G1", "i": 1, "j": 2, "phase": "OIL"}])
    assert text == "WELSPECS\n  'P1' 'G1' 1 2 1* 'OIL' /\n/\n\n"


def test_render_fixed_size_keyword():
    assert render(DIMENS, [{"nx": 3, "ny": 2, "nz": 1}]) == "DIMENS\n  3 2 1 /\n\n"
    assert render(SCHEDULE) == "SCHEDULE\n\n"


def test_render_wrong_record_count_raises():
    with pytest.raises(ValueError, match="DIMENS takes 1 record"):
        render(DIMENS, [])


def test_render_unknown_item_raises():
    with pytest.raises(ValueError, match="DIMENS has no item 'nk'"):
        render(DIMENS, [{"nx": 3, "nk": 1}])


def test_find_several_keywords_in_file_order(tmp_path):
    found = read(tmp_path).find(WELSPECS, DIMENS, SCHEDULE)
    assert [kw.spec.name for kw in found] == ["DIMENS", "SCHEDULE", "WELSPECS"]


def test_variadic_item_collects_the_rest_of_the_record(tmp_path):
    wbhp = next(spec for spec in WELL_SUMMARY if spec.name == "WBHP")
    (keyword,) = read(tmp_path).find(wbhp)
    assert keyword.records[0].items == {"wells": ("PROD", "INJ")}


def test_render_variadic_item():
    wopr = next(spec for spec in WELL_SUMMARY if spec.name == "WOPR")
    assert render(wopr, [{"wells": ("P1", "P2")}]) == "WOPR\n  'P1' 'P2' /\n\n"


def test_render_wraps_long_records():
    text = render(DIMENS, [{"nx": 10**75, "ny": 2, "nz": 3}])
    assert text == f"DIMENS\n  {10**75} 2\n  3 /\n\n"
