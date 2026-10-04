from datetime import date, datetime

import pytest

from resdeck import Deck, add_dates, report_dates
from resdeck.schedule import insertion_point

TEXT = """\
RUNSPEC
START
   1 'JAN' 2020 /
SCHEDULE
TSTEP
   2*10 31 /
DATES
   1 'MAR' 2020 /
   15 'MAR' 2020 '12:00:00' /
/
END
"""


def read(tmp_path, text=TEXT) -> Deck:
    path = tmp_path / "CASE.DATA"
    path.write_text(text, newline="\n")
    return Deck.read(path)


def test_report_dates_from_tstep_and_dates(tmp_path):
    assert report_dates(read(tmp_path)) == [
        datetime(2020, 1, 1),
        datetime(2020, 1, 11),
        datetime(2020, 1, 21),
        datetime(2020, 2, 21),
        datetime(2020, 3, 1),
        datetime(2020, 3, 15, 12),
    ]


def test_report_dates_without_steps_is_the_start(tmp_path):
    deck = read(tmp_path, "START\n 1 'JAN' 2020 /\nSCHEDULE\n")
    assert report_dates(deck) == [datetime(2020, 1, 1)]


def test_report_dates_need_start(tmp_path):
    with pytest.raises(ValueError, match="no START date"):
        report_dates(read(tmp_path, "SCHEDULE\nTSTEP\n 10 /\n"))


def test_dates_must_increase(tmp_path):
    deck = read(tmp_path, TEXT.replace("1 'MAR' 2020", "1 'FEB' 2020"))
    with pytest.raises(ValueError, match="DATES 01 Feb 2020 is not after 21 Feb 2020"):
        report_dates(deck)


def test_add_date_splits_a_tstep_value(tmp_path):
    deck = add_dates(read(tmp_path), [date(2020, 1, 15)])
    assert "TSTEP\n  10 4 6 31 /\n" in deck.text
    assert datetime(2020, 1, 15) in report_dates(deck)


def test_add_date_inserts_a_dates_record(tmp_path):
    deck = add_dates(read(tmp_path), [date(2020, 3, 10)])
    assert "  1 'MAR' 2020 /\n  10 'MAR' 2020 /\n  15 'MAR' 2020 '12:00:00' /\n" in deck.text


def test_add_dates_after_the_end_are_appended_before_end(tmp_path):
    deck = add_dates(read(tmp_path), [date(2020, 5, 1), date(2020, 4, 1)])
    assert deck.text.endswith("DATES\n  1 'APR' 2020 /\n  1 'MAY' 2020 /\n/\nEND\n")


def test_add_dates_to_an_empty_schedule(tmp_path):
    deck = add_dates(read(tmp_path, "START\n 1 'JAN' 2020 /\nSCHEDULE"), [date(2020, 2, 1)])
    assert deck.text == "START\n 1 'JAN' 2020 /\nSCHEDULE\nDATES\n  1 'FEB' 2020 /\n/\n"


def test_existing_report_date_is_not_added_again(tmp_path):
    deck = read(tmp_path)
    assert add_dates(deck, [date(2020, 1, 11)]).text == deck.text


def test_date_before_start_raises(tmp_path):
    with pytest.raises(ValueError, match="before the START date"):
        add_dates(read(tmp_path), [date(2019, 12, 31)])


def test_insertion_at_start_is_before_the_first_step(tmp_path):
    deck = read(tmp_path)
    same, offset = insertion_point(deck)
    assert same is deck
    assert deck.text[offset:].startswith("TSTEP\n")


def test_insertion_inside_a_tstep_splits_it(tmp_path):
    deck, offset = insertion_point(read(tmp_path), date(2020, 1, 11))
    assert "TSTEP\n  10 /\nTSTEP\n  10 31 /\n" in deck.text
    assert deck.text[offset:].startswith("TSTEP\n  10 31 /\n")


def test_insertion_at_the_end_of_a_keyword_needs_no_split(tmp_path):
    deck = read(tmp_path)
    same, offset = insertion_point(deck, date(2020, 2, 21))
    assert same.text == deck.text
    assert deck.text[offset:].startswith("DATES\n")


def test_insertion_at_a_date_that_is_no_report_step_raises(tmp_path):
    with pytest.raises(ValueError, match="no report step at 15 Jan 2020"):
        insertion_point(read(tmp_path), date(2020, 1, 15))


def test_insertion_at_the_end_of_the_run_raises(tmp_path):
    with pytest.raises(ValueError, match="end of the run"):
        insertion_point(read(tmp_path), datetime(2020, 3, 15, 12))
