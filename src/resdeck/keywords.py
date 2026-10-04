"""The Eclipse / OPM Flow keywords resdeck reads and writes.

Each keyword is a ``KeywordSpec`` constant that gives the items of a record
readable names, in file order. The deck reader and the well writer both go
through these specs, so the rest of the library only sees the readable names.
Item order and types follow the OPM keyword definitions
(https://github.com/OPM/opm-flow-editor-support).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# KeywordSpec.size for keywords that take any number of records, closed by a lone "/".
LIST = None

SECTION_NAMES = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE")


def number(token: str) -> float:
    """A deck number; Fortran exponents such as ``1.5D3`` are accepted."""
    return float(token.upper().replace("D", "E"))


def uda(token: str) -> float | str:
    """A number, or the name of a user-defined quantity."""
    try:
        return number(token)
    except ValueError:
        return token


@dataclass(frozen=True, eq=False)
class KeywordSpec:
    """Structure of one keyword.

    ``items`` maps the readable name of each record item, in file order, to
    the function that converts its token. ``size`` is the number of records:
    0 for a keyword without data, ``LIST`` for any number of records. With
    ``variadic``, the last item takes all remaining tokens of the record and
    its value is a tuple.
    """

    name: str
    label: str
    sections: tuple[str, ...]
    size: int | None = 0
    items: dict[str, Callable] = field(default_factory=dict)
    variadic: bool = False


RUNSPEC = KeywordSpec("RUNSPEC", "run specification section", ("RUNSPEC",))
GRID = KeywordSpec("GRID", "grid section", ("GRID",))
EDIT = KeywordSpec("EDIT", "grid edit section", ("EDIT",))
PROPS = KeywordSpec("PROPS", "fluid and rock properties section", ("PROPS",))
REGIONS = KeywordSpec("REGIONS", "regions section", ("REGIONS",))
SOLUTION = KeywordSpec("SOLUTION", "initial state section", ("SOLUTION",))
SUMMARY = KeywordSpec("SUMMARY", "summary output section", ("SUMMARY",))
SCHEDULE = KeywordSpec("SCHEDULE", "wells and time stepping section", ("SCHEDULE",))

SECTIONS = (RUNSPEC, GRID, EDIT, PROPS, REGIONS, SOLUTION, SUMMARY, SCHEDULE)

DIMENS = KeywordSpec(
    "DIMENS",
    "grid dimensions",
    ("RUNSPEC",),
    size=1,
    items={"nx": int, "ny": int, "nz": int},
)

START = KeywordSpec(
    "START",
    "simulation start date",
    ("RUNSPEC",),
    size=1,
    items={"day": int, "month": str, "year": int, "time": str},
)

WELLDIMS = KeywordSpec(
    "WELLDIMS",
    "well and group dimensions",
    ("RUNSPEC",),
    size=1,
    items={
        "max_wells": int,
        "max_connections": int,
        "max_groups": int,
        "max_wells_per_group": int,
        "max_stages": int,
        "max_streams": int,
        "max_mixtures": int,
        "max_separators": int,
        "max_mixture_items": int,
        "max_completion_x": int,
        "max_well_lists_per_well": int,
        "max_dynamic_well_lists": int,
        "max_secondary_wells": int,
        "max_gpp_rows": int,
    },
)

INCLUDE = KeywordSpec(
    "INCLUDE",
    "include another file",
    SECTION_NAMES,
    size=1,
    items={"path": str},
)

WELSPECS = KeywordSpec(
    "WELSPECS",
    "well specification",
    ("SCHEDULE",),
    size=LIST,
    items={
        "well": str,
        "group": str,
        "i": int,
        "j": int,
        "ref_depth": number,
        "phase": str,
        "drainage_radius": number,
        "inflow_equation": str,
        "auto_shutin": str,
        "crossflow": str,
        "pvt_table": int,
        "density_calc": str,
        "fip_region": int,
        "frontsim1": str,
        "frontsim2": str,
        "well_model": str,
        "polymer_table": int,
    },
)

COMPDAT = KeywordSpec(
    "COMPDAT",
    "well connections",
    ("SCHEDULE",),
    size=LIST,
    items={
        "well": str,
        "i": int,
        "j": int,
        "k_top": int,
        "k_bottom": int,
        "status": str,
        "sat_table": int,
        "connection_factor": number,
        "diameter": number,
        "kh": number,
        "skin": number,
        "d_factor": number,
        "direction": str,
        "pressure_radius": number,
    },
)

# Items 13 to 20 are compositional (E300) only and are left out.
WCONPROD = KeywordSpec(
    "WCONPROD",
    "production well controls",
    ("SCHEDULE",),
    size=LIST,
    items={
        "well": str,
        "status": str,
        "mode": str,
        "oil_rate": uda,
        "water_rate": uda,
        "gas_rate": uda,
        "liquid_rate": uda,
        "resv_rate": uda,
        "bhp": uda,
        "thp": uda,
        "vfp_table": int,
        "alq": uda,
    },
)

# Items 10 to 15 are oil, steam and multi-phase options that OPM Flow ignores.
WCONINJE = KeywordSpec(
    "WCONINJE",
    "injection well controls",
    ("SCHEDULE",),
    size=LIST,
    items={
        "well": str,
        "fluid": str,
        "status": str,
        "mode": str,
        "rate": uda,
        "resv_rate": uda,
        "bhp": uda,
        "thp": uda,
        "vfp_table": int,
    },
)

# Without I, J, K or completion numbers the status applies to the whole well.
WELOPEN = KeywordSpec(
    "WELOPEN",
    "open or shut wells",
    ("SCHEDULE",),
    size=LIST,
    items={
        "well": str,
        "status": str,
        "i": int,
        "j": int,
        "k": int,
        "first_completion": int,
        "last_completion": int,
    },
)

DATES = KeywordSpec(
    "DATES",
    "advance to report dates",
    ("SCHEDULE",),
    size=LIST,
    items={"day": int, "month": str, "year": int, "time": str},
)

TSTEP = KeywordSpec(
    "TSTEP",
    "advance by report steps in days",
    ("SCHEDULE",),
    size=1,
    items={"steps": number},
    variadic=True,
)

END = KeywordSpec("END", "end of input", SECTION_NAMES)


# SUMMARY keywords that take one record listing well names (or patterns such as
# 'P*'); an empty list means every well. Well vectors with other record layouts
# (WOPRL, ...) are not included.
WELL_SUMMARY_NAMES = (
    "WALQ",
    "WAMIR",
    "WAMIT",
    "WAMPR",
    "WAMPT",
    "WBGLR",
    "WBHP",
    "WBHPH",
    "WBHPT",
    "WBP",
    "WBP4",
    "WBP5",
    "WBP9",
    "WCIC",
    "WCIR",
    "WCIT",
    "WCPC",
    "WCPR",
    "WCPT",
    "WEFF",
    "WEFFG",
    "WEPR",
    "WEPT",
    "WGIGR",
    "WGIR",
    "WGIRH",
    "WGIRT",
    "WGIT",
    "WGITH",
    "WGLIR",
    "WGLIT",
    "WGLR",
    "WGLRH",
    "WGMIR",
    "WGMIT",
    "WGMPR",
    "WGMPT",
    "WGOR",
    "WGORH",
    "WGPGR",
    "WGPI",
    "WGPI2",
    "WGPP",
    "WGPP2",
    "WGPPF",
    "WGPPF2",
    "WGPPS",
    "WGPPS2",
    "WGPR",
    "WGPRF",
    "WGPRH",
    "WGPRS",
    "WGPRT",
    "WGPT",
    "WGPTF",
    "WGPTH",
    "WGPTS",
    "WGVIR",
    "WGVPR",
    "WINFC",
    "WINJFVR",
    "WINJFVT",
    "WLPR",
    "WLPRH",
    "WLPRT",
    "WLPT",
    "WLPTH",
    "WMCON",
    "WMCTL",
    "WMMIR",
    "WMMIT",
    "WMMPR",
    "WMMPT",
    "WMOIR",
    "WMOIT",
    "WMOPR",
    "WMOPT",
    "WMUIR",
    "WMUIT",
    "WMUPR",
    "WMUPT",
    "WMVFP",
    "WNIR",
    "WNIT",
    "WNPR",
    "WNPT",
    "WOGLR",
    "WOGR",
    "WOGRH",
    "WOIGR",
    "WOIR",
    "WOIRH",
    "WOIRT",
    "WOIT",
    "WOITH",
    "WOPGR",
    "WOPI",
    "WOPI2",
    "WOPP",
    "WOPP2",
    "WOPR",
    "WOPRF",
    "WOPRH",
    "WOPRS",
    "WOPRT",
    "WOPT",
    "WOPTF",
    "WOPTH",
    "WOPTS",
    "WPI",
    "WPI1",
    "WPI4",
    "WPI5",
    "WPI9",
    "WPIG",
    "WPIL",
    "WPIO",
    "WPIW",
    "WSIC",
    "WSIR",
    "WSIT",
    "WSPC",
    "WSPR",
    "WSPT",
    "WSTAT",
    "WTHP",
    "WTHPH",
    "WTIC",
    "WTICF",
    "WTICHEA",
    "WTICS",
    "WTIR",
    "WTIRANI",
    "WTIRCAT",
    "WTIRF",
    "WTIRFOA",
    "WTIRHEA",
    "WTIRS",
    "WTIT",
    "WTITANI",
    "WTITCAT",
    "WTITF",
    "WTITFOA",
    "WTITHEA",
    "WTITS",
    "WTPC",
    "WTPCF",
    "WTPCHEA",
    "WTPCS",
    "WTPR",
    "WTPRANI",
    "WTPRCAT",
    "WTPRF",
    "WTPRFOA",
    "WTPRHEA",
    "WTPRS",
    "WTPT",
    "WTPTANI",
    "WTPTCAT",
    "WTPTF",
    "WTPTFOA",
    "WTPTHEA",
    "WTPTS",
    "WU",
    "WVIGR",
    "WVIR",
    "WVIRT",
    "WVIT",
    "WVPGR",
    "WVPI",
    "WVPI2",
    "WVPP",
    "WVPP2",
    "WVPR",
    "WVPRT",
    "WVPT",
    "WWCT",
    "WWCTH",
    "WWGR",
    "WWGRH",
    "WWIGR",
    "WWIR",
    "WWIRH",
    "WWIRT",
    "WWIT",
    "WWITH",
    "WWPGR",
    "WWPI",
    "WWPI2",
    "WWPIR",
    "WWPP",
    "WWPP2",
    "WWPR",
    "WWPRH",
    "WWPRT",
    "WWPT",
    "WWPTH",
    "WWVIR",
    "WWVIT",
)

WELL_SUMMARY = tuple(
    KeywordSpec(name, "well summary vector", ("SUMMARY",), 1, {"wells": str}, variadic=True)
    for name in WELL_SUMMARY_NAMES
)
