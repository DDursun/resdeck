"""Playground: read a simple case (SPE1), look at it, add wells, write a new deck.

Run from the project root:

    uv run python example.py
"""

from pathlib import Path

import numpy as np

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
from resdeck import keywords as kw

ROOT = Path(__file__).resolve().parent.parent  # the Research3 folder
DECK = ROOT / "opm-data" / "spe1" / "SPE1CASE1.DATA"
EGRID = ROOT / "runs" / "spe1_test" / "SPE1CASE1.EGRID"
INIT = ROOT / "runs" / "spe1_test" / "SPE1CASE1.INIT"
OUT = ROOT / "runs" / "spe1_p2" / "SPE1_P2.DATA"

# --- 1. Read the deck -------------------------------------------------------
# Only keywords listed in resdeck/keywords.py are parsed. Each record is a
# dict of readable item name -> value, None where the deck leaves a default.
deck = Deck.read(DECK)

dimens = deck.find(kw.DIMENS)[0].records[0].items
start = deck.find(kw.START)[0].records[0].items
print(f"Deck:  {deck.path.name}")
print(f"Grid:  {dimens['nx']} x {dimens['ny']} x {dimens['nz']}")
print(f"Start: {start['day']} {start['month']} {start['year']}")

print("\nWells in the deck:")
for keyword in deck.find(kw.WELSPECS):
    for record in keyword.records:
        w = record.items
        print(f"  {w['well']:<6} group {w['group']}  cell ({w['i']}, {w['j']})  {w['phase']}")

print("\nProducer controls:")
for keyword in deck.find(kw.WCONPROD):
    for record in keyword.records:
        c = record.items
        print(f"  {c['well']:<6} {c['mode']}  oil rate {c['oil_rate']}  bhp limit {c['bhp']}")

# --- 2. Read the grid and properties ----------------------------------------
# All arrays are (nx, ny, nz); simulator cell (i, j, k) is array[i-1, j-1, k-1].
model = Model.read(EGRID, INIT)

print(f"\nModel: {model.nx} x {model.ny} x {model.nz}, {int(model.active.sum())} active cells")
print(f"Properties: {', '.join(sorted(model.props))}")

permx, dz = model.props["PERMX"], model.props["DZ"]
kh = np.nansum(permx * dz, axis=2)  # (nx, ny) map of permeability-thickness
print(f"PERMX by layer at (1, 1): {permx[0, 0, :]}")
print(f"kh map: min {kh.min():.0f}, max {kh.max():.0f}")
print(f"Centre of cell (5, 5, 1): x={model.x[4, 4, 0]}, y={model.y[4, 4, 0]}, z={model.z[4, 4, 0]}")

# --- 3. Define new wells ----------------------------------------------------
# I, J and K are 1-based, as in the simulator. Units are the deck's (FIELD here).
# The control decides the role: ProducerControl or InjectorControl.
producer = Well(
    name="P2",
    i=5,
    j=5,
    completions=(Completion(k_top=1, k_bottom=3, diameter=0.5),),
    control=ProducerControl("ORAT", oil_rate=5000, bhp=1000),
    group="G1",
)

# Gas injector in cell (2, 9, 1): 100000 Mscf/d target, BHP limit 9014 psi, the
# same controls as SPE1's own injector. (Water barely flows in SPE1: its water
# relative permeability is at most 1e-5, so a water injector hits its BHP limit.)
injector = Well(
    name="I1",
    i=2,
    j=9,
    completions=(Completion(k_top=1, k_bottom=1, diameter=0.5),),
    control=InjectorControl("GAS", "RATE", rate=100000, bhp=9014),
    group="G1",
)

print("\nText that will be added to the deck:\n")
for well in (producer, injector):
    print(format_well(well))

# --- 4. Add them and write a new deck ---------------------------------------
# add_well returns a new deck; the original is not changed. Passing the model
# checks that the well is on the grid and connects to an active cell. By
# default the well is also added to the SUMMARY well vectors (WOPR, WBHP, ...);
# pass add_to_summary=False to leave SUMMARY as it is.
new_deck = deck
for well in (producer, injector):
    new_deck = add_well(new_deck, well, model)

OUT.parent.mkdir(parents=True, exist_ok=True)
new_deck.write(OUT)
print(f"Written: {OUT}")

welldims = new_deck.find(kw.WELLDIMS)[0].records[0].items
print(f"WELLDIMS is now {[welldims[name] for name in list(welldims)[:4]]}")

print("\nTo run it in Flow, from the Research3 folder:")
print("  WSL (bash):")
print(f"    flow runs/spe1_p2/{OUT.name} --output-dir=runs/spe1_p2/out")
print("  Windows (PowerShell):")
print(rf"    tools\flow.cmd runs\spe1_p2\{OUT.name} --output-dir=runs\spe1_p2\out")
