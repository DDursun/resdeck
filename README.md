# resdeck

resdeck is a Python library for generating field development scenarios for
reservoir simulation. Given a geological realization, it produces well
placements and operating schedules and writes them as simulator input decks.

Ensembles of geological realizations are commonly simulated under a single
development: one set of wells and one schedule. Models trained on such data see
variation in geology but almost none in well count, well location or operating
conditions. resdeck supplies that variation, with many development scenarios
per realization.

A scenario is stored as plain data, a list of dated well events, and contains
no simulator syntax. A separate writer translates it into the input format of a
simulator. The current writer produces Eclipse-format decks and is tested with
OPM Flow; other simulators can be added without changing the scenarios.

Intended uses are training data for surrogate and foundation models of
reservoir flow, and initial development scenarios and baselines for well
placement and well control optimization.

## Install

```
pip install git+https://github.com/DDursun/resdeck
```

Python 3.10 or newer; depends on numpy and scipy.

## Usage

```python
from datetime import date

from resdeck import (
    Deck, InjectorControl, Model, ProducerControl, Scenario, Shut,
    apply, build_pattern, save_scenario,
)

deck = Deck.read("CASE.DATA")
model = Model.read("CASE.EGRID", "CASE.INIT")

# A producer at column (16, 42) with four injectors 100 m away.
wells = build_pattern(
    model,
    "five_spot",
    anchor=(16, 42),
    radius=100.0,
    producer=ProducerControl("BHP", bhp=395.0),
    injector=InjectorControl("WATER", "RATE", rate=79.5, bhp=420.0),
    diameter=0.2,
)

scenario = Scenario(
    events=wells + (Shut("I1", date(2012, 1, 1)),),
    dates=(date(2013, 1, 1),),  # end of the run
)

apply(deck, scenario, model).write("CASE_0001.DATA")
save_scenario(scenario, "CASE_0001.json")
```

The events are `Drill`, `SetControl`, `Shut` and `Open`. The patterns are
`single`, `four_spot`, `five_spot`, `seven_spot` and `nine_spot`; pass
`inverted=True` for an injector in the centre. Running the simulator is left to
the user.

---

[Dursun Dashdamirov](https://ddursun.github.io/), PhD student at UT Austin
