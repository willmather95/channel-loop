PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
# Use sibling checkouts (../ads-guardrail-agent, ../landing-page-lab) when present, else pinned git URLs.
REQ := $(if $(wildcard ../ads-guardrail-agent/pyproject.toml),requirements-local.txt,requirements.txt)

.PHONY: install demo compare test test-fast hero clean

$(BIN)/channel-loop:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q -r $(REQ)
	$(BIN)/pip install -q -e . --no-deps

install: $(BIN)/channel-loop

demo: install        ## SIMULATED: 12 weeks, loop vs doing nothing, one seed (~30 s)
	$(BIN)/channel-loop demo --out demo/out

compare: install     ## SIMULATED: 30 seeds x 5 policies (~5 min on 8 cores)
	$(BIN)/channel-loop compare --seeds 30 --out results

test: install
	$(BIN)/python -m pytest

test-fast: install   ## skips the many-seed learning test
	$(BIN)/python -m pytest -m "not slow"

hero: install        ## docs/hero.png + docs/architecture.png from results/ and demo/out (needs Chrome)
	$(BIN)/python scripts/render.py --results results/summary.json --demo demo/out --docs docs
	cp results/summary.json docs/results-summary.json
	cp demo/out/summary.json docs/demo-summary.json
	cp demo/out/decisions.csv docs/demo-decisions.csv

clean:
	rm -rf demo/out results .pytest_cache
