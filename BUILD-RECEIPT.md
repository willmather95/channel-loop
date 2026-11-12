# Build receipt: channel-loop

Built 2026-11-12. Capstone of the willmather95 marketing-engineer repos. Local commits only: nothing pushed, no GitHub repo created, `gh` not run.

## What works

- **Daily tick** (`channel-loop tick` / `catch-up` / `serve`): ingest, read the landing test, measure, observe and hypothesize, choose, act, mirror to the warehouse. Idempotent, and it resumes after a crash. A process lock stops two runners at once.
- **TRUTH:** DuckDB warehouse using ads-guardrail-agent's own `ads_daily` / `ads_entities` / `ads_load_meta` DDL, plus `ads_settings_daily`, `web_daily`, the marketing-warehouse `gsc_*` tables and `loop_decisions`. Readable by ads-guardrail-agent's `WarehouseReader` (tested).
- **HANDS:** budget and bid changes go through `ads_guardrail.external_proposal` + `apply` + `rollback` (12 caps, two-key writes, kill switch, audit log). Landing tests go through `landing_page_lab.create_experiment` + `readout` (planned looks only). Creative refreshes go to an approval queue.
- **Agent:** five kinds of change, each with a $/day estimate. A Bayesian linear model per kind (exact conjugate update, weighted by each measurement's standard error). Posterior-mean decision with a per-locked-campaign bar. Thompson sampling is available as `loop_ts`.
- **Attribution:** ratio diff-in-diff against untouched campaigns, with a day bootstrap standard error and a flagged pre/post fallback. Landing tests are measured by their randomized control arm.
- **Journal:** hash-chained JSONL (ads-guardrail-agent format). It is the only state. `journal replay` re-derives every decision, score and belief.
- **Simulator:** seasonality, Thanksgiving dip, AR(1) shocks, diminishing returns, bid pacing, creative fatigue, conversion lag, learning resets, hidden landing-page lifts. Common random numbers across policies. Hidden truth is used only to grade decisions.
- **Experiment harness:** `channel-loop compare`, with paired sign-flip tests and bootstrap CIs across seeds and policies.
- **Docs:** README, `docs/hero.png` (rendered from `results/summary.json` + `demo/out`), `docs/architecture.png`, mermaid diagram, CI, cron and GitHub Actions HOME examples.
- A fresh clone with `requirements-local.txt` installed and ran `channel-loop demo` in 36 seconds.

## Tests

`.venv/bin/python -m pytest` gives **51 passed** in about 2 minutes on 8 cores. `-m "not slow"` gives 45 passed.

Covers tick idempotency and crash-resume, guardrails end to end (every write re-checked from the audit log), kill switch and write switch, learning across 8 seeds (sign-flip p < 0.01), journal replay and forgery detection, conjugate math, DiD recovery of known effects, simulator properties, and CLI exit codes.

## Results (SIMULATED, `channel-loop compare --seeds 30`, deterministic)

- Loop vs doing nothing: +$44,982 over 12 weeks (95% CI $33,818 to $56,750, p < 0.0001, 28/30 seeds).
- Loop vs random on total profit: **-$7,153, not significant** (CI -$21,035 to +$5,683). Loop vs greedy: +$426, not significant. Reported as a tie in the README and hero.
- Decision quality by hidden truth (ad changes): loop $49/day per change vs random $29 (+$18, p < 0.0001, 27/30), vs greedy $38 (+$10, p < 0.0001). Harmful share: 16% vs 22% for random.
- Measurement vs truth: r = 0.86, 95% interval coverage 91% (n = 445).
- Design changes made during the build, and why. All are disclosed in the README:
  1. Split landing tests and ad changes into separate slots.
  2. Centered the prior on "trust my estimates".
  3. Added the per-campaign bar.
  4. Made posterior-mean the default after Thompson sampling underperformed.

  The simulator was not tuned to make the loop win.

## Scrub results

- `gitleaks dir . --no-banner --redact`: no leaks found
- `gitleaks git . --no-banner --redact`: no leaks found
- `DRY=1 bash publish-private.sh . "x"`: gitleaks dir/git clean, scrub grep **clean** (stopped before any push)
- No client names, no real ids, no keys. `.env.example` has placeholders only. Fieldnote, its account ids and campaigns are fictional.

## Commit

Code and docs HEAD before this receipt: `8000a93b3e75f449260c785510499a4553f78173`. This receipt is committed on top of it. Run `git rev-parse HEAD` for the final hash.

## Simulated, fixture or not run

- The market, every metric, profit and "truth" are **simulated**. The ad account is ads-guardrail-agent's fake in-memory account. The flag service, site host and creative approvals are local JSON stand-ins. Approvals are simulated and logged as such. PostHog calls are dry-run.
- Never run against a real ad account, PostHog project or host. The CLI only drives the simulated account.
- **CI caveat:** `requirements.txt` installs the sibling repos from `github.com/willmather95/{ads-guardrail-agent,landing-page-lab}` at pinned commits (281243e, 4434109). CI will fail until those repos are published at those commits (or public). Locally, `requirements-local.txt` uses the sibling folders.

## Paid-spend ask

None. Everything runs offline at $0. No paid API, model or ad spend is needed to reproduce any number here.
