"""channel-loop command line. Everything here runs on the SIMULATED account, offline.

    channel-loop demo                       12 simulated weeks, loop vs doing nothing, one seed (~30 s)
    channel-loop compare --seeds 30         loop vs loop_ts vs greedy vs random vs static, many seeds
    channel-loop init --state S             make a state dir for scheduled ticks
    channel-loop tick --state S --date D    one tick (cron / GitHub Actions entrypoint)
    channel-loop catch-up --state S         every missing tick up to --through (default: end of run)
    channel-loop serve --state S            long-running runner, one simulated day per wake-up
    channel-loop journal verify|replay|show --state S
    channel-loop kill on|off|status --state S

Exit codes: 0 ok, 1 error, 3 a tick was halted by the kill switch, 5 another runner is busy,
6 journal check failed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from .config import ARMS, POLICIES, load_loop_config
from .experiment import START, run_matrix, run_one, RunSpec, summarize, write_json
from .home.runner import Meta, RunnerBusy, build_loop, catch_up, serve
from .loop import TickResult
from .replay import replay
from .stats import fmt_money

REPO = Path(__file__).resolve().parents[2]


def _default(path: str) -> str:
    """Config files: prefer ./config, then the repo's own config dir."""
    p = Path(path)
    if p.exists():
        return str(p)
    q = REPO / path
    return str(q) if q.exists() else str(p)


def _print_tick(r: TickResult) -> None:
    acts = ", ".join(f"{d['slot']}: {d['arm'] or 'hold'}" + (f" ({d['target']})" if d.get("target") else "")
                     for d in r.decisions if d.get("arm") or d["slot"] == "ads")
    meas = "; ".join(f"measured {m['arm']} {m['y']:+.0f} +/- {m['se']:.0f} $/day" for m in r.measured)
    extra = f" | {r.readout['headline']}" if r.readout else ""
    halt = f" | HALTED: {r.halted}" if r.halted else ""
    print(f"{r.day} {r.status:4} {acts}{(' | ' + meas) if meas else ''}{extra}{halt}")


def cmd_demo(a: argparse.Namespace) -> int:
    cfg = load_loop_config(a.loop_config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"SIMULATED: 12 weeks from {START}, seed {a.seed}. Fictional account, fake ad account, no network.\n")
    res = {}
    for policy in ("loop", "static"):
        spec = RunSpec(a.seed, policy, str(out / policy), a.guard_config, a.brief, cfg, 12,
                       build_sites=a.build_sites and policy == "loop")
        res[policy] = run_one(spec)
    loop, static = res["loop"], res["static"]
    print("Decisions the loop made (acted on), with what it measured and the hidden truth:")
    print(f"{'day':10} {'arm':16} {'target':52} {'estimate':>9} {'measured':>16} {'truth':>8}")
    rows = []
    for d in loop["decisions"]:
        if d["outcome"] in ("blocked", "dry_run", "halted"):
            continue
        meas = "" if d["y"] is None else f"{d['y']:+.0f} +/- {d['se']:.0f}"
        tr = "" if d["truth"] is None else f"{d['truth']:+.0f}"
        print(f"{d['day']:10} {d['arm']:16} {d['target'][:52]:52} {d['x']:+9.0f} {meas:>16} {tr:>8}")
        rows.append(d)
    with (out / "decisions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "day", "arm", "target", "x", "outcome", "y", "se", "method", "truth"])
        w.writeheader()
        w.writerows(loop["decisions"])
    with (out / "daily.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "loop_profit", "static_profit"])
        for i, (p1, p0) in enumerate(zip(loop["daily_profit"], static["daily_profit"])):
            w.writerow([(START + timedelta(days=i)).isoformat(), round(p1, 2), round(p0, 2)])
    diff = loop["total_profit"] - static["total_profit"]
    blocked = sum(1 for d in loop["decisions"] if d["outcome"] == "blocked")
    summary = {"label": "SIMULATED", "seed": a.seed, "start": START.isoformat(), "weeks": 12,
               "loop_total_profit": round(loop["total_profit"], 2), "static_total_profit": round(static["total_profit"], 2),
               "difference": round(diff, 2), "actions": len(rows), "blocked_by_guardrails": blocked,
               "ads_writes": loop["ads_writes"], "landing_ships": loop["ships"], "journal": loop["journal"],
               "beliefs": loop["beliefs"]}
    write_json(out / "summary.json", summary)
    rep = replay(build_loop_journal(out / "loop"), cfg.bandit)
    print(f"\nprofit over 12 weeks: loop {fmt_money(loop['total_profit'])}, doing nothing "
          f"{fmt_money(static['total_profit'])}, difference {fmt_money(diff)} (one seed; see `compare` for many)")
    print(f"actions: {len(rows)}; proposals blocked by guardrails: {blocked}; ad writes through ads-guardrail-agent: "
          f"{loop['ads_writes']}; landing winners shipped: {len(loop['ships'])}")
    print(f"journal: {rep.summary()}")
    print(f"\nwrote {out}/summary.json, decisions.csv, daily.csv; state dirs {out}/loop and {out}/static")
    print("SIMULATED. Every number above comes from a made-up market, not a real account.")
    return 0 if rep.ok else 6


def build_loop_journal(state: Path):
    from .journal import Journal
    return Journal(state / "journal.jsonl")


def cmd_compare(a: argparse.Namespace) -> int:
    cfg = load_loop_config(a.loop_config)
    seeds = list(range(a.first_seed, a.first_seed + a.seeds))
    policies = tuple(a.policies.split(",")) if a.policies else POLICIES
    workers = a.workers or max(1, (os.cpu_count() or 2))
    out = Path(a.out)
    if a.from_runs:
        results = json.loads(Path(a.from_runs).read_text())
        print(f"re-summarizing {a.from_runs} (no new runs)")
    else:
        print(f"SIMULATED: {len(seeds)} seeds x {len(policies)} policies x 12 weeks on {workers} workers ...", flush=True)
        results = run_matrix(seeds, policies, cfg, a.guard_config, a.brief, workers=workers)
        results = [{k: r[k] for k in ("seed", "policy", "total_profit", "daily_profit", "ads_writes", "ads_blocked",
                                      "ships", "journal_ok", "beliefs", "decisions")} for r in results]
        write_json(out / "runs.json", results)
    s = summarize(results)
    write_json(out / "summary.json", s)
    print(f"\n{'policy':8} {'mean 12-week profit':>20} {'actions':>8} {'ad writes':>10}")
    for p, v in s["policies"].items():
        print(f"{p:8} {fmt_money(v['mean_total_profit']):>20} {v['actions']:8.1f} {v['ads_writes']:10.1f}")
    print("\npaired by seed (same market, same noise):")
    for k, v in s["comparisons"].items():
        print(f"  {k:18} mean {fmt_money(v['mean']):>10}  95% CI {fmt_money(v['ci95'][0])} to {fmt_money(v['ci95'][1])}"
              f"  one-sided p {v['p_one_sided']:.4f}  wins {v['wins']}/{v['n']}")
    if "attribution" in s:
        at = s["attribution"]
        print("\nad-change quality, graded by the hidden truth (landing tests excluded):")
        for p_, v in s["ad_change_quality"]["policies"].items():
            print(f"  {p_:8} {v['changes_per_run']:5.1f} changes/run, true value {fmt_money(v['true_value_per_change'])}"
                  f"/day per change, {v['share_harmful']:.0%} harmful; weeks 1-6 {fmt_money(v['weeks_1_6_value_per_change'])}"
                  f", weeks 7-12 {fmt_money(v['weeks_7_12_value_per_change'])}")
        for k, v in s["ad_change_quality"]["comparisons"].items():
            print(f"  {k:34} mean {fmt_money(v['mean'])}/day  95% CI {fmt_money(v['ci95'][0])} to "
                  f"{fmt_money(v['ci95'][1])}  p {v['p_one_sided']:.4f}  wins {v['wins']}/{v['n']}")
        print(f"\nloop measurement vs hidden truth: n {at['n']}, r {at['pearson_r']}, mean abs error "
              f"{fmt_money(at['mean_abs_error'])}/day, 95% interval covers truth {at['ci95_coverage']:.0%}")
    print(f"\nwrote {out}/summary.json and {out}/runs.json. SIMULATED.")
    return 0


def cmd_init(a: argparse.Namespace) -> int:
    state = Path(a.state)
    if (state / "loop-meta.json").exists() and not a.force:
        print(f"{state} already initialised (use --force to overwrite the meta file)")
        return 1
    Meta(a.seed, a.start, a.weeks * 7, a.policy, str(Path(a.guard_config).resolve()),
         str(Path(a.brief).resolve())).save(state)
    print(f"initialised {state}: seed {a.seed}, start {a.start}, {a.weeks} weeks, policy {a.policy} (SIMULATED)")
    return 0


def _run_ticks(a: argparse.Namespace, through: date) -> int:
    cfg = load_loop_config(a.loop_config)
    try:
        res = catch_up(Path(a.state), cfg, through, on_tick=_print_tick)
    except RunnerBusy as exc:
        print(f"busy: {exc}", file=sys.stderr)
        return 5
    return 3 if any(r.halted for r in res) else 0


def cmd_tick(a: argparse.Namespace) -> int:
    meta = Meta.load(Path(a.state))
    day = date.fromisoformat(a.date) if a.date else meta.start_date
    loop = build_loop(Path(a.state), load_loop_config(a.loop_config), meta)
    last = loop.journal.state().last_tick
    if day in loop.journal.state().finished_ticks:
        print(f"{day} noop (already finished)")
        return 0
    expected = meta.start_date if last is None else last + timedelta(days=1)
    if day != expected:
        print(f"next tick is {expected}; use catch-up to run up to {day}", file=sys.stderr)
        return 1
    return _run_ticks(a, day)


def cmd_catch_up(a: argparse.Namespace) -> int:
    meta = Meta.load(Path(a.state))
    return _run_ticks(a, date.fromisoformat(a.through) if a.through else meta.end_date)


def cmd_serve(a: argparse.Namespace) -> int:
    try:
        return serve(Path(a.state), load_loop_config(a.loop_config), a.interval, a.max_wakeups, on_tick=_print_tick)
    except RunnerBusy as exc:
        print(f"busy: {exc}", file=sys.stderr)
        return 5


def cmd_journal(a: argparse.Namespace) -> int:
    j = build_loop_journal(Path(a.state))
    if a.action == "verify":
        ok, msg = j.verify()
        print(msg)
        return 0 if ok else 6
    if a.action == "replay":
        rep = replay(j, load_loop_config(a.loop_config).bandit)
        print(rep.summary())
        for m in rep.mismatches[:10]:
            print(" ", m)
        return 0 if rep.ok else 6
    for r in j.records():
        if r["event"] in ("decision", "acted", "measured", "lp_readout", "lp_concluded", "halted"):
            keep = {k: r[k] for k in ("seq", "event", "date", "decision_id", "arm", "target", "outcome", "y", "se",
                                      "method", "headline", "reason") if k in r and r[k] not in (None, "")}
            print(json.dumps(keep))
    return 0


def cmd_kill(a: argparse.Namespace) -> int:
    f = Path(a.state) / "KILL_SWITCH"
    if a.action == "on":
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text((a.note or "engaged by channel-loop kill on") + "\n")
        print(f"kill switch ON ({f}). No ad, landing or creative change will be made.")
    elif a.action == "off":
        f.unlink(missing_ok=True)
        print("kill switch OFF")
    else:
        env = os.environ.get("CHANNEL_LOOP_KILL", "")
        print(f"file: {'present' if f.exists() else 'absent'}; env CHANNEL_LOOP_KILL={env!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="channel-loop", description=__doc__.split("\n\n")[0])
    p.add_argument("--loop-config", default=_default("config/loop.toml"))
    p.add_argument("--guard-config", default=_default("config/guardrail.toml"))
    p.add_argument("--brief", default=_default("config/brief.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="12 simulated weeks: loop vs doing nothing, one seed")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--out", default="demo/out")
    d.add_argument("--build-sites", action="store_true", help="also build the landing-page variants as static HTML")
    d.set_defaults(fn=cmd_demo)

    c = sub.add_parser("compare", help="many seeds, every policy, paired stats")
    c.add_argument("--seeds", type=int, default=30)
    c.add_argument("--first-seed", type=int, default=1)
    c.add_argument("--policies", default="")
    c.add_argument("--workers", type=int, default=0)
    c.add_argument("--out", default="results")
    c.add_argument("--from-runs", help="re-summarize an existing runs.json instead of running")
    c.set_defaults(fn=cmd_compare)

    i = sub.add_parser("init", help="create a state dir for scheduled ticks")
    i.add_argument("--state", required=True)
    i.add_argument("--seed", type=int, default=7)
    i.add_argument("--start", default=START.isoformat())
    i.add_argument("--weeks", type=int, default=12)
    i.add_argument("--policy", default="loop", choices=POLICIES)
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)

    t = sub.add_parser("tick", help="run one tick (idempotent)")
    t.add_argument("--state", required=True)
    t.add_argument("--date")
    t.set_defaults(fn=cmd_tick)

    cu = sub.add_parser("catch-up", help="run every missing tick, in order")
    cu.add_argument("--state", required=True)
    cu.add_argument("--through")
    cu.set_defaults(fn=cmd_catch_up)

    s = sub.add_parser("serve", help="long-running runner: one simulated day per wake-up")
    s.add_argument("--state", required=True)
    s.add_argument("--interval", type=float, default=60.0)
    s.add_argument("--max-wakeups", type=int)
    s.set_defaults(fn=cmd_serve)

    j = sub.add_parser("journal", help="verify the hash chain, replay decisions, or show the log")
    j.add_argument("action", choices=["verify", "replay", "show"])
    j.add_argument("--state", required=True)
    j.set_defaults(fn=cmd_journal)

    k = sub.add_parser("kill", help="kill switch for every write the loop makes")
    k.add_argument("action", choices=["on", "off", "status"])
    k.add_argument("--state", required=True)
    k.add_argument("--note")
    k.set_defaults(fn=cmd_kill)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
