"""Render docs/hero.png and docs/architecture.png from the repo's own outputs.

    python scripts/render.py --results results/summary.json --demo demo/out --docs docs

The hero chart is drawn as SVG from `channel-loop compare` (30 seeds) and annotated with
the decisions from `channel-loop demo` (one seed). House style follows world-class-charts:
the takeaway in the title, one highlighted series, direct labels, a written source line.
Screenshots use headless Chrome (set CHROME_PATH if it is not on PATH).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from datetime import date, timedelta
from html import escape
from pathlib import Path

W, H = 1600, 900
T = {
    "page": "#f2f2f1", "surface": "#ffffff", "grid": "#ececeb", "hairline": "#e4e4e3", "axis": "#8a8a8a",
    "ink": "#141414", "ink2": "#3f3f3f", "muted": "#6b6b6b", "accent": "#0a72e8", "accent_soft": "#d8e8fc",
    "accent_ink": "#0657b8", "compare": "#8c8c8c", "compare_soft": "#e3e3e3", "warn_ink": "#8a5300",
    "pos_ink": "#0b6e34", "neg_ink": "#b42329",
}
START = date(2026, 9, 7)
ARM_LABEL = {"landing_test": "Landing test", "creative_refresh": "Creative refresh", "budget_shift": "Budget shift",
             "bid_up": "Bid up", "bid_down": "Bid down"}

CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;height:{H}px;background:{T['page']};font-family:Geist,Inter,system-ui,sans-serif;color:{T['ink']};
-webkit-font-smoothing:antialiased}}
.mono{{font-family:'Geist Mono',ui-monospace,monospace}}
.card{{background:{T['surface']};border-radius:24px;border:1px solid {T['hairline']};
box-shadow:0 1px 2px rgba(0,0,0,.04),0 8px 24px -12px rgba(0,0,0,.08)}}
"""


def money(v: float, sign: bool = False) -> str:
    s = "-" if v < 0 else ("+" if sign and v > 0 else "")
    v = abs(v)
    return f"{s}${v / 1000:,.1f}k" if v >= 10_000 else f"{s}${v:,.0f}"


def pfmt(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def chrome() -> str:
    for c in (os.environ.get("CHROME_PATH"), "google-chrome", "chromium", "chromium-browser"):
        if c and shutil.which(c):
            return shutil.which(c)
    raise SystemExit("Chrome not found; set CHROME_PATH")


def shoot(html: str, out: Path, w: int = W, h: int = H) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "page.html"
        page.write_text(html)
        subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                        f"--window-size={w},{h}", "--force-device-scale-factor=2", f"--screenshot={out}",
                        page.as_uri()], check=True, capture_output=True, timeout=120)


# --- hero -------------------------------------------------------------------------------------
def chart_svg(summary: dict, decisions: list[dict]) -> str:
    cw, ch = 900, 492
    left, right, top = 118, 132, 24
    plot_h = 300
    strip_top = top + plot_h + 64
    pw = cw - left - right
    loop = summary["comparisons"]["loop_vs_static"]["cumulative_weekly"]
    rnd = summary["comparisons"]["random_vs_static"]["cumulative_weekly"]
    weeks = len(loop["mean"])
    ymax = max(max(loop["hi"]), max(rnd["hi"])) * 1.08
    ymin = min(0.0, min(loop["lo"]))
    step = 10_000 if ymax < 80_000 else 20_000

    def x(w: float) -> float:  # w in weeks from start (0..weeks)
        return left + pw * w / weeks

    def y(v: float) -> float:
        return top + plot_h - (v - ymin) / (ymax - ymin) * plot_h

    o = [f'<svg viewBox="0 0 {cw} {ch}" width="100%" role="img" '
         'aria-label="Cumulative extra profit versus doing nothing, loop and random policy, 12 simulated weeks">']
    v = 0
    while v <= ymax:
        o.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{T["axis"] if v == 0 else T["grid"]}"/>')
        o.append(f'<text x="{left - 10}" y="{y(v) + 5:.1f}" text-anchor="end" class="tick">{money(v)}</text>')
        v += step
    for w in range(0, weeks + 1, 2):
        o.append(f'<text x="{x(w):.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">'
                 f'{"Week " if w == 0 else ""}{w}</text>')
    pts = [(0.0, 0.0, 0.0, 0.0)] + [(i + 1, loop["mean"][i], loop["lo"][i], loop["hi"][i]) for i in range(weeks)]
    band = " ".join(f"{x(w):.1f},{y(hi):.1f}" for w, _, _, hi in pts) + " " + \
        " ".join(f"{x(w):.1f},{y(lo):.1f}" for w, _, lo, _ in reversed(pts))
    o.append(f'<polygon points="{band}" fill="{T["accent_soft"]}" opacity=".9"/>')
    rp = [(0, 0.0)] + [(i + 1, rnd["mean"][i]) for i in range(weeks)]
    o.append(f'<polyline points="{" ".join(f"{x(w):.1f},{y(m):.1f}" for w, m in rp)}" fill="none" '
             f'stroke="{T["compare"]}" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(f'<polyline points="{" ".join(f"{x(w):.1f},{y(m):.1f}" for w, m, _, _ in pts)}" fill="none" '
             f'stroke="{T["accent"]}" stroke-width="3.5" stroke-linejoin="round"/>')
    # direct labels at the right edge
    ly, ry = y(loop["mean"][-1]), y(rnd["mean"][-1])
    if abs(ly - ry) < 34:
        ly, ry = (ly - 17, ry + 17) if ly <= ry else (ly + 17, ry - 17)
    o.append(f'<text x="{left + pw + 12}" y="{ly - 2:.1f}" class="lab strong" fill="{T["accent_ink"]}">The loop</text>')
    o.append(f'<text x="{left + pw + 12}" y="{ly + 17:.1f}" class="lab" fill="{T["accent_ink"]}">{money(loop["mean"][-1], True)}</text>')
    o.append(f'<text x="{left + pw + 12}" y="{ry - 2:.1f}" class="lab strong" fill="{T["ink2"]}">Random policy</text>')
    o.append(f'<text x="{left + pw + 12}" y="{ry + 17:.1f}" class="lab" fill="{T["muted"]}">{money(rnd["mean"][-1], True)}</text>')
    o.append(f'<text x="{x(weeks):.1f}" y="{y(0) - 8:.1f}" text-anchor="end" class="lab" fill="{T["muted"]}">Doing nothing = $0</text>')
    # decision strip for the demo seed
    rows = ["landing_test", "creative_refresh", "budget_shift", "bid_up", "bid_down"]
    o.append(f'<text x="{left}" y="{strip_top - 12}" class="lab strong" fill="{T["ink2"]}">'
             f'Every change the loop made in one run (seed {decisions[0].get("seed", "")}), and what it was really worth</text>')
    rh = 16
    for i, arm in enumerate(rows):
        yy = strip_top + i * rh
        o.append(f'<text x="{left - 10}" y="{yy + 4}" text-anchor="end" class="small">{ARM_LABEL[arm]}</text>')
        o.append(f'<line x1="{left}" x2="{left + pw}" y1="{yy}" y2="{yy}" stroke="{T["grid"]}"/>')
        for d in decisions:
            if d["arm"] != arm or d["outcome"] in ("blocked", "dry_run", "halted"):
                continue
            dd = (date.fromisoformat(d["day"]) - START).days / 7
            t = d.get("truth")
            col = T["compare"] if t in (None, "") else (T["accent"] if float(t) >= 0 else T["neg_ink"])
            o.append(f'<circle cx="{x(dd):.1f}" cy="{yy}" r="5" fill="{col}" stroke="#fff" stroke-width="1.5"/>')
    ly2 = strip_top + len(rows) * rh + 6
    o.append(f'<circle cx="{left + 6}" cy="{ly2}" r="5" fill="{T["accent"]}"/><text x="{left + 16}" y="{ly2 + 4}" class="small">helped (hidden truth)</text>')
    o.append(f'<circle cx="{left + 156}" cy="{ly2}" r="5" fill="{T["neg_ink"]}"/><text x="{left + 166}" y="{ly2 + 4}" class="small">hurt</text>')
    o.append(f'<circle cx="{left + 216}" cy="{ly2}" r="5" fill="{T["compare"]}"/><text x="{left + 226}" y="{ly2 + 4}" class="small">not measured by week 12</text>')
    o.append("</svg>")
    return "\n".join(o)


def hero_html(summary: dict, demo: dict, decisions: list[dict]) -> str:
    cmp_ = summary["comparisons"]
    q = summary["ad_change_quality"]
    ls = cmp_["loop_vs_static"]
    lr = cmp_["loop_vs_random"]
    pq = q["comparisons"]["loop_vs_random_value_per_change"]
    n = ls["n"]
    lp, rp = q["policies"]["loop"], q["policies"]["random"]
    kpis = [
        ("Extra profit vs doing nothing", money(ls["mean"], True), f"12 weeks, mean of {n} seeds, won {ls['wins']} of {n}"),
        ("Value per ad change", f"{money(lp['true_value_per_change'])}/day",
         f"random: {money(rp['true_value_per_change'])}/day, p {pfmt(pq['p_one_sided'])}"),
        ("Harmful ad changes", f"{lp['share_harmful']:.0%}", f"random: {rp['share_harmful']:.0%}, graded by hidden truth"),
        ("Total profit vs random", "Tie", f"{money(lr['mean'], True)}, 95% CI {money(lr['ci95'][0], True)} to {money(lr['ci95'][1], True)}"),
    ]
    k_html = "".join(f'<div class="kpi"><div class="kl">{escape(a)}</div><div class="kv">{escape(b)}</div>'
                     f'<div class="ks">{escape(c)}</div></div>' for a, b, c in kpis)
    css = CSS + f"""
body{{display:grid;grid-template-columns:500px 1fr;gap:34px;padding:46px 46px 40px 54px}}
.left{{display:flex;flex-direction:column;padding-top:6px}}
.url{{font-size:16px;color:{T['muted']}}}
h1{{font-size:56px;line-height:1.03;letter-spacing:-2px;font-weight:650;margin-top:26px}}
.sub{{font-size:20.5px;line-height:1.45;color:{T['ink2']};margin-top:22px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}}
.chip{{font-size:14px;padding:6px 11px;border-radius:999px;background:#fff;border:1px solid {T['hairline']};color:{T['ink2']}}}
.flow{{list-style:none;margin-top:24px;display:grid;gap:8px;counter-reset:f}}
.flow li{{counter-increment:f;font-size:16px;color:{T['ink2']}}}
.flow li::before{{content:counter(f);font-family:'Geist Mono',monospace;font-size:12px;color:{T['accent_ink']};
background:{T['accent_soft']};border-radius:6px;padding:2px 7px;margin-right:11px;display:inline-block}}
.flow b{{color:{T['ink']};font-weight:600}}
.foot{{margin-top:auto;font-size:14.5px;line-height:1.5;color:{T['muted']}}}
.foot b{{color:{T['warn_ink']};font-weight:650}}
.right{{display:flex;flex-direction:column;gap:18px;min-width:0}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);padding:20px 8px}}
.kpi{{padding:4px 22px;border-left:1px solid {T['hairline']}}}.kpi:first-child{{border-left:0}}
.kl{{font-size:14.5px;color:{T['ink2']}}}.kv{{font-size:38px;font-weight:650;letter-spacing:-1px;margin-top:8px}}
.ks{{font-size:13.5px;color:{T['muted']};margin-top:6px;line-height:1.35}}
.chart{{padding:24px 30px 18px}}
.ct{{font-size:23px;font-weight:650;letter-spacing:-.3px}}.cs{{font-size:16px;color:{T['ink2']};margin-top:6px}}
.src{{font-size:12.5px;color:{T['muted']};margin-top:4px}}
svg .tick{{font-size:13px;fill:{T['muted']}}} svg .lab{{font-size:14px}} svg .strong{{font-weight:650}}
svg .small{{font-size:12px;fill:{T['ink2']}}}
"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
<div class="left">
  <div class="url mono">github.com/willmather95/channel-loop</div>
  <h1>A channel loop that learns which changes pay.</h1>
  <div class="sub">It reads the warehouse, makes one change a day through hard guardrails, measures it against
  campaigns it did not touch, and updates what it believes.</div>
  <div class="chips"><span class="chip">ads-guardrail-agent caps</span><span class="chip">landing-page-lab stats</span>
  <span class="chip">diff-in-diff holdout</span><span class="chip">idempotent daily tick</span>
  <span class="chip">hash-chained journal</span></div>
  <ol class="flow">
    <li><b>Observe</b> 14 days of DuckDB warehouse data</li>
    <li><b>Hypothesize</b> one change per kind, with a $ estimate</li>
    <li><b>Choose</b> by Bayesian belief in each kind of estimate</li>
    <li><b>Act</b> only through the guardrails and the kill switch</li>
    <li><b>Measure</b> 9 days later vs untouched campaigns</li>
    <li><b>Update</b> beliefs; replay any decision from the log</li>
  </ol>
  <div class="foot"><b>SIMULATED.</b> A made-up market with hidden true effects, a fictional product and a fake ad
  account. {n} seeds per policy. Every number comes from <span class="mono">channel-loop compare</span> and
  <span class="mono">channel-loop demo</span>.</div>
</div>
<div class="right">
  <div class="card kpis">{k_html}</div>
  <div class="card chart">
    <div class="ct">Extra profit vs doing nothing, 12 weeks</div>
    <div class="cs">The loop and a random policy both beat doing nothing. The loop's changes are worth more each, and it makes fewer of them.</div>
    {chart_svg(summary, decisions)}
    <div class="src">Source: results/summary.json (mean and 95% bootstrap band over {n} seeds) and demo/out/decisions.csv. Profit = signups x $60 assumed value, minus ad spend.</div>
  </div>
</div>
</body></html>"""


# --- architecture -----------------------------------------------------------------------------------
def architecture_html() -> str:
    css = CSS + f"""
body{{width:1600px;height:900px;padding:44px 56px;display:flex;flex-direction:column}}
h2{{font-size:34px;letter-spacing:-1px;font-weight:650}} .lede{{font-size:18px;color:{T['ink2']};margin-top:6px}}
.stage{{position:relative;flex:1;margin-top:22px}}
.box{{position:absolute;background:#fff;border:1px solid {T['hairline']};border-radius:18px;padding:14px 16px;
box-shadow:0 1px 2px rgba(0,0,0,.04),0 8px 24px -14px rgba(0,0,0,.10)}}
.box h3{{font-size:16.5px;font-weight:650}} .box p{{font-size:13.5px;color:{T['ink2']};margin-top:5px;line-height:1.4}}
.tag{{font-family:'Geist Mono',monospace;font-size:11.5px;letter-spacing:.6px;color:{T['accent_ink']};background:{T['accent_soft']};
border-radius:6px;padding:2px 7px;display:inline-block;margin-bottom:7px}}
.zone{{position:absolute;border:1.5px dashed {T['axis']};border-radius:24px}}
.zl{{position:absolute;font-family:'Geist Mono',monospace;font-size:12.5px;color:{T['muted']};background:{T['page']};padding:0 8px}}
.sim{{color:{T['warn_ink']};background:#fbf1e1}}
svg{{position:absolute;left:0;top:0}}
"""
    boxes = [
        (20, 250, 230, "TRUTH", "DuckDB warehouse", "ads_daily, ads_settings_daily, web_daily, loop_decisions. ads-guardrail and marketing-warehouse schema."),
        (330, 60, 250, "AGENT", "Observe + hypothesize", "Marginal return, pacing, CTR decay, site signups. One concrete change per kind, with a $/day estimate."),
        (330, 250, 250, "AGENT", "Choose (Bayesian)", "Per kind: effect = w0 + w1 x estimate. Act on the best posterior mean above the bar, or hold."),
        (330, 440, 250, "AGENT", "Measure + update", "Diff-in-diff vs untouched campaigns, or the A/B control arm. Each result updates one posterior."),
        (680, 40, 270, "HANDS", "ads-guardrail-agent", "external_proposal, then 12 caps, two-key writes, kill switch, audit log, rollback."),
        (680, 220, 270, "HANDS", "landing-page-lab", "create_experiment, then readout on planned looks only. Ship a winner or keep control."),
        (680, 400, 270, "HANDS", "Creative queue", "Refresh request with a brief for ugc-content-machine. A person approves."),
        (1050, 40, 230, "PLATFORM", "Ad account", "Fake in-memory account, rebuilt from the audit log each tick."),
        (1050, 220, 230, "PLATFORM", "Site flag + pages", "platform/site.json: live split, production page."),
        (1050, 400, 230, "PLATFORM", "Creatives", "platform/creatives.json: queue and live creative."),
        (1330, 220, 200, "SIMULATED", "Market", "Hidden curves, fatigue, lag, seasonality, noise. Writes yesterday into the warehouse."),
        (330, 610, 620, "HOME", "Daily tick: cron, GitHub Actions or serve", "Idempotent: a finished tick is a no-op, a crashed tick resumes at its first unfinished step. One runner at a time (flock)."),
        (1050, 610, 480, "MEMORY", "Decision journal (JSONL, hash chain)", "The loop's only state. replay re-derives every decision, score and belief from it."),
    ]
    html = []
    for x, y, w, tag, title, body in boxes:
        cls = "tag sim" if tag == "SIMULATED" else "tag"
        html.append(f'<div class="box" style="left:{x}px;top:{y}px;width:{w}px"><span class="{cls}">{tag}</span>'
                    f'<h3>{escape(title)}</h3><p>{escape(body)}</p></div>')
    arrows = [
        (250, 310, 330, 140), (250, 320, 330, 320), (250, 330, 330, 500), (455, 175, 455, 250), (455, 365, 455, 440),
        (580, 300, 680, 120), (580, 310, 680, 290), (580, 320, 680, 460), (950, 110, 1050, 110), (950, 290, 1050, 290),
        (950, 470, 1050, 470), (1280, 110, 1330, 260), (1280, 290, 1330, 290), (1280, 470, 1330, 320),
        (1430, 360, 140, 380), (640, 610, 640, 560), (950, 660, 1050, 660),
    ]
    svg = ['<svg width="1488" height="760" viewBox="0 0 1488 760"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{T["axis"]}"/></marker></defs>']
    for x1, y1, x2, y2 in arrows:
        if (x1, y1) == (1430, 360):  # market -> warehouse, routed along the bottom
            svg.append(f'<path d="M1430,360 L1430,585 L135,585 L135,{y2 + 40}" fill="none" stroke="{T["accent"]}" '
                       'stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#a)"/>')
            continue
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{T["axis"]}" stroke-width="1.4" marker-end="url(#a)"/>')
    svg.append(f'<text x="700" y="578" font-size="12.5" fill="{T["accent_ink"]}" font-family="Geist Mono">nightly load of yesterday (SIMULATED here; real loaders in live use)</text>')
    svg.append("</svg>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
<h2>channel-loop: how one tick flows</h2>
<div class="lede">TRUTH in, one guarded change out, measured later, beliefs updated. Every change to the right of the agent goes through another repo's limits.</div>
<div class="stage">{''.join(svg)}{''.join(html)}</div></body></html>"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/summary.json")
    p.add_argument("--demo", default="demo/out")
    p.add_argument("--docs", default="docs")
    a = p.parse_args()
    docs = Path(a.docs)
    summary = json.loads(Path(a.results).read_text())
    demo = json.loads((Path(a.demo) / "summary.json").read_text())
    with (Path(a.demo) / "decisions.csv").open() as f:
        decisions = [dict(r, seed=demo["seed"]) for r in csv.DictReader(f)]
    html = hero_html(summary, demo, decisions)
    (docs / "hero.html").write_text(html)
    shoot(html, docs / "hero.png")
    arch = architecture_html()
    (docs / "architecture.html").write_text(arch)
    shoot(arch, docs / "architecture.png")
    print(f"wrote {docs}/hero.png and {docs}/architecture.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
