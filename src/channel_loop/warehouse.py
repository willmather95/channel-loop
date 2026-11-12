"""TRUTH: the DuckDB warehouse the agent reads, and nothing else.

Conventions follow marketing-warehouse: one local DuckDB file, daily fact tables keyed by
date, a load-meta table, and idempotent loads (delete the date, then insert). The ad tables
use ads-guardrail-agent's own DDL, so either repo can read the other's file.

Tables
  ads_daily, ads_entities, ads_load_meta   (from ads_guardrail.warehouse, unchanged)
  ads_settings_daily   budget, bid and live creative per campaign per day
  web_daily            landing-page sessions and signups by experiment, variant and source
  gsc_page_query, gsc_load_meta            (marketing-warehouse schema, for the nightly GSC load)
  loop_decisions       the loop's own decisions and measured results, mirrored from the journal
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
from ads_guardrail.warehouse import DDL as ADS_DDL

LOOP_DDL = """
CREATE TABLE IF NOT EXISTS ads_settings_daily (
  entity_key VARCHAR NOT NULL,
  date DATE NOT NULL,
  daily_budget DOUBLE NOT NULL,
  bid DOUBLE,
  creative_id VARCHAR,
  creative_launched_on DATE,
  PRIMARY KEY (entity_key, date)
);
CREATE TABLE IF NOT EXISTS web_daily (
  date DATE NOT NULL,
  page VARCHAR NOT NULL,
  experiment_key VARCHAR NOT NULL,
  variant VARCHAR NOT NULL,
  source VARCHAR NOT NULL,
  sessions BIGINT NOT NULL,
  signups BIGINT NOT NULL,
  PRIMARY KEY (date, page, experiment_key, variant, source)
);
CREATE TABLE IF NOT EXISTS gsc_page_query (
  property VARCHAR NOT NULL, brand VARCHAR NOT NULL, date DATE NOT NULL, page VARCHAR NOT NULL,
  query VARCHAR NOT NULL, clicks DOUBLE NOT NULL, impressions DOUBLE NOT NULL, ctr DOUBLE NOT NULL,
  position DOUBLE NOT NULL, pulled_at TIMESTAMP NOT NULL, PRIMARY KEY (property, date, page, query)
);
CREATE TABLE IF NOT EXISTS gsc_load_meta (
  property VARCHAR PRIMARY KEY, brand VARCHAR NOT NULL, min_date DATE, max_date DATE, row_count BIGINT,
  last_pulled_at TIMESTAMP, notes VARCHAR
);
CREATE TABLE IF NOT EXISTS loop_decisions (
  decision_id VARCHAR PRIMARY KEY,
  date DATE NOT NULL,
  arm VARCHAR NOT NULL,
  target VARCHAR,
  hypothesis VARCHAR,
  estimate_per_day DOUBLE,
  outcome VARCHAR,
  measured_per_day DOUBLE,
  measured_se DOUBLE,
  method VARCHAR
);
"""


@dataclass(frozen=True)
class EntityWindow:
    key: str
    name: str
    account_id: str
    days: int
    spend: float
    impressions: int
    clicks: int
    conversions: float
    value: float
    avg_budget: float
    avg_bid: float | None


class Warehouse:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.path)
        self.con.execute(ADS_DDL)
        self.con.execute(LOOP_DDL)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Warehouse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- loads (idempotent: delete the day, then insert) -------------------------------------
    def load_day(self, day: date, ads: list[dict], settings: list[dict], web: list[dict]) -> int:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        c = self.con
        c.execute("BEGIN")
        try:
            for t in ("ads_daily", "ads_settings_daily", "web_daily"):
                c.execute(f"DELETE FROM {t} WHERE date = ?", [day])
            if ads:
                c.executemany("INSERT INTO ads_daily VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
                    [r["platform"], r["account_id"], r["entity_id"], r["entity_name"], r["date"], r["spend"],
                     r["impressions"], r["clicks"], r["conversions"], r["conversion_value"], now] for r in ads])
            if settings:
                c.executemany("INSERT INTO ads_settings_daily VALUES (?,?,?,?,?,?)", [
                    [r["entity_key"], r["date"], r["daily_budget"], r["bid"], r["creative_id"],
                     r["creative_launched_on"]] for r in settings])
            if web:
                c.executemany("INSERT INTO web_daily VALUES (?,?,?,?,?,?,?)", [
                    [r["date"], r["page"], r["experiment_key"], r["variant"], r["source"], r["sessions"],
                     r["signups"]] for r in web])
            names = {f"{r['platform']}:{r['account_id']}:{r['entity_id']}": r["entity_name"] for r in ads}
            for r in settings:  # latest settings snapshot, in ads-guardrail-agent's ads_entities shape
                plat, acct, ent = r["entity_key"].split(":", 2)
                c.execute("INSERT OR REPLACE INTO ads_entities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          [plat, acct, ent, names.get(r["entity_key"], ""), r["daily_budget"], r["bid"], False, False,
                           r["bid"], None, "ENABLED", "", now])
            c.execute("""
                INSERT OR REPLACE INTO ads_load_meta
                SELECT platform, account_id, min(date), max(date), count(*), ?, 'channel-loop load'
                FROM ads_daily GROUP BY platform, account_id""", [now])
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        return len(ads) + len(settings) + len(web)

    def has_day(self, day: date) -> bool:
        return bool(self.con.execute("SELECT count(*) FROM ads_daily WHERE date = ?", [day]).fetchone()[0])

    def max_day(self) -> date | None:
        return self.con.execute("SELECT max(date) FROM ads_daily").fetchone()[0]

    # --- reads used by the agent ---------------------------------------------------------------
    def entity_windows(self, start: date, end: date) -> dict[str, EntityWindow]:
        rows = self.con.execute("""
            SELECT a.platform || ':' || a.account_id || ':' || a.entity_id AS k, any_value(a.entity_name),
                   any_value(a.account_id), count(*), sum(a.spend), sum(a.impressions), sum(a.clicks),
                   sum(a.conversions), sum(a.conversion_value), avg(s.daily_budget), avg(s.bid)
            FROM ads_daily a LEFT JOIN ads_settings_daily s
              ON s.entity_key = a.platform || ':' || a.account_id || ':' || a.entity_id AND s.date = a.date
            WHERE a.date BETWEEN ? AND ? GROUP BY 1 ORDER BY 1""", [start, end]).fetchall()
        return {r[0]: EntityWindow(r[0], r[1], r[2], int(r[3]), float(r[4]), int(r[5]), int(r[6]), float(r[7]),
                                   float(r[8]), float(r[9] or 0), None if r[10] is None else float(r[10]))
                for r in rows}

    def daily_entity(self, start: date, end: date) -> dict[str, dict[date, tuple[float, float]]]:
        """{entity_key: {day: (spend, conversion_value)}}"""
        out: dict[str, dict[date, tuple[float, float]]] = {}
        for k, d, sp, v in self.con.execute("""
                SELECT platform || ':' || account_id || ':' || entity_id, date, spend, conversion_value
                FROM ads_daily WHERE date BETWEEN ? AND ? ORDER BY 1, 2""", [start, end]).fetchall():
            out.setdefault(k, {})[d] = (float(sp), float(v))
        return out

    def settings_on(self, day: date) -> dict[str, dict[str, Any]]:
        rows = self.con.execute("""SELECT entity_key, daily_budget, bid, creative_id, creative_launched_on
                                   FROM ads_settings_daily WHERE date = ?""", [day]).fetchall()
        return {r[0]: dict(budget=r[1], bid=r[2], creative_id=r[3], launched_on=r[4]) for r in rows}

    def ctr_series(self, key: str, start: date, end: date) -> list[tuple[date, int, int, str]]:
        return self.con.execute("""
            SELECT a.date, a.impressions, a.clicks, s.creative_id FROM ads_daily a JOIN ads_settings_daily s
              ON s.entity_key = a.platform || ':' || a.account_id || ':' || a.entity_id AND s.date = a.date
            WHERE s.entity_key = ? AND a.date BETWEEN ? AND ? ORDER BY 1""", [key, start, end]).fetchall()

    def experiment_counts(self, key: str, start: date, end: date) -> dict[str, tuple[int, int]]:
        rows = self.con.execute("""SELECT variant, sum(sessions), sum(signups) FROM web_daily
                                   WHERE experiment_key = ? AND date BETWEEN ? AND ? GROUP BY 1 ORDER BY 1""",
                                [key, start, end]).fetchall()
        return {r[0]: (int(r[1]), int(r[2])) for r in rows}

    def site_daily(self, start: date, end: date) -> tuple[float, float]:
        """Average (sessions, signups) per day on the landing page, all sources."""
        r = self.con.execute("""SELECT sum(sessions), sum(signups), count(DISTINCT date) FROM web_daily
                                WHERE date BETWEEN ? AND ?""", [start, end]).fetchone()
        n = max(1, int(r[2] or 0))
        return float(r[0] or 0) / n, float(r[1] or 0) / n

    def totals_by_day(self) -> list[tuple[date, float, float, float]]:
        """(day, ad spend, ad-attributed value, site signups) for charts and results."""
        return self.con.execute("""
            WITH a AS (SELECT date, sum(spend) sp, sum(conversion_value) v FROM ads_daily GROUP BY 1),
                 w AS (SELECT date, sum(signups) s FROM web_daily GROUP BY 1)
            SELECT a.date, a.sp, a.v, coalesce(w.s, 0) FROM a LEFT JOIN w USING (date) ORDER BY 1""").fetchall()

    # --- loop history mirror ---------------------------------------------------------------------
    def upsert_decisions(self, rows: Iterable[dict[str, Any]]) -> None:
        cols = ("decision_id", "date", "arm", "target", "hypothesis", "estimate_per_day", "outcome",
                "measured_per_day", "measured_se", "method")
        data = [[r.get(c) for c in cols] for r in rows]
        if not data:
            return
        self.con.execute("BEGIN")
        self.con.executemany("INSERT OR REPLACE INTO loop_decisions VALUES (?,?,?,?,?,?,?,?,?,?)", data)
        self.con.execute("COMMIT")
