# Blue Ballers Analytics — Sleeper Sync (MVP)
#
# Paste this entire file into a single Colab code cell and run it.
# First run will prompt Google Drive authorization — that's where
# blue_ballers.db lives, so history survives across Colab sessions.

get_ipython().system('pip install -q requests sqlalchemy google-genai')

import os
import re
import time
import json
import sqlite3
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, JSON, BigInteger,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/BlueBallersAnalytics"
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = f"{DATA_DIR}/blue_ballers.db"

LEAGUE_ID = "1312250088587812865"  # Blue Ballers — current season league_id
SLEEPER_BASE = "https://api.sleeper.app/v1"
MAX_WEEK = 18            # regular season + playoffs
MAX_SEASON_HOPS = 30     # safety cap when walking the previous_league_id chain
REFRESH_PLAYERS = True   # the full player index is ~14 MB — set False on reruns to skip it

# External dynasty valuation sources (DynastyProcess + FantasyCalc) — see
# project notes for why: replaces the in-house age-curve heuristic with real
# market consensus, and gives historical grading real point-in-time values
# instead of re-deriving them from a formula.
SYNC_EXTERNAL_VALUES = True
DYNASTYPROCESS_VALUES_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/values.csv"
DYNASTYPROCESS_IDS_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
FANTASYCALC_URL = "https://api.fantasycalc.com/values/current"
FANTASYCALC_PARAMS = {"isDynasty": "true", "numQbs": 2, "numTeams": 8, "ppr": 1}  # matches this league's real superflex/8-team format

# One-time historical backfill of DynastyProcess snapshots via GitHub's commit
# history (weekly cadence, history goes back to 2019 — confirmed well past
# this league's 2024 start). Set True once, run, then set back to False —
# it's slow (one raw-file fetch per weekly commit) and only needs to run once;
# regular syncs going forward only need SYNC_EXTERNAL_VALUES above.
BACKFILL_EXTERNAL_HISTORY = False
BACKFILL_SINCE = "2024-01-01"

# Real NFL injury-report history (nflverse) — feeds the durability-risk signal.
# Unlike the value archive, each season's file is already a complete historical
# snapshot (not a daily diff), so this just re-fetches + upserts every sync,
# no backfill script needed. A fixed rolling window of REAL NFL seasons,
# deliberately independent of which seasons this dynasty league has synced --
# a player's injury history from before this league existed is still real
# signal (e.g. a chronically-injured veteran).
SYNC_INJURY_REPORTS = True
INJURY_SEASONS_BACK = 3
NFLVERSE_INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"

# Real snap-count presence (nflverse, sourced from Pro Football Reference) --
# a player who's on a team's roster but has NO snap-count row for a week
# that team played a real game almost certainly missed that game (IR,
# inactive, healthy scratch). Confirmed empirically: this catches full
# season-long IR stints the weekly injury-report tag alone misses, since a
# player often drops off that report entirely once on IR (no more weekly
# Doubtful/Out tags) -- e.g. a real 2024 case correctly showed 13 of 17
# missed games via snap-count absence vs. only 2 tagged weeks via injuries.
SYNC_SNAP_COUNTS = True
NFLVERSE_SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"

# AI-generated League News (Google Gemini free tier) — one editorial recap
# article per (league_id, week), written once at sync time and cached in the
# league_news table; the deployed dashboard only ever reads it, never calls
# Gemini itself. Key comes from Colab's own secrets store (key icon in the
# left sidebar) -- add GEMINI_API_KEY there once and grant this notebook
# access the first time it prompts. Never hardcode the key here.
# Publish the finished database to a GitHub Release, which is what the deployed dashboard
# downloads. Drive stays the working copy (Colab writes straight to it), but Drive's
# anonymous-download limit is enforced per source IP and Streamlit Cloud shares its egress with
# thousands of apps, so the deployed app could sit throttled for days while the same link worked
# fine elsewhere. GitHub documents no bandwidth limit on release assets.
# Needs a GitHub token with `repo` scope in Colab's secrets (key icon, left sidebar) named
# GITHUB_TOKEN. Without one this step is skipped and the sync is otherwise unaffected.
PUBLISH_TO_GITHUB = True
GITHUB_REPO = "tommykoreen-commits/blue-ballers-analytics"
GITHUB_RELEASE_TAG = "data"

try:
    from google.colab import userdata as _colab_userdata
    GITHUB_TOKEN = _colab_userdata.get("GITHUB_TOKEN")
except Exception:
    GITHUB_TOKEN = None

SYNC_LEAGUE_NEWS = True
GEMINI_MODEL = "gemini-3.7-flash"
GEMINI_PACING_SECONDS = 4  # stay well under the free tier's per-minute cap during a ~28-week backfill

try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY  # genai.Client() picks this up implicitly

print("Database will be stored at:", DB_PATH)

engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# Schema — each season has its own distinct league_id in Sleeper (linked via
# previous_league_id), so keying tables by (league_id, ...) gives us
# historical storage for free: syncing the current season never touches
# last season's rows.
# ---------------------------------------------------------------------------
class LeagueSeason(Base):
    __tablename__ = "league_seasons"
    league_id = Column(String, primary_key=True)
    season = Column(String, nullable=False)
    name = Column(String)
    previous_league_id = Column(String)
    status = Column(String)
    total_rosters = Column(Integer)
    draft_id = Column(String)
    settings = Column(JSON)
    scoring_settings = Column(JSON)
    roster_positions = Column(JSON)
    synced_at = Column(String)


class Manager(Base):
    __tablename__ = "managers"
    user_id = Column(String, primary_key=True)
    display_name = Column(String)
    avatar = Column(String)


class RosterTeamName(Base):
    """Team name a manager used in a given season — names change season to season."""
    __tablename__ = "roster_team_names"
    league_id = Column(String, primary_key=True)
    user_id = Column(String, primary_key=True)
    team_name = Column(String)


class Roster(Base):
    __tablename__ = "rosters"
    league_id = Column(String, primary_key=True)
    roster_id = Column(Integer, primary_key=True)
    owner_id = Column(String)
    players = Column(JSON)
    starters = Column(JSON)
    reserve = Column(JSON)
    taxi = Column(JSON)
    wins = Column(Integer)
    losses = Column(Integer)
    ties = Column(Integer)
    fpts = Column(Float)
    fpts_against = Column(Float)
    waiver_position = Column(Integer)
    waiver_budget_used = Column(Integer)
    total_moves = Column(Integer)


class Matchup(Base):
    __tablename__ = "matchups"
    league_id = Column(String, primary_key=True)
    week = Column(Integer, primary_key=True)
    roster_id = Column(Integer, primary_key=True)
    matchup_id = Column(Integer)
    points = Column(Float)
    starters = Column(JSON)
    players = Column(JSON)
    players_points = Column(JSON)


class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(String, primary_key=True)
    league_id = Column(String, nullable=False)
    week = Column(Integer)
    type = Column(String)  # trade / waiver / free_agent
    status = Column(String)
    creator = Column(String)
    roster_ids = Column(JSON)
    adds = Column(JSON)
    drops = Column(JSON)
    draft_picks = Column(JSON)
    waiver_budget = Column(JSON)
    created = Column(BigInteger)


class Draft(Base):
    __tablename__ = "drafts"
    draft_id = Column(String, primary_key=True)
    league_id = Column(String, nullable=False)
    season = Column(String)
    type = Column(String)
    status = Column(String)
    settings = Column(JSON)
    draft_order = Column(JSON)
    start_time = Column(BigInteger)   # scheduled start (epoch ms) -- fallback anchor
    last_picked = Column(BigInteger)  # when the final pick was actually made (epoch ms) --
    # the real "draft day" anchor for grading picks against value at that moment, not a guess


class DraftPick(Base):
    __tablename__ = "draft_picks"
    draft_id = Column(String, primary_key=True)
    pick_no = Column(Integer, primary_key=True)
    round = Column(Integer)
    roster_id = Column(Integer)
    player_id = Column(String)
    picked_by = Column(String)
    is_keeper = Column(Boolean)


class TradedPick(Base):
    __tablename__ = "traded_picks"
    league_id = Column(String, primary_key=True)
    season = Column(String, primary_key=True)
    round = Column(Integer, primary_key=True)
    roster_id = Column(Integer, primary_key=True)  # original pick owner
    owner_id = Column(Integer)                      # current owner roster_id
    previous_owner_id = Column(Integer)


class Player(Base):
    __tablename__ = "players"
    player_id = Column(String, primary_key=True)
    full_name = Column(String)
    first_name = Column(String)
    last_name = Column(String)
    position = Column(String)
    team = Column(String)
    status = Column(String)
    active = Column(Boolean)
    birth_date = Column(String)
    years_exp = Column(Integer)


class ExternalPlayerValueSnapshot(Base):
    """One dated snapshot of one external source's value for one player.
    `source` is one of dynastyprocess_value / dynastyprocess_ecr / fantasycalc
    (ecr is lower-is-better rank, not a dollar value — inverted at read time)."""
    __tablename__ = "external_player_value_snapshots"
    source = Column(String, primary_key=True)
    sleeper_player_id = Column(String, primary_key=True)
    scrape_date = Column(String, primary_key=True)  # ISO date
    raw_value = Column(Float)


class ExternalPickValueSnapshot(Base):
    """One dated snapshot of one external source's value for one future pick.
    `granularity` distinguishes how precisely the source located the pick:
    exact_slot (round+slot both known, e.g. current draft class), tier
    (round + early/mid/late bucket, e.g. FantasyCalc's out-year picks), or
    round (only a single per-round number, e.g. 2028+ picks). `slot`/`tier`
    are sentinel 0/"" when not applicable to that row's granularity."""
    __tablename__ = "external_pick_value_snapshots"
    source = Column(String, primary_key=True)
    season = Column(String, primary_key=True)
    round = Column(Integer, primary_key=True)
    granularity = Column(String, primary_key=True)
    slot = Column(Integer, primary_key=True)
    tier = Column(String, primary_key=True)
    scrape_date = Column(String, primary_key=True)
    raw_value = Column(Float)


class InjuryReport(Base):
    """One real weekly NFL injury-report appearance (nflverse). Only rows
    with a genuine report_status (Questionable/Doubtful/Out) are stored --
    a player not appearing at all that week means no injury designation,
    not "definitely healthy" (nflverse doesn't publish a healthy row)."""
    __tablename__ = "injury_reports"
    season = Column(String, primary_key=True)
    week = Column(Integer, primary_key=True)
    sleeper_player_id = Column(String, primary_key=True)
    report_status = Column(String)


class SnapCount(Base):
    """One real weekly snap-count presence row (nflverse, via PFR) -- a
    player who's absent from this table for a week their own team otherwise
    has rows for (i.e. that team played a real game) almost certainly missed
    that game entirely. `team` lets the reader reconstruct "did this
    player's team play a real game that week" from the table itself, no
    separate schedule table needed."""
    __tablename__ = "snap_counts"
    season = Column(String, primary_key=True)
    week = Column(Integer, primary_key=True)
    sleeper_player_id = Column(String, primary_key=True)
    team = Column(String)


class LeagueNews(Base):
    """One AI-generated weekly recap article (Google Gemini), written once at
    sync time and never regenerated automatically -- an existing (league_id,
    week) row is left untouched by every later sync run, including backfill,
    so articles don't churn once published and the free-tier API budget
    isn't re-spent on weeks that already have a recap. No `season` column --
    league_id already uniquely identifies the season (same convention as
    Matchup/Roster/Transaction). `source_facts` snapshots the exact MVP/
    upset/headline numbers fed into the prompt, so a bad or stale-looking
    article can be diagnosed -- or hand-regenerated by deleting the row --
    without re-deriving the facts from the raw matchup tables."""
    __tablename__ = "league_news"
    league_id = Column(String, primary_key=True)
    week = Column(Integer, primary_key=True)
    article = Column(String)
    model = Column(String)
    generated_at = Column(String)
    source_facts = Column(JSON)


Base.metadata.create_all(engine)
print("Tables ready:", list(Base.metadata.tables.keys()))


def ensure_column(engine, table, column, sql_type):
    """SQLAlchemy's create_all() above only creates missing TABLES -- it never
    ALTERs an already-existing table to add a new column, so adding a Column
    to a model (like Draft.start_time/last_picked) silently does nothing for
    a DB that already has that table. This adds any column that's genuinely
    missing, one ALTER TABLE per (table, column), safe to re-run every sync
    since it checks PRAGMA table_info first."""
    with engine.connect() as conn:
        existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            conn.commit()
            print(f"  migrated: added {table}.{column}")


ensure_column(engine, "drafts", "start_time", "BIGINT")
ensure_column(engine, "drafts", "last_picked", "BIGINT")

# ---------------------------------------------------------------------------
# Sleeper API client
# ---------------------------------------------------------------------------
def sleeper_get(path):
    url = f"{SLEEPER_BASE}{path}"
    resp = None
    for attempt in range(3):
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
    resp.raise_for_status()


def get_league(league_id):
    return sleeper_get(f"/league/{league_id}")


def get_users(league_id):
    return sleeper_get(f"/league/{league_id}/users") or []


def get_rosters(league_id):
    return sleeper_get(f"/league/{league_id}/rosters") or []


def get_matchups(league_id, week):
    return sleeper_get(f"/league/{league_id}/matchups/{week}") or []


def get_transactions(league_id, week):
    return sleeper_get(f"/league/{league_id}/transactions/{week}") or []


def get_traded_picks(league_id):
    return sleeper_get(f"/league/{league_id}/traded_picks") or []


def get_drafts(league_id):
    return sleeper_get(f"/league/{league_id}/drafts") or []


def get_draft_picks(draft_id):
    return sleeper_get(f"/draft/{draft_id}/picks") or []


def get_all_players():
    """Sleeper's global player index — ~14 MB, refresh sparingly (once a day is plenty)."""
    return sleeper_get("/players/nfl") or {}


def discover_season_chain(current_league_id):
    """Walk previous_league_id backwards to find every season's league_id, oldest last."""
    chain = []
    league_id = current_league_id
    seen = set()
    for _ in range(MAX_SEASON_HOPS):
        if not league_id or league_id in seen or league_id == "0":
            break
        seen.add(league_id)
        league = get_league(league_id)
        if not league:
            break
        chain.append(league)
        league_id = league.get("previous_league_id")
    return chain


# ---------------------------------------------------------------------------
# Sync functions — each uses session.merge(), which SQLAlchemy resolves by
# primary key: insert if new, update in place if it already exists. Safe to
# re-run any time.
# ---------------------------------------------------------------------------
def sync_league_season(session, league):
    session.merge(LeagueSeason(
        league_id=league["league_id"],
        season=league.get("season"),
        name=league.get("name"),
        previous_league_id=league.get("previous_league_id"),
        status=league.get("status"),
        total_rosters=league.get("total_rosters"),
        draft_id=league.get("draft_id"),
        settings=league.get("settings"),
        scoring_settings=league.get("scoring_settings"),
        roster_positions=league.get("roster_positions"),
        synced_at=datetime.now(timezone.utc).isoformat(),
    ))


def sync_users(session, league_id, users):
    for u in users:
        session.merge(Manager(
            user_id=u["user_id"],
            display_name=u.get("display_name"),
            avatar=u.get("avatar"),
        ))
        team_name = (u.get("metadata") or {}).get("team_name")
        session.merge(RosterTeamName(
            league_id=league_id,
            user_id=u["user_id"],
            team_name=team_name,
        ))


def sync_rosters(session, league_id, rosters):
    for r in rosters:
        s = r.get("settings") or {}
        session.merge(Roster(
            league_id=league_id,
            roster_id=r["roster_id"],
            owner_id=r.get("owner_id"),
            players=r.get("players"),
            starters=r.get("starters"),
            reserve=r.get("reserve"),
            taxi=r.get("taxi"),
            wins=s.get("wins", 0),
            losses=s.get("losses", 0),
            ties=s.get("ties", 0),
            fpts=s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100,
            fpts_against=s.get("fpts_against", 0) + s.get("fpts_against_decimal", 0) / 100,
            waiver_position=s.get("waiver_position"),
            waiver_budget_used=s.get("waiver_budget_used"),
            total_moves=s.get("total_moves"),
        ))


def sync_matchups(session, league_id, max_week=None):
    total = 0
    for week in range(1, (max_week or MAX_WEEK) + 1):
        for m in get_matchups(league_id, week):
            if m.get("roster_id") is None:
                continue
            session.merge(Matchup(
                league_id=league_id,
                week=week,
                roster_id=m["roster_id"],
                matchup_id=m.get("matchup_id"),
                points=m.get("points"),
                starters=m.get("starters"),
                players=m.get("players"),
                players_points=m.get("players_points"),
            ))
            total += 1
    return total


def sync_transactions(session, league_id, max_week=None):
    total = 0
    for week in range(1, (max_week or MAX_WEEK) + 1):
        for t in get_transactions(league_id, week):
            session.merge(Transaction(
                transaction_id=t["transaction_id"],
                league_id=league_id,
                week=week,
                type=t.get("type"),
                status=t.get("status"),
                creator=t.get("creator"),
                roster_ids=t.get("roster_ids"),
                adds=t.get("adds"),
                drops=t.get("drops"),
                draft_picks=t.get("draft_picks"),
                waiver_budget=t.get("waiver_budget"),
                created=t.get("created"),
            ))
            total += 1
    return total


def sync_traded_picks(session, league_id):
    picks = get_traded_picks(league_id)
    for p in picks:
        session.merge(TradedPick(
            league_id=league_id,
            season=p["season"],
            round=p["round"],
            roster_id=p["roster_id"],
            owner_id=p.get("owner_id"),
            previous_owner_id=p.get("previous_owner_id"),
        ))
    return len(picks)


def sync_drafts(session, league_id):
    drafts = get_drafts(league_id)
    pick_count = 0
    for d in drafts:
        session.merge(Draft(
            draft_id=d["draft_id"],
            league_id=league_id,
            season=d.get("season"),
            type=d.get("type"),
            status=d.get("status"),
            settings=d.get("settings"),
            draft_order=d.get("draft_order"),
            start_time=d.get("start_time"),
            last_picked=d.get("last_picked"),
        ))
        for p in get_draft_picks(d["draft_id"]):
            if p.get("pick_no") is None:
                continue
            session.merge(DraftPick(
                draft_id=d["draft_id"],
                pick_no=p["pick_no"],
                round=p.get("round"),
                roster_id=p.get("roster_id"),
                player_id=p.get("player_id"),
                picked_by=p.get("picked_by"),
                is_keeper=p.get("is_keeper"),
            ))
            pick_count += 1
    return len(drafts), pick_count


def sync_players(session, players_map):
    count = 0
    for player_id, p in players_map.items():
        full_name = p.get("full_name") or f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip()
        session.merge(Player(
            player_id=player_id,
            full_name=full_name,
            first_name=p.get("first_name"),
            last_name=p.get("last_name"),
            position=p.get("position"),
            team=p.get("team"),
            status=p.get("status"),
            active=p.get("active"),
            birth_date=p.get("birth_date"),
            years_exp=p.get("years_exp"),
        ))
        count += 1
    return count


# ---------------------------------------------------------------------------
# External dynasty valuation — DynastyProcess (values.csv, weekly, GPL-3.0)
# + FantasyCalc (live JSON API, current only). Both cover players; both also
# embed pick rows on the same value scale as players (DynastyProcess's
# values.csv folds its 2026 pick rows in directly — values-picks.csv turned
# out to be a redundant duplicate of those same rows, so it's skipped).
# ---------------------------------------------------------------------------
PICK_LABEL_EXACT = re.compile(r"^(\d{4}) Pick (\d+)\.(\d+)$")
PICK_LABEL_TIER = re.compile(r"^(\d{4}) (\d+)(?:st|nd|rd|th) \((Early|Mid|Late)\)$")
PICK_LABEL_ROUND = re.compile(r"^(\d{4}) (\d+)(?:st|nd|rd|th)$")


def parse_pick_label(label):
    """Parse a pick label into (season, round, granularity, slot, tier).
    Handles all 3 granularities seen in the wild: '2026 Pick 1.01' (exact
    slot — this league's upcoming draft), '2027 1st (Early)' (tier — next
    year's draft, FantasyCalc only), '2028 1st' (round only — no pick this
    far out has a slot or tier estimate yet). Returns None for player rows."""
    if not isinstance(label, str):
        return None
    m = PICK_LABEL_EXACT.match(label)
    if m:
        season, round_no, slot = m.groups()
        return season, int(round_no), "exact_slot", int(slot), ""
    m = PICK_LABEL_TIER.match(label)
    if m:
        season, round_no, tier = m.groups()
        return season, int(round_no), "tier", 0, tier.lower()
    m = PICK_LABEL_ROUND.match(label)
    if m:
        season, round_no = m.groups()
        return season, int(round_no), "round", 0, ""
    return None


def fetch_dynastyprocess_values():
    return pd.read_csv(DYNASTYPROCESS_VALUES_URL)


def fetch_dynastyprocess_ids():
    return pd.read_csv(DYNASTYPROCESS_IDS_URL)


def fetch_fantasycalc_values():
    resp = requests.get(FANTASYCALC_URL, params=FANTASYCALC_PARAMS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def crosswalk_to_sleeper(values_df, ids_df, id_col="fp_id", ids_id_col="fantasypros_id"):
    """Join key defaults to DynastyProcess's fp_id -> db_playerids.fantasypros_id
    -> sleeper_id, but any db_playerids.csv ID column works the same way (e.g.
    id_col="gsis_id", ids_id_col="gsis_id" for nflverse injury data, which uses
    the same column name on both sides). Both null columns MUST be dropped
    before merging — a left merge on a column containing NaN keys matches
    every NaN against every other NaN (cartesian blowup), silently producing
    millions of bogus rows and a near-zero apparent join rate. Confirmed
    empirically: skipping this drop turned a real ~87% join rate into a false
    0% overlap with real Sleeper IDs."""
    values_valid = values_df.dropna(subset=[id_col])
    ids_valid = ids_df.dropna(subset=[ids_id_col])
    merged = values_valid.merge(
        ids_valid[[ids_id_col, "sleeper_id"]],
        left_on=id_col, right_on=ids_id_col, how="left",
    ).dropna(subset=["sleeper_id"])
    merged["sleeper_id"] = merged["sleeper_id"].astype(int).astype(str)
    return merged


def sync_external_player_values(session, scrape_date_today):
    dp_ids = fetch_dynastyprocess_ids()
    dp_values = fetch_dynastyprocess_values()
    dp_players = crosswalk_to_sleeper(dp_values[dp_values["pos"] != "PICK"], dp_ids)

    count = 0
    for _, row in dp_players.iterrows():
        sid, scrape = row["sleeper_id"], str(row["scrape_date"])
        for source, col in (("dynastyprocess_value", "value_2qb"), ("dynastyprocess_ecr", "ecr_2qb")):
            if pd.notna(row.get(col)):
                session.merge(ExternalPlayerValueSnapshot(
                    source=source, sleeper_player_id=sid, scrape_date=scrape, raw_value=float(row[col]),
                ))
                count += 1

    for d in fetch_fantasycalc_values():
        p = d.get("player") or {}
        if p.get("position") == "PICK" or not p.get("sleeperId"):
            continue
        session.merge(ExternalPlayerValueSnapshot(
            source="fantasycalc", sleeper_player_id=str(p["sleeperId"]),
            scrape_date=scrape_date_today, raw_value=float(d["value"]),
        ))
        count += 1
    return count


def sync_external_pick_values(session, scrape_date_today):
    dp_values = fetch_dynastyprocess_values()
    dp_picks = dp_values[dp_values["pos"] == "PICK"]

    count = 0
    for _, row in dp_picks.iterrows():
        parsed = parse_pick_label(row["player"])
        if not parsed:
            continue
        season, round_no, granularity, slot, tier = parsed
        scrape = str(row["scrape_date"])
        for source, col in (("dynastyprocess_value", "value_2qb"), ("dynastyprocess_ecr", "ecr_2qb")):
            if pd.notna(row.get(col)):
                session.merge(ExternalPickValueSnapshot(
                    source=source, season=season, round=round_no, granularity=granularity,
                    slot=slot, tier=tier, scrape_date=scrape, raw_value=float(row[col]),
                ))
                count += 1

    for d in fetch_fantasycalc_values():
        p = d.get("player") or {}
        if p.get("position") != "PICK":
            continue
        parsed = parse_pick_label(p.get("name", ""))
        if not parsed:
            continue
        season, round_no, granularity, slot, tier = parsed
        session.merge(ExternalPickValueSnapshot(
            source="fantasycalc", season=season, round=round_no, granularity=granularity,
            slot=slot, tier=tier, scrape_date=scrape_date_today, raw_value=float(d["value"]),
        ))
        count += 1
    return count


# ---------------------------------------------------------------------------
# Real NFL injury-report history (nflverse) — feeds the durability-risk
# signal. Each season's file is already a complete historical snapshot to
# date, so there's no separate backfill step like the value archive needed --
# just re-fetch + upsert a rolling window of real NFL seasons every sync.
# ---------------------------------------------------------------------------
def fetch_nfl_injury_reports(season):
    return pd.read_csv(NFLVERSE_INJURIES_URL.format(season=season))


def sync_injury_reports(session, dp_ids):
    current_year = datetime.now(timezone.utc).year
    total = 0
    for season in range(current_year - INJURY_SEASONS_BACK, current_year + 1):
        try:
            df = fetch_nfl_injury_reports(season)
        except Exception as e:
            print(f"  no injury data for {season} yet, skipping ({e})")
            continue
        reported = df.dropna(subset=["report_status"])
        matched = crosswalk_to_sleeper(reported, dp_ids, id_col="gsis_id", ids_id_col="gsis_id")
        for _, row in matched.iterrows():
            session.merge(InjuryReport(
                season=str(row["season"]), week=int(row["week"]),
                sleeper_player_id=row["sleeper_id"], report_status=row["report_status"],
            ))
            total += 1
    return total


def fetch_nfl_snap_counts(season):
    return pd.read_csv(NFLVERSE_SNAP_COUNTS_URL.format(season=season))


def sync_snap_counts(session, dp_ids):
    """Every player's weekly snap-count presence (not just injured players --
    absence itself is the signal here, so every real row matters, unlike
    injury_reports which only keeps rows with a genuine designation)."""
    current_year = datetime.now(timezone.utc).year
    total = 0
    for season in range(current_year - INJURY_SEASONS_BACK, current_year + 1):
        try:
            df = fetch_nfl_snap_counts(season)
        except Exception as e:
            print(f"  no snap count data for {season} yet, skipping ({e})")
            continue
        matched = crosswalk_to_sleeper(df, dp_ids, id_col="pfr_player_id", ids_id_col="pfr_id")
        for _, row in matched.iterrows():
            session.merge(SnapCount(
                season=str(row["season"]), week=int(row["week"]),
                sleeper_player_id=row["sleeper_id"], team=row["team"],
            ))
            total += 1
    return total


# ---------------------------------------------------------------------------
# One-time historical backfill — reconstructs DynastyProcess snapshots from
# GitHub's commit history for values.csv (weekly cadence, confirmed history
# back to 2019). Only run when BACKFILL_EXTERNAL_HISTORY is True; leave False
# on normal runs. FantasyCalc has no historical API — pre-launch trades only
# get DynastyProcess's two signals (value_2qb + ecr_2qb), not the 3-source
# blend live values get from launch day onward.
# ---------------------------------------------------------------------------
GITHUB_COMMITS_API = "https://api.github.com/repos/dynastyprocess/data/commits"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dynastyprocess/data"


def list_dynastyprocess_commits(path, since):
    commits, page = [], 1
    while True:
        resp = requests.get(GITHUB_COMMITS_API, params={
            "path": path, "since": f"{since}T00:00:00Z", "per_page": 100, "page": page,
        }, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        commits.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(1)
    return commits


def backfill_dynastyprocess_history(session, since=BACKFILL_SINCE):
    dp_ids = fetch_dynastyprocess_ids()  # today's crosswalk — fp_id<->sleeper_id identity
    # mappings are stable over time, so reusing today's is fine (avoids doubling
    # request volume by re-fetching db_playerids.csv at every historical commit)

    commits = list_dynastyprocess_commits("files/values.csv", since)
    print(f"Backfill: found {len(commits)} values.csv commits since {since}")

    total = 0
    for i, c in enumerate(commits):
        sha = c["sha"]
        try:
            df = pd.read_csv(f"{GITHUB_RAW_BASE}/{sha}/files/values.csv")
        except Exception as e:
            print(f"  skip commit {sha[:8]} ({c['commit']['committer']['date'][:10]}): {e}")
            continue

        players = crosswalk_to_sleeper(df[df["pos"] != "PICK"], dp_ids)
        for _, row in players.iterrows():
            sid, scrape = row["sleeper_id"], str(row["scrape_date"])
            for source, col in (("dynastyprocess_value", "value_2qb"), ("dynastyprocess_ecr", "ecr_2qb")):
                if pd.notna(row.get(col)):
                    session.merge(ExternalPlayerValueSnapshot(
                        source=source, sleeper_player_id=sid, scrape_date=scrape, raw_value=float(row[col]),
                    ))
                    total += 1

        picks = df[df["pos"] == "PICK"]
        for _, row in picks.iterrows():
            parsed = parse_pick_label(row["player"])
            if not parsed:
                continue
            season, round_no, granularity, slot, tier = parsed
            scrape = str(row["scrape_date"])
            for source, col in (("dynastyprocess_value", "value_2qb"), ("dynastyprocess_ecr", "ecr_2qb")):
                if pd.notna(row.get(col)):
                    session.merge(ExternalPickValueSnapshot(
                        source=source, season=season, round=round_no, granularity=granularity,
                        slot=slot, tier=tier, scrape_date=scrape, raw_value=float(row[col]),
                    ))
                    total += 1

        if (i + 1) % 10 == 0:
            session.commit()
            print(f"  processed {i + 1}/{len(commits)} commits, {total} snapshot rows so far")
        time.sleep(0.3)  # polite pacing against raw.githubusercontent.com

    session.commit()
    print(f"Backfill complete: {total} historical snapshot rows from {len(commits)} weekly commits")
    return total


# ---------------------------------------------------------------------------
# Orchestrator — one-click sync of every season
# ---------------------------------------------------------------------------
def sync_all_seasons(current_league_id):
    session = SessionLocal()
    summary = []
    try:
        chain = discover_season_chain(current_league_id)
        print(f"Discovered {len(chain)} season(s): " +
              ", ".join(f"{lg['season']}({lg['league_id']})" for lg in chain))

        for league in chain:
            league_id = league["league_id"]
            season = league.get("season")
            print(f"\nSyncing season {season} — league_id {league_id}")

            sync_league_season(session, league)

            users = get_users(league_id)
            sync_users(session, league_id, users)

            rosters = get_rosters(league_id)
            sync_rosters(session, league_id, rosters)

            matchup_rows = sync_matchups(session, league_id)
            txn_rows = sync_transactions(session, league_id)
            pick_rows = sync_traded_picks(session, league_id)
            draft_count, draft_pick_count = sync_drafts(session, league_id)

            session.commit()

            row = {
                "season": season,
                "league_id": league_id,
                "managers": len(users),
                "rosters": len(rosters),
                "matchup_rows": matchup_rows,
                "transactions": txn_rows,
                "traded_picks": pick_rows,
                "drafts": draft_count,
                "draft_picks": draft_pick_count,
            }
            summary.append(row)
            print("  " + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("season", "league_id")))
    finally:
        session.close()
    return summary


summary = sync_all_seasons(LEAGUE_ID)
print(pd.DataFrame(summary))

# ---------------------------------------------------------------------------
# Player index — full name/position/team lookup. Set REFRESH_PLAYERS = False
# above on reruns if you don't want to re-download the ~14 MB index every time.
# ---------------------------------------------------------------------------
if REFRESH_PLAYERS:
    print("Fetching full Sleeper player index (~14 MB)...")
    session = SessionLocal()
    try:
        player_count = sync_players(session, get_all_players())
        session.commit()
    finally:
        session.close()
    print(f"Synced {player_count} players")
else:
    print("Skipping player index refresh (REFRESH_PLAYERS=False) — using whatever's already in the DB")

# ---------------------------------------------------------------------------
# External dynasty valuation — DynastyProcess + FantasyCalc. Runs every sync
# (this IS the going-forward snapshot archive: each run appends today's
# values, it never overwrites past dates). Wrapped defensively so a network
# hiccup or upstream format change here can't take down the core league sync
# above, which already succeeded by this point.
# ---------------------------------------------------------------------------
if SYNC_EXTERNAL_VALUES:
    print("\nSyncing external dynasty valuations (DynastyProcess + FantasyCalc)...")
    session = SessionLocal()
    try:
        scrape_date_today = datetime.now(timezone.utc).date().isoformat()
        player_rows = sync_external_player_values(session, scrape_date_today)
        pick_rows = sync_external_pick_values(session, scrape_date_today)
        session.commit()
        print(f"  external player value rows: {player_rows}, external pick value rows: {pick_rows}")
    except Exception as e:
        session.rollback()
        print(f"  WARNING: external valuation sync failed, skipping ({e}) — league data above is unaffected")
    finally:
        session.close()
else:
    print("Skipping external valuation sync (SYNC_EXTERNAL_VALUES=False)")

if SYNC_INJURY_REPORTS:
    print("\nSyncing real NFL injury-report history (nflverse)...")
    session = SessionLocal()
    try:
        dp_ids = fetch_dynastyprocess_ids()
        injury_rows = sync_injury_reports(session, dp_ids)
        session.commit()
        print(f"  injury report rows: {injury_rows}")
    except Exception as e:
        session.rollback()
        print(f"  WARNING: injury report sync failed, skipping ({e}) — everything else above is unaffected")
    finally:
        session.close()
else:
    print("Skipping injury report sync (SYNC_INJURY_REPORTS=False)")

if SYNC_SNAP_COUNTS:
    print("\nSyncing real NFL snap-count history (nflverse)...")
    session = SessionLocal()
    try:
        dp_ids = fetch_dynastyprocess_ids()
        snap_rows = sync_snap_counts(session, dp_ids)
        session.commit()
        print(f"  snap count rows: {snap_rows}")
    except Exception as e:
        session.rollback()
        print(f"  WARNING: snap count sync failed, skipping ({e}) — everything else above is unaffected")
    finally:
        session.close()
else:
    print("Skipping snap count sync (SYNC_SNAP_COUNTS=False)")

# ---------------------------------------------------------------------------
# AI-generated League News (Google Gemini) — ported copies of dashboard_app.py's
# real Weekly MVP / Biggest Upset logic (kept byte-for-byte identical to their
# dashboard originals so this stays easy to cross-check against the live app),
# feeding a grounded, data-only prompt into Gemini. Runs once per (league_id,
# week) ever -- an existing article is never regenerated or overwritten.
# ---------------------------------------------------------------------------
def load_table(query, params=None):
    """sync.py's own copy of dashboard_app.py's load_table (sqlite3 + pd.read_sql_query) --
    reads through this script's own DB_PATH file rather than the SQLAlchemy engine, so every
    ported dashboard function below can keep its qmark-param SQL unchanged. No @st.cache_data
    since this whole script only runs once per sync."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def parse_json_field(value, default):
    if not value:
        return default
    parsed = json.loads(value)
    return parsed if parsed is not None else default


SHRINKAGE_GAMES = 4
RECENCY_DECAY = 0.9
UPSET_TRIALS = 2000


def get_last_completed_week(league_id):
    query = """
        SELECT week FROM matchups
        WHERE league_id = ?
        GROUP BY week HAVING SUM(points) > 0
        ORDER BY week DESC LIMIT 1
    """
    df = load_table(query, params=(league_id,))
    return int(df.iloc[0]["week"]) if not df.empty else None


def get_week_matchups(league_id, week):
    query = """
        SELECT m.matchup_id, m.roster_id, m.points,
               COALESCE(t.team_name, mg.display_name) AS team_name
        FROM matchups m
        LEFT JOIN rosters r ON r.league_id = m.league_id AND r.roster_id = m.roster_id
        LEFT JOIN managers mg ON mg.user_id = r.owner_id
        LEFT JOIN roster_team_names t ON t.league_id = m.league_id AND t.user_id = r.owner_id
        WHERE m.league_id = ? AND m.week = ?
        ORDER BY m.matchup_id
    """
    return load_table(query, params=(league_id, week))


def compute_week_positional_averages(league_id, week, players_df):
    query = "SELECT starters, players_points FROM matchups WHERE league_id = ? AND week = ?"
    df = load_table(query, params=(league_id, week))
    totals, counts = {}, {}
    for _, row in df.iterrows():
        starter_ids = [pid for pid in parse_json_field(row["starters"], []) if pid != "0"]
        points_map = parse_json_field(row["players_points"], {})
        for pid in starter_ids:
            info = players_df.loc[pid] if pid in players_df.index else None
            pos = info["position"] if info is not None else None
            if not pos:
                continue
            totals[pos] = totals.get(pos, 0.0) + points_map.get(pid, 0.0)
            counts[pos] = counts.get(pos, 0) + 1
    return {pos: totals[pos] / counts[pos] for pos in totals if counts.get(pos)}


def get_week_mvp(league_id, week, players_df):
    pos_avg = compute_week_positional_averages(league_id, week, players_df)
    df = load_table(
        "SELECT starters, players_points FROM matchups WHERE league_id = ? AND week = ?",
        params=(league_id, week),
    )
    best = None
    for _, row in df.iterrows():
        starter_ids = [pid for pid in parse_json_field(row["starters"], []) if pid != "0"]
        points_map = parse_json_field(row["players_points"], {})
        for pid in starter_ids:
            info = players_df.loc[pid] if pid in players_df.index else None
            pos = info["position"] if info is not None else None
            if not pos:
                continue
            pts = points_map.get(pid, 0.0)
            margin = pts - pos_avg.get(pos, pts)
            if best is None or margin > best["margin"]:
                best = {"player": info["full_name"], "position": pos, "points": pts, "margin": margin}
    return best


def get_playoff_settings(league_id):
    query = "SELECT settings FROM league_seasons WHERE league_id = ?"
    df = load_table(query, params=(league_id,))
    settings = parse_json_field(df.iloc[0]["settings"], {}) if not df.empty else {}
    playoff_week_start = settings.get("playoff_week_start", 15)
    return {
        "playoff_teams": settings.get("playoff_teams", 4),
        "regular_season_weeks": playoff_week_start - 1,
        "round1_week": playoff_week_start,
        "round2_weeks": (playoff_week_start + 1, playoff_week_start + 2),
        "league_average_match": bool(settings.get("league_average_match", 0)),
    }


def get_full_schedule(league_id, max_week):
    query = "SELECT week, matchup_id, roster_id, points FROM matchups WHERE league_id = ? AND week <= ?"
    return load_table(query, params=(league_id, max_week))


def compute_actual_week_scores(schedule, roster_ids, week):
    week_rows = schedule[schedule["week"] == week]
    if week_rows.empty or week_rows["points"].sum() <= 0:
        return None
    scores = dict(zip(week_rows["roster_id"], week_rows["points"]))
    return {rid: scores.get(rid, 0.0) for rid in roster_ids}


def compute_record_through_week(league_id, roster_ids, week):
    playoff_settings = get_playoff_settings(league_id)
    max_week = min(week, playoff_settings["regular_season_weeks"])
    schedule = get_full_schedule(league_id, max_week)
    record = {rid: {"w": 0, "l": 0, "t": 0} for rid in roster_ids}
    for _, week_rows in schedule.groupby("week"):
        if week_rows["points"].sum() <= 0:
            continue
        for _, pair in week_rows.groupby("matchup_id"):
            if len(pair) != 2:
                continue
            a, b = pair.iloc[0], pair.iloc[1]
            rid_a, rid_b = int(a["roster_id"]), int(b["roster_id"])
            if rid_a not in record or rid_b not in record:
                continue
            if a["points"] > b["points"]:
                record[rid_a]["w"] += 1
                record[rid_b]["l"] += 1
            elif a["points"] < b["points"]:
                record[rid_a]["l"] += 1
                record[rid_b]["w"] += 1
            else:
                record[rid_a]["t"] += 1
                record[rid_b]["t"] += 1
        if playoff_settings["league_average_match"]:
            median = week_rows["points"].median()
            for _, r in week_rows.iterrows():
                rid = int(r["roster_id"])
                if rid not in record:
                    continue
                if r["points"] > median:
                    record[rid]["w"] += 1
                elif r["points"] < median:
                    record[rid]["l"] += 1
                else:
                    record[rid]["t"] += 1
    return record


def get_week_pairings(schedule, week):
    week_rows = schedule[schedule["week"] == week]
    return {mid: group["roster_id"].tolist() for mid, group in week_rows.groupby("matchup_id")}


def get_previous_league_id(league_id):
    df = load_table("SELECT previous_league_id FROM league_seasons WHERE league_id = ?", params=(league_id,))
    if df.empty or not df.iloc[0]["previous_league_id"]:
        return None
    return df.iloc[0]["previous_league_id"]


def get_owner_scores(league_id, owner_id):
    if not league_id:
        return []
    query = """
        SELECT m.points FROM matchups m
        JOIN rosters r ON r.league_id = m.league_id AND r.roster_id = m.roster_id
        WHERE m.league_id = ? AND r.owner_id = ? AND m.points > 0
    """
    return load_table(query, params=(league_id, owner_id))["points"].tolist()


def get_standings(league_id):
    query = """
        SELECT r.roster_id, r.owner_id, r.wins, r.losses, r.ties, r.fpts, r.fpts_against,
               COALESCE(t.team_name, m.display_name) AS team_name, m.display_name AS manager
        FROM rosters r
        LEFT JOIN managers m ON m.user_id = r.owner_id
        LEFT JOIN roster_team_names t ON t.league_id = r.league_id AND t.user_id = r.owner_id
        WHERE r.league_id = ?
    """
    df = load_table(query, params=(league_id,))
    games = df["wins"] + df["losses"] + df["ties"]
    df["win_pct"] = df["wins"] / games.replace(0, 1)
    return df.sort_values(["wins", "fpts"], ascending=False).reset_index(drop=True)


def estimate_team_distributions(league_id, standings, through_week=None):
    prev_league_id = get_previous_league_id(league_id)
    week_filter = " AND week <= ?" if through_week is not None else ""
    week_params = (through_week,) if through_week is not None else ()
    league_scores = load_table(
        f"SELECT points FROM matchups WHERE league_id = ? AND points > 0{week_filter}",
        params=(league_id, *week_params),
    )["points"].tolist()
    fallback_mean = float(np.mean(league_scores)) if league_scores else 100.0
    fallback_std = float(np.std(league_scores)) if len(league_scores) >= 8 else 25.0

    distributions = {}
    for _, row in standings.iterrows():
        own_rows = load_table(
            f"SELECT week, points FROM matchups WHERE league_id = ? AND roster_id = ? AND points > 0{week_filter}",
            params=(league_id, int(row["roster_id"]), *week_params),
        )
        own_scores = own_rows["points"].to_numpy()
        prior_scores = get_owner_scores(prev_league_id, row["owner_id"])
        prior_mean = float(np.mean(prior_scores)) if prior_scores else fallback_mean
        prior_std = float(np.std(prior_scores)) if len(prior_scores) >= 3 else fallback_std

        games = len(own_scores)
        weight = games / (games + SHRINKAGE_GAMES)

        if games:
            weeks_ago = own_rows["week"].max() - own_rows["week"].to_numpy()
            recency_weights = RECENCY_DECAY ** weeks_ago
            own_mean = float(np.average(own_scores, weights=recency_weights))
            own_std = float(np.std(own_scores)) if games >= 2 else prior_std
        else:
            own_mean, own_std = 0.0, prior_std

        mean = weight * own_mean + (1 - weight) * prior_mean
        std = weight * own_std + (1 - weight) * prior_std

        distributions[int(row["roster_id"])] = (mean, max(std, 1.0))
    return distributions


def sample_weekly_scores(rng, means, stds, size):
    shape = (means / stds) ** 2
    scale = (stds ** 2) / means
    return rng.gamma(shape, scale, size=size)


def get_global_team_lookup():
    df = load_table("""
        SELECT r.league_id, r.roster_id, COALESCE(t.team_name, m.display_name) AS team_name
        FROM rosters r
        LEFT JOIN managers m ON m.user_id = r.owner_id
        LEFT JOIN roster_team_names t ON t.league_id = r.league_id AND t.user_id = r.owner_id
    """)
    return {(row["league_id"], row["roster_id"]): row["team_name"] for _, row in df.iterrows()}


def build_upset_history():
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    global_team_lookup = get_global_team_lookup()
    rng = np.random.default_rng()

    rows = []
    for _, s in all_seasons.iterrows():
        lid, season = s["league_id"], s["season"]
        standings_local = get_standings(lid)
        reg_weeks = get_playoff_settings(lid)["regular_season_weeks"]
        schedule = get_full_schedule(lid, max_week=reg_weeks)
        roster_ids = [int(r) for r in standings_local["roster_id"]]

        for week in range(1, reg_weeks + 1):
            actual = compute_actual_week_scores(schedule, roster_ids, week)
            if not actual:
                continue
            distributions = estimate_team_distributions(lid, standings_local, through_week=week - 1)
            for _, pair in get_week_pairings(schedule, week).items():
                if len(pair) != 2:
                    continue
                a, b = int(pair[0]), int(pair[1])
                mean_a, std_a = distributions.get(a, (100.0, 25.0))
                mean_b, std_b = distributions.get(b, (100.0, 25.0))
                samples_a = sample_weekly_scores(rng, np.array([mean_a]), np.array([std_a]), size=(UPSET_TRIALS, 1))
                samples_b = sample_weekly_scores(rng, np.array([mean_b]), np.array([std_b]), size=(UPSET_TRIALS, 1))
                prob_a = float(np.mean(samples_a > samples_b))

                winner, loser = (a, b) if actual[a] > actual[b] else (b, a)
                winner_prob = prob_a if winner == a else (1 - prob_a)
                rows.append({
                    "season": season, "week": week,
                    "winner": global_team_lookup.get((lid, winner), f"Roster {winner}"),
                    "loser": global_team_lookup.get((lid, loser), f"Roster {loser}"),
                    "winner_pregame_prob": round(winner_prob, 3),
                    "winner_score": actual[winner], "loser_score": actual[loser],
                })
    return pd.DataFrame(rows)


class QuotaExhausted(Exception):
    """The free tier's daily request budget for GEMINI_MODEL is used up, confirmed by
    Google's own 'exceeded your current quota' message -- distinct from a short-lived
    429 blip, this will not clear up within the same run no matter how long a retry
    waits (confirmed empirically: 2026-08-20's run kept hitting this same wall for 30
    straight weeks after the first few succeeded). The caller stops the whole backfill
    immediately on this rather than wasting retries/time on every remaining week."""


def generate_recap_text(client, prompt):
    """One Gemini call with 429 retry, mirroring sleeper_get's backoff pattern -- but
    only for a genuinely transient 429. A quota-exceeded 429 (see QuotaExhausted above)
    is raised immediately with no retry, since backing off can't fix a daily cap."""
    last_err = None
    for attempt in range(3):
        try:
            resp = client.interactions.create(model=GEMINI_MODEL, input=prompt)
            text = (resp.output_text or "").strip()
            if not text:
                raise RuntimeError("Gemini returned empty output_text")
            return text
        except Exception as e:
            if "429" in str(e) and "exceeded your current quota" in str(e).lower():
                raise QuotaExhausted(str(e))
            last_err = e
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise last_err


def build_week_facts(league_id, season, week, week_matchups, players_df, upset_history):
    """Structured, JSON-serializable facts for one week -- the ONLY things the recap
    prompt is allowed to talk about. All numpy/pandas scalars cast to plain float/int/str
    so this is safe to store in the JSON source_facts column and safe to embed in an
    f-string prompt."""
    highest = week_matchups.loc[week_matchups["points"].idxmax()]
    lowest = week_matchups.loc[week_matchups["points"].idxmin()]

    two_team_games, matchup_lines = [], []
    for _, group in week_matchups.groupby("matchup_id"):
        if len(group) == 2:
            a, b = group.iloc[0], group.iloc[1]
            two_team_games.append({"a": a, "b": b, "margin": abs(a["points"] - b["points"])})
            matchup_lines.append(f"{a['team_name']} vs {b['team_name']}: {a['points']:.1f}-{b['points']:.1f}")
        else:
            for _, row in group.iterrows():
                matchup_lines.append(f"{row['team_name']}: {row['points']:.1f} (no head-to-head opponent this week)")
    closest = min(two_team_games, key=lambda g: g["margin"]) if two_team_games else None
    blowout = max(two_team_games, key=lambda g: g["margin"]) if two_team_games else None

    mvp = get_week_mvp(league_id, week, players_df)

    week_upsets = upset_history[(upset_history["season"] == season) & (upset_history["week"] == week)]
    upset = week_upsets.loc[week_upsets["winner_pregame_prob"].idxmin()] if not week_upsets.empty else None
    if upset is not None and upset["winner_pregame_prob"] >= 0.5:
        upset = None

    roster_ids = [int(r) for r in week_matchups["roster_id"].unique()]
    record = compute_record_through_week(league_id, roster_ids, week)
    name_by_roster = dict(zip(week_matchups["roster_id"], week_matchups["team_name"]))
    records = [
        {"team": name_by_roster.get(rid, f"Roster {rid}"), "w": r["w"], "l": r["l"], "t": r["t"]}
        for rid, r in record.items()
    ]

    return {
        "season": str(season), "week": int(week),
        "matchup_lines": matchup_lines,
        "top_score": {"team": highest["team_name"], "points": round(float(highest["points"]), 1)},
        "lowest_score": {"team": lowest["team_name"], "points": round(float(lowest["points"]), 1)},
        "closest_game": ({"a": closest["a"]["team_name"], "b": closest["b"]["team_name"],
                           "margin": round(float(closest["margin"]), 1)} if closest else None),
        "biggest_blowout": ({"a": blowout["a"]["team_name"], "b": blowout["b"]["team_name"],
                              "margin": round(float(blowout["margin"]), 1)} if blowout else None),
        "mvp": ({"player": mvp["player"], "position": mvp["position"],
                  "points": round(float(mvp["points"]), 1), "margin": round(float(mvp["margin"]), 1)}
                 if mvp else None),
        "biggest_upset": ({"winner": upset["winner"], "loser": upset["loser"],
                            "winner_pregame_prob": round(float(upset["winner_pregame_prob"]), 3),
                            "winner_score": round(float(upset["winner_score"]), 1),
                            "loser_score": round(float(upset["loser_score"]), 1)}
                            if upset is not None else None),
        "records_through_week": records,
    }


def build_recap_prompt(facts):
    lines = [
        "You are a beat writer for a friend-group dynasty fantasy football league called "
        "\"Blue Ballers\". Write a short, engaging weekly recap using ONLY the facts below -- "
        "do not invent any player, team, score, or statistic not listed. 3-4 short paragraphs "
        "of plain prose, no markdown headers, no bullet lists.",
        "", f"Season {facts['season']}, Week {facts['week']}", "", "Final scores this week:",
    ]
    lines += [f"- {m}" for m in facts["matchup_lines"]]
    lines.append("")
    lines.append(f"Top score: {facts['top_score']['team']} with {facts['top_score']['points']}")
    lines.append(f"Lowest score: {facts['lowest_score']['team']} with {facts['lowest_score']['points']}")
    if facts["closest_game"]:
        cg = facts["closest_game"]
        lines.append(f"Closest game: {cg['a']} vs {cg['b']}, decided by {cg['margin']} points")
    if facts["biggest_blowout"] and facts["biggest_blowout"] != facts["closest_game"]:
        bo = facts["biggest_blowout"]
        lines.append(f"Biggest blowout: {bo['a']} vs {bo['b']}, decided by {bo['margin']} points")
    if facts["mvp"]:
        mv = facts["mvp"]
        lines.append(f"Weekly MVP: {mv['player']} ({mv['position']}), {mv['points']} points, "
                      f"{mv['margin']} above the league {mv['position']} average that week")
    if facts["biggest_upset"]:
        up = facts["biggest_upset"]
        lines.append(f"Biggest upset: {up['winner']} beat {up['loser']} "
                      f"({up['winner_score']}-{up['loser_score']}) despite only a "
                      f"{up['winner_pregame_prob']:.0%} pre-game win chance")
    lines.append("")
    lines.append("Team records through this week:")
    for r in facts["records_through_week"]:
        lines.append(f"- {r['team']}: {r['w']}-{r['l']}-{r['t']}")
    lines.append("")
    lines.append("Write the recap now.")
    return "\n".join(lines)


def sync_league_news(session, client):
    existing = load_table("SELECT league_id, week FROM league_news")
    existing_keys = set(zip(existing["league_id"], existing["week"]))

    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    players_df = load_table(
        "SELECT player_id, full_name, position, team, birth_date FROM players"
    ).set_index("player_id")
    upset_history = build_upset_history()  # whole synced history, computed once, reused for every week

    generated = 0
    quota_exhausted = False
    for _, s in all_seasons.iterrows():
        if quota_exhausted:
            break
        lid, season = s["league_id"], s["season"]
        last_week = get_last_completed_week(lid)
        if last_week is None:
            continue
        last_week = min(last_week, get_playoff_settings(lid)["round2_weeks"][1])

        for week in range(1, last_week + 1):
            if (lid, week) in existing_keys:
                continue  # never overwrite, never re-spend a Gemini call on an existing article
            week_matchups = get_week_matchups(lid, week)
            if week_matchups.empty:
                continue
            try:
                facts = build_week_facts(lid, season, week, week_matchups, players_df, upset_history)
                prompt = build_recap_prompt(facts)
                article_text = generate_recap_text(client, prompt)
            except QuotaExhausted as e:
                print(f"  Gemini free-tier quota exhausted for {GEMINI_MODEL} -- stopping League News "
                      f"generation for this run ({generated} article(s) written). The rest of the "
                      f"backfill will pick up automatically on a future sync once the quota resets: {e}")
                quota_exhausted = True
                break
            except Exception as e:
                print(f"  WARNING: League News generation failed for {season} week {week}, "
                      f"skipping ({e}) -- will retry on next sync")
                continue
            session.merge(LeagueNews(
                league_id=lid, week=week, article=article_text, model=GEMINI_MODEL,
                generated_at=datetime.now(timezone.utc).isoformat(), source_facts=facts,
            ))
            session.commit()  # per-article commit -- a mid-backfill crash loses nothing already written
            generated += 1
            print(f"  generated League News: {season} week {week}")
            time.sleep(GEMINI_PACING_SECONDS)
    return generated


if SYNC_LEAGUE_NEWS and GEMINI_API_KEY:
    print("\nSyncing AI-generated League News (Google Gemini)...")
    session = SessionLocal()
    try:
        from google import genai
        client = genai.Client()
        news_rows = sync_league_news(session, client)
        print(f"  league news articles generated this run: {news_rows}")
    except Exception as e:
        session.rollback()
        print(f"  WARNING: League News sync failed, skipping ({e}) — everything else above is unaffected")
    finally:
        session.close()
elif SYNC_LEAGUE_NEWS:
    print("Skipping League News sync — no GEMINI_API_KEY found in Colab secrets "
          "(add it via the key icon in the left sidebar, then grant this notebook access)")
else:
    print("Skipping League News sync (SYNC_LEAGUE_NEWS=False)")

if BACKFILL_EXTERNAL_HISTORY:
    print(f"\nRunning ONE-TIME DynastyProcess historical backfill since {BACKFILL_SINCE}...")
    session = SessionLocal()
    try:
        backfill_dynastyprocess_history(session, since=BACKFILL_SINCE)
    finally:
        session.close()
    print("Backfill done — set BACKFILL_EXTERNAL_HISTORY back to False before your next regular sync run.")

# ---------------------------------------------------------------------------
# Verify — row counts across the whole historical database
# ---------------------------------------------------------------------------
with engine.connect() as conn:
    for table in ["league_seasons", "managers", "roster_team_names", "rosters",
                  "matchups", "transactions", "drafts", "draft_picks", "traded_picks", "players",
                  "external_player_value_snapshots", "external_pick_value_snapshots", "injury_reports",
                  "snap_counts", "league_news"]:
        count = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
        print(f"{table:20s} {count}")

# ---------------------------------------------------------------------------
# Quick preview — current-season standings (a taste of the future power
# rankings page)
# ---------------------------------------------------------------------------
session = SessionLocal()
try:
    rows = (
        session.query(Roster, Manager, RosterTeamName)
        .join(Manager, Roster.owner_id == Manager.user_id)
        .outerjoin(RosterTeamName, (RosterTeamName.league_id == Roster.league_id) & (RosterTeamName.user_id == Roster.owner_id))
        .filter(Roster.league_id == LEAGUE_ID)
        .all()
    )
    standings = sorted(
        [{
            "team": (team.team_name if team else None) or mgr.display_name,
            "manager": mgr.display_name,
            "wins": r.wins,
            "losses": r.losses,
            "fpts": round(r.fpts or 0, 2),
        } for r, mgr, team in rows],
        key=lambda x: (-x["wins"], -x["fpts"]),
    )
finally:
    session.close()

print(pd.DataFrame(standings))

# ---------------------------------------------------------------------------
# Quick preview — a roster's starters resolved to real names. Proves the
# player-name join actually works end to end.
# ---------------------------------------------------------------------------
session = SessionLocal()
try:
    sample_roster = (
        session.query(Roster, RosterTeamName)
        .outerjoin(RosterTeamName, (RosterTeamName.league_id == Roster.league_id) & (RosterTeamName.user_id == Roster.owner_id))
        .filter(Roster.league_id == LEAGUE_ID)
        .first()
    )
    if sample_roster and sample_roster[0].starters:
        roster, team = sample_roster
        players_by_id = {
            p.player_id: p for p in session.query(Player).filter(Player.player_id.in_(roster.starters)).all()
        }
        starters_named = [
            f"{players_by_id[pid].full_name} ({players_by_id[pid].position})" if pid in players_by_id else pid
            for pid in roster.starters
            if pid != "0"  # Sleeper's placeholder for an empty/unset starting slot
        ]
        print(f"Sample starting lineup — {team.team_name if team else roster.owner_id}:")
        for name in starters_named:
            print(" ", name)
finally:
    session.close()


# ---------------------------------------------------------------------------
# Publish to GitHub Releases — the deployed dashboard reads the database from here.
# Replaces the single asset on the fixed GITHUB_RELEASE_TAG release so the download URL
# never changes. Wrapped defensively like every other external step: a failure here leaves
# the synced database on Drive untouched.
# ---------------------------------------------------------------------------
def publish_db_to_github(token, repo, tag, db_path):
    api = "https://api.github.com"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    resp = requests.get(f"{api}/repos/{repo}/releases/tags/{tag}", headers=headers, timeout=30)
    if resp.status_code == 404:
        resp = requests.post(f"{api}/repos/{repo}/releases", headers=headers, timeout=30,
                              json={"tag_name": tag, "name": "League database",
                                    "body": "Latest synced Blue Ballers database."})
    resp.raise_for_status()
    release = resp.json()

    name = os.path.basename(db_path)
    for asset in release.get("assets", []):
        if asset["name"] == name:
            # A release can't hold two assets with the same name, so the old one goes first.
            deleted = requests.delete(f"{api}/repos/{repo}/releases/assets/{asset['id']}",
                                       headers=headers, timeout=30)
            if deleted.status_code == 403:
                # Reading a public repo needs no auth at all, so everything above succeeds even
                # with a read-only token; this is the first call that actually requires write.
                raise PermissionError(
                    "GitHub refused to replace the existing database asset (403). The token in "
                    "Colab's GITHUB_TOKEN secret can read this repo but can't write to it. A "
                    "classic token needs the `repo` scope; a fine-grained token needs Contents: "
                    "Read and write on tommykoreen-commits/blue-ballers-analytics.")
            deleted.raise_for_status()

    with open(db_path, "rb") as fh:
        upload = requests.post(
            f"https://uploads.github.com/repos/{repo}/releases/{release['id']}/assets",
            params={"name": name}, data=fh, timeout=600,
            headers={**headers, "Content-Type": "application/octet-stream"},
        )
    upload.raise_for_status()
    return upload.json()["browser_download_url"]


if PUBLISH_TO_GITHUB and GITHUB_TOKEN:
    print("\nPublishing database to GitHub Releases...")
    try:
        url = publish_db_to_github(GITHUB_TOKEN, GITHUB_REPO, GITHUB_RELEASE_TAG, DB_PATH)
        print(f"  published: {url}")
        print("  the deployed dashboard will pick this up on its next refresh")
    except Exception as e:
        print(f"  WARNING: publish failed, skipping ({e}) — the database on Drive is fine, "
              f"but the deployed app won't see this sync until a publish succeeds")
elif PUBLISH_TO_GITHUB:
    print("\nSkipping GitHub publish — no GITHUB_TOKEN in Colab secrets "
          "(key icon in the left sidebar; needs a token with `repo` scope)")
else:
    print("\nSkipping GitHub publish (PUBLISH_TO_GITHUB=False)")
