# Blue Ballers Analytics — Sleeper Sync (MVP)
#
# Paste this entire file into a single Colab code cell and run it.
# First run will prompt Google Drive authorization — that's where
# blue_ballers.db lives, so history survives across Colab sessions.

get_ipython().system('pip install -q requests sqlalchemy')

import os
import re
import time
from datetime import datetime, timezone

import requests
import pandas as pd
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


Base.metadata.create_all(engine)
print("Tables ready:", list(Base.metadata.tables.keys()))

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


def crosswalk_to_sleeper(values_df, ids_df):
    """DynastyProcess join key is fp_id -> db_playerids.fantasypros_id ->
    sleeper_id. Both null columns MUST be dropped before merging — a left
    merge on a column containing NaN keys matches every NaN against every
    other NaN (cartesian blowup), silently producing millions of bogus rows
    and a near-zero apparent join rate. Confirmed empirically: skipping this
    drop turned a real ~87% join rate into a false 0% overlap with real
    Sleeper IDs."""
    values_valid = values_df.dropna(subset=["fp_id"])
    ids_valid = ids_df.dropna(subset=["fantasypros_id"])
    merged = values_valid.merge(
        ids_valid[["fantasypros_id", "sleeper_id"]],
        left_on="fp_id", right_on="fantasypros_id", how="left",
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
                  "external_player_value_snapshots", "external_pick_value_snapshots"]:
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
