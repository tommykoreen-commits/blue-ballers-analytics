# Blue Ballers Analytics — Streamlit Dashboard
#
# Deployed on Streamlit Community Cloud, reading a SQLite database that this app
# downloads from Google Drive on its own (see ensure_db() below) — tommy's existing
# blue_ballers_sync.py/.ipynb notebook keeps writing to Drive exactly as it always
# has, with no extra manual step. For local/Colab testing instead of the deployed
# copy, use blue_ballers_dashboard.py, which mounts Drive directly.

import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

N_TRIALS = 10000
SHRINKAGE_GAMES = 4  # weight given to a team's own results vs. its prior-season baseline
RECENCY_DECAY = 0.9  # per-week decay on how much a past week counts toward "current form"

# Player value heuristic — no external dynasty-value source, so this is a rough in-house
# estimate: position scarcity baseline x age-curve x on-field performance. Good for comparing
# teams *relative to each other in this league*, not a market-calibrated trade value.
POSITION_BASELINE = {"QB": 1.4, "RB": 1.0, "WR": 1.0, "TE": 0.75, "K": 0.2}  # QB boosted: this is a 2QB/superflex league
POSITION_PEAK_AGE = {"QB": 29, "RB": 25, "WR": 27, "TE": 27, "K": 30}
POSITION_DECLINE_RATE = {"QB": 0.04, "RB": 0.12, "WR": 0.07, "TE": 0.08, "K": 0.05}  # value lost per year past peak
ROUND_BASE_VALUE = {1: 100, 2: 50, 3: 25, 4: 12}

DB_PATH = "blue_ballers.db"
DB_REFRESH_SECONDS = 600  # how often the deployed app checks Drive for a fresher sync

# Bump this string with every edit — shown in the sidebar so it's obvious at a glance
# whether the deployed app is actually running the latest code.
APP_BUILD = "2026-08-17-robust-drive-download"

st.set_page_config(page_title="Blue Ballers Analytics", layout="wide")


def download_db():
    """One attempt at pulling the latest database from Google Drive, with a single
    retry — Drive's anonymous-link downloads are occasionally flaky/rate-limited
    (seen in production as a bare gdown.exceptions.FileURLRetrievalError), and a
    fresh container hitting this on every cold start is a real risk, not a
    hypothetical one. Returns True on success, False on failure (caller decides
    whether a stale local copy is an acceptable fallback)."""
    import gdown
    for attempt in range(2):
        try:
            gdown.download(id=st.secrets["drive_file_id"], output=DB_PATH, quiet=True, fuzzy=True)
            return True
        except Exception as e:
            last_error = e
            time.sleep(2)
    st.session_state["db_download_error"] = str(last_error)
    return False


def ensure_db():
    """Downloads the latest synced database from Google Drive if the local copy is
    missing or older than DB_REFRESH_SECONDS. The Drive file itself is kept fresh by
    tommy's existing blue_ballers_sync.py/.ipynb notebook — this just means the
    deployed app periodically catches up on its own, no manual re-upload step. Falls
    back to a stale local copy (with a visible warning) rather than hard-crashing the
    whole app if Drive is temporarily unreachable/rate-limited — only a truly cold
    start with no local copy AND a failed download is fatal."""
    have_local_copy = os.path.exists(DB_PATH)
    stale = not have_local_copy or (time.time() - os.path.getmtime(DB_PATH)) > DB_REFRESH_SECONDS
    if not stale:
        return
    if "drive_file_id" not in st.secrets:
        st.error(
            "Missing `drive_file_id` in Streamlit secrets — set it to the Google Drive "
            "file ID of blue_ballers.db (Share > Anyone with the link > copy the ID out "
            "of the URL) in the app's Settings > Secrets."
        )
        st.stop()
    if download_db():
        return
    if have_local_copy:
        st.warning(
            "Couldn't refresh data from Google Drive just now (it may be temporarily "
            "rate-limited) — showing the last successfully downloaded copy instead."
        )
    else:
        st.error(f"Couldn't download the database from Google Drive: {st.session_state.get('db_download_error')}")
        st.stop()


ensure_db()


@st.cache_data(ttl=300)
def load_table(query, params=None):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def escape_markdown(text):
    """Escape markdown special characters in a user-supplied display string (team
    name, manager name) before embedding it in an f-string passed to st.write/
    st.markdown. Some real Sleeper team names contain literal '*' as a personal
    styling flourish, which collides with this app's own **bold**/:color[] markdown
    wrapping and can garble the whole line — including swallowing an adjacent
    colored arrow span entirely. Only needed for markdown-rendering calls; leave
    raw for st.metric/st.dataframe/st.selectbox, which don't parse markdown and
    would otherwise show literal backslashes."""
    if text is None:
        return text
    # .strip() first: a name with trailing whitespace (real Sleeper data seen in this
    # league) puts a space right before the closing ** when the caller bold-wraps it,
    # which breaks CommonMark's rule that a closing delimiter can't be preceded by
    # whitespace — the bold silently fails to parse and shows literal asterisks.
    return str(text).strip().replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")


@st.cache_data(ttl=1800)
def get_manager_name_lookup():
    """Single source of truth for user_id -> display_name, used everywhere instead
    of ad-hoc per-callsite queries. If a manager renames their team and the
    `managers` table ends up with more than one row per user_id, different parts
    of the app picking first-vs-last row inconsistently is exactly the
    "names keep switching" symptom reported — dict(zip(...)) always keeps the
    LAST row per key (the most recently synced name, since rows are inserted in
    sync order), so every caller of this function agrees."""
    managers_df = load_table("SELECT user_id, display_name FROM managers")
    return dict(zip(managers_df["user_id"], managers_df["display_name"]))


def metric_block(container, label, value, delta=None):
    """Like st.metric, but for variable-length text (player/team names) that
    st.metric would otherwise truncate with an ellipsis in the UI — st.write
    wraps instead of clipping. Both label and value are escaped since either
    position might hold a real team/player name (some contain literal '*')."""
    container.write(f"**{escape_markdown(label)}**")
    container.write(escape_markdown(str(value)))
    if delta:
        container.caption(delta)


def ensure_rivalry_submissions_table():
    """Manager-declared 'biggest rivalry' picks — dashboard-owned data, not synced
    from Sleeper, so it lives in its own small table rather than the sync schema."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manager_rivalries (
                owner_id TEXT PRIMARY KEY,
                rival_owner_id TEXT NOT NULL,
                note TEXT,
                submitted_at TEXT
            )
        """)
        conn.commit()


def submit_rivalry(owner_id, rival_owner_id, note):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO manager_rivalries (owner_id, rival_owner_id, note, submitted_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(owner_id) DO UPDATE SET
                rival_owner_id = excluded.rival_owner_id,
                note = excluded.note,
                submitted_at = excluded.submitted_at
            """,
            (owner_id, rival_owner_id, note, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    load_table.clear()


# Declared rivalries tommy relayed directly, seeded once so they show up before anyone's
# used the submission form. (display_name, display_name, note) — INSERT OR IGNORE means
# this never overwrites a real submission from that manager, only fills the gap if absent.
KNOWN_RIVALRY_DECLARATIONS = [
    ("beastboy9112", "Tweeko69", "Big rivalry this year"),
    ("D00z", "teamsanford", "Always competitive"),
    ("Budman13", "4thand49ers", "Friendly rivalry"),
]


def seed_rivalry_declarations():
    owner_by_name = {name: owner for owner, name in get_manager_name_lookup().items()}
    with sqlite3.connect(DB_PATH) as conn:
        for name1, name2, note in KNOWN_RIVALRY_DECLARATIONS:
            owner1, owner2 = owner_by_name.get(name1), owner_by_name.get(name2)
            if owner1 and owner2:
                conn.execute(
                    "INSERT OR IGNORE INTO manager_rivalries (owner_id, rival_owner_id, note, submitted_at) "
                    "VALUES (?, ?, ?, ?)",
                    (owner1, owner2, note, datetime.now(timezone.utc).isoformat()),
                )
        conn.commit()
    load_table.clear()


@st.cache_data(ttl=3600)
def get_players_df():
    return load_table(
        "SELECT player_id, full_name, position, team, birth_date FROM players"
    ).set_index("player_id")


def safe_zscore(series):
    std = series.std(ddof=0)
    if not std or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def get_seasons():
    return load_table("SELECT league_id, season, synced_at FROM league_seasons ORDER BY season DESC")


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


def compute_expected_win_pct(league_id, through_week=None):
    """Per-roster expected win% this season, from points-rank each week — purges
    matchup-pairing luck (same 'deserved record' concept as the Hall of Fame luck index).
    `through_week` caps the window (e.g. to reconstruct last week's power rankings for
    a week-over-week movement indicator); defaults to the whole regular season."""
    reg_weeks = get_playoff_settings(league_id)["regular_season_weeks"]
    max_week = min(through_week, reg_weeks) if through_week else reg_weeks
    schedule = load_table(
        "SELECT week, roster_id, points FROM matchups WHERE league_id = ? AND week <= ?",
        params=(league_id, max_week),
    )
    expected_sum, games = {}, {}
    for week, week_rows in schedule.groupby("week"):
        if week_rows["points"].sum() <= 0:
            continue
        n = len(week_rows)
        for _, r in week_rows.iterrows():
            rid = int(r["roster_id"])
            better = int((week_rows["points"] < r["points"]).sum())
            expected_sum[rid] = expected_sum.get(rid, 0.0) + (better / (n - 1) if n > 1 else 0.0)
            games[rid] = games.get(rid, 0) + 1
    return {rid: expected_sum[rid] / games[rid] for rid in expected_sum if games.get(rid)}


def compute_power_rankings(league_id, standings, through_week=None):
    """Predictive power score: 0.6x a shrinkage-blended true-strength estimate (the
    same per-team mean the season simulator uses, so a small early-season sample
    can't overweight a hot or cold start) + 0.4x a luck-purged expected win%
    (from points-rank each week, not actual results — actual wins bake in
    matchup-pairing luck, which the Hall of Fame Luck Index already shows is
    real in this league). `through_week` recomputes the ranking as of an earlier
    week (e.g. last week, for a week-over-week movement indicator)."""
    df = standings.copy()
    distributions = estimate_team_distributions(league_id, df, through_week)
    expected_win_pct = compute_expected_win_pct(league_id, through_week)
    fallback_win_pct = df["win_pct"].mean()
    roster_ids_int = df["roster_id"].astype(int)
    df["strength_score"] = roster_ids_int.map(lambda rid: distributions.get(rid, (0.0, 1.0))[0])
    df["expected_win_pct"] = roster_ids_int.map(lambda rid: expected_win_pct.get(rid, fallback_win_pct))
    df["power_score"] = 0.6 * safe_zscore(df["strength_score"]) + 0.4 * safe_zscore(df["expected_win_pct"])
    return df.sort_values("power_score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_power_rankings(league_id, cache_key):
    return compute_power_rankings(league_id, get_standings(league_id))


@st.cache_data(ttl=1800)
def get_previous_week_power_rankings(league_id, cache_key):
    """Power rankings as of last week, for a week-over-week movement indicator.
    None if there isn't a prior completed week yet to compare against."""
    last_week = get_last_completed_week(league_id)
    if last_week is None or last_week <= 1:
        return None
    return compute_power_rankings(league_id, get_standings(league_id), through_week=last_week - 1)


def compute_standings_through_week(league_id, standings, through_week):
    """Reconstructs wins/losses/ties/fpts as of a given week, for a true historical
    Power Rankings snapshot. The live `standings` table only ever holds the
    CURRENT/final season totals — reusing it as-is for a past-week view would show
    today's record and points total next to a past week's power score, which reads
    as internally inconsistent (and throws off the luck-index blurb, which compares
    actual wins against the through-week expected win%)."""
    roster_ids = [int(rid) for rid in standings["roster_id"]]
    record = compute_record_through_week(league_id, roster_ids, through_week)
    schedule = get_full_schedule(league_id, through_week)
    real_weeks = schedule.groupby("week")["points"].sum()
    real_weeks = real_weeks[real_weeks > 0].index
    fpts_through = schedule[schedule["week"].isin(real_weeks)].groupby("roster_id")["points"].sum()

    snap = standings.copy()
    rid_int = snap["roster_id"].astype(int)
    snap["wins"] = rid_int.map(lambda rid: record.get(rid, {}).get("w", 0))
    snap["losses"] = rid_int.map(lambda rid: record.get(rid, {}).get("l", 0))
    snap["ties"] = rid_int.map(lambda rid: record.get(rid, {}).get("t", 0))
    snap["fpts"] = rid_int.map(lambda rid: fpts_through.get(rid, 0.0))
    games = snap["wins"] + snap["losses"] + snap["ties"]
    snap["win_pct"] = snap["wins"] / games.replace(0, 1)
    return snap


@st.cache_data(ttl=1800)
def get_power_rankings_as_of(league_id, through_week, cache_key):
    """Power rankings snapshot as of a specific past week — powers the Home page's
    historical Power Rankings viewer."""
    snapshot_standings = compute_standings_through_week(league_id, get_standings(league_id), through_week)
    return compute_power_rankings(league_id, snapshot_standings, through_week=through_week)


def power_ranking_blurb(row, rank_change):
    """Short NFL.com-style line combining week-over-week movement with whether
    the team's actual record matches what their scoring says it should be —
    deterministic/templated (no LLM integration exists in this app yet)."""
    if rank_change is None:
        movement = "New to the rankings. "
    elif rank_change > 0:
        movement = f"Up {rank_change} from last week. "
    elif rank_change < 0:
        movement = f"Down {abs(rank_change)} from last week. "
    else:
        movement = "Holding steady. "

    games = row["wins"] + row["losses"] + row["ties"]
    expected_wins = row["expected_win_pct"] * games if games else 0.0
    luck_gap = row["wins"] - expected_wins
    if luck_gap > 1:
        form = "winning more than their scoring says they should"
    elif luck_gap < -1:
        form = "still due for better luck than their record shows"
    else:
        form = "playing about as well as their record shows"
    return movement + form + "."


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


def get_week_top_performer(league_id, week, players_df):
    """Highest individual player score across the whole league for one week."""
    df = load_table(
        "SELECT roster_id, players_points FROM matchups WHERE league_id = ? AND week = ?",
        params=(league_id, week),
    )
    best = None
    for _, row in df.iterrows():
        pts_map = parse_json_field(row["players_points"], {})
        for pid, pts in pts_map.items():
            if best is None or pts > best["points"]:
                info = players_df.loc[pid] if pid in players_df.index else None
                best = {
                    "player": info["full_name"] if info is not None else pid,
                    "position": info["position"] if info is not None else "",
                    "points": pts,
                }
    return best


POSITION_LABELS = {"QB": "quarterback", "RB": "running back", "WR": "wide receiver",
                    "TE": "tight end", "DEF": "defense", "K": "kicker"}


def compute_week_positional_averages(league_id, week, players_df):
    """Average points scored by STARTERS (not bench) at each position, across every
    roster in the league for one week — the baseline a CBS-style recap compares
    a standout performance against ('topped the quarterback average by 6.6')."""
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


def describe_matchup(team_a, points_a, team_b, points_b):
    winner, winner_pts = (team_a, points_a) if points_a > points_b else (team_b, points_b)
    loser, loser_pts = (team_b, points_b) if points_a > points_b else (team_a, points_a)
    margin = winner_pts - loser_pts
    if margin < 5:
        flavor = "survived a nailbiter against"
    elif margin < 15:
        flavor = "edged out"
    elif margin < 40:
        flavor = "beat"
    else:
        flavor = "blew out"
    return (f"**{escape_markdown(winner)}** {flavor} **{escape_markdown(loser)}**, "
            f"{winner_pts:.1f}-{loser_pts:.1f}.")


def get_week_lineup(league_id, week, roster_id, players_df):
    """One roster's boxscore for one week: starters (in the slot order Sleeper
    returns them, e.g. QB/RB/RB/WR.../FLEX/DEF/K — not re-sorted, since that
    order reflects the actual starting slots) and bench (sorted by points
    since it has no slot structure)."""
    query = "SELECT starters, players, players_points FROM matchups WHERE league_id = ? AND week = ? AND roster_id = ?"
    df = load_table(query, params=(league_id, week, roster_id))
    if df.empty:
        return pd.DataFrame(columns=["player", "position", "points"]), pd.DataFrame(columns=["player", "position", "points"])

    row = df.iloc[0]
    starter_ids = [pid for pid in parse_json_field(row["starters"], []) if pid != "0"]
    all_ids = parse_json_field(row["players"], [])
    points_map = parse_json_field(row["players_points"], {})

    def resolve(pid):
        info = players_df.loc[pid] if pid in players_df.index else None
        return {
            "player": info["full_name"] if info is not None else pid,
            "position": info["position"] if info is not None else "",
            "points": points_map.get(pid, 0.0),
        }

    starters = pd.DataFrame([resolve(pid) for pid in starter_ids])
    starter_set = set(starter_ids)
    bench_ids = [pid for pid in all_ids if pid not in starter_set]
    bench = pd.DataFrame([resolve(pid) for pid in bench_ids]).sort_values("points", ascending=False) \
        if bench_ids else pd.DataFrame(columns=["player", "position", "points"])
    return starters, bench


def get_recent_transactions(league_id, limit=15):
    query = """
        SELECT transaction_id, week, type, status, adds, drops, roster_ids, created
        FROM transactions
        WHERE league_id = ? AND type IN ('trade', 'waiver', 'free_agent')
        ORDER BY created DESC
        LIMIT ?
    """
    return load_table(query, params=(league_id, limit))


def parse_json_field(value, default):
    """Parse a JSON column, treating SQLAlchemy's JSON-encoded null (the literal
    text "null" — stored whenever the source value was Python None) the same as
    a genuinely empty/missing field instead of returning None."""
    if not value:
        return default
    parsed = json.loads(value)
    return parsed if parsed is not None else default


def format_player_moves(json_field, players_df, team_lookup):
    """Returns (name, team) pairs pre-escaped for markdown — every call site feeds
    these straight into an f-string passed to st.write."""
    ids = parse_json_field(json_field, {})
    if not ids:
        return []
    lines = []
    for player_id, roster_id in ids.items():
        name = players_df.loc[player_id, "full_name"] if player_id in players_df.index else player_id
        team = team_lookup.get(roster_id, f"Roster {roster_id}")
        lines.append((escape_markdown(name), escape_markdown(team)))
    return lines


def get_roster(league_id, roster_id):
    query = "SELECT players, starters FROM rosters WHERE league_id = ? AND roster_id = ?"
    df = load_table(query, params=(league_id, roster_id))
    if df.empty:
        return [], set()
    player_ids = parse_json_field(df.iloc[0]["players"], [])
    starter_ids = set(parse_json_field(df.iloc[0]["starters"], []))
    return player_ids, starter_ids


# ---------------------------------------------------------------------------
# Championship / Playoff odds — Monte Carlo season simulator.
#
# Bracket shape (4-team single elimination) was reverse-engineered from the
# league's actual winners_bracket history, not guessed: round 1 is seed1-vs-
# seed4 and seed2-vs-seed3 decided by a single week's score; round 2 (the
# championship) is decided by the *combined* score across the following two
# weeks. This matches this league's playoff_teams=4 / playoff_round_type=1
# settings for all 3 synced seasons.
# ---------------------------------------------------------------------------
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
    """dict roster_id -> points if this week has actually been played, else None."""
    week_rows = schedule[schedule["week"] == week]
    if week_rows.empty or week_rows["points"].sum() <= 0:
        return None
    scores = dict(zip(week_rows["roster_id"], week_rows["points"]))
    return {rid: scores.get(rid, 0.0) for rid in roster_ids}


def compute_record_through_week(league_id, roster_ids, week):
    """Reconstructs each roster's real win-loss-tie record through a given week by
    replaying actual historical scores under this league's real rules: head-to-head
    result, PLUS a bonus win/loss for beating/missing that week's league median
    (league_average_match=1 in this league — see the Championship Odds simulator's
    design notes for the same rule) — matches Sleeper's own standings math rather
    than just counting head-to-head results."""
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


def find_previous_meeting(league_id, before_week, roster_a, roster_b):
    """An earlier meeting this season between the same two rosters, if any — powers
    the 'revenge game' angle real recaps call out."""
    schedule = get_full_schedule(league_id, before_week - 1)
    for wk, week_rows in schedule.groupby("week"):
        for _, pair in week_rows.groupby("matchup_id"):
            if len(pair) != 2:
                continue
            ids = set(int(x) for x in pair["roster_id"])
            if ids == {roster_a, roster_b}:
                a_row = pair[pair["roster_id"].astype(int) == roster_a].iloc[0]
                b_row = pair[pair["roster_id"].astype(int) == roster_b].iloc[0]
                return {"week": int(wk), "points_a": a_row["points"], "points_b": b_row["points"]}
    return None


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


def estimate_team_distributions(league_id, standings, through_week=None):
    """Per-roster (mean, std) for a team's weekly score, blending this season's
    own results with last season's average as a prior (more prior weight early
    in the season, less as more of this season's games are in the books).
    The mean is recency-weighted (a team's last few weeks count more than its
    week 1) so the projection reflects current form, not just a flat season
    average. The std is shrunk toward a prior the same way the mean is —
    otherwise a team with only 3-4 games gets its raw, noisy sample std used
    directly, which produces unstable, inconsistently-scaled variance
    estimates team-to-team early in the season.
    `through_week` caps the window, e.g. to reconstruct last week's power
    rankings for a week-over-week movement indicator."""
    prev_league_id = get_previous_league_id(league_id)
    week_filter = " AND week <= ?" if through_week else ""
    week_params = (through_week,) if through_week else ()
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


FLEX_ELIGIBLE = {"FLEX": {"RB", "WR", "TE"}, "SUPER_FLEX": {"QB", "RB", "WR", "TE"}}


def get_roster_positions(league_id):
    df = load_table("SELECT roster_positions FROM league_seasons WHERE league_id = ?", params=(league_id,))
    return parse_json_field(df.iloc[0]["roster_positions"], []) if not df.empty else []


def compute_optimal_lineup_score(roster_positions, entries):
    """Best possible lineup score for one team-week: greedily fills the most
    position-restrictive slots first (QB/RB/WR/TE/K), then FLEX (RB/WR/TE),
    then SUPER_FLEX (any offensive position) last. Because FLEX's eligible
    set is a subset of SUPER_FLEX's, this restrictive-first order is
    provably optimal here, not just a heuristic."""
    starting_slots = [s for s in roster_positions if s != "BN"]
    slot_order = sorted(starting_slots, key=lambda s: {"FLEX": 1, "SUPER_FLEX": 2}.get(s, 0))

    available = sorted(entries, key=lambda e: -e[0])
    used = [False] * len(available)
    total = 0.0

    for slot in slot_order:
        eligible = FLEX_ELIGIBLE.get(slot, {slot})
        for i, (pts, pos) in enumerate(available):
            if not used[i] and pos in eligible:
                used[i] = True
                total += pts
                break
    return total


def find_best_bench_swap(roster_positions, starters, bench):
    """The single highest-value bench-for-starter swap available this week — mirrors
    the 'if Coach had started X instead of Y' angle a real recap calls out. Only
    considers one swap at a time (not a full lineup reshuffle), matching that
    narrative's granularity. `starters` must be in Sleeper's real slot order (as
    returned by get_week_lineup) so it lines up with `roster_positions`."""
    starting_slots = [s for s in roster_positions if s != "BN"]
    best = None
    for slot, (_, starter) in zip(starting_slots, starters.iterrows()):
        eligible = FLEX_ELIGIBLE.get(slot, {slot})
        for _, b in bench.iterrows():
            if b["position"] not in eligible:
                continue
            gain = b["points"] - starter["points"]
            if gain > 0 and (best is None or gain > best["gain"]):
                best = {
                    "out_player": starter["player"], "out_points": starter["points"],
                    "in_player": b["player"], "in_points": b["points"], "gain": gain,
                }
    return best


def build_matchup_recap(league_id, week, team_a, team_b, players_df):
    """CBS-style narrative recap for one matchup: result + record + revenge-game
    context, a standout performer vs. their position's weekly average for each
    team, and a lineup-management angle (perfect lineup, or the single best swap
    left on the bench) — modeled on a real CBS Fantasy 'Around the League' recap
    email tommy sent as a reference. Returns a list of markdown-ready lines."""
    roster_a, roster_b = int(team_a["roster_id"]), int(team_b["roster_id"])
    pts_a, pts_b = team_a["points"], team_b["points"]
    winner, loser = (team_a, team_b) if pts_a > pts_b else (team_b, team_a)

    lines = [describe_matchup(team_a["team_name"], pts_a, team_b["team_name"], pts_b)]

    records = compute_record_through_week(league_id, [roster_a, roster_b], week)
    rec_a, rec_b = records[roster_a], records[roster_b]
    rec_str = lambda r: f"{r['w']}-{r['l']}" + (f"-{r['t']}" if r["t"] else "")
    lines.append(
        f"{escape_markdown(team_a['team_name'])} are now {rec_str(rec_a)}; "
        f"{escape_markdown(team_b['team_name'])} are {rec_str(rec_b)}."
    )

    rematch = find_previous_meeting(league_id, week, roster_a, roster_b)
    if rematch:
        prev_winner = team_a["team_name"] if rematch["points_a"] > rematch["points_b"] else team_b["team_name"]
        hi, lo = sorted([rematch["points_a"], rematch["points_b"]], reverse=True)
        lines.append(
            f"This is a rematch of Week {rematch['week']}, when **{escape_markdown(prev_winner)}** won "
            f"{hi:.1f}-{lo:.1f}."
        )

    pos_avg = compute_week_positional_averages(league_id, week, players_df)
    for team in (team_a, team_b):
        starters, _ = get_week_lineup(league_id, week, int(team["roster_id"]), players_df)
        if starters.empty:
            continue
        vs_avg = starters["points"] - starters["position"].map(pos_avg).fillna(starters["points"])
        best_idx = vs_avg.idxmax()
        if vs_avg.loc[best_idx] > 0:
            best = starters.loc[best_idx]
            label = POSITION_LABELS.get(best["position"], best["position"])
            lines.append(
                f"**{escape_markdown(best['player'])}** topped the {label} average by "
                f"{vs_avg.loc[best_idx]:.1f} for {escape_markdown(team['team_name'])}."
            )

    roster_positions = get_roster_positions(league_id)
    for team, is_loser in ((winner, False), (loser, True)):
        starters, bench = get_week_lineup(league_id, week, int(team["roster_id"]), players_df)
        if starters.empty:
            continue
        actual_total = starters["points"].sum()
        optimal_total = compute_optimal_lineup_score(
            roster_positions,
            [(r["points"], r["position"]) for _, r in pd.concat([starters, bench]).iterrows()],
        )
        if abs(actual_total - optimal_total) < 0.05:
            lines.append(f"**{escape_markdown(team['team_name'])}** set a perfect lineup this week.")
        elif is_loser:
            swap = find_best_bench_swap(roster_positions, starters, bench)
            if swap:
                hypothetical = team["points"] + swap["gain"]
                outcome = (
                    f"would have won {hypothetical:.1f}-{winner['points']:.1f}"
                    if hypothetical > winner["points"]
                    else f"still would have fallen short, {hypothetical:.1f}-{winner['points']:.1f}"
                )
                lines.append(
                    f"If **{escape_markdown(team['team_name'])}** had started "
                    f"**{escape_markdown(swap['in_player'])}** ({swap['in_points']:.1f}) over "
                    f"**{escape_markdown(swap['out_player'])}** ({swap['out_points']:.1f}), they {outcome}."
                )
    return lines


def compute_actual_optimal_lineup_totals(league_id, roster_ids, reg_weeks, players_df):
    """Sum of each roster's best-possible ('Max PF') weekly lineup score,
    using real player-by-player scoring — only weeks that have actually been
    played contribute (players_points is empty for future weeks), so this
    naturally covers exactly the season-to-date portion."""
    roster_positions = get_roster_positions(league_id)
    schedule_points = load_table(
        "SELECT week, roster_id, players_points FROM matchups WHERE league_id = ? AND week <= ?",
        params=(league_id, reg_weeks),
    )
    totals = {rid: 0.0 for rid in roster_ids}
    for _, row in schedule_points.iterrows():
        pts_map = parse_json_field(row["players_points"], {})
        if not pts_map:
            continue
        entries = [(pts, players_df.loc[pid, "position"]) for pid, pts in pts_map.items()
                   if pid in players_df.index and players_df.loc[pid, "position"]]
        rid = int(row["roster_id"])
        if rid in totals:
            totals[rid] += compute_optimal_lineup_score(roster_positions, entries)
    return totals


@st.cache_data(ttl=1800)
def get_max_pf(league_id, cache_key):
    reg_weeks = get_playoff_settings(league_id)["regular_season_weeks"]
    roster_ids = [int(rid) for rid in get_standings(league_id)["roster_id"].tolist()]
    return compute_actual_optimal_lineup_totals(league_id, roster_ids, reg_weeks, get_players_df())


def sample_weekly_scores(rng, means, stds, size):
    """Right-skewed, non-negative weekly score draws matched to (mean, std) via
    method-of-moments Gamma parameters. A clipped Normal is symmetric and piles
    artificial mass at/near 0 for high-variance teams; real weekly fantasy
    scores are right-skewed with a floor near 0 (occasional blowout weeks, no
    negative scores) — Gamma matches that shape without an arbitrary clip."""
    safe_means = np.maximum(means, 1.0)  # Gamma needs mean > 0; matches the existing std floor below
    shape = (safe_means / stds) ** 2
    scale = stds ** 2 / safe_means
    return rng.gamma(shape, scale, size=size)


def run_simulation(league_id, standings, distributions, n_trials):
    rng = np.random.default_rng()
    playoff_settings = get_playoff_settings(league_id)
    reg_weeks = playoff_settings["regular_season_weeks"]
    round1_week = playoff_settings["round1_week"]
    round2_weeks = playoff_settings["round2_weeks"]
    league_average_match = playoff_settings["league_average_match"]

    roster_ids = [int(rid) for rid in standings["roster_id"].tolist()]
    idx = {rid: i for i, rid in enumerate(roster_ids)}
    n = len(roster_ids)
    means = np.array([distributions[rid][0] for rid in roster_ids])
    stds = np.array([distributions[rid][1] for rid in roster_ids])

    schedule = get_full_schedule(league_id, max_week=round2_weeks[1])
    pairings_by_week = {w: get_week_pairings(schedule, w) for w in range(1, reg_weeks + 1)}

    def week_scores(week):
        actual = compute_actual_week_scores(schedule, roster_ids, week)
        if actual:
            row = np.array([actual[rid] for rid in roster_ids])
            return np.tile(row, (n_trials, 1))
        return sample_weekly_scores(rng, means, stds, size=(n_trials, n))

    wins = np.zeros((n_trials, n))
    fpts = np.zeros((n_trials, n))
    future_fpts = np.zeros((n_trials, n))  # only unplayed-week contributions — see Max PF below

    for week in range(1, reg_weeks + 1):
        actual = compute_actual_week_scores(schedule, roster_ids, week)
        if actual:
            row = np.array([actual[rid] for rid in roster_ids])
            scores = np.tile(row, (n_trials, 1))
        else:
            scores = sample_weekly_scores(rng, means, stds, size=(n_trials, n))
            future_fpts += scores
        fpts += scores

        for matchup_id, pair in pairings_by_week[week].items():
            if len(pair) == 2:
                a, b = idx[int(pair[0])], idx[int(pair[1])]
                a_wins = scores[:, a] > scores[:, b]
                wins[:, a] += a_wins
                wins[:, b] += ~a_wins

        if league_average_match:
            median = np.median(scores, axis=1, keepdims=True)
            wins += scores > median

    # Final regular-season seeding: most wins, points-for as tiebreaker.
    combined = wins * 1e6 + fpts
    order = np.argsort(-combined, axis=1)
    top_n = playoff_settings["playoff_teams"]
    seed_idx = order[:, :top_n]
    bottom_idx = order[:, top_n:]

    playoff_counts = np.array([np.sum(np.any(seed_idx == i, axis=1)) for i in range(n)])

    # Final-standing rank per roster per trial (1 = best), via inverse permutation of `order`.
    ranks = np.empty((n_trials, n), dtype=int)
    rank_values = np.tile(np.arange(1, n + 1), (n_trials, 1))
    np.put_along_axis(ranks, order, rank_values, axis=1)
    expected_finish = ranks.mean(axis=0)
    finish_distribution = [
        (np.bincount(ranks[:, i], minlength=n + 1)[1:] / n_trials).tolist() for i in range(n)
    ]

    trial_range = np.arange(n_trials)
    s1, s2, s3, s4 = (seed_idx[:, i] for i in range(4))

    r1_scores = week_scores(round1_week)
    m1_winner = np.where(r1_scores[trial_range, s1] > r1_scores[trial_range, s4], s1, s4)
    m1_loser = np.where(m1_winner == s1, s4, s1)
    m2_winner = np.where(r1_scores[trial_range, s2] > r1_scores[trial_range, s3], s2, s3)
    m2_loser = np.where(m2_winner == s2, s3, s2)

    r2a = week_scores(round2_weeks[0])
    r2b = week_scores(round2_weeks[1])
    m1_total = r2a[trial_range, m1_winner] + r2b[trial_range, m1_winner]
    m2_total = r2a[trial_range, m2_winner] + r2b[trial_range, m2_winner]
    champion_idx = np.where(m1_total > m2_total, m1_winner, m2_winner)
    runner_up = np.where(champion_idx == m1_winner, m2_winner, m1_winner)

    # 3rd-place game: the two semifinal losers, same combined-two-week format as the championship.
    m1_loser_total = r2a[trial_range, m1_loser] + r2b[trial_range, m1_loser]
    m2_loser_total = r2a[trial_range, m2_loser] + r2b[trial_range, m2_loser]
    third_place = np.where(m1_loser_total > m2_loser_total, m1_loser, m2_loser)
    fourth_place = np.where(m1_loser_total > m2_loser_total, m2_loser, m1_loser)

    champion_counts = np.array([np.sum(champion_idx == i) for i in range(n)])

    # Rookie draft order for THIS league: bottom 4 (non-playoff) teams pick 1-4, ordered by
    # "Max PF" (best-ball / optimal-lineup points, not actual scoring — so benching studs to
    # tank doesn't work) with the LOWEST Max PF picking 1st. Top 4 (playoff) teams pick 5-8 in
    # straight reverse finish order: 4th place picks 5th, champion picks 8th (last).
    real_optimal_totals = compute_actual_optimal_lineup_totals(league_id, roster_ids, reg_weeks, get_players_df())
    real_optimal_array = np.array([real_optimal_totals.get(rid, 0.0) for rid in roster_ids])
    max_pf_proxy = real_optimal_array[None, :] + future_fpts

    bottom_pf = np.take_along_axis(max_pf_proxy, bottom_idx, axis=1)
    bottom_order_within = np.argsort(bottom_pf, axis=1)  # ascending: lowest Max PF first
    bottom_slots = np.take_along_axis(bottom_idx, bottom_order_within, axis=1)

    draft_slot = np.zeros((n_trials, n), dtype=int)
    for slot_pos in range(bottom_slots.shape[1]):
        draft_slot[trial_range, bottom_slots[:, slot_pos]] = slot_pos + 1
    draft_slot[trial_range, fourth_place] = top_n + 1
    draft_slot[trial_range, third_place] = top_n + 2
    draft_slot[trial_range, runner_up] = top_n + 3
    draft_slot[trial_range, champion_idx] = top_n + 4

    expected_draft_slot = draft_slot.mean(axis=0)
    draft_slot_distribution = [
        (np.bincount(draft_slot[:, i], minlength=n + 1)[1:] / n_trials).tolist() for i in range(n)
    ]

    return {
        roster_ids[i]: {
            "playoff_odds": playoff_counts[i] / n_trials,
            "championship_odds": champion_counts[i] / n_trials,
            "expected_finish": float(expected_finish[i]),
            "finish_distribution": finish_distribution[i],
            "expected_draft_slot": float(expected_draft_slot[i]),
            "draft_slot_distribution": draft_slot_distribution[i],
        }
        for i in range(n)
    }


@st.cache_data(ttl=1800)
def simulate_odds(league_id, cache_key, n_trials=N_TRIALS):
    standings_local = get_standings(league_id)
    distributions = estimate_team_distributions(league_id, standings_local)
    return run_simulation(league_id, standings_local, distributions, n_trials)


# ---------------------------------------------------------------------------
# Team Pages — heuristic player values, team grades, age curve, draft capital,
# and full transaction history for one team.
# ---------------------------------------------------------------------------
def age_curve_multiplier(position, age):
    peak = POSITION_PEAK_AGE.get(position, 27)
    decline = POSITION_DECLINE_RATE.get(position, 0.08)
    if age is None:
        return 0.8
    if age <= peak:
        return max(0.85, 1 - 0.02 * (peak - age))
    return max(0.05, 1 - decline * (age - peak))


def compute_age(birth_date, as_of_year):
    if not birth_date:
        return None
    try:
        return as_of_year - int(str(birth_date)[:4])
    except ValueError:
        return None


def get_all_rostered_player_ids(league_id):
    df = load_table("SELECT players FROM rosters WHERE league_id = ?", params=(league_id,))
    ids = set()
    for blob in df["players"]:
        ids.update(parse_json_field(blob, []))
    return ids


def get_player_points_map(league_id):
    """Recency-weighted PPG per player — mirrors RECENCY_DECAY, the same per-week
    decay `estimate_team_distributions` already uses for team strength, so a
    player who was hot for a few weeks early in the season and has since been
    cut/benched/declined doesn't still show a high value off just that early
    production (the reported Ayomanor case). Only counts real (played) weeks,
    via the same 'SUM(points) > 0 that week' signal `get_last_completed_week`
    uses — Sleeper pre-populates the WHOLE season's matchup rows upfront with
    every future week's players_points zeroed out, so a per-row truthy-dict
    check doesn't catch it (the dict itself isn't empty, just all zeros)."""
    if not league_id:
        return {}
    real_weeks_df = load_table(
        "SELECT week FROM matchups WHERE league_id = ? GROUP BY week HAVING SUM(points) > 0",
        params=(league_id,),
    )
    real_weeks = set(real_weeks_df["week"].tolist())
    if not real_weeks:
        return {}
    max_week = max(real_weeks)

    df = load_table(
        "SELECT week, players_points FROM matchups WHERE league_id = ? AND players_points IS NOT NULL",
        params=(league_id,),
    )
    totals, weight_sums = {}, {}
    for _, row in df.iterrows():
        if row["week"] not in real_weeks:
            continue
        pts_map = parse_json_field(row["players_points"], {})
        if not pts_map:
            continue
        decay = RECENCY_DECAY ** (max_week - row["week"])
        for pid, pts in pts_map.items():
            totals[pid] = totals.get(pid, 0.0) + pts * decay
            weight_sums[pid] = weight_sums.get(pid, 0.0) + decay
    return {pid: totals[pid] / weight_sums[pid] for pid in totals}


def get_blended_player_ppg(league_id):
    prior_league_id = get_previous_league_id(league_id)
    current = get_player_points_map(league_id)
    prior = get_player_points_map(prior_league_id) if prior_league_id else {}
    blended = {}
    for pid in set(current) | set(prior):
        cur_val, prior_val = current.get(pid), prior.get(pid)
        if cur_val is not None and prior_val is not None:
            blended[pid] = 0.7 * cur_val + 0.3 * prior_val
        else:
            blended[pid] = cur_val if cur_val is not None else prior_val
    return blended


def compute_performance_multipliers(ppg_map, players_df):
    pos_groups = {}
    for pid, ppg in ppg_map.items():
        if pid not in players_df.index:
            continue
        pos = players_df.loc[pid, "position"]
        if pos:
            pos_groups.setdefault(pos, []).append((pid, ppg))

    multipliers = {}
    for entries in pos_groups.values():
        values = np.array([e[1] for e in entries])
        if len(values) < 2 or values.std() == 0:
            for pid, _ in entries:
                multipliers[pid] = 1.0
            continue
        for pid, ppg in entries:
            z = (ppg - values.mean()) / values.std()
            multipliers[pid] = float(np.clip(1.0 + 0.25 * z, 0.5, 1.6))
    return multipliers


def compute_player_value(position, age, ppg_multiplier):
    baseline = POSITION_BASELINE.get(position, 0.5)
    return round(100 * baseline * age_curve_multiplier(position, age) * ppg_multiplier, 1)


def build_value_table(league_id, season_year, players_df):
    rostered_ids = get_all_rostered_player_ids(league_id)
    relevant = players_df.loc[players_df.index.intersection(rostered_ids)]
    ppg_map = get_blended_player_ppg(league_id)
    ppg_map = {pid: ppg for pid, ppg in ppg_map.items() if pid in relevant.index}
    multipliers = compute_performance_multipliers(ppg_map, relevant)

    rows = []
    for pid, info in relevant.iterrows():
        age = compute_age(info["birth_date"], season_year)
        value = compute_player_value(info["position"], age, multipliers.get(pid, 1.0))
        rows.append({"player_id": pid, "full_name": info["full_name"], "position": info["position"],
                      "age": age, "value": value})
    return pd.DataFrame(rows).set_index("player_id")


@st.cache_data(ttl=1800)
def get_value_table(league_id, season_year, cache_key):
    return build_value_table(league_id, season_year, get_players_df())


def get_career_player_ppg():
    """PPG blended across every synced season (not just current+prior, like
    get_blended_player_ppg) — used only for historical grading, where we want a
    player's real production over their whole time in this league, not a
    recency-weighted snapshot. Only counts real (played) weeks, via the same
    'SUM(points) > 0 that week' signal get_last_completed_week/get_player_points_map
    use — Sleeper pre-populates the WHOLE season's matchup rows upfront with every
    future week's players_points zeroed out, so a per-row truthy-dict check doesn't
    catch it (the dict itself isn't empty, just all zeros). Without this filter, the
    currently in-progress season's future zero-weeks would dilute every rostered
    player's career PPG for as long as that season stays in progress."""
    league_ids = load_table("SELECT league_id FROM league_seasons")["league_id"].tolist()
    totals, counts = {}, {}
    for lid in league_ids:
        real_weeks_df = load_table(
            "SELECT week FROM matchups WHERE league_id = ? GROUP BY week HAVING SUM(points) > 0",
            params=(lid,),
        )
        real_weeks = set(real_weeks_df["week"].tolist())
        if not real_weeks:
            continue
        df = load_table(
            "SELECT week, players_points FROM matchups WHERE league_id = ? AND players_points IS NOT NULL",
            params=(lid,),
        )
        for _, row in df.iterrows():
            if row["week"] not in real_weeks:
                continue
            pts_map = parse_json_field(row["players_points"], {})
            if not pts_map:
                continue
            for pid, pts in pts_map.items():
                totals[pid] = totals.get(pid, 0.0) + pts
                counts[pid] = counts.get(pid, 0) + 1
    return {pid: totals[pid] / counts[pid] for pid in totals}


def build_historical_value_table(season_year, players_df):
    """Value table for grading HISTORICAL picks/trades — spans every player ever
    rostered in this league, not just currently-rostered ones, valued off real
    career-wide PPG. The 'current' value table above stays roster-scoped on
    purpose for market-value use cases (Team Pages, live trade value); this one
    exists so a since-dropped player (e.g. cut before breaking out elsewhere)
    doesn't get graded as a flat 0 just because nobody currently owns them."""
    all_ids = get_all_ever_rostered_ids()
    relevant = players_df.loc[players_df.index.intersection(all_ids)]
    ppg_map = get_career_player_ppg()
    ppg_map = {pid: ppg for pid, ppg in ppg_map.items() if pid in relevant.index}
    multipliers = compute_performance_multipliers(ppg_map, relevant)

    rows = []
    for pid, info in relevant.iterrows():
        age = compute_age(info["birth_date"], season_year)
        value = compute_player_value(info["position"], age, multipliers.get(pid, 1.0))
        rows.append({"player_id": pid, "full_name": info["full_name"], "position": info["position"],
                      "age": age, "value": value})
    return pd.DataFrame(rows).set_index("player_id")


@st.cache_data(ttl=1800)
def get_historical_value_table(latest_season_year, cache_key):
    return build_historical_value_table(latest_season_year, get_players_df())


# ---------------------------------------------------------------------------
# Stock Market — every ever-rostered player's value as a real time series
# (recomputed at every past week using only PPG accumulated up to that point),
# not just a single end-of-season snapshot. Reuses the exact same value formula
# as everywhere else in the app (position baseline x age curve x performance
# multiplier); the only new piece is computing PPG point-in-time instead of a
# full-season/blended average. Cumulative PPG resets at each new season — a
# fresh season's role/opportunity is a real reset, not a continuation of last
# year's, and this matches how build_value_table already treats seasons.
# ---------------------------------------------------------------------------
def build_player_ppg_timeline():
    """Cumulative PPG per player at every real (season, week) point. A week only
    counts once its matchups have real scoring (SUM(points) > 0), the same bar
    get_last_completed_week uses elsewhere — Sleeper pre-populates future weeks'
    matchup rows with each roster's player list and all-zero players_points
    before real games happen, and without this gate those zero-filled weeks
    would masquerade as real (if quiet) games and drag every player's PPG down."""
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    timeline = []
    for _, row in all_seasons.iterrows():
        lid, season = row["league_id"], row["season"]
        reg_weeks = get_playoff_settings(lid)["regular_season_weeks"]
        matchups = load_table(
            "SELECT week, points, players_points FROM matchups WHERE league_id = ? AND week <= ? "
            "AND players_points IS NOT NULL",
            params=(lid, reg_weeks),
        )
        totals, counts = {}, {}
        for week in range(1, reg_weeks + 1):
            week_rows = matchups[matchups["week"] == week]
            if week_rows.empty or week_rows["points"].sum() <= 0:
                continue
            for blob in week_rows["players_points"]:
                pts_map = parse_json_field(blob, {})
                if not pts_map:
                    continue
                for pid, pts in pts_map.items():
                    totals[pid] = totals.get(pid, 0.0) + pts
                    counts[pid] = counts.get(pid, 0) + 1
            if not totals:
                continue
            for pid in totals:
                timeline.append({
                    "season": season, "week": week, "player_id": pid,
                    "cum_ppg": totals[pid] / counts[pid],
                })
    return pd.DataFrame(timeline)


@st.cache_data(ttl=1800)
def get_player_ppg_timeline(cache_key):
    return build_player_ppg_timeline()


def build_stock_market_history(players_df):
    ppg_timeline = build_player_ppg_timeline()
    if ppg_timeline.empty:
        return pd.DataFrame(columns=["season", "week", "player_id", "value"])

    all_ids = get_all_ever_rostered_ids()
    relevant = players_df.loc[players_df.index.intersection(all_ids)]

    rows = []
    for (season, week), group in ppg_timeline.groupby(["season", "week"]):
        ppg_map = {pid: ppg for pid, ppg in zip(group["player_id"], group["cum_ppg"])
                   if pid in relevant.index}
        multipliers = compute_performance_multipliers(ppg_map, relevant)
        for pid, ppg in ppg_map.items():
            info = relevant.loc[pid]
            age = compute_age(info["birth_date"], int(season))
            value = compute_player_value(info["position"], age, multipliers.get(pid, 1.0))
            rows.append({"season": season, "week": week, "player_id": pid, "value": value})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def get_stock_market_history(cache_key):
    return build_stock_market_history(get_players_df())


def team_overview_metrics(value_table, player_ids, starter_ids):
    team_values = value_table.loc[value_table.index.intersection(player_ids)]
    starters = team_values.loc[team_values.index.intersection(starter_ids)]
    bench = team_values.drop(starters.index, errors="ignore")

    youth_bonus = team_values["value"] * (1 + (27 - team_values["age"].fillna(27)).clip(lower=0) * 0.03)

    return {
        "dynasty_score": team_values["value"].sum(),
        "contender_score": starters["value"].sum(),
        "future_score": youth_bonus.sum(),
        "avg_age": team_values["age"].mean(),
        "starter_value": starters["value"].sum(),
        "bench_value": bench["value"].sum(),
    }


def to_grade(percentile_series):
    bins = [0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.01]
    labels = ["F", "D", "C", "C+", "B", "A"]
    return pd.cut(percentile_series, bins=bins, labels=labels, include_lowest=True)


GRADE_COLORS = {"F": "#DC2626", "D": "#EA580C", "C": "#F59E0B", "C+": "#CA8A04", "B": "#65A30D", "A": "#16A34A"}
GRADE_ORDER = {"F": 0, "D": 1, "C": 2, "C+": 3, "B": 4, "A": 5}


def dynasty_recommendation(state):
    """Actionable read on team-building strategy from the same contend/rebuild/balanced
    signal Trade Center's Context Fit uses — ties the timeline read to a concrete next
    step instead of leaving it as just a date range."""
    if state == "contend":
        return ("This is your window — consider trading future draft capital for proven "
                "win-now upgrades rather than sitting on picks you won't need for years.")
    if state == "rebuild":
        return ("Rebuilding — prioritize acquiring youth and future picks. Selling aging "
                "win-now assets now, while they still have value, usually beats riding them "
                "into a lost season.")
    return ("No urgent timeline pressure — keep building depth and let the roster develop "
            "naturally rather than forcing a move either direction.")


def style_grades(df, columns):
    """Color-code letter-grade columns (status palette: red->green) on a dataframe for display."""

    def _color(v):
        color = GRADE_COLORS.get(str(v))
        return f"background-color: {color}; color: white;" if color else ""

    return df.style.applymap(_color, subset=columns)


def build_league_grades(league_id, value_table, standings):
    records = []
    for _, row in standings.iterrows():
        roster_id = int(row["roster_id"])
        player_ids, starter_ids = get_roster(league_id, roster_id)
        metrics = team_overview_metrics(value_table, player_ids, starter_ids)
        metrics["roster_id"] = roster_id
        records.append(metrics)
    league_df = pd.DataFrame(records).set_index("roster_id")

    league_df["contender_pct"] = league_df["contender_score"].rank(pct=True)
    league_df["future_pct"] = league_df["future_score"].rank(pct=True)

    composite = (
        league_df["dynasty_score"].rank(pct=True)
        + league_df["contender_pct"]
        + league_df["future_pct"]
    ) / 3
    league_df["overall_grade"] = to_grade(composite)
    league_df["starter_grade"] = to_grade(league_df["starter_value"].rank(pct=True))
    league_df["bench_grade"] = to_grade(league_df["bench_value"].rank(pct=True))
    return league_df


@st.cache_data(ttl=1800)
def get_league_grades(league_id, season_year, cache_key):
    value_table = get_value_table(league_id, season_year, cache_key)
    standings_local = get_standings(league_id)
    return build_league_grades(league_id, value_table, standings_local)


def build_positional_grades(league_id, value_table, standings):
    rows = []
    for _, row in standings.iterrows():
        roster_id = int(row["roster_id"])
        player_ids, starter_ids = get_roster(league_id, roster_id)
        team_values = value_table.loc[value_table.index.intersection(player_ids)]
        for pos in ["QB", "RB", "WR", "TE"]:
            rows.append({"roster_id": roster_id, "position": pos,
                         "value": team_values.loc[team_values["position"] == pos, "value"].sum()})
        bench_ids = set(player_ids) - set(starter_ids)
        rows.append({"roster_id": roster_id, "position": "Bench",
                     "value": team_values.loc[team_values.index.intersection(bench_ids), "value"].sum()})

    df = pd.DataFrame(rows)
    df["grade"] = to_grade(df.groupby("position")["value"].rank(pct=True))
    return df


@st.cache_data(ttl=1800)
def get_positional_grades(league_id, season_year, cache_key):
    value_table = get_value_table(league_id, season_year, cache_key)
    standings_local = get_standings(league_id)
    return build_positional_grades(league_id, value_table, standings_local)


def team_competitive_state(contender_pct, future_pct):
    """contender_pct/future_pct are each team's league-wide percentile rank (0-1) on
    win-now starter value vs. age-adjusted roster value — comparable scales, unlike
    the raw sums (future_score sums the whole roster with an age multiplier on top,
    so it dwarfs contender_score's starters-only total for nearly every team if
    compared directly). Returns "contend", "rebuild", or "balanced"."""
    edge = contender_pct - future_pct
    if edge > 0.25:
        return "contend"
    if edge < -0.25:
        return "rebuild"
    return "balanced"


def estimate_championship_window(contender_pct, future_pct, season_year):
    state = team_competitive_state(contender_pct, future_pct)
    if state == "contend":
        return season_year, season_year + 1
    if state == "rebuild":
        return season_year + 2, season_year + 4
    return season_year, season_year + 2


def project_team_value_curve(league_id, roster_id, season_year, players_df, years=5):
    """Per-POSITION value trajectory, not a single aggregate line — an aggregate
    curve looks qualitatively similar for almost every roster (ages blend into a
    smooth average decline regardless of team), while per-position lines actually
    show what makes each team's aging profile distinct (e.g. a young WR corps
    holding value while an aging RB room craters in 2 years)."""
    player_ids, _ = get_roster(league_id, roster_id)
    relevant = players_df.loc[players_df.index.intersection(player_ids)]
    ppg_map = get_blended_player_ppg(league_id)
    ppg_map = {pid: ppg for pid, ppg in ppg_map.items() if pid in relevant.index}
    multipliers = compute_performance_multipliers(ppg_map, relevant)

    base_ages = {pid: compute_age(info["birth_date"], season_year) for pid, info in relevant.iterrows()}

    trajectory = []
    for offset in range(years + 1):
        year = season_year + offset
        pos_totals = {}
        for pid, info in relevant.iterrows():
            base_age = base_ages[pid]
            if base_age is None:
                continue
            pos = info["position"] or "Other"
            value = compute_player_value(info["position"], base_age + offset, multipliers.get(pid, 1.0))
            pos_totals[pos] = pos_totals.get(pos, 0.0) + value
        for pos, total in pos_totals.items():
            trajectory.append({"year": year, "position": pos, "projected_value": total})
    return pd.DataFrame(trajectory)


def get_draft_rounds(league_id):
    df = load_table("SELECT settings FROM league_seasons WHERE league_id = ?", params=(league_id,))
    settings = parse_json_field(df.iloc[0]["settings"], {}) if not df.empty else {}
    return settings.get("draft_rounds", 4)


def get_traded_picks_map(league_id):
    df = load_table(
        "SELECT season, round, roster_id, owner_id FROM traded_picks WHERE league_id = ?",
        params=(league_id,),
    )
    return {(row["season"], row["round"], row["roster_id"]): row["owner_id"] for _, row in df.iterrows()}


def build_pick_inventory(league_id, season_year, standings, seasons_ahead=3):
    draft_rounds = get_draft_rounds(league_id)
    traded = get_traded_picks_map(league_id)
    roster_ids = [int(rid) for rid in standings["roster_id"].tolist()]

    picks = []
    for offset in range(seasons_ahead):
        pick_season = str(season_year + offset)
        for round_no in range(1, draft_rounds + 1):
            for original_roster in roster_ids:
                owner = traded.get((pick_season, round_no, original_roster), original_roster)
                picks.append({"season": pick_season, "round": round_no,
                              "original_roster_id": original_roster, "owner_roster_id": int(owner)})
    return pd.DataFrame(picks)


@st.cache_data(ttl=1800)
def get_pick_inventory(league_id, season_year, cache_key):
    standings_local = get_standings(league_id)
    return build_pick_inventory(league_id, season_year, standings_local)


PICK_SLOT_DECAY = {1: 0.80, 2: 0.88, 3: 0.94}  # per-slot decay within a round — steeper
# in earlier rounds, where pick 1.01 vs 1.08 is a massive real-world gap; later rounds
# default to PICK_SLOT_DECAY_DEFAULT below, much flatter (late-round dynasty rookie
# picks are all fairly interchangeable low-value assets regardless of exact slot).
PICK_SLOT_DECAY_DEFAULT = 0.97


def pick_slot_value(round_no, slot):
    """Value of one numbered pick within a round (slot 1 = earliest/most valuable —
    this league's worst-record team by Max PF picks first). Decays exponentially per
    slot rather than linearly, and steeper in earlier rounds, mirroring real dynasty
    rookie-pick value charts: 1.01 vs 1.08 is a massive gap, 4.01 vs 4.08 is nearly
    nothing — NOT the same gap just proportionally scaled down."""
    base = ROUND_BASE_VALUE.get(round_no, 6)
    decay = PICK_SLOT_DECAY.get(round_no, PICK_SLOT_DECAY_DEFAULT)
    return base * (decay ** (slot - 1))


def pick_value(round_no, original_roster_power_percentile, slot_distribution=None):
    """Value of a future (undrafted) pick. When a real simulated `slot_distribution`
    is available (this league's immediate upcoming draft, from the Championship Odds
    simulator's `draft_slot_distribution`), values the pick as its expectation across
    that actual distribution — correct under Jensen's inequality for pick_slot_value's
    convex shape, and a real improvement over a single power-percentile point estimate.
    Falls back to an estimated slot from the team's current power percentile for picks
    in seasons too far out to simulate (this is a fixed 8-team league: weakest team ≈
    slot 1, strongest ≈ slot 8)."""
    if slot_distribution:
        return round(
            sum(p * pick_slot_value(round_no, slot) for slot, p in enumerate(slot_distribution, start=1)), 1
        )
    estimated_slot = 1 + (1 - original_roster_power_percentile) * 7
    return round(pick_slot_value(round_no, estimated_slot), 1)


def value_pick_row(row, power_pct, current_season_str, odds):
    """Value one row of a pick inventory table. `current_season_str` is whichever
    season `odds` (the Championship Odds simulator's output) was run for — only a
    pick in THAT season has a real simulated slot distribution to use; anything
    further out falls back to pick_value's percentile-estimated slot."""
    if row["season"] == current_season_str:
        dist = odds.get(int(row["original_roster_id"]), {}).get("draft_slot_distribution")
        if dist:
            return pick_value(row["round"], power_pct.get(row["original_roster_id"], 0.5), slot_distribution=dist)
    return pick_value(row["round"], power_pct.get(row["original_roster_id"], 0.5))


def get_global_team_lookup():
    df = load_table("""
        SELECT r.league_id, r.roster_id, COALESCE(t.team_name, m.display_name) AS team_name
        FROM rosters r
        LEFT JOIN managers m ON m.user_id = r.owner_id
        LEFT JOIN roster_team_names t ON t.league_id = r.league_id AND t.user_id = r.owner_id
    """)
    return {(row["league_id"], row["roster_id"]): row["team_name"] for _, row in df.iterrows()}


def format_moves_global(json_field, players_df, league_id, global_team_lookup):
    """Returns (name, team) pairs pre-escaped for markdown — see format_player_moves."""
    ids = parse_json_field(json_field, {})
    if not ids:
        return []
    lines = []
    for player_id, roster_id in ids.items():
        name = players_df.loc[player_id, "full_name"] if player_id in players_df.index else player_id
        team = global_team_lookup.get((league_id, roster_id), f"Roster {roster_id}")
        lines.append((escape_markdown(name), escape_markdown(team)))
    return lines


def build_team_timeline(owner_id):
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season DESC")
    frames = []
    for _, season_row in all_seasons.iterrows():
        lid = season_row["league_id"]
        roster_row = load_table(
            "SELECT roster_id FROM rosters WHERE league_id = ? AND owner_id = ?", params=(lid, owner_id)
        )
        if roster_row.empty:
            continue
        roster_id = int(roster_row.iloc[0]["roster_id"])
        txns = load_table(
            """
            SELECT transaction_id, week, type, status, adds, drops, roster_ids,
                   draft_picks, waiver_budget, created
            FROM transactions WHERE league_id = ? AND type IN ('trade', 'waiver', 'free_agent')
            ORDER BY created DESC
            """,
            params=(lid,),
        )
        involved = txns[txns["roster_ids"].apply(
            lambda blob: roster_id in parse_json_field(blob, [])
        )].copy()
        involved["season"] = season_row["season"]
        involved["league_id"] = lid
        frames.append(involved)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=1800)
def get_team_timeline(owner_id, cache_key):
    return build_team_timeline(owner_id)


# ---------------------------------------------------------------------------
# Trade Center — grades every trade using the same value heuristic as Team
# Pages, plus a per-player ownership chain ("trade tree").
# ---------------------------------------------------------------------------
def build_all_trades():
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season DESC")
    frames = []
    for _, row in all_seasons.iterrows():
        lid = row["league_id"]
        txns = load_table(
            """
            SELECT transaction_id, week, type, status, adds, drops, roster_ids,
                   draft_picks, waiver_budget, created
            FROM transactions WHERE league_id = ? AND type = 'trade'
            ORDER BY created DESC
            """,
            params=(lid,),
        )
        txns = txns.copy()
        txns["season"] = row["season"]
        txns["league_id"] = lid
        frames.append(txns)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=1800)
def get_all_trades(cache_key):
    return build_all_trades()


@st.cache_data(ttl=1800)
def get_season_power_pct(league_id, cache_key):
    rankings_local = get_power_rankings(league_id, cache_key)
    return dict(zip(rankings_local["roster_id"].astype(int), rankings_local["power_score"].rank(pct=True)))


def future_weighted_value(value, age):
    age = age if pd.notna(age) else 27
    return value * (1 + max(0, 27 - age) * 0.03)


def ratio_to_grade(ratio):
    if ratio >= 1.3:
        return "A"
    if ratio >= 1.1:
        return "B"
    if ratio >= 0.9:
        return "C"
    if ratio >= 0.7:
        return "D"
    return "F"


def grade_trade(txn, value_table, power_pct, global_team_lookup, league_id):
    roster_ids = parse_json_field(txn["roster_ids"], [])
    adds = parse_json_field(txn["adds"], {})
    drops = parse_json_field(txn["drops"], {})
    draft_picks = parse_json_field(txn["draft_picks"], [])
    waiver_budget = parse_json_field(txn["waiver_budget"], [])

    received = {rid: 0.0 for rid in roster_ids}
    given = {rid: 0.0 for rid in roster_ids}
    future_received = {rid: 0.0 for rid in roster_ids}
    future_given = {rid: 0.0 for rid in roster_ids}

    def player_value_and_age(player_id):
        if player_id in value_table.index:
            row = value_table.loc[player_id]
            return row["value"], row["age"]
        return 20.0, None  # unrostered/unknown player — small flat fallback

    for player_id, roster_id in adds.items():
        if roster_id in received:
            val, age = player_value_and_age(player_id)
            received[roster_id] += val
            future_received[roster_id] += future_weighted_value(val, age)

    for player_id, roster_id in drops.items():
        if roster_id in given:
            val, age = player_value_and_age(player_id)
            given[roster_id] += val
            future_given[roster_id] += future_weighted_value(val, age)

    for pick in draft_picks:
        val = pick_value(pick.get("round", 4), power_pct.get(pick.get("roster_id"), 0.5))
        new_owner, prev_owner = pick.get("owner_id"), pick.get("previous_owner_id")
        if new_owner in received:
            received[new_owner] += val
            future_received[new_owner] += val * 1.3  # picks are pure future assets
        if prev_owner in given:
            given[prev_owner] += val
            future_given[prev_owner] += val * 1.3

    for wb in waiver_budget:
        faab_val = wb.get("amount", 0) * 0.3
        receiver, sender = wb.get("receiver"), wb.get("sender")
        if receiver in received:
            received[receiver] += faab_val
        if sender in given:
            given[sender] += faab_val

    results = []
    for rid in roster_ids:
        recv, giv = received[rid], given[rid]
        ratio = recv / giv if giv > 0 else (2.0 if recv > 0 else 1.0)
        results.append({
            "roster_id": rid,
            "team_name": global_team_lookup.get((league_id, rid), f"Roster {rid}"),
            "grade": ratio_to_grade(ratio),
            "win_now_impact": round(recv - giv, 1),
            "future_impact": round(future_received[rid] - future_given[rid], 1),
        })

    if len(results) >= 2:
        results.sort(key=lambda r: r["win_now_impact"], reverse=True)
        if abs(results[0]["win_now_impact"] - results[-1]["win_now_impact"]) < 5:
            for r in results:
                r["role"] = "Even"
        else:
            results[0]["role"] = "Winner"
            results[-1]["role"] = "Loser"
            for r in results[1:-1]:
                r["role"] = "Even"
    else:
        for r in results:
            r["role"] = "—"

    return results


CONTEXT_FIT_THRESHOLD = 10.0  # ignore trivial win-now/future imbalances as noise


def trade_context_fit(state, win_now_impact, future_impact):
    """Whether a trade fits the team's own timeline (e.g. a rebuilding team giving up
    future assets to chase a marginal win-now upgrade doesn't make sense even if the
    raw value is fair) — a ⚠️ result here downgrades the displayed letter grade one
    full step (see downgrade_grade), since a value-fair trade that fights a team's
    own timeline isn't actually the A it looks like on paper."""
    if state == "rebuild" and win_now_impact > CONTEXT_FIT_THRESHOLD and future_impact < -CONTEXT_FIT_THRESHOLD:
        return "⚠️ Win-now move for a rebuilding team"
    if state == "contend" and future_impact > CONTEXT_FIT_THRESHOLD and win_now_impact < -CONTEXT_FIT_THRESHOLD:
        return "⚠️ Future-focused move for a team that should be contending"
    return "✅ Fits team timeline"


TRADE_GRADE_STEPS = ["F", "D", "C", "B", "A"]  # ascending order


def downgrade_grade(grade, steps=1):
    """Step a trade letter grade down (A->B->C->D->F), floored at F."""
    idx = TRADE_GRADE_STEPS.index(grade)
    return TRADE_GRADE_STEPS[max(0, idx - steps)]


def ordinal(n):
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th', etc."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def humanize_list(items):
    """['A'] -> 'A'; ['A','B'] -> 'A and B'; ['A','B','C'] -> 'A, B, and C'."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def deterministic_index(key, n):
    """A stable pseudo-random index in range(n) for `key` — same trade always reads
    the same way on every reload, but different trades land on different phrasing.
    Deliberately NOT Python's hash(): str hashing is randomized per-process
    (PYTHONHASHSEED), which would reshuffle every trade's wording on every Streamlit
    restart."""
    return sum(ord(c) for c in str(key)) % n


TRADE_TEMPLATES = [
    "**{a}** acquired {gains_a} from **{b}** in exchange for {gains_b}.",
    "**{a}** landed {gains_a}, sending {gains_b} to **{b}**.",
    "**{b}** sent {gains_a} to **{a}** for {gains_b}.",
    "**{a}** and **{b}** swapped assets — **{a}** gets {gains_a}, **{b}** gets {gains_b}.",
]


def describe_trade(txn, players_df, league_id, global_team_lookup):
    """Varied, readable trade language ('X acquired Y from Z in exchange for W')
    replacing the flat Added:/Dropped: list — reads naturally because it names both
    sides of the exchange, which only makes sense for a genuine two-party trade.
    Anything else (3+ rosters, or a leg with nothing on one side) falls back to a
    plain per-team receipt list rather than forcing an ill-fitting sentence."""
    adds = parse_json_field(txn["adds"], {})
    picks = parse_json_field(txn["draft_picks"], [])
    faab = parse_json_field(txn["waiver_budget"], [])

    gains = {}

    def add_gain(roster_id, text):
        gains.setdefault(roster_id, []).append(text)

    for player_id, roster_id in adds.items():
        name = players_df.loc[player_id, "full_name"] if player_id in players_df.index else player_id
        add_gain(roster_id, escape_markdown(name))
    for p in picks:
        add_gain(p.get("owner_id"), f"a {p.get('season')} {ordinal(p.get('round'))}")
    for wb in faab:
        add_gain(wb.get("receiver"), f"${wb.get('amount')} FAAB")

    roster_ids = list(gains.keys())
    if len(roster_ids) != 2:
        lines = []
        for rid, items in gains.items():
            team = escape_markdown(global_team_lookup.get((league_id, rid), f"Roster {rid}"))
            lines.append(f"**{team}** received {humanize_list(items)}.")
        return lines

    a, b = roster_ids
    team_a = escape_markdown(global_team_lookup.get((league_id, a), f"Roster {a}"))
    team_b = escape_markdown(global_team_lookup.get((league_id, b), f"Roster {b}"))
    template = TRADE_TEMPLATES[deterministic_index(txn["transaction_id"], len(TRADE_TEMPLATES))]
    return [template.format(a=team_a, b=team_b, gains_a=humanize_list(gains[a]), gains_b=humanize_list(gains[b]))]


def get_all_ever_rostered_ids():
    league_ids = load_table("SELECT league_id FROM league_seasons")["league_id"].tolist()
    ids = set()
    for lid in league_ids:
        ids |= get_all_rostered_player_ids(lid)
    return ids


def build_player_ownership_history(player_id):
    events = []

    draft_rows = load_table(
        """
        SELECT d.season, dp.roster_id, d.league_id
        FROM draft_picks dp JOIN drafts d ON d.draft_id = dp.draft_id
        WHERE dp.player_id = ?
        ORDER BY d.season ASC
        """,
        params=(player_id,),
    )
    for _, row in draft_rows.iterrows():
        events.append({"season": row["season"], "league_id": row["league_id"],
                        "week": None, "type": "draft", "roster_id": row["roster_id"], "order": (row["season"], 0)})

    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    for _, row in all_seasons.iterrows():
        lid = row["league_id"]
        txns = load_table(
            "SELECT week, type, adds, created FROM transactions WHERE league_id = ? ORDER BY created ASC",
            params=(lid,),
        )
        for _, txn in txns.iterrows():
            adds = parse_json_field(txn["adds"], {})
            if player_id in adds:
                events.append({"season": row["season"], "league_id": lid, "week": txn["week"],
                                "type": txn["type"], "roster_id": adds[player_id],
                                "order": (row["season"], txn["created"])})

    events.sort(key=lambda e: e["order"])
    return events


@st.cache_data(ttl=1800)
def get_player_ownership_history(player_id, cache_key):
    return build_player_ownership_history(player_id)


# ---------------------------------------------------------------------------
# Rookie Draft Center — grades every rookie draft (excludes the 2024 startup
# draft; distinguished via Sleeper's own settings.player_type: 0 = startup/
# all players, 1 = rookies only) using the same value heuristic as Team
# Pages, plus a league-wide view of future draft pick value.
# ---------------------------------------------------------------------------
def expected_pick_value(pick_no):
    """Smooth per-pick decay matching ROUND_BASE_VALUE at round boundaries
    (pick 1 = 100, pick 9 = 50, pick 17 = 25, pick 25 = 12.5 for an 8-team league)."""
    return 100 * (0.5 ** ((pick_no - 1) / 8))


def build_rookie_draft_list():
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season DESC")
    rows = []
    for _, row in all_seasons.iterrows():
        lid = row["league_id"]
        drafts_df = load_table("SELECT draft_id, settings FROM drafts WHERE league_id = ?", params=(lid,))
        for _, d in drafts_df.iterrows():
            settings = parse_json_field(d["settings"], {})
            if settings.get("player_type") == 1:
                rows.append({"draft_id": d["draft_id"], "season": row["season"], "league_id": lid})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def get_rookie_draft_list(cache_key):
    return build_rookie_draft_list()


def get_draft_picks_detail(draft_id):
    query = "SELECT pick_no, round, roster_id, player_id FROM draft_picks WHERE draft_id = ? ORDER BY pick_no"
    return load_table(query, params=(draft_id,))


def grade_rookie_draft(draft_id, league_id, career_value_table, global_team_lookup, players_df):
    picks = get_draft_picks_detail(draft_id)
    rows = []
    for _, p in picks.iterrows():
        pid = p["player_id"]
        expected = expected_pick_value(p["pick_no"])
        career_row = career_value_table.loc[pid] if pid in career_value_table.index else None
        career = career_row["value"] if career_row is not None else 0.0
        info = players_df.loc[pid] if pid in players_df.index else None
        roster_id = int(p["roster_id"])
        rows.append({
            "pick_no": int(p["pick_no"]),
            "round": int(p["round"]),
            "roster_id": roster_id,
            "team": global_team_lookup.get((league_id, roster_id), f"Roster {roster_id}"),
            "player": info["full_name"] if info is not None else pid,
            "position": info["position"] if info is not None else "",
            "expected_value": round(expected, 1),
            "career_value": round(career, 1),
            "delta": round(career - expected, 1),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def get_rookie_draft_grades(draft_id, league_id, latest_season_year, cache_key):
    career_value_table = get_historical_value_table(latest_season_year, cache_key)
    return grade_rookie_draft(draft_id, league_id, career_value_table, get_global_team_lookup(), get_players_df())


def summarize_team_grades(draft_board):
    team_deltas = draft_board.groupby("team")["delta"].sum().reset_index()
    team_deltas["grade"] = to_grade(team_deltas["delta"].rank(pct=True))
    return team_deltas.sort_values("delta", ascending=False)


# ---------------------------------------------------------------------------
# GM Profiles — seven ratings per manager, each a 0-100 percentile score
# among the league's managers, built entirely from data/logic already
# established elsewhere in this file (Rookie Draft/Trade grading, Team Page
# metrics, the season simulator's deterministic replay of completed seasons).
# No injury data exists yet, so Risk Taking is an activity/volatility proxy,
# not a true risk measure — same caveat as Team Pages/Trade Center.
# ---------------------------------------------------------------------------
GM_RATING_LABELS = ["Drafting", "Trading", "Waivers", "Roster Construction",
                    "Risk Taking", "Player Development", "Clutch"]


def get_owner_lookup():
    df = load_table("SELECT league_id, roster_id, owner_id FROM rosters")
    return {(row["league_id"], row["roster_id"]): row["owner_id"] for _, row in df.iterrows()}


def grade_waiver_move(txn, value_table):
    """Net value change (added - dropped) per roster for one waiver/free-agent move."""
    roster_ids = parse_json_field(txn["roster_ids"], [])
    adds = parse_json_field(txn["adds"], {})
    drops = parse_json_field(txn["drops"], {})
    net = {rid: 0.0 for rid in roster_ids}

    def value_of(pid):
        return value_table.loc[pid, "value"] if pid in value_table.index else 15.0

    for pid, rid in adds.items():
        if rid in net:
            net[rid] += value_of(pid)
    for pid, rid in drops.items():
        if rid in net:
            net[rid] -= value_of(pid)
    return net


def build_all_waiver_moves():
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season DESC")
    frames = []
    for _, row in all_seasons.iterrows():
        lid = row["league_id"]
        txns = load_table(
            """
            SELECT transaction_id, week, type, adds, drops, roster_ids, created
            FROM transactions WHERE league_id = ? AND type IN ('waiver', 'free_agent')
            """,
            params=(lid,),
        )
        txns = txns.copy()
        txns["season"] = row["season"]
        txns["league_id"] = lid
        frames.append(txns)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def is_season_complete(league_id):
    settings = get_playoff_settings(league_id)
    last_week = get_last_completed_week(league_id)
    return last_week is not None and last_week >= settings["round2_weeks"][1]


PLACEMENT_POINTS = {"champion": 100, "runner_up": 60, "third": 35, "fourth": 15, "missed_playoffs": 0}


def get_completed_season_placements(league_id):
    """Exact champion/runner-up/3rd/4th/missed-playoffs for a fully-completed
    season, read off the existing simulator run with n_trials=1 — exact
    (not approximate) because every week is real data, so there's no
    randomness left for the simulator to resolve."""
    standings_local = get_standings(league_id)
    distributions = estimate_team_distributions(league_id, standings_local)
    result = run_simulation(league_id, standings_local, distributions, n_trials=1)
    top_n = get_playoff_settings(league_id)["playoff_teams"]
    slot_label = {top_n + 4: "champion", top_n + 3: "runner_up", top_n + 2: "third", top_n + 1: "fourth"}
    placements = {}
    for rid, info in result.items():
        dist = info["draft_slot_distribution"]
        slot = dist.index(1.0) + 1 if 1.0 in dist else None
        placements[rid] = slot_label.get(slot, "missed_playoffs")
    return placements


def to_score_100(raw_series):
    if raw_series.empty or raw_series.nunique() <= 1:
        return pd.Series([50] * len(raw_series), index=raw_series.index)
    return (raw_series.rank(pct=True) * 100).round().astype(int)


def build_gm_profiles(latest_league_id, latest_season_year, cache_key):
    owner_lookup = get_owner_lookup()
    global_team_lookup_local = get_global_team_lookup()
    players_df_local = get_players_df()
    manager_name_lookup = get_manager_name_lookup()

    drafting = {}
    dev_hits, dev_total = {}, {}

    for _, d in build_rookie_draft_list().iterrows():
        board = get_rookie_draft_grades(d["draft_id"], d["league_id"], latest_season_year, cache_key)
        for _, pick in board.iterrows():
            owner = owner_lookup.get((d["league_id"], pick["roster_id"]))
            if not owner:
                continue
            drafting[owner] = drafting.get(owner, 0.0) + pick["delta"]
            dev_total[owner] = dev_total.get(owner, 0) + 1
            dev_hits[owner] = dev_hits.get(owner, 0) + (1 if pick["delta"] > 0 else 0)

    # Trading score is a win-rate (Winner=1/Even=0.5/Loser=0 per trade side), not a raw value-swing
    # sum — a raw sum let one or two lopsided trades dominate a manager's whole career score
    # regardless of how often they actually come out ahead. Shrunk toward the league-average rate
    # so a manager with very few trades can't look like the league's best or worst trader off a
    # tiny sample (same shrinkage idea as the power-rankings/simulator prior elsewhere in this file).
    TRADE_ROLE_OUTCOME = {"Winner": 1.0, "Even": 0.5, "Loser": 0.0}
    TRADE_SHRINKAGE = 3  # phantom trades at the league-average rate
    trading_wins, trading_total = {}, {}
    for _, txn in get_all_trades(cache_key).iterrows():
        vt = get_historical_value_table(int(txn["season"]), cache_key)
        pp = get_season_power_pct(txn["league_id"], cache_key)
        for side in grade_trade(txn, vt, pp, global_team_lookup_local, txn["league_id"]):
            owner = owner_lookup.get((txn["league_id"], side["roster_id"]))
            if owner:
                trading_total[owner] = trading_total.get(owner, 0) + 1
                trading_wins[owner] = trading_wins.get(owner, 0.0) + TRADE_ROLE_OUTCOME.get(side["role"], 0.5)

    league_avg_trade_rate = (
        sum(trading_wins.values()) / sum(trading_total.values()) if trading_total else 0.5
    )
    trading = {
        owner: (trading_wins[owner] + TRADE_SHRINKAGE * league_avg_trade_rate)
        / (trading_total[owner] + TRADE_SHRINKAGE)
        for owner in trading_total
    }

    waivers = {}
    for _, txn in build_all_waiver_moves().iterrows():
        vt = get_historical_value_table(int(txn["season"]), cache_key)
        for rid, net in grade_waiver_move(txn, vt).items():
            owner = owner_lookup.get((txn["league_id"], rid))
            if not owner:
                continue
            waivers[owner] = waivers.get(owner, 0.0) + net
            dev_total[owner] = dev_total.get(owner, 0) + 1
            dev_hits[owner] = dev_hits.get(owner, 0) + (1 if net > 0 else 0)

    # Roster Construction: current lineup efficiency (starter value share of total roster value).
    latest_standings = get_standings(latest_league_id)
    latest_value_table = get_value_table(latest_league_id, latest_season_year, cache_key)
    roster_construction = {}
    risk_taking_age = {}
    for _, row in latest_standings.iterrows():
        roster_id = int(row["roster_id"])
        owner = row["owner_id"]
        player_ids, starter_ids = get_roster(latest_league_id, roster_id)
        metrics = team_overview_metrics(latest_value_table, player_ids, starter_ids)
        total = metrics["starter_value"] + metrics["bench_value"]
        roster_construction[owner] = metrics["starter_value"] / total if total else 0.5
        risk_taking_age[owner] = metrics["avg_age"]

    # Risk Taking: transaction frequency (all-time) + how far below the league-average
    # age their roster sits (younger roster = more boom/bust risk in dynasty terms).
    activity_counts = {}
    for _, txn in pd.concat([get_all_trades(cache_key), build_all_waiver_moves()], ignore_index=True).iterrows():
        for rid in parse_json_field(txn["roster_ids"], []):
            owner = owner_lookup.get((txn["league_id"], rid))
            if owner:
                activity_counts[owner] = activity_counts.get(owner, 0) + 1

    # Neilee is Buddy's co-manager (viewing access only, not a real GM) — exclude her.
    all_owners = sorted(
        owner for owner, name in manager_name_lookup.items() if "neilee" not in name.lower()
    )
    league_avg_age = np.nanmean(list(risk_taking_age.values())) if risk_taking_age else 27.0
    risk_raw = {
        owner: activity_counts.get(owner, 0) * 1.0 + max(0.0, league_avg_age - risk_taking_age.get(owner, league_avg_age)) * 5
        for owner in all_owners
    }

    # Clutch: sum of playoff-placement points across every fully-completed season —
    # evaluates purely on playoff finish, not regular-season performance at all.
    championship = {owner: 0 for owner in all_owners}
    for _, row in load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC").iterrows():
        lid = row["league_id"]
        if not is_season_complete(lid):
            continue
        placements = get_completed_season_placements(lid)
        season_standings = get_standings(lid)
        for _, r in season_standings.iterrows():
            owner = r["owner_id"]
            roster_id = int(r["roster_id"])
            label = placements.get(roster_id, "missed_playoffs")
            championship[owner] = championship.get(owner, 0) + PLACEMENT_POINTS[label]

    dev_hit_rate = {owner: (dev_hits.get(owner, 0) / dev_total[owner]) for owner in dev_total if dev_total[owner] > 0}

    profile_rows = []
    for owner in all_owners:
        profile_rows.append({
            "owner_id": owner,
            "manager": manager_name_lookup.get(owner, owner),
            "Drafting": drafting.get(owner, 0.0),
            "Trading": trading.get(owner, 0.0),
            "Waivers": waivers.get(owner, 0.0),
            "Roster Construction": roster_construction.get(owner, 0.5),
            "Risk Taking": risk_raw.get(owner, 0.0),
            "Player Development": dev_hit_rate.get(owner, 0.0),
            "Clutch": championship.get(owner, 0),
        })
    profiles = pd.DataFrame(profile_rows)

    for label in GM_RATING_LABELS:
        profiles[label] = to_score_100(profiles[label])
    profiles["Overall GM"] = profiles[GM_RATING_LABELS].mean(axis=1).round().astype(int)
    return profiles.sort_values("Overall GM", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_gm_profiles(latest_league_id, latest_season_year, cache_key):
    return build_gm_profiles(latest_league_id, latest_season_year, cache_key)


# ---------------------------------------------------------------------------
# Hall of Fame / Hall of Shame — career records and single-event superlatives,
# built by reusing Rookie Draft/Trade grading and the completed-season replay
# already established above. "Most Injury Luck" is skipped — no injury data
# is synced yet, same caveat as Risk Taking/Roster Risk elsewhere.
# ---------------------------------------------------------------------------
def build_regular_season_log():
    """One row per (season, week, roster) played: points, expected win share
    (fraction of the other 7 teams they outscored that week — an all-play-all
    proxy for 'deserved' win probability), and actual head-to-head result."""
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    rows = []
    for _, season_row in all_seasons.iterrows():
        lid = season_row["league_id"]
        reg_weeks = get_playoff_settings(lid)["regular_season_weeks"]
        schedule = load_table(
            "SELECT week, matchup_id, roster_id, points FROM matchups WHERE league_id = ? AND week <= ?",
            params=(lid, reg_weeks),
        )
        roster_owner = load_table("SELECT roster_id, owner_id FROM rosters WHERE league_id = ?", params=(lid,))
        owner_map = dict(zip(roster_owner["roster_id"], roster_owner["owner_id"]))

        for week, week_rows in schedule.groupby("week"):
            if week_rows["points"].sum() <= 0:
                continue
            n = len(week_rows)
            result_map = {}
            for _, group in week_rows.groupby("matchup_id"):
                if len(group) == 2:
                    a, b = group.iloc[0], group.iloc[1]
                    if a["points"] > b["points"]:
                        result_map[a["roster_id"]], result_map[b["roster_id"]] = "W", "L"
                    elif b["points"] > a["points"]:
                        result_map[a["roster_id"]], result_map[b["roster_id"]] = "L", "W"
                    else:
                        result_map[a["roster_id"]] = result_map[b["roster_id"]] = "T"

            for _, r in week_rows.iterrows():
                rid = int(r["roster_id"])
                better = int((week_rows["points"] < r["points"]).sum())
                expected = better / (n - 1) if n > 1 else 0.0
                rows.append({
                    "season": season_row["season"], "week": int(week), "league_id": lid, "roster_id": rid,
                    "owner_id": owner_map.get(rid), "points": r["points"],
                    "expected_win": expected, "result": result_map.get(rid, "T"),
                })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def get_regular_season_log(cache_key):
    return build_regular_season_log()


def compute_streaks(game_log):
    rows = []
    for owner, group in game_log.sort_values(["season", "week"]).groupby("owner_id"):
        best_win = best_loss = cur_win = cur_loss = 0
        for r in group["result"]:
            if r == "W":
                cur_win += 1
                cur_loss = 0
            elif r == "L":
                cur_loss += 1
                cur_win = 0
            else:
                cur_win = cur_loss = 0
            best_win = max(best_win, cur_win)
            best_loss = max(best_loss, cur_loss)
        rows.append({"owner_id": owner, "best_win_streak": best_win, "best_loss_streak": best_loss})
    return pd.DataFrame(rows)


def compute_luck_index(game_log):
    luck = game_log.groupby("owner_id").agg(
        actual_wins=("result", lambda s: (s == "W").sum()),
        expected_wins=("expected_win", "sum"),
    ).reset_index()
    luck["luck_index"] = luck["actual_wins"] - luck["expected_wins"]
    return luck


PLAYOFF_WINS_BY_PLACEMENT = {"champion": 2, "runner_up": 1, "third": 1, "fourth": 0, "missed_playoffs": 0}
PLACEMENT_RANK = {"champion": 1, "runner_up": 2, "third": 3, "fourth": 4}
PLACEMENT_LABEL = {"champion": "1st", "runner_up": "2nd", "third": "3rd", "fourth": "4th",
                    "missed_playoffs": "missed the playoffs"}


def compute_playoff_records():
    """Championships/runner-ups/playoff wins per owner, plus finals/playoff
    appearance counts (used by League Records) and 'choke' candidates:
    playoff exits where a team's seed (regular-season rank) was much better
    than where they ended up finishing."""
    champions, runner_ups, playoff_wins = {}, {}, {}
    finals_appearances, playoff_appearances = {}, {}
    choke_candidates = []

    for _, row in load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC").iterrows():
        lid = row["league_id"]
        if not is_season_complete(lid):
            continue
        placements = get_completed_season_placements(lid)
        season_standings = get_standings(lid)
        seed_rank_map = {int(r["roster_id"]): i + 1 for i, (_, r) in enumerate(season_standings.iterrows())}

        for _, r in season_standings.iterrows():
            roster_id = int(r["roster_id"])
            owner = r["owner_id"]
            label = placements.get(roster_id, "missed_playoffs")

            if label == "champion":
                champions[owner] = champions.get(owner, 0) + 1
            if label == "runner_up":
                runner_ups[owner] = runner_ups.get(owner, 0) + 1
            playoff_wins[owner] = playoff_wins.get(owner, 0) + PLAYOFF_WINS_BY_PLACEMENT[label]

            if label in ("champion", "runner_up"):
                finals_appearances[owner] = finals_appearances.get(owner, 0) + 1
            if label in PLACEMENT_RANK:
                playoff_appearances[owner] = playoff_appearances.get(owner, 0) + 1
                seed = seed_rank_map[roster_id]
                choke_score = PLACEMENT_RANK[label] - seed
                if choke_score > 0:
                    choke_candidates.append({
                        "season": row["season"], "owner_id": owner, "seed": seed,
                        "placement": label, "choke_score": choke_score,
                    })

    return champions, runner_ups, playoff_wins, finals_appearances, playoff_appearances, choke_candidates


def compute_draft_and_trade_superlatives(latest_league_id, latest_season_year, cache_key):
    all_picks = []
    for _, d in build_rookie_draft_list().iterrows():
        board = get_rookie_draft_grades(d["draft_id"], d["league_id"], latest_season_year, cache_key).copy()
        board["season"] = d["season"]
        all_picks.append(board)
    picks_df = pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()

    best_pick = picks_df.loc[picks_df["delta"].idxmax()] if not picks_df.empty else None
    worst_pick = picks_df.loc[picks_df["delta"].idxmin()] if not picks_df.empty else None
    best_draft_team = None
    if not picks_df.empty:
        team_totals = picks_df.groupby(["season", "team"])["delta"].sum().reset_index()
        best_draft_team = team_totals.loc[team_totals["delta"].idxmax()]

    global_team_lookup_local = get_global_team_lookup()
    players_df_local = get_players_df()
    trade_rows = []
    for _, txn in get_all_trades(cache_key).iterrows():
        vt = get_historical_value_table(int(txn["season"]), cache_key)
        pp = get_season_power_pct(txn["league_id"], cache_key)
        trade_summary = " ".join(describe_trade(txn, players_df_local, txn["league_id"], global_team_lookup_local))
        for side in grade_trade(txn, vt, pp, global_team_lookup_local, txn["league_id"]):
            trade_rows.append({
                "season": txn["season"], "team": side["team_name"],
                "net_value": side["win_now_impact"] + side["future_impact"],
                "summary": trade_summary,
            })
    trades_df = pd.DataFrame(trade_rows)
    greatest_trade = trades_df.loc[trades_df["net_value"].idxmax()] if not trades_df.empty else None
    worst_trade = trades_df.loc[trades_df["net_value"].idxmin()] if not trades_df.empty else None

    return best_pick, worst_pick, best_draft_team, greatest_trade, worst_trade


def build_hall_of_fame(latest_league_id, latest_season_year, cache_key):
    career = load_table("SELECT league_id, roster_id, owner_id, wins, losses, ties, fpts FROM rosters")
    career_totals = career.groupby("owner_id").agg(
        career_wins=("wins", "sum"), career_losses=("losses", "sum"),
        career_ties=("ties", "sum"), career_points=("fpts", "sum"),
    ).reset_index()

    scored = load_table("""
        SELECT m.league_id, m.week, m.roster_id, m.points, r.owner_id, ls.season
        FROM matchups m
        JOIN rosters r ON r.league_id = m.league_id AND r.roster_id = m.roster_id
        JOIN league_seasons ls ON ls.league_id = m.league_id
        WHERE m.points > 0
    """)
    # Bound each season to its own real final week (round2_weeks[1]) — real-NFL weeks
    # beyond that (e.g. week 18 in a league whose fantasy season ends week 17) aren't
    # real fantasy matchups for this league and shouldn't count toward these records.
    max_week_by_league = {lid: get_playoff_settings(lid)["round2_weeks"][1] for lid in scored["league_id"].unique()}
    scored = scored[scored.apply(lambda r: r["week"] <= max_week_by_league[r["league_id"]], axis=1)]
    highest_week = scored.loc[scored["points"].idxmax()]
    lowest_week = scored.loc[scored["points"].idxmin()]

    game_log = get_regular_season_log(cache_key)
    streaks = compute_streaks(game_log)
    luck = compute_luck_index(game_log)

    champions, runner_ups, playoff_wins, _finals_appearances, _playoff_appearances, choke_candidates = \
        compute_playoff_records()
    best_pick, worst_pick, best_draft_team, greatest_trade, worst_trade = \
        compute_draft_and_trade_superlatives(latest_league_id, latest_season_year, cache_key)

    manager_name_lookup = get_manager_name_lookup()
    global_team_lookup_local = get_global_team_lookup()

    def manager_name(owner_id):
        return manager_name_lookup.get(owner_id, owner_id)

    def team_name(league_id, roster_id):
        return global_team_lookup_local.get((league_id, roster_id), f"Roster {roster_id}")

    leaderboard = career_totals.copy()
    leaderboard["manager"] = leaderboard["owner_id"].map(manager_name)
    leaderboard["championships"] = leaderboard["owner_id"].map(champions).fillna(0).astype(int)
    leaderboard["runner_ups"] = leaderboard["owner_id"].map(runner_ups).fillna(0).astype(int)
    leaderboard["playoff_wins"] = leaderboard["owner_id"].map(playoff_wins).fillna(0).astype(int)
    leaderboard = leaderboard.merge(streaks, on="owner_id", how="left").merge(luck, on="owner_id", how="left")
    leaderboard = leaderboard.sort_values("career_wins", ascending=False)

    biggest_choke = max(choke_candidates, key=lambda c: c["choke_score"]) if choke_candidates else None
    if biggest_choke:
        biggest_choke = {**biggest_choke, "manager": manager_name(biggest_choke["owner_id"])}
    most_unlucky = leaderboard.loc[leaderboard["luck_index"].idxmin()] if not leaderboard.empty else None

    # NOTE: don't put closures (e.g. `manager_name`) in this dict — st.cache_data pickles
    # the return value, and Python can't pickle local closures. Resolve names above instead.
    return {
        "leaderboard": leaderboard,
        "highest_week": {"team": team_name(highest_week["league_id"], highest_week["roster_id"]),
                          "season": highest_week["season"], "week": int(highest_week["week"]),
                          "points": highest_week["points"]},
        "lowest_week": {"team": team_name(lowest_week["league_id"], lowest_week["roster_id"]),
                        "season": lowest_week["season"], "week": int(lowest_week["week"]),
                        "points": lowest_week["points"]},
        "best_pick": best_pick, "worst_pick": worst_pick, "best_draft_team": best_draft_team,
        "greatest_trade": greatest_trade, "worst_trade": worst_trade,
        "biggest_choke": biggest_choke, "most_unlucky": most_unlucky,
    }


@st.cache_data(ttl=1800)
def get_hall_of_fame(latest_league_id, latest_season_year, cache_key):
    return build_hall_of_fame(latest_league_id, latest_season_year, cache_key)


# ---------------------------------------------------------------------------
# League Records — a straightforward "record book" of ranked leaderboards and
# single-season bests, distinct from Hall of Fame's narrative "best/worst
# moment ever" framing. Reuses Hall of Fame's underlying data (career
# leaderboard, streaks, weekly high/low) rather than recomputing it, and adds
# the counting stats Hall of Fame doesn't track: trade volume, single-season
# totals, and finals/playoff appearance counts.
# ---------------------------------------------------------------------------
def compute_trade_records(cache_key):
    """Career trade count per manager, plus the single largest trade ever by
    total assets exchanged (players + picks + FAAB) — a different axis than
    Hall of Fame's Greatest/Worst Trade, which ranks by value swing, not size."""
    trades_df = get_all_trades(cache_key)
    if trades_df.empty:
        return pd.DataFrame(columns=["manager", "trades"]), None

    roster_owner = load_table("SELECT league_id, roster_id, owner_id FROM rosters")
    owner_map = {(row.league_id, row.roster_id): row.owner_id for row in roster_owner.itertuples()}
    manager_lookup = get_manager_name_lookup()
    global_team_lookup_local = get_global_team_lookup()

    trade_counts = {}
    largest_trade, max_assets = None, -1

    for _, txn in trades_df.iterrows():
        roster_ids = parse_json_field(txn["roster_ids"], [])
        adds = parse_json_field(txn["adds"], {})
        draft_picks = parse_json_field(txn["draft_picks"], [])
        waiver_budget = parse_json_field(txn["waiver_budget"], [])

        for rid in roster_ids:
            owner = owner_map.get((txn["league_id"], rid))
            if owner is not None:
                trade_counts[owner] = trade_counts.get(owner, 0) + 1

        n_assets = len(adds) + len(draft_picks) + len(waiver_budget)
        if n_assets > max_assets:
            max_assets = n_assets
            largest_trade = {
                "season": txn["season"], "n_assets": n_assets,
                "teams": [global_team_lookup_local.get((txn["league_id"], rid), f"Roster {rid}")
                          for rid in roster_ids],
            }

    trade_counts_df = pd.DataFrame([
        {"manager": manager_lookup.get(owner, owner), "trades": count}
        for owner, count in trade_counts.items()
    ]).sort_values("trades", ascending=False)

    return trade_counts_df, largest_trade


def compute_single_season_records():
    """Most Points and Best Record in a single season — each row in `rosters`
    is already one team's totals for one season, so no new aggregation is
    needed beyond joining in the season year and team name."""
    season_rows = load_table("""
        SELECT r.league_id, r.roster_id, r.owner_id, r.wins, r.losses, r.ties, r.fpts, ls.season
        FROM rosters r JOIN league_seasons ls ON ls.league_id = r.league_id
    """)
    if season_rows.empty:
        return None, None

    global_team_lookup_local = get_global_team_lookup()
    manager_lookup = get_manager_name_lookup()

    season_rows["team"] = [
        global_team_lookup_local.get((lid, rid), f"Roster {rid}")
        for lid, rid in zip(season_rows["league_id"], season_rows["roster_id"])
    ]
    season_rows["manager"] = season_rows["owner_id"].map(manager_lookup).fillna(season_rows["owner_id"])

    most_points_row = season_rows.loc[season_rows["fpts"].idxmax()]
    best_record_row = season_rows.loc[season_rows["wins"].idxmax()]

    most_points = {"team": most_points_row["team"], "manager": most_points_row["manager"],
                   "season": most_points_row["season"], "points": most_points_row["fpts"]}
    best_record = {"team": best_record_row["team"], "manager": best_record_row["manager"],
                   "season": best_record_row["season"], "wins": int(best_record_row["wins"]),
                   "losses": int(best_record_row["losses"]), "ties": int(best_record_row["ties"])}
    return most_points, best_record


def build_league_records(latest_league_id, latest_season_year, cache_key):
    hof = get_hall_of_fame(latest_league_id, latest_season_year, cache_key)
    _champions, _runner_ups, _playoff_wins, finals_appearances, playoff_appearances, _choke = \
        compute_playoff_records()

    manager_lookup = get_manager_name_lookup()

    def appearances_df(counts):
        if not counts:
            return pd.DataFrame(columns=["manager", "count"])
        return pd.DataFrame([
            {"manager": manager_lookup.get(owner, owner), "count": count}
            for owner, count in counts.items()
        ]).sort_values("count", ascending=False)

    trade_counts_df, largest_trade = compute_trade_records(cache_key)
    most_points_season, best_record_season = compute_single_season_records()

    return {
        "hof_leaderboard": hof["leaderboard"],
        "highest_week": hof["highest_week"], "lowest_week": hof["lowest_week"],
        "championships": appearances_df({
            owner: count for owner, count in
            zip(hof["leaderboard"]["owner_id"], hof["leaderboard"]["championships"]) if count > 0
        }),
        "finals_appearances": appearances_df(finals_appearances),
        "playoff_appearances": appearances_df(playoff_appearances),
        "trade_counts": trade_counts_df, "largest_trade": largest_trade,
        "most_points_season": most_points_season, "best_record_season": best_record_season,
    }


@st.cache_data(ttl=1800)
def get_league_records(latest_league_id, latest_season_year, cache_key):
    return build_league_records(latest_league_id, latest_season_year, cache_key)


# ---------------------------------------------------------------------------
# Rivalries — automatically tracked head-to-head history for every pair of
# managers who've ever played each other, regular season and playoffs, across
# every synced season.
# ---------------------------------------------------------------------------
H2H_COLUMNS = ["season", "week", "league_id", "is_playoff", "round_label", "owner_a", "points_a", "owner_b", "points_b"]


def build_regular_h2h_games():
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    rows = []
    for _, season_row in all_seasons.iterrows():
        lid = season_row["league_id"]
        reg_weeks = get_playoff_settings(lid)["regular_season_weeks"]
        schedule = load_table(
            "SELECT week, matchup_id, roster_id, points FROM matchups WHERE league_id = ? AND week <= ?",
            params=(lid, reg_weeks),
        )
        roster_owner = load_table("SELECT roster_id, owner_id FROM rosters WHERE league_id = ?", params=(lid,))
        owner_map = dict(zip(roster_owner["roster_id"], roster_owner["owner_id"]))

        for week, week_rows in schedule.groupby("week"):
            if week_rows["points"].sum() <= 0:
                continue
            for _, group in week_rows.groupby("matchup_id"):
                if len(group) != 2:
                    continue
                a, b = group.iloc[0], group.iloc[1]
                owner_a, owner_b = owner_map.get(int(a["roster_id"])), owner_map.get(int(b["roster_id"]))
                if not owner_a or not owner_b:
                    continue
                rows.append({
                    "season": season_row["season"], "week": int(week), "league_id": lid,
                    "is_playoff": False, "round_label": f"Week {week}",
                    "owner_a": owner_a, "points_a": a["points"],
                    "owner_b": owner_b, "points_b": b["points"],
                })
    return pd.DataFrame(rows, columns=H2H_COLUMNS)


def get_playoff_h2h_games(league_id, season):
    """Actual playoff head-to-heads for a completed season, derived directly from this
    league's real bracket format (seed1-vs-seed4, seed2-vs-seed3, then combined-two-week
    championship/3rd-place games) — same shape already established for the simulator."""
    if not is_season_complete(league_id):
        return []
    settings = get_playoff_settings(league_id)
    standings_local = get_standings(league_id)
    roster_ids = standings_local["roster_id"].astype(int).tolist()
    owner_map = dict(zip(standings_local["roster_id"].astype(int), standings_local["owner_id"]))
    if len(roster_ids) < 4:
        return []
    s1, s2, s3, s4 = roster_ids[:4]

    def week_points(week):
        df = load_table("SELECT roster_id, points FROM matchups WHERE league_id = ? AND week = ?", params=(league_id, week))
        return dict(zip(df["roster_id"].astype(int), df["points"]))

    r1 = week_points(settings["round1_week"])
    games = [
        {"season": season, "week": settings["round1_week"], "league_id": league_id, "is_playoff": True,
         "round_label": "Round 1", "owner_a": owner_map[s1], "points_a": r1.get(s1, 0.0),
         "owner_b": owner_map[s4], "points_b": r1.get(s4, 0.0)},
        {"season": season, "week": settings["round1_week"], "league_id": league_id, "is_playoff": True,
         "round_label": "Round 1", "owner_a": owner_map[s2], "points_a": r1.get(s2, 0.0),
         "owner_b": owner_map[s3], "points_b": r1.get(s3, 0.0)},
    ]

    m1_winner, m1_loser = (s1, s4) if r1.get(s1, 0.0) > r1.get(s4, 0.0) else (s4, s1)
    m2_winner, m2_loser = (s2, s3) if r1.get(s2, 0.0) > r1.get(s3, 0.0) else (s3, s2)

    r2a = week_points(settings["round2_weeks"][0])
    r2b = week_points(settings["round2_weeks"][1])

    def combined(rid):
        return r2a.get(rid, 0.0) + r2b.get(rid, 0.0)

    games.append({"season": season, "week": settings["round2_weeks"][0], "league_id": league_id, "is_playoff": True,
                   "round_label": "Championship", "owner_a": owner_map[m1_winner], "points_a": combined(m1_winner),
                   "owner_b": owner_map[m2_winner], "points_b": combined(m2_winner)})
    games.append({"season": season, "week": settings["round2_weeks"][0], "league_id": league_id, "is_playoff": True,
                   "round_label": "3rd Place Game", "owner_a": owner_map[m1_loser], "points_a": combined(m1_loser),
                   "owner_b": owner_map[m2_loser], "points_b": combined(m2_loser)})
    return games


def build_all_h2h_games():
    frames = [build_regular_h2h_games()]
    for _, row in load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC").iterrows():
        playoff_games = get_playoff_h2h_games(row["league_id"], row["season"])
        if playoff_games:
            frames.append(pd.DataFrame(playoff_games, columns=H2H_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def build_rivalries():
    games = build_all_h2h_games()
    if games.empty:
        return pd.DataFrame()
    games = games.copy()
    games["pair"] = games.apply(lambda r: tuple(sorted([r["owner_a"], r["owner_b"]])), axis=1)

    name_lookup = get_manager_name_lookup()

    rows = []
    for pair, group in games.groupby("pair"):
        owner1, owner2 = pair
        wins1 = wins2 = ties = 0
        playoff_wins1 = playoff_wins2 = 0
        margins = []
        highest_scoring = None
        results_seq = []

        for _, row in group.sort_values(["season", "week"]).iterrows():
            if row["owner_a"] == owner1:
                p1, p2 = row["points_a"], row["points_b"]
            else:
                p1, p2 = row["points_b"], row["points_a"]
            margins.append(abs(p1 - p2))
            total = p1 + p2
            if highest_scoring is None or total > highest_scoring["total"]:
                highest_scoring = {"season": row["season"], "round_label": row["round_label"], "total": total}

            if p1 > p2:
                wins1 += 1
                results_seq.append("W")
                if row["is_playoff"]:
                    playoff_wins1 += 1
            elif p2 > p1:
                wins2 += 1
                results_seq.append("L")
                if row["is_playoff"]:
                    playoff_wins2 += 1
            else:
                ties += 1
                results_seq.append("T")

        best_streak_owner, best_streak_len = None, 0
        cur_owner, cur_len = None, 0
        for r in results_seq:
            if r == "T":
                cur_owner, cur_len = None, 0
                continue
            who = owner1 if r == "W" else owner2
            cur_len = cur_len + 1 if who == cur_owner else 1
            cur_owner = who
            if cur_len > best_streak_len:
                best_streak_len, best_streak_owner = cur_len, cur_owner

        games_played = wins1 + wins2 + ties
        avg_margin = float(np.mean(margins)) if margins else 0.0
        balance = 1 - abs(wins1 - wins2) / games_played if games_played else 0.0

        rows.append({
            "owner1": owner1, "owner2": owner2,
            "manager1": name_lookup.get(owner1, owner1), "manager2": name_lookup.get(owner2, owner2),
            "wins1": wins1, "wins2": wins2, "ties": ties,
            "playoff_wins1": playoff_wins1, "playoff_wins2": playoff_wins2,
            "games_played": games_played, "avg_margin": round(avg_margin, 1),
            "highest_scoring_total": round(highest_scoring["total"], 1) if highest_scoring else None,
            "highest_scoring_season": highest_scoring["season"] if highest_scoring else None,
            "highest_scoring_round": highest_scoring["round_label"] if highest_scoring else None,
            "streak_manager": name_lookup.get(best_streak_owner, best_streak_owner) if best_streak_owner else None,
            "streak_len": best_streak_len,
            "rivalry_rating_raw": games_played * 2 + balance * 20 - avg_margin * 0.1,
        })

    rivalries_df = pd.DataFrame(rows)
    rivalries_df["rivalry_rating"] = to_score_100(rivalries_df["rivalry_rating_raw"])
    return rivalries_df.sort_values("rivalry_rating", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_rivalries(cache_key):
    return build_rivalries()


st.title("Blue Ballers Analytics")

seasons = get_seasons()
if seasons.empty:
    st.warning("No data found yet — run the sync notebook first.")
    st.stop()

season_choice = st.sidebar.selectbox("Season", seasons["season"].tolist())
league_id = seasons.loc[seasons["season"] == season_choice, "league_id"].iloc[0]
synced_at = seasons.loc[seasons["season"] == season_choice, "synced_at"].iloc[0]
st.sidebar.caption(f"Last synced: {synced_at}")
st.sidebar.caption(f"Build: {APP_BUILD}")
if st.sidebar.button("Refresh Data"):
    with st.spinner("Downloading latest league data..."):
        refreshed = download_db()
    if refreshed:
        load_table.clear()
        st.rerun()
    else:
        st.sidebar.error(f"Refresh failed: {st.session_state.get('db_download_error')}")

standings = get_standings(league_id)
rankings = get_power_rankings(league_id, synced_at)
team_lookup = dict(zip(standings["roster_id"], standings["team_name"]))
players_df = get_players_df()
global_team_lookup = get_global_team_lookup()
odds = simulate_odds(league_id, synced_at)
ensure_rivalry_submissions_table()
seed_rivalry_declarations()
latest_league_id = seasons.iloc[0]["league_id"]
latest_season_year = int(seasons.iloc[0]["season"])


def page_home():
    col_rankings, col_standings = st.columns(2)
    
    with col_rankings:
        st.header("Power Rankings")
        st.caption(
            "0.6x true-strength estimate (blends this season's scoring with last season's, so a hot "
            "or cold start early on doesn't overweight) + 0.4x luck-purged expected win% (from "
            "points-rank each week, not real results). Aims to answer 'who's actually the best team' "
            "rather than just replaying the standings."
        )
        home_last_week = get_last_completed_week(league_id)
        week_options = (
            ["Current"] + [str(w) for w in range(home_last_week, 0, -1)] if home_last_week else ["Current"]
        )
        week_choice = st.selectbox("As of", week_options, key="power_rankings_snapshot_week")

        if week_choice == "Current":
            display_rankings = rankings
            prev_rankings = get_previous_week_power_rankings(league_id, synced_at)
        else:
            snapshot_week = int(week_choice)
            display_rankings = get_power_rankings_as_of(league_id, snapshot_week, synced_at)
            prev_rankings = (
                get_power_rankings_as_of(league_id, snapshot_week - 1, synced_at) if snapshot_week > 1 else None
            )
        prev_rank_by_roster = (
            {int(r): i + 1 for i, r in enumerate(prev_rankings["roster_id"])}
            if prev_rankings is not None else {}
        )
        for i, row in display_rankings.iterrows():
            rank = i + 1
            prev_rank = prev_rank_by_roster.get(int(row["roster_id"]))
            rank_change = (prev_rank - rank) if prev_rank is not None else None
            if rank_change is None:
                arrow = ""
            elif rank_change > 0:
                arrow = f" :green[▲{rank_change}]"
            elif rank_change < 0:
                arrow = f" :red[▼{abs(rank_change)}]"
            else:
                arrow = " ▬"
            st.write(f"**{rank}. {escape_markdown(row['team_name'])}**{arrow} — "
                     f"{escape_markdown(row['manager'])} "
                     f"({int(row['wins'])}-{int(row['losses'])}, {row['fpts']:.1f} pts)")
            st.caption(power_ranking_blurb(row, rank_change))
    
    with col_standings:
        st.header("Standings")
        max_pf = get_max_pf(league_id, synced_at)
        standings_display = standings[
            ["roster_id", "team_name", "manager", "wins", "losses", "ties", "fpts", "fpts_against"]
        ].copy()
        standings_display["max_pf"] = standings_display["roster_id"].map(
            lambda rid: max_pf.get(int(rid), 0.0)
        )
        st.dataframe(
            standings_display[["team_name", "manager", "wins", "losses", "ties", "fpts", "fpts_against", "max_pf"]]
            .rename(columns={"team_name": "Team", "manager": "Manager", "wins": "W", "losses": "L",
                              "ties": "T", "fpts": "PF", "fpts_against": "PA", "max_pf": "Max PF"}),
            use_container_width=True, hide_index=True,
        )
        st.caption("Max PF = best-possible optimal-lineup points scored so far — the stat this "
                   "league's rookie draft order uses for the non-playoff teams, to prevent tanking.")
    
    st.header("Championship & Playoff Odds")
    st.caption(
        f"{N_TRIALS:,}-trial Monte Carlo simulation of the rest of the season and playoffs. "
        "Unplayed weeks are simulated from each team's own scoring so far this season (recent "
        "weeks weighted more than early ones, for current form), blended with their scoring last "
        "season (more weight to this season as more games are played — same shrinkage applies to "
        "variance, not just the average). Weekly scores are drawn from a right-skewed distribution "
        "matched to each team's own mean/variance, not a symmetric bell curve, so it doesn't "
        "understate real blowout-week upside."
    )
    odds_df = standings[["roster_id", "team_name", "manager"]].copy()
    odds_df["Playoff Odds"] = odds_df["roster_id"].map(lambda rid: odds[int(rid)]["playoff_odds"])
    odds_df["Championship Odds"] = odds_df["roster_id"].map(lambda rid: odds[int(rid)]["championship_odds"])
    odds_df = odds_df.sort_values("Championship Odds", ascending=False)
    st.dataframe(
        odds_df[["team_name", "manager", "Playoff Odds", "Championship Odds"]]
        .rename(columns={"team_name": "Team", "manager": "Manager"})
        .style.format({"Playoff Odds": "{:.1%}", "Championship Odds": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )
    
    st.header("Weekly Recap")
    last_week = get_last_completed_week(league_id)
    if last_week is None:
        st.info("No completed weeks yet this season.")
    else:
        week_choice = st.selectbox("Week", list(range(last_week, 0, -1)))
        week_matchups = get_week_matchups(league_id, week_choice)
    
        if not week_matchups.empty:
            highest = week_matchups.loc[week_matchups["points"].idxmax()]
            lowest = week_matchups.loc[week_matchups["points"].idxmin()]
            top_performer = get_week_top_performer(league_id, week_choice, players_df)
    
            two_team_games = []
            for _, group in week_matchups.groupby("matchup_id"):
                if len(group) == 2:
                    a, b = group.iloc[0], group.iloc[1]
                    two_team_games.append({"a": a, "b": b, "margin": abs(a["points"] - b["points"])})
            closest_game = min(two_team_games, key=lambda g: g["margin"]) if two_team_games else None
            biggest_blowout = max(two_team_games, key=lambda g: g["margin"]) if two_team_games else None
    
            st.subheader(f"Week {week_choice} Headlines")
            c1, c2, c3, c4 = st.columns(4)
            metric_block(c1, "Top Score", highest["team_name"], f"{highest['points']:.1f} pts")
            metric_block(c2, "Lowest Score", lowest["team_name"], f"{lowest['points']:.1f} pts")
            if closest_game:
                metric_block(c3, "Closest Game",
                             f"{closest_game['a']['team_name']} vs {closest_game['b']['team_name']}",
                             f"{closest_game['margin']:.1f} pt margin")
            if top_performer:
                metric_block(c4, "Top Performer", f"{top_performer['player']} ({top_performer['position']})",
                             f"{top_performer['points']:.1f} pts")
    
            st.subheader("Matchup Recaps")
            for _, group in week_matchups.groupby("matchup_id"):
                if len(group) == 2:
                    a, b = group.iloc[0], group.iloc[1]
                    st.write(describe_matchup(a["team_name"], a["points"], b["team_name"], b["points"]))
                else:
                    for _, row in group.iterrows():
                        st.write(f"{escape_markdown(row['team_name'])}: {row['points']:.1f}")

            if biggest_blowout and biggest_blowout is not closest_game:
                st.caption(f"Blowout of the week: {escape_markdown(biggest_blowout['a']['team_name'])} vs "
                           f"{escape_markdown(biggest_blowout['b']['team_name'])} "
                           f"({biggest_blowout['margin']:.1f} pt margin)")


def page_recent_transactions():
    st.header("Recent Transactions")
    transactions = get_recent_transactions(league_id)
    if transactions.empty:
        st.info("No transactions recorded yet.")
    else:
        for _, txn in transactions.iterrows():
            st.subheader(f"{txn['type'].replace('_', ' ').title()} — Week {txn['week']}")
            for name, team in format_player_moves(txn["adds"], players_df, team_lookup):
                st.write(f"Added: {name} ({team})")
            for name, team in format_player_moves(txn["drops"], players_df, team_lookup):
                st.write(f"Dropped: {name} ({team})")
            st.divider()


def page_matchup_center():
    st.header("Matchup Center")
    st.caption(
        "Pick any matchup from any synced week for a full recap — narrative writeup plus the "
        "boxscore, starting lineup vs. starting lineup, player by player, plus each team's bench. "
        "For whole-week aggregate stats (top score, closest game, top performer), see the Home "
        "page's Weekly Recap instead."
    )

    mc_last_week = get_last_completed_week(league_id)
    if mc_last_week is None:
        st.info("No completed weeks yet this season.")
    else:
        mc_week = st.selectbox("Week", list(range(mc_last_week, 0, -1)), key="matchup_center_week")
        mc_week_matchups = get_week_matchups(league_id, mc_week)

        two_team_games = [
            (group.iloc[0], group.iloc[1])
            for _, group in mc_week_matchups.groupby("matchup_id") if len(group) == 2
        ]

        if not two_team_games:
            st.info("No head-to-head matchups found for this week.")
        else:
            labels = [f"{a['team_name']} vs {b['team_name']}" for a, b in two_team_games]
            matchup_choice = st.selectbox("Matchup", labels, key="matchup_center_pick")
            team_a, team_b = two_team_games[labels.index(matchup_choice)]

            st.subheader("Recap")
            for line in build_matchup_recap(league_id, mc_week, team_a, team_b, players_df):
                st.write(line)

            st.subheader("Boxscore")
            col_a, col_b = st.columns(2)
            for col, team in zip((col_a, col_b), (team_a, team_b)):
                with col:
                    metric_block(col, team["team_name"], f"{team['points']:.1f} pts")
                    starters, bench = get_week_lineup(league_id, mc_week, int(team["roster_id"]), players_df)
                    st.write("**Starters**")
                    st.dataframe(
                        starters.rename(columns={"player": "Player", "position": "Pos", "points": "Points"}),
                        use_container_width=True, hide_index=True,
                    )
                    if not bench.empty:
                        with st.expander("Bench"):
                            st.dataframe(
                                bench.rename(columns={"player": "Player", "position": "Pos", "points": "Points"}),
                                use_container_width=True, hide_index=True,
                            )

def page_team_pages():
    st.header("Team Page")
    st.caption(
        "Grades and scores below come from an in-house heuristic (position scarcity x age curve x "
        "on-field performance) since no external dynasty-value source is wired in yet — good for "
        "comparing teams within this league, not a market-calibrated trade value."
    )
    team_choice = st.selectbox("Team", standings["team_name"].tolist())
    sel = standings.loc[standings["team_name"] == team_choice].iloc[0]
    roster_id = int(sel["roster_id"])
    owner_id = sel["owner_id"]
    season_year = int(season_choice)
    
    player_ids, starter_ids = get_roster(league_id, roster_id)
    value_table = get_value_table(league_id, season_year, synced_at)
    league_grades = get_league_grades(league_id, season_year, synced_at)
    positional_grades = get_positional_grades(league_id, season_year, synced_at)
    
    metrics = team_overview_metrics(value_table, player_ids, starter_ids)
    grade_row = league_grades.loc[roster_id]
    
    st.subheader("Team Overview")
    n_teams = len(league_grades)
    dynasty_rank = int(league_grades["dynasty_score"].rank(ascending=False, method="min").loc[roster_id])
    contender_rank = int(league_grades["contender_score"].rank(ascending=False, method="min").loc[roster_id])
    future_rank = int(league_grades["future_score"].rank(ascending=False, method="min").loc[roster_id])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Grade", str(grade_row["overall_grade"]))
    c2.metric("Dynasty Score", f"{metrics['dynasty_score']:.0f}", f"#{dynasty_rank} of {n_teams}")
    c3.metric("Contender Score", f"{metrics['contender_score']:.0f}", f"#{contender_rank} of {n_teams}")
    c4.metric("Future Score", f"{metrics['future_score']:.0f}", f"#{future_rank} of {n_teams}")

    c5, c6, c7 = st.columns(3)
    avg_age = metrics["avg_age"]
    c5.metric("Average Age", f"{avg_age:.1f}" if pd.notna(avg_age) else "—")
    c6.metric("Starting Lineup Grade", str(grade_row["starter_grade"]))
    c7.metric("Bench Grade", str(grade_row["bench_grade"]))

    state = team_competitive_state(grade_row["contender_pct"], grade_row["future_pct"])
    window_start, window_end = estimate_championship_window(
        grade_row["contender_pct"], grade_row["future_pct"], season_year
    )
    st.write(f"**Championship Window:** {window_start}–{window_end} "
             "*(rough estimate from this roster's win-now vs. future-asset balance)*")
    st.caption(dynasty_recommendation(state))

    st.subheader("Positional Grades")
    team_pos_grades = positional_grades[positional_grades["roster_id"] == roster_id]
    st.dataframe(
        style_grades(
            team_pos_grades[["position", "value", "grade"]]
            .rename(columns={"position": "Position", "value": "Value", "grade": "Grade"}),
            ["Grade"],
        ),
        use_container_width=True, hide_index=True,
    )
    non_bench = team_pos_grades[team_pos_grades["position"] != "Bench"]
    if not non_bench.empty:
        weakest = non_bench.loc[non_bench["grade"].map(GRADE_ORDER).idxmin()]
        st.caption(f"Weakest position: **{weakest['position']}** ({weakest['grade']}) — "
                   "the most likely spot to target via trade or draft capital.")
    
    st.subheader("Age Curve")
    curve_df = project_team_value_curve(league_id, roster_id, season_year, players_df)
    fig = px.line(curve_df, x="year", y="projected_value", color="position", markers=True,
                  labels={"year": "Season", "projected_value": "Projected Value", "position": "Position"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Per-position value forward assuming no trades or additions — shows which parts of "
               "the roster age well vs. crater, instead of one aggregate line.")
    
    st.subheader("Future Draft Capital")
    pick_inventory = get_pick_inventory(league_id, season_year, synced_at)
    team_picks = pick_inventory[pick_inventory["owner_roster_id"] == roster_id].copy()
    power_pct = dict(zip(rankings["roster_id"], rankings["power_score"].rank(pct=True)))
    team_picks["Value"] = team_picks.apply(
        lambda r: value_pick_row(r, power_pct, str(season_year), odds), axis=1
    )
    team_picks["Original Team"] = team_picks["original_roster_id"].map(team_lookup)
    team_picks = team_picks.sort_values(["season", "round"])
    st.dataframe(
        team_picks[["season", "round", "Original Team", "Value"]]
        .rename(columns={"season": "Season", "round": "Round"}),
        use_container_width=True, hide_index=True,
    )
    
    st.subheader("Roster")
    roster_rows = []
    for pid in player_ids:
        info = players_df.loc[pid] if pid in players_df.index else None
        value_row = value_table.loc[pid] if pid in value_table.index else None
        roster_rows.append({
            "Player": info["full_name"] if info is not None else pid,
            "Pos": info["position"] if info is not None else "",
            "NFL Team": info["team"] if info is not None else "",
            "Age": value_row["age"] if value_row is not None else None,
            "Value": value_row["value"] if value_row is not None else None,
            "Starter": pid in starter_ids,
        })
    roster_df = pd.DataFrame(roster_rows).sort_values("Starter", ascending=False)
    st.dataframe(roster_df, use_container_width=True, hide_index=True)
    
    st.subheader("Team Timeline")
    st.caption("Every trade, waiver, and free agent move this manager has made across every synced season.")
    timeline = get_team_timeline(owner_id, synced_at)
    if timeline.empty:
        st.info("No transactions found for this team.")
    else:
        timeline = timeline.copy()
        # Sleeper stamps every transaction's "week" with the CURRENT week at the time
        # it happened, which is 1 during the whole offseason (no games played yet) —
        # so `not week` never fires and nothing ever lands in "Offseason". Go by the
        # actual calendar date instead: NFL offseason runs Feb-Aug regardless of what
        # week number Sleeper reports.
        created_dt = pd.to_datetime(timeline["created"], unit="ms", errors="coerce")
        is_offseason = created_dt.dt.month.between(2, 8).fillna(False)
        timeline["period"] = [
            f"{season} Offseason" if off else f"{season} Week {int(week)}"
            for season, week, off in zip(timeline["season"], timeline["week"], is_offseason)
        ]
        timeline["_sort_week"] = [-1 if off else week for week, off in zip(timeline["week"], is_offseason)]
        ordered_periods = timeline.sort_values(["season", "_sort_week"], ascending=[False, False])["period"]
        period_options = ["All"] + list(dict.fromkeys(ordered_periods))
        period_choice = st.selectbox("Filter by season/week", period_options, key="team_timeline_period")
        filtered = timeline if period_choice == "All" else timeline[timeline["period"] == period_choice]

        for _, txn in filtered.iterrows():
            period_suffix = txn["period"].split(" ", 1)[1]  # "Offseason" or "Week N"
            st.write(f"**{txn['season']} — {txn['type'].replace('_', ' ').title()} ({period_suffix})**")
            if txn["type"] == "trade":
                for line in describe_trade(txn, players_df, txn["league_id"], global_team_lookup):
                    st.write(line)
            else:
                for name, team in format_moves_global(txn["adds"], players_df, txn["league_id"], global_team_lookup):
                    st.write(f"Added: {name} ({team})")
                for name, team in format_moves_global(txn["drops"], players_df, txn["league_id"], global_team_lookup):
                    st.write(f"Dropped: {name} ({team})")
            st.divider()

def page_stock_market():
    st.header("Stock Market")
    st.caption(
        "Every player who's ever been rostered in this league, valued with the same "
        "position/age/performance heuristic used everywhere else in the app — but recomputed "
        "week by week using only production accumulated up to that point, so it moves like a "
        "real price instead of sitting at a single end-of-season snapshot. Resets each new "
        "season, since a fresh season's role/opportunity is a real reset, not a continuation "
        "of last year's."
    )

    history = get_stock_market_history(synced_at)
    if history.empty:
        st.info("No player performance data synced yet.")
        return

    periods = sorted(history[["season", "week"]].drop_duplicates().itertuples(index=False, name=None))
    latest_period = periods[-1]
    LOOKBACK = 4
    lookback_period = periods[max(0, len(periods) - 1 - LOOKBACK)]

    now_df = (history[(history["season"] == latest_period[0]) & (history["week"] == latest_period[1])]
              [["player_id", "value"]].rename(columns={"value": "now"}))
    before_df = (history[(history["season"] == lookback_period[0]) & (history["week"] == lookback_period[1])]
                 [["player_id", "value"]].rename(columns={"value": "before"}))
    movers = now_df.merge(before_df, on="player_id", how="left")
    movers["before"] = movers["before"].fillna(movers["now"])
    movers["change"] = movers["now"] - movers["before"]
    movers["full_name"] = movers["player_id"].map(players_df["full_name"])
    movers["position"] = movers["player_id"].map(players_df["position"])
    movers = movers.dropna(subset=["full_name"])

    st.subheader("Biggest Movers")
    st.caption(f"Value change from Season {lookback_period[0]} Week {lookback_period[1]} to "
               f"Season {latest_period[0]} Week {latest_period[1]}.")
    col_up, col_down = st.columns(2)
    with col_up:
        st.write("**Top Gainers**")
        gainers = movers.sort_values("change", ascending=False).head(10)
        st.dataframe(
            gainers[["full_name", "position", "now", "change"]]
            .rename(columns={"full_name": "Player", "position": "Pos", "now": "Value", "change": "Change"}),
            use_container_width=True, hide_index=True,
        )
    with col_down:
        st.write("**Top Fallers**")
        fallers = movers.sort_values("change", ascending=True).head(10)
        st.dataframe(
            fallers[["full_name", "position", "now", "change"]]
            .rename(columns={"full_name": "Player", "position": "Pos", "now": "Value", "change": "Change"}),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Player Trend")
    player_options = movers.sort_values("full_name")[["player_id", "full_name", "position"]].copy()
    player_options["label"] = player_options["full_name"] + " (" + player_options["position"].fillna("") + ")"
    selected_label = st.selectbox("Player", player_options["label"].tolist())
    selected_pid = player_options.loc[player_options["label"] == selected_label, "player_id"].iloc[0]

    player_history = history[history["player_id"] == selected_pid].sort_values(["season", "week"]).copy()
    player_history["period_label"] = player_history["season"].astype(str) + " Wk " + player_history["week"].astype(str)

    current_value = player_history["value"].iloc[-1]
    prior_value = (player_history["value"].iloc[-1 - LOOKBACK] if len(player_history) > LOOKBACK
                   else player_history["value"].iloc[0])
    metric_block(st, selected_label, f"{current_value:.0f}",
                 f"{current_value - prior_value:+.0f} vs {LOOKBACK} weeks ago")

    fig = px.line(player_history, x="period_label", y="value", markers=True,
                  labels={"period_label": "Season / Week", "value": "Value"})
    st.plotly_chart(fig, use_container_width=True)


def page_trade_center():
    st.header("Trade Center")
    st.caption(
        "Every trade across every synced season, auto-graded with hindsight — each player is valued "
        "off their real career-wide production, so a since-dropped player still counts instead of "
        "scoring 0. Grade starts from value received vs. given up per side, then gets knocked down a "
        "full letter if it doesn't fit that team's own timeline at the time (e.g. a rebuilding team "
        "giving up future assets for a marginal win-now piece drops from an A to a B, even if the raw "
        "value was fair) — see the Context Fit note under each grade. Winner/Loser is based on net "
        "value gained, not the letter grade."
    )
    
    st.subheader("Trade Tree")
    st.caption("Ownership history for any player who's been rostered in this league — draft, trades, and waiver moves.")
    ever_rostered = get_all_ever_rostered_ids()
    player_options = players_df.loc[players_df.index.intersection(ever_rostered)].dropna(subset=["full_name"]).copy()
    player_options["label"] = player_options["full_name"] + " (" + player_options["position"].fillna("") + ")"
    player_options = player_options.sort_values("full_name")

    selected_label = st.selectbox("Player", player_options["label"].tolist())
    selected_player_id = player_options.index[player_options["label"] == selected_label][0]

    history = get_player_ownership_history(selected_player_id, synced_at)
    if not history:
        st.info("No ownership history found for this player.")
    else:
        for event in history:
            team = global_team_lookup.get((event["league_id"], event["roster_id"]), f"Roster {event['roster_id']}")
            label = "Drafted by" if event["type"] == "draft" else event["type"].replace("_", " ").title() + " to"
            week_str = f", Week {event['week']}" if event["week"] else ""
            st.write(f"{event['season']}{week_str} — {label} **{escape_markdown(team)}**")

    st.divider()

    st.subheader("Trade History")
    search = st.text_input("Search trades (player or team name)")
    all_trades = get_all_trades(synced_at)

    if all_trades.empty:
        st.info("No trades recorded yet.")
    else:
        if search:
            search_lower = search.lower()

            def trade_matches(row):
                adds = parse_json_field(row["adds"], {})
                drops = parse_json_field(row["drops"], {})
                names = [players_df.loc[pid, "full_name"] for pid in list(adds) + list(drops)
                         if pid in players_df.index]
                roster_ids = parse_json_field(row["roster_ids"], [])
                teams = [global_team_lookup.get((row["league_id"], rid), "") for rid in roster_ids]
                return search_lower in " ".join(names + teams).lower()

            all_trades = all_trades[all_trades.apply(trade_matches, axis=1)]

        st.caption(f"{len(all_trades)} trade(s)")
        for _, txn in all_trades.iterrows():
            trade_value_table = get_historical_value_table(int(txn["season"]), synced_at)
            trade_power_pct = get_season_power_pct(txn["league_id"], synced_at)
            sides = grade_trade(txn, trade_value_table, trade_power_pct, global_team_lookup, txn["league_id"])
            trade_league_grades = get_league_grades(txn["league_id"], int(txn["season"]), synced_at)

            st.write(f"**{txn['season']} — Week {txn['week']}**")
            for line in describe_trade(txn, players_df, txn["league_id"], global_team_lookup):
                st.write(line)

            if sides:
                side_cols = st.columns(len(sides))
                for col, side in zip(side_cols, sides):
                    display_grade = side["grade"]
                    fit_note = None
                    if side["roster_id"] in trade_league_grades.index:
                        team_row = trade_league_grades.loc[side["roster_id"]]
                        state = team_competitive_state(team_row["contender_pct"], team_row["future_pct"])
                        fit_note = trade_context_fit(state, side["win_now_impact"], side["future_impact"])
                        if fit_note.startswith("⚠️"):
                            display_grade = downgrade_grade(side["grade"])
                            if display_grade != side["grade"]:
                                fit_note = f"{fit_note} (grade downgraded {side['grade']}→{display_grade})"
                    metric_block(col, f"{side['team_name']} — {side['role']}", display_grade,
                                 f"{side['win_now_impact']:+.0f} now / {side['future_impact']:+.0f} future")
                    if fit_note:
                        col.caption(fit_note)
            st.divider()

def page_rookie_draft():
    st.header("Rookie Draft Center")

    rookie_drafts = get_rookie_draft_list(synced_at)

    if rookie_drafts.empty:
        st.info("No rookie drafts found yet.")
    else:
        st.subheader("Draft Grades")
        st.caption(
            "Career Value uses the same position/age/performance heuristic as Team Pages, but scored "
            "off each player's real career PPG across every synced season — so a player later dropped "
            "still gets credit for the production they actually delivered, instead of scoring 0 just "
            "because nobody currently rosters them. Expected Value is a smooth pick-position curve, "
            "not a real consensus rookie ranking."
        )
        draft_season_choice = st.selectbox("Rookie draft season", rookie_drafts["season"].tolist())
        draft_row = rookie_drafts.loc[rookie_drafts["season"] == draft_season_choice].iloc[0]
    
        draft_board = get_rookie_draft_grades(
            draft_row["draft_id"], draft_row["league_id"], latest_season_year, synced_at
        )
    
        if draft_board.empty:
            st.info("No picks found for this draft.")
        else:
            team_grades = summarize_team_grades(draft_board)
            st.dataframe(
                style_grades(
                    team_grades.rename(columns={"team": "Team", "delta": "Total Value Over Expected", "grade": "Grade"}),
                    ["Grade"],
                ),
                use_container_width=True, hide_index=True,
            )
    
            best_pick = draft_board.loc[draft_board["delta"].idxmax()]
            worst_pick = draft_board.loc[draft_board["delta"].idxmin()]
            col_best, col_worst = st.columns(2)
            metric_block(col_best, "Best Pick", f"{best_pick['player']} ({best_pick['team']})",
                          f"+{best_pick['delta']:.0f} vs. expected")
            metric_block(col_worst, "Worst Pick", f"{worst_pick['player']} ({worst_pick['team']})",
                          f"{worst_pick['delta']:.0f} vs. expected")
    
            steals = draft_board[draft_board["delta"] > 15].sort_values("delta", ascending=False)
            reaches = draft_board[draft_board["delta"] < -15].sort_values("delta")
            col_steals, col_reaches = st.columns(2)
            with col_steals:
                st.write("**Steals**")
                if steals.empty:
                    st.caption("None this class.")
                for _, row in steals.iterrows():
                    st.write(f"Pick {row['pick_no']} ({escape_markdown(row['team'])}): "
                             f"{escape_markdown(row['player'])} ({row['position']}) — +{row['delta']:.0f}")
            with col_reaches:
                st.write("**Reaches**")
                if reaches.empty:
                    st.caption("None this class.")
                for _, row in reaches.iterrows():
                    st.write(f"Pick {row['pick_no']} ({escape_markdown(row['team'])}): "
                             f"{escape_markdown(row['player'])} ({row['position']}) — {row['delta']:.0f}")
    
            st.write("**Full Draft Board**")
            st.dataframe(
                draft_board[["pick_no", "round", "team", "player", "position", "expected_value",
                             "career_value", "delta"]]
                .rename(columns={"pick_no": "Pick", "round": "Round", "team": "Team", "player": "Player",
                                  "position": "Pos", "expected_value": "Expected", "career_value": "Career",
                                  "delta": "Delta"}),
                use_container_width=True, hide_index=True,
            )
    
    st.subheader("Draft Pick Value")
    st.caption(
        "Every future pick across the league. Value curves steeply within a round early (pick 1.01 vs "
        "1.08 is a big gap) and flattens out in later rounds, like a real dynasty pick-value chart, not "
        "a flat linear scale. For this season's picks, value is the expectation over the Championship "
        "Odds simulator's actual projected draft-slot odds (below); picks further out fall back to an "
        "estimated slot from the original team's current strength, since there's nothing left to simulate "
        "that far ahead."
    )
    league_pick_inventory = get_pick_inventory(league_id, int(season_choice), synced_at).copy()
    league_power_pct = dict(zip(rankings["roster_id"], rankings["power_score"].rank(pct=True)))
    league_pick_inventory["Value"] = league_pick_inventory.apply(
        lambda r: value_pick_row(r, league_power_pct, str(season_choice), odds), axis=1
    )
    league_pick_inventory["Original Team"] = league_pick_inventory["original_roster_id"].map(team_lookup)
    league_pick_inventory["Current Owner"] = league_pick_inventory["owner_roster_id"].map(team_lookup)
    league_pick_inventory = league_pick_inventory.sort_values(
        ["season", "round", "Value"], ascending=[True, True, False]
    )
    st.dataframe(
        league_pick_inventory[["season", "round", "Original Team", "Current Owner", "Value"]]
        .rename(columns={"season": "Season", "round": "Round"}),
        use_container_width=True, hide_index=True,
    )
    
    current_season_picks = league_pick_inventory[league_pick_inventory["season"] == season_choice]
    if not current_season_picks.empty:
        st.write(f"**{season_choice} Season — Projected Draft Position**")
        st.caption(
            "This league's actual rookie draft order: the 4 non-playoff teams pick 1-4, ordered by "
            "'Max PF' (best-ball / optimal-lineup points, not actual scoring, so it can't be tanked) "
            "with the lowest Max PF picking 1st; the 4 playoff teams pick 5-8 in straight reverse "
            "finish (champion picks 8th, last)."
        )
        proj_rows = []
        for rid in standings["roster_id"].tolist():
            rid = int(rid)
            team_odds = odds.get(rid, {})
            expected_slot = team_odds.get("expected_draft_slot")
            slot_dist = team_odds.get("draft_slot_distribution", [])
            odds_of_pick_1 = slot_dist[0] if slot_dist else None
            proj_rows.append({
                "Team": team_lookup.get(rid, f"Roster {rid}"),
                "Expected Draft Slot": round(expected_slot, 1) if expected_slot is not None else None,
                "Odds of Pick #1": odds_of_pick_1,
            })
        proj_df = pd.DataFrame(proj_rows).sort_values("Expected Draft Slot")
        st.dataframe(
            proj_df.style.format({"Odds of Pick #1": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )

def page_gm_profiles():
    st.header("GM Profiles")
    st.caption(
        "Every manager rated 0-100 (percentile among the league's managers) across seven dimensions, "
        "built from the same grading logic as Rookie Draft Center, Trade Center, and Team Pages. "
        "Risk Taking is an activity/roster-age proxy, not a true risk measure — no injury data is "
        "synced yet. Clutch is based purely on playoff finish (fully-completed seasons only) — "
        "regular-season performance doesn't factor in at all."
    )
    
    gm_profiles = get_gm_profiles(latest_league_id, latest_season_year, synced_at)
    
    gm_display_cols = ["Overall GM"] + GM_RATING_LABELS
    st.dataframe(
        gm_profiles[["manager"] + gm_display_cols].rename(columns={"manager": "Manager"})
        .style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100, subset=gm_display_cols),
        use_container_width=True, hide_index=True,
    )
    
    gm_choice = st.selectbox("Manager", gm_profiles["manager"].tolist())
    gm_row = gm_profiles.loc[gm_profiles["manager"] == gm_choice].iloc[0]
    
    radar_values = [int(gm_row[label]) for label in GM_RATING_LABELS]
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_values + [radar_values[0]],
        theta=GM_RATING_LABELS + [GM_RATING_LABELS[0]],
        fill="toself",
        name=gm_choice,
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=f"{gm_choice} — Overall GM: {int(gm_row['Overall GM'])}",
    )
    st.plotly_chart(radar_fig, use_container_width=True)

def page_hall_of_fame():
    st.header("Hall of Fame & Hall of Shame")
    st.caption(
        "Career records and single-event superlatives across every synced season. Best/Worst Draft "
        "Pick and Greatest/Worst Trade reuse the same value grading as Rookie Draft Center and Trade "
        "Center. Most Injury Luck isn't shown — no injury-status data is synced yet."
    )
    
    hof = get_hall_of_fame(latest_league_id, latest_season_year, synced_at)
    hof_leaderboard = hof["leaderboard"]
    
    st.subheader("Career Leaderboard")
    st.dataframe(
        hof_leaderboard[["manager", "career_wins", "career_losses", "career_ties", "career_points",
                         "championships", "runner_ups", "playoff_wins", "best_win_streak", "luck_index"]]
        .rename(columns={"manager": "Manager", "career_wins": "W", "career_losses": "L", "career_ties": "T",
                          "career_points": "Career Points", "championships": "Championships",
                          "runner_ups": "Runner-Ups", "playoff_wins": "Playoff Wins",
                          "best_win_streak": "Longest Win Streak", "luck_index": "Luck Index"}),
        use_container_width=True, hide_index=True,
    )
    
    col_fame, col_shame = st.columns(2)
    
    with col_fame:
        st.subheader("Hall of Fame")
        hw = hof["highest_week"]
        metric_block(st, "Highest Weekly Score", hw["team"],
                     f"{hw['points']:.1f} pts (Season {hw['season']}, Week {hw['week']})")

        if hof_leaderboard["best_win_streak"].notna().any():
            streak_row = hof_leaderboard.loc[hof_leaderboard["best_win_streak"].idxmax()]
            metric_block(st, "Longest Win Streak", streak_row["manager"],
                         f"{int(streak_row['best_win_streak'])} games")

        if hof["best_draft_team"] is not None:
            bdt = hof["best_draft_team"]
            metric_block(st, "Best Draft", f"{bdt['team']} ({bdt['season']})",
                         f"+{bdt['delta']:.0f} value over expected")

        if hof["best_pick"] is not None:
            bp = hof["best_pick"]
            metric_block(st, "Best Draft Pick", f"{bp['player']} — {bp['team']} (Pick {bp['pick_no']})",
                         f"+{bp['delta']:.0f}")

        if hof["greatest_trade"] is not None:
            gt = hof["greatest_trade"]
            metric_block(st, "Greatest Trade", f"{gt['team']} ({gt['season']})",
                         f"+{gt['net_value']:.0f} value gained")
            st.caption(escape_markdown(gt["summary"]))

    with col_shame:
        st.subheader("Hall of Shame")
        lw = hof["lowest_week"]
        metric_block(st, "Lowest Weekly Score", lw["team"],
                     f"{lw['points']:.1f} pts (Season {lw['season']}, Week {lw['week']})")

        if hof_leaderboard["best_loss_streak"].notna().any():
            loss_streak_row = hof_leaderboard.loc[hof_leaderboard["best_loss_streak"].idxmax()]
            metric_block(st, "Longest Losing Streak", loss_streak_row["manager"],
                         f"{int(loss_streak_row['best_loss_streak'])} games")

        if hof["worst_pick"] is not None:
            wp = hof["worst_pick"]
            metric_block(st, "Worst Draft Pick", f"{wp['player']} — {wp['team']} (Pick {wp['pick_no']})",
                         f"{wp['delta']:.0f}")

        if hof["worst_trade"] is not None:
            wt = hof["worst_trade"]
            metric_block(st, "Worst Trade", f"{wt['team']} ({wt['season']})", f"{wt['net_value']:.0f} value lost")
            st.caption(escape_markdown(wt["summary"]))

        if hof["biggest_choke"] is not None:
            bc = hof["biggest_choke"]
            finish_label = PLACEMENT_LABEL.get(bc["placement"], bc["placement"].replace("_", " ").title())
            metric_block(st, "Biggest Playoff Choke", f"{bc['manager']} ({bc['season']})",
                         f"Seed #{bc['seed']} → finished {finish_label}")
            st.caption(f"Entered the playoffs as the #{bc['seed']} seed but finished {finish_label} — "
                       "the biggest seed-to-finish drop in this league's history.")

        if hof["most_unlucky"] is not None:
            mu = hof["most_unlucky"]
            metric_block(st, "Most Unlucky Team", mu["manager"],
                         f"{mu['luck_index']:.1f} wins below expected (career)")

def page_league_records():
    st.header("League Records")
    st.caption(
        "The record book: single-season bests and career counting stats, as ranked leaderboards. "
        "For narrative 'best/worst moment ever' superlatives (Greatest Trade, Best Draft Pick, "
        "Biggest Choke), see Hall of Fame instead."
    )

    records = get_league_records(latest_league_id, latest_season_year, synced_at)

    st.subheader("Single-Season Records")
    col_points, col_record = st.columns(2)
    with col_points:
        mp = records["most_points_season"]
        if mp is not None:
            metric_block(st, "Most Points in a Season", f"{mp['team']} ({mp['season']})", f"{mp['points']:.1f} pts")
    with col_record:
        br = records["best_record_season"]
        if br is not None:
            record_str = f"{br['wins']}-{br['losses']}" + (f"-{br['ties']}" if br["ties"] else "")
            metric_block(st, "Best Record in a Season", f"{br['team']} ({br['season']})", record_str)

    col_hw, col_lw = st.columns(2)
    with col_hw:
        hw = records["highest_week"]
        metric_block(st, "Highest Weekly Score", hw["team"],
                     f"{hw['points']:.1f} pts (Season {hw['season']}, Week {hw['week']})")
    with col_lw:
        lw = records["lowest_week"]
        metric_block(st, "Lowest Weekly Score", lw["team"],
                     f"{lw['points']:.1f} pts (Season {lw['season']}, Week {lw['week']})")

    hof_leaderboard = records["hof_leaderboard"]
    col_ws, col_ls = st.columns(2)
    with col_ws:
        if hof_leaderboard["best_win_streak"].notna().any():
            streak_row = hof_leaderboard.loc[hof_leaderboard["best_win_streak"].idxmax()]
            metric_block(st, "Longest Win Streak", streak_row["manager"],
                         f"{int(streak_row['best_win_streak'])} games")
    with col_ls:
        if hof_leaderboard["best_loss_streak"].notna().any():
            loss_streak_row = hof_leaderboard.loc[hof_leaderboard["best_loss_streak"].idxmax()]
            metric_block(st, "Longest Losing Streak", loss_streak_row["manager"],
                         f"{int(loss_streak_row['best_loss_streak'])} games")

    if records["largest_trade"] is not None:
        lt = records["largest_trade"]
        metric_block(st, "Largest Trade", f"{' ↔ '.join(lt['teams'])} (Season {lt['season']})",
                     f"{lt['n_assets']} assets exchanged")

    st.subheader("Career Leaderboards")
    col_champs, col_finals, col_playoffs, col_trades = st.columns(4)
    with col_champs:
        st.write("**Most Championships**")
        st.dataframe(records["championships"].rename(columns={"manager": "Manager", "count": "Championships"}),
                     use_container_width=True, hide_index=True)
    with col_finals:
        st.write("**Most Finals Appearances**")
        st.dataframe(records["finals_appearances"].rename(columns={"manager": "Manager", "count": "Finals"}),
                     use_container_width=True, hide_index=True)
    with col_playoffs:
        st.write("**Most Playoff Appearances**")
        st.dataframe(records["playoff_appearances"].rename(columns={"manager": "Manager", "count": "Playoffs"}),
                     use_container_width=True, hide_index=True)
    with col_trades:
        st.write("**Most Trades**")
        st.dataframe(records["trade_counts"].rename(columns={"manager": "Manager", "trades": "Trades"}),
                     use_container_width=True, hide_index=True)

# Known off-Sleeper context tommy provided directly (family relationships don't change
# season to season, so these are hardcoded rather than run through the submission form
# below). Pairs are (display_name1, display_name2).
SIBLING_PAIRS = [("rpfau", "dpfau24"), ("D00z", "beastboy9112")]  # Ryder/Dalt, Jonah/tommy


def page_rivalries():
    st.header("Rivalries")
    st.caption(
        "Every pair of managers who've played each other, regular season and playoffs, across "
        "every synced season. Rivalry Rating rewards pairs who've played often, kept it close, "
        "and stayed competitive — a lopsided record or blowout-heavy history scores lower."
    )

    if SIBLING_PAIRS:
        st.caption("👨‍👦 Family: " + "; ".join(f"{a} & {b}" for a, b in SIBLING_PAIRS))

    rivalries = get_rivalries(synced_at)
    if rivalries.empty:
        st.info("No head-to-head history found yet.")
    else:
        st.subheader("Top Rivalries")
        st.dataframe(
            rivalries[["manager1", "manager2", "wins1", "wins2", "ties", "games_played",
                       "avg_margin", "rivalry_rating"]]
            .rename(columns={"manager1": "Manager", "manager2": "Opponent", "wins1": "W", "wins2": "L",
                              "ties": "T", "games_played": "Games", "avg_margin": "Avg Margin",
                              "rivalry_rating": "Rivalry Rating"}),
            use_container_width=True, hide_index=True,
        )
    
        st.subheader("Head-to-Head Detail")
        pair_labels = (rivalries["manager1"] + " vs " + rivalries["manager2"]).tolist()
        pair_choice = st.selectbox("Matchup", pair_labels)
        riv = rivalries.iloc[pair_labels.index(pair_choice)]
    
        record_str = f"{int(riv['wins1'])} — {int(riv['wins2'])}" + (f" ({int(riv['ties'])} ties)" if riv["ties"] else "")
        st.write(f"**Overall Record:** {escape_markdown(riv['manager1'])} {record_str} "
                 f"{escape_markdown(riv['manager2'])}")

        if riv["playoff_wins1"] or riv["playoff_wins2"]:
            st.write(f"**Playoff Record:** {escape_markdown(riv['manager1'])} {int(riv['playoff_wins1'])} — "
                     f"{int(riv['playoff_wins2'])} {escape_markdown(riv['manager2'])}")
        else:
            st.caption("No playoff meetings yet.")
    
        c1, c2, c3 = st.columns(3)
        c1.metric("Rivalry Rating", int(riv["rivalry_rating"]))
        c2.metric("Average Margin", f"{riv['avg_margin']:.1f} pts")
        c3.metric("Games Played", int(riv["games_played"]))
    
        if riv["highest_scoring_total"] is not None:
            st.write(f"**Highest Scoring Matchup:** {riv['highest_scoring_total']:.1f} combined points "
                     f"({riv['highest_scoring_season']} — {riv['highest_scoring_round']})")
        if riv["streak_manager"]:
            st.write(f"**Longest Win Streak:** {escape_markdown(riv['streak_manager'])} — "
                     f"{int(riv['streak_len'])} straight")

    st.subheader("Manager's Pick")
    st.caption("Every manager's own call on their biggest rivalry — separate from the Rivalry "
               "Rating above, which is purely algorithmic. Submitting again updates your pick.")

    name_by_owner = get_manager_name_lookup()
    owner_by_name = {name: owner for owner, name in name_by_owner.items()}
    sorted_names = sorted(name_by_owner.values())

    with st.form("rivalry_submission_form"):
        me_name = st.selectbox("You are", sorted_names)
        rival_options = [n for n in sorted_names if n != me_name]
        rival_name = st.selectbox("Your biggest rival", rival_options)
        note = st.text_input("Why? (optional)")
        if st.form_submit_button("Submit"):
            submit_rivalry(owner_by_name[me_name], owner_by_name[rival_name], note)
            st.success(f"Saved — {me_name}'s biggest rival is {rival_name}.")

    submissions = load_table("SELECT owner_id, rival_owner_id, note FROM manager_rivalries")
    if submissions.empty:
        st.caption("No one's submitted a pick yet.")
    else:
        for _, sub in submissions.iterrows():
            who = escape_markdown(name_by_owner.get(sub["owner_id"], sub["owner_id"]))
            rival = escape_markdown(name_by_owner.get(sub["rival_owner_id"], sub["rival_owner_id"]))
            line = f"**{who}** says their biggest rival is **{rival}**"
            if sub["note"]:
                line += f" — \"{escape_markdown(sub['note'])}\""
            st.write(line)

pages = [
    st.Page(page_home, title="Home", icon="🏠", default=True),
    st.Page(page_matchup_center, title="Matchup Center", icon="🆚"),
    st.Page(page_team_pages, title="Team Pages", icon="👤"),
    st.Page(page_stock_market, title="Stock Market", icon="📈"),
    st.Page(page_trade_center, title="Trade Center", icon="🔄"),
    st.Page(page_rookie_draft, title="Rookie Draft", icon="📋"),
    st.Page(page_gm_profiles, title="GM Profiles", icon="🧠"),
    st.Page(page_hall_of_fame, title="Hall of Fame", icon="🏆"),
    st.Page(page_league_records, title="League Records", icon="📜"),
    st.Page(page_rivalries, title="Rivalries", icon="⚔️"),
    st.Page(page_recent_transactions, title="Recent Transactions", icon="📰"),
]
pg = st.navigation(pages)
pg.run()

st.divider()
st.caption(
    "Not yet built: Weekly MVP and Biggest Upset (needs a win-probability model), Risk Rating "
    "and Probability of Regret (both need injury-status data, not synced yet), and "
    "AI-generated League News. These are separate features on the roadmap."
)
