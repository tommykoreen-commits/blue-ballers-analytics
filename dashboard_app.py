# Blue Ballers Analytics — Streamlit Dashboard
#
# Deployed on Streamlit Community Cloud, reading a SQLite database that this app
# downloads from Google Drive on its own (see ensure_db() below) — tommy's existing
# blue_ballers_sync.py/.ipynb notebook keeps writing to Drive exactly as it always
# has, with no extra manual step. For local/Colab testing instead of the deployed
# copy, use blue_ballers_dashboard.py, which mounts Drive directly.

import bisect
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

N_TRIALS = 10000
SHRINKAGE_GAMES = 4  # weight given to a team's own results vs. its prior-season baseline
RECENCY_DECAY = 0.9  # per-week decay on how much a past week counts toward "current form"
MAX_WEEKS_PER_SEASON = 18  # NFL's real week range (regular season + postseason). Folds a
# (season-rank, week) pair into one continuous integer sequence across season boundaries, so
# a player's recency decay doesn't reset to "week 1 = most recent" every time a new league_id
# (new season) starts. Doesn't need to be exact -- just >= any real week number that appears
# in matchups.week.
PLAYER_SHRINK_GAMES = 8  # per-PLAYER analogue of SHRINKAGE_GAMES: how hard a small real-game
# sample is pulled toward that position's median. Without it a backup who happened to be
# rostered during one hot 9-game stretch (real case: Jacoby Brissett, 19.0 PPG over 9 games,
# no byes) outranks a genuine star with a longer, more representative record.
MARKET_BLEND = 0.5  # how much of a player's projected strength comes from real external
# dynasty-market consensus vs. his own real in-league production. Production alone
# systematically misreads players whose recent real weeks were byes/injuries/bench time, and
# can't see talent a player hasn't had the chance to show yet; market consensus alone ignores
# how a player is actually producing right now. An even split scored closest to a real
# human read of the league (verified against tommy's own gut ranking of all 8 teams).
BYE_WEEK_SHARE = 1 / 17  # every NFL team sits out one week a season — real unavailability a
# "best 12 players" sum silently ignores. Simulated per NFL TEAM, not per player, because a
# bye takes every player on that team out the same week; that clustering is exactly what
# punishes a roster stacked with several starters from the same NFL team.
MAX_INJURY_RATE = 0.5  # cap on a single player's modeled injury miss rate, so someone coming
# off a lost season isn't projected as more absent than present.
BASE_RATE_MIN_GAMES = 8  # a player needs this many real games played before he counts toward
# his position's BASE injury rate. Every NFL position carries a long tail of fringe players who
# are inactive most weeks; pooling them in put the "normal" rate for a QB over 50%, which is
# nonsense for anyone who actually starts, and would have handed every unproven rookie that
# same inflated risk.
INJURY_SHRINK_GAMES = 24  # ~1.5 real seasons' worth of pull toward the positional base rate.
# A clean injury record is NOT evidence of zero future risk -- every RB carries real risk
# whatever his history, and taking observed rates at face value handed genuine workhorses a
# 0% chance of ever sitting, which quietly erased the cost of a thin bench behind them.
DEPTH_SIM_WEEKS = 150  # simulated weeks per team for the depth model. Enough to price
# multi-absence weeks (the case that actually separates a deep roster from a top-heavy one)
# without a meaningful hit to page load.
DEPTH_SIM_SEED = 20260820  # fixed so rankings don't wobble between identical page loads.
PROJECTION_UNCERTAINTY = 16.0  # points/week of honest doubt about a roster projection itself,
# applied only in proportion to how little of THIS season has actually been played. Projecting
# a roster that has never taken the field is a genuinely uncertain estimate, and treating it as
# exact truth made the preseason favorite look far more inevitable than any real projection
# should. Fades to nothing as real results accumulate and the estimate stops being a guess.

# Live/current player and pick values come from real external market consensus
# (DynastyProcess + FantasyCalc — see compute_consensus_player_values/compute_consensus_pick_value
# below). Stock Market's per-week history also reads from that same archive now (see
# build_stock_market_history/week_end_date). What's left of the old in-house heuristic is only
# the age-curve SHAPE, still load-bearing in 2 places no external source can price:
# project_team_value_curve's FUTURE age-projection (anchored to the real consensus value at
# year 0, but no external source prices a future season), and pick_value's fallback for picks far
# enough out that even FantasyCalc's coarse round-level number doesn't reach yet.
POSITION_PEAK_AGE = {"QB": 29, "RB": 25, "WR": 27, "TE": 27, "K": 30}
POSITION_DECLINE_RATE = {"QB": 0.04, "RB": 0.12, "WR": 0.07, "TE": 0.08, "K": 0.05}  # value lost per year past peak
ROUND_BASE_VALUE = {1: 100, 2: 50, 3: 25, 4: 12}

DB_PATH = "blue_ballers.db"
DB_REFRESH_SECONDS = 14400  # how often the deployed app checks Drive for a fresher sync (4h —
# tommy's data only changes when he manually re-syncs, so there's no benefit to checking more
# often, only more risk of tripping Google Drive's anonymous-download rate limit. Use the
# sidebar's "Refresh Data" button for an immediate pull right after a sync instead.

# Bump this string with every edit — shown in the sidebar so it's obvious at a glance
# whether the deployed app is actually running the latest code.
APP_BUILD = "2026-08-23-stale-copy-age"

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
            gdown.download(id=st.secrets["drive_file_id"], output=DB_PATH, quiet=True)
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
        # Say how old the fallback is: "couldn't refresh" alone doesn't tell you whether it
        # matters, and a copy pulled an hour ago is a non-event while one from last week isn't.
        age_hours = (time.time() - os.path.getmtime(DB_PATH)) / 3600
        age = (f"{age_hours:.0f} hours ago" if age_hours < 48
               else f"{age_hours / 24:.0f} days ago")
        st.warning(
            f"Couldn't refresh data from Google Drive just now (it may be temporarily "
            f"rate-limited) — showing the copy downloaded {age}. Drive's anonymous-download "
            f"limit usually clears within the hour; the sidebar's Last synced shows when that "
            f"copy was actually built."
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


# ---------------------------------------------------------------------------
# External dynasty valuation — blends DynastyProcess (value_2qb + ecr_2qb as
# an ADP proxy) and FantasyCalc into one market-consensus value, replacing
# the old in-house age-curve heuristic. blue_ballers_sync.py is the sole
# writer of these two tables (into the same Drive-hosted DB everything else
# reads from) — this app only ever reads them, going forward AND historically
# (the sync script also runs a one-time backfill from DynastyProcess's git
# history), exactly like every other table here.
#
# Reconciliation: each source's raw value is percentile-ranked against that
# SAME source's own combined players+picks pool (mirroring how DynastyProcess
# and FantasyCalc already place players and picks on one unified scale
# internally — confirmed empirically), then the percentiles are combined via
# median. This gives a 0-100 "value" on a scale directly comparable between
# players and picks, which the rest of the app depends on for trade grading.
# ---------------------------------------------------------------------------
EXTERNAL_PICK_SLOTS_PER_ROUND = 12  # DynastyProcess/FantasyCalc both label picks on the
# industry-standard 12-team draft-slot scale (1.01-1.12) regardless of query params —
# confirmed empirically (FantasyCalc still returned slots up to 1.12 with numTeams=8
# passed). This league's real 8-team slot needs mapping onto that scale before any
# lookup — see _map_to_external_pick.
EXTERNAL_PICK_TIER_SLOT = {"early": 2, "mid": 6, "late": 10}  # representative external
# slot per FantasyCalc out-year tier bucket, used as anchor points


@st.cache_data(ttl=1800)
def get_external_player_snapshots(cache_key):
    return load_table(
        "SELECT source, sleeper_player_id, scrape_date, raw_value FROM external_player_value_snapshots"
    )


@st.cache_data(ttl=1800)
def get_external_pick_snapshots(cache_key):
    return load_table(
        "SELECT source, season, round, granularity, slot, tier, scrape_date, raw_value "
        "FROM external_pick_value_snapshots"
    )


def _latest_per_source_entity(df, key_col, as_of_date=None):
    """Collapse a snapshot history down to one row per (source, key_col) —
    the most recent scrape_date on/before as_of_date, or the most recent
    overall if as_of_date is None (i.e. "current")."""
    if df.empty:
        return df
    filtered = df[df["scrape_date"] <= as_of_date] if as_of_date else df
    if filtered.empty:
        return filtered
    idx = filtered.groupby(["source", key_col])["scrape_date"].idxmax()
    return filtered.loc[idx]


def _latest_per_source_pick(df, as_of_date=None):
    if df.empty:
        return df
    filtered = df[df["scrape_date"] <= as_of_date] if as_of_date else df
    if filtered.empty:
        return filtered
    idx = filtered.groupby(["source", "season", "round", "granularity", "slot", "tier"])["scrape_date"].idxmax()
    return filtered.loc[idx]


def _external_source_pools(player_df, pick_df, as_of_date=None):
    """Per-source pooled raw-value arrays (players + that source's picks) —
    the population every individual percentile below is computed against."""
    player_latest = _latest_per_source_entity(player_df, "sleeper_player_id", as_of_date)
    pick_latest = _latest_per_source_pick(pick_df, as_of_date)
    sources = (set(player_latest["source"]) if not player_latest.empty else set()) | \
              (set(pick_latest["source"]) if not pick_latest.empty else set())

    pools = {}
    for source in sources:
        parts = []
        if not player_latest.empty:
            parts.append(player_latest.loc[player_latest["source"] == source, "raw_value"].to_numpy())
        if not pick_latest.empty:
            parts.append(pick_latest.loc[pick_latest["source"] == source, "raw_value"].to_numpy())
        pools[source] = np.sort(np.concatenate(parts)) if parts else np.array([])
    return pools, player_latest, pick_latest


def _percentile_of_value(sorted_pool, x, higher_is_better=True):
    """Percentile (0-1) of x within a sorted 1-D array via linear
    interpolation against the empirical distribution — not a discrete rank —
    so an interpolated in-between pick value still gets a sensible
    percentile instead of being forced onto an existing player's rank."""
    n = len(sorted_pool)
    if n == 0:
        return None
    pct = np.searchsorted(sorted_pool, x, side="left") / n
    return pct if higher_is_better else 1 - pct


def compute_consensus_player_values(cache_key, as_of_date=None):
    """Blended external consensus value (0-100) per Sleeper player_id, as of
    as_of_date (None = current/latest). A player absent from every source
    (rare — deep bench/practice-squad players nobody's ranking service
    tracks) simply isn't a key in the returned dict; callers should treat a
    missing player as a value-of-0 floor, not an error."""
    player_df = get_external_player_snapshots(cache_key)
    pick_df = get_external_pick_snapshots(cache_key)
    pools, player_latest, _ = _external_source_pools(player_df, pick_df, as_of_date)
    if player_latest.empty:
        return {}

    percentile_lookup = {}
    for source, group in player_latest.groupby("source"):
        pooled = pools.get(source, np.array([]))
        if len(pooled) == 0:
            continue
        higher_is_better = source != "dynastyprocess_ecr"  # ecr is lower-is-better rank
        percentile_lookup[source] = {
            pid: _percentile_of_value(pooled, val, higher_is_better)
            for pid, val in zip(group["sleeper_player_id"], group["raw_value"])
        }

    all_ids = set().union(*percentile_lookup.values()) if percentile_lookup else set()
    consensus = {}
    for pid in all_ids:
        pcts = [m[pid] for m in percentile_lookup.values() if pid in m]
        if pcts:
            consensus[pid] = round(100 * float(np.median(pcts)), 1)
    return consensus


def _map_to_external_pick(our_round, our_slot, our_team_count):
    """Map one of this league's picks onto the external 12-team scale by OVERALL pick number,
    returning (external_round, external_slot).

    By overall number, not by position within the round: the Nth pick of any rookie draft
    selects from the same depth of talent pool regardless of league size, so overall number is
    what actually transfers between a 8-team league and the 12-team scale the market prices on.
    Stretching each round proportionally instead (this league's slot 8 of 8 -> external slot 12
    of 12) broke cross-round comparisons — our last first-rounder is overall pick 8 and our
    first second-rounder is overall pick 9, adjacent picks, but proportional mapping priced them
    as a 12-team 1.12 and 2.01 and had the second-rounder come out AHEAD. Mapping by overall
    number puts them at 1.08 and 1.09, correctly near-equal and correctly ordered."""
    if our_team_count < 1:
        return our_round, float(our_slot)
    overall = (our_round - 1) * our_team_count + our_slot
    ext_round = int(np.ceil(overall / EXTERNAL_PICK_SLOTS_PER_ROUND))
    ext_slot = overall - (ext_round - 1) * EXTERNAL_PICK_SLOTS_PER_ROUND
    return ext_round, float(ext_slot)


def _overall_to_round_slot(overall):
    """External overall pick number -> (round, slot) on the external 12-slot scale."""
    ext_round = int(np.ceil(overall / EXTERNAL_PICK_SLOTS_PER_ROUND))
    return max(ext_round, 1), overall - (max(ext_round, 1) - 1) * EXTERNAL_PICK_SLOTS_PER_ROUND


def _shape_at_overall(overall):
    """In-house decay shape read at an external OVERALL pick number, so the shape can be used
    to extend a curve past the last real anchor without re-introducing a per-round seam."""
    ext_round, ext_slot = _overall_to_round_slot(overall)
    return pick_slot_value(ext_round, ext_slot)


def _value_at_overall(overall, anchors_by_overall):
    """Value at an external overall pick number, given real market anchors keyed by overall
    pick number. Between two anchors, interpolates log-linearly (geometric) directly between
    them — NOT via the in-house decay shape's curvature, which is markedly steeper than the
    market's real early-to-late gap and produced non-monotonic results when forced onto real
    anchors. Outside the anchors' span, extends by the in-house shape's *relative* decay ratio
    from the nearest anchor: real data sets the level, in-house math only extends the tail.

    Keyed by OVERALL pick rather than per-round slot so one continuous curve spans every round.
    Pricing each round against its own anchors left a seam at the round boundary — a mid-second
    could come out above an early-second one pick earlier — since the two rounds' anchor sets
    carry no guarantee of joining smoothly."""
    anchor_points = sorted(anchors_by_overall)
    if not anchor_points:
        return None
    if len(anchor_points) == 1 or overall <= anchor_points[0] or overall >= anchor_points[-1]:
        nearest = min(anchor_points, key=lambda s: abs(s - overall))
        shape_nearest = _shape_at_overall(nearest)
        if not shape_nearest:
            return anchors_by_overall[nearest]
        return anchors_by_overall[nearest] * (_shape_at_overall(overall) / shape_nearest)
    log_vals = np.log([anchors_by_overall[s] for s in anchor_points])
    return float(np.exp(np.interp(overall, anchor_points, log_vals)))


def _pick_raw_value(pick_latest, source, season, round_no, our_slot, our_team_count, higher_is_better=True):
    """One source's raw value for one of this league's specific picks.

    Every anchor this source has for that season — at whatever granularity it carries — is
    placed on a single OVERALL-pick-number axis and read as one continuous curve
    (_value_at_overall), rather than each round being priced against only its own anchors:
      - exact_slot (current draft class, both sources): the listed slot itself.
      - tier (next season out): the tier's representative slot, per EXTERNAL_PICK_TIER_SLOT.
      - round (2+ seasons out): the round's midpoint. Both sources carry these — not
        FantasyCalc alone; DynastyProcess's values.csv includes round-level entries for
        2027/2028 too, found only once real out-year data was tested (confirmed via
        `sorted(pick_df[pick_df["source"]=="dynastyprocess_value"]["season"].unique())`
        showing 2025-2028).
    Finer granularity wins where two would land on the same overall pick. Returns None when
    this source has nothing for that season at all — caller then falls back to the plain
    in-house formula for that source's contribution.

    The anchor math assumes "higher raw value = better"; for the ecr signal (lower rank =
    better) values pass through as their reciprocal so the curve math stays in a consistent
    direction, then invert back.
    """
    rows = pick_latest[(pick_latest["source"] == source) & (pick_latest["season"] == str(season))]
    if rows.empty:
        return None

    to_goodness = (lambda v: v) if higher_is_better else (lambda v: 1.0 / v)
    from_goodness = to_goodness  # reciprocal is its own inverse

    def overall_of(ext_round, ext_slot):
        return (int(ext_round) - 1) * EXTERNAL_PICK_SLOTS_PER_ROUND + float(ext_slot)

    anchors = {}
    for _, r in rows[rows["granularity"] == "exact_slot"].iterrows():
        anchors[overall_of(r["round"], r["slot"])] = to_goodness(float(r["raw_value"]))
    for _, r in rows[rows["granularity"] == "tier"].iterrows():
        if r["tier"] in EXTERNAL_PICK_TIER_SLOT:
            anchors.setdefault(overall_of(r["round"], EXTERNAL_PICK_TIER_SLOT[r["tier"]]),
                               to_goodness(float(r["raw_value"])))
    for _, r in rows[rows["granularity"] == "round"].iterrows():
        anchors.setdefault(overall_of(r["round"], (EXTERNAL_PICK_SLOTS_PER_ROUND + 1) / 2),
                           to_goodness(float(r["raw_value"])))
    if not anchors:
        return None

    ext_round, ext_slot = _map_to_external_pick(round_no, our_slot, our_team_count)
    result = _value_at_overall(overall_of(ext_round, ext_slot), anchors)
    return None if result is None else from_goodness(result)

    return None


def compute_consensus_pick_value(season, round_no, our_slot, cache_key, our_team_count=8, as_of_date=None):
    """Blended external consensus value (0-100) for one of this league's
    specific picks, on the SAME scale as compute_consensus_player_values.
    Returns None if no external source has anything for that season at all
    (e.g. picks far enough out that even FantasyCalc's coarse round-level
    number doesn't reach) — caller falls back entirely to the in-house
    pick_slot_value/CLASS_STRENGTH_MULTIPLIER formula in that case."""
    player_df = get_external_player_snapshots(cache_key)
    pick_df = get_external_pick_snapshots(cache_key)
    pools, _, pick_latest = _external_source_pools(player_df, pick_df, as_of_date)

    percentiles = []
    for source, pooled in pools.items():
        if len(pooled) == 0:
            continue
        higher_is_better = source != "dynastyprocess_ecr"
        raw = _pick_raw_value(pick_latest, source, season, round_no, our_slot, our_team_count, higher_is_better)
        if raw is None:
            continue
        percentiles.append(_percentile_of_value(pooled, raw, higher_is_better))

    if not percentiles:
        return None
    return round(100 * float(np.median(percentiles)), 1)


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


def get_week_mvp(league_id, week, players_df):
    """Best individual performance league-wide for one week: the STARTER (not
    bench — same real-lineup-only rule compute_week_positional_averages uses)
    whose points beat their own position's average that week by the most,
    across every roster in the league — not just within one matchup, unlike
    build_matchup_recap's per-team version of this same idea. Replaces the
    old get_week_top_performer, which compared raw points with no positional
    context and included bench players, so a garbage-time backup kicker
    could out-rank a real standout starter."""
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


def get_league_news(league_id, week):
    """Cached AI-generated weekly recap, written once by blue_ballers_sync.py
    at sync time -- this only ever reads a pre-written row, never calls
    Gemini itself. Bare try/except like get_draft_row/compute_games_missed_risk:
    the league_news table won't exist on the live Drive DB until tommy's next
    real Colab sync creates it, so a stale deployed app must degrade to
    "nothing to show" instead of crashing the Home page."""
    try:
        df = load_table(
            "SELECT article FROM league_news WHERE league_id = ? AND week = ?",
            params=(league_id, week),
        )
    except Exception:
        return None
    return df.iloc[0]["article"] if not df.empty else None


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


def _season_chain_up_to(league_id):
    """All league_ids in this dynasty's lineage from oldest to `league_id` itself, walking
    the previous_league_id chain all the way back instead of just one hop -- a player's real
    production from two seasons ago is still real signal for build_player_form_index below."""
    chain = [league_id]
    cur = league_id
    while True:
        prev = get_previous_league_id(cur)
        if not prev:
            break
        chain.append(prev)
        cur = prev
    return list(reversed(chain))  # oldest -> newest; league_id is always chain[-1]


def build_player_form_index(league_id, through_week=None):
    """Per-player real recent production: {player_id: (recency_weighted_ppg, games_played)}
    across all synced seasons through `league_id`/`through_week`. Feeds
    project_roster_distribution's roster-based prior below.

    Counts ONLY weeks the player actually scored (points > 0). A 0.0 entry in
    players_points overwhelmingly means "didn't play" -- bye week, inactive, IR, or a week
    that hasn't happened yet -- not "played and was shut out", and averaging those in
    silently guts real starters (real case: Trevor Lawrence's last 9 recorded weeks are all
    0.0 while he was on IR, dragging a genuine 18.4 PPG starter down to 13.8). The
    tradeoff -- a true 0-point game by a healthy starter is dropped too -- is far smaller
    than the distortion it prevents, since real 0-point starts are rare and bye/IR zeros are
    not. `games_played` is returned alongside so callers can shrink small samples
    (PLAYER_SHRINK_GAMES).

    Reuses RECENCY_DECAY exactly the way estimate_team_distributions' own own_mean does
    (weight = RECENCY_DECAY ** weeks_ago), extended across season boundaries via
    MAX_WEEKS_PER_SEASON so production isn't judged against a decay clock that resets every
    year. `weeks_ago` is measured against the cutoff itself, not each player's own last game
    -- someone who hasn't played in weeks should read as stale relative to right now.

    One query total, independent of player/team count -- safe to call once per
    estimate_team_distributions() invocation even though build_upset_history calls that once
    per historical (season, week)."""
    chain = _season_chain_up_to(league_id)
    league_rank = {lid: i for i, lid in enumerate(chain)}

    placeholders = ",".join("?" * len(chain))
    query = f"SELECT league_id, week, players_points FROM matchups WHERE league_id IN ({placeholders}) AND points > 0"
    params = list(chain)
    if through_week is not None:
        query += " AND NOT (league_id = ? AND week > ?)"
        params += [league_id, through_week]
    rows = load_table(query, params=tuple(params))

    records = []
    for _, r in rows.iterrows():
        pts_map = parse_json_field(r["players_points"], {})
        if not pts_map:
            continue
        seq = league_rank[r["league_id"]] * MAX_WEEKS_PER_SEASON + int(r["week"])
        for pid, pts in pts_map.items():
            if pts > 0:
                records.append((pid, float(pts), seq))
    if not records:
        return {}

    flat = pd.DataFrame(records, columns=["player_id", "points", "seq"])
    cutoff_seq = (league_rank[league_id] * MAX_WEEKS_PER_SEASON + through_week
                  if through_week is not None else int(flat["seq"].max()))
    flat["weight"] = RECENCY_DECAY ** (cutoff_seq - flat["seq"]).clip(lower=0)
    flat["weighted_pts"] = flat["points"] * flat["weight"]
    agg = flat.groupby("player_id").agg(
        weighted_pts=("weighted_pts", "sum"), weight=("weight", "sum"), games=("points", "size"))
    return {pid: (row["weighted_pts"] / row["weight"], int(row["games"]))
            for pid, row in agg.iterrows()}


def compute_position_baselines(player_form, players_df):
    """Per-position (median, replacement_floor) of real per-game production among players who
    DO have history. The median is the shrinkage target for players with a thin real sample
    (PLAYER_SHRINK_GAMES); the floor -- bottom quartile -- is the value used for players with
    NO synced history at all (true rookies from this year's draft). The floor is deliberately
    not 0.0 (most rookies who make an active dynasty roster are usable, not truly
    replacement-level) and deliberately not the position average (that would credit a total
    unknown with league-average production). Both self-calibrate to this league's real
    scoring settings instead of hand-picked constants needing re-tuning every year."""
    by_pos = {}
    for pid, (ppg, _games) in player_form.items():
        pos = players_df.loc[pid, "position"] if pid in players_df.index else None
        if pos:
            by_pos.setdefault(pos, []).append(ppg)
    medians = {pos: float(np.median(vals)) for pos, vals in by_pos.items() if vals}
    floors = {pos: float(np.percentile(vals, 25)) for pos, vals in by_pos.items() if vals}
    all_vals = [v for vals in by_pos.values() for v in vals]
    medians["_overall"] = float(np.median(all_vals)) if all_vals else 0.0
    floors["_overall"] = float(np.percentile(all_vals, 25)) if all_vals else 0.0
    return medians, floors


def _most_common_team(df, key_col):
    """Each player's usual NFL team that season. Counting group sizes and taking the largest is
    the same answer as a per-player mode but ~24x faster on this league's snap-count archive --
    the lambda version ran once per estimate_team_distributions call, which several pages make
    repeatedly."""
    counts = df.groupby([key_col, "team"], observed=True).size()
    return counts.groupby(level=0, observed=True).idxmax().map(lambda key: key[1])


def build_player_injury_rate(players_df):
    """Per-player probability of missing a given game, plus the per-POSITION base rate to fall
    back on. Built from snap-count ABSENCE — the same signal compute_games_missed_risk uses,
    but the raw RATE here rather than a percentile rank, since depth math needs a real
    probability. Byes are NOT included (a bye leaves no snap-count rows for the whole team, so
    those weeks aren't in the denominator either); compute_depth_adjusted_distribution
    simulates them separately, per NFL team.

    Each player's observed rate is shrunk toward his position's real base rate by
    INJURY_SHRINK_GAMES, and a player with no history at all (true rookie) gets that base rate
    outright. Taking observed rates at face value is the trap here: a genuine workhorse who
    happens not to have missed a game reads as 0% risk forever, which makes a thin bench behind
    him look free. Positions differ enough in real risk (a RB is not a QB) that one league-wide
    base rate would be worse than a per-position one.

    `snap_counts` is a newer table — same missing-table fallback as get_draft_row/
    compute_games_missed_risk (an empty result leaves the depth model with byes only)."""
    try:
        df = load_table("SELECT season, week, sleeper_player_id, team FROM snap_counts")
    except Exception:
        return {}, {}
    if df.empty:
        return {}, {}

    missed, played = {}, {}
    for _season, season_df in df.groupby("season"):
        team_weeks = season_df.groupby("team")["week"].apply(set).to_dict()
        player_weeks = season_df.groupby("sleeper_player_id")["week"].apply(set)
        player_team = _most_common_team(season_df, "sleeper_player_id")
        for pid, weeks_played in player_weeks.items():
            real_weeks = team_weeks.get(player_team[pid], set())
            missed[pid] = missed.get(pid, 0) + len(real_weeks - weeks_played)
            played[pid] = played.get(pid, 0) + len(weeks_played)

    pos_missed, pos_total = {}, {}
    for pid, miss in missed.items():
        pos = players_df.loc[pid, "position"] if pid in players_df.index else None
        if not pos or played.get(pid, 0) < BASE_RATE_MIN_GAMES:
            continue
        pos_missed[pos] = pos_missed.get(pos, 0) + miss
        pos_total[pos] = pos_total.get(pos, 0) + miss + played.get(pid, 0)
    base_by_pos = {pos: pos_missed[pos] / pos_total[pos] for pos in pos_total if pos_total[pos]}
    overall = (sum(pos_missed.values()) / sum(pos_total.values())) if sum(pos_total.values()) else 0.0
    base_by_pos["_overall"] = overall

    rates = {}
    for pid, miss in missed.items():
        total = miss + played.get(pid, 0)
        if not total:
            continue
        pos = players_df.loc[pid, "position"] if pid in players_df.index else None
        base = base_by_pos.get(pos, overall)
        shrunk = (miss + base * INJURY_SHRINK_GAMES) / (total + INJURY_SHRINK_GAMES)
        rates[pid] = min(shrunk, MAX_INJURY_RATE)
    return rates, base_by_pos


def compute_depth_adjusted_distribution(entries, roster_positions, injury_rates, injury_base,
                                         nfl_team_of, rng):
    """Expected weekly points AND week-to-week spread once byes and injuries are priced in,
    instead of assuming every starter plays every week. Simulates DEPTH_SIM_WEEKS weeks: each
    NFL team independently on bye (BYE_WEEK_SHARE), each remaining player independently out at
    his own real injury rate, then the best lineup from whoever's left.

    This is what makes depth matter, and why it's simulated rather than priced one absence at
    a time: absences overlap. A bye takes out every player from the same NFL team at once, and
    injuries land on top of that. A deep roster plugs the hole with someone comparable; a
    top-heavy one is forced down to replacement level, and the second and third simultaneous
    hole cost it far more than the first — a convexity that per-player expected values can't
    see. The same simulation yields the spread, so a thin roster also correctly reads as more
    volatile week to week, not just slightly worse on average."""
    if not entries:
        return 0.0, 0.0
    nfl_teams = {nfl_team_of.get(e[2]) for e in entries if nfl_team_of.get(e[2])}
    # Resolved once per roster, not once per simulated week: a player with no snap-count
    # history still carries his position's real base risk, never zero.
    out_rate = [injury_rates.get(e[2], injury_base.get(e[1], injury_base.get("_overall", 0.0)))
                for e in entries]
    totals = []
    for _ in range(DEPTH_SIM_WEEKS):
        on_bye = {t for t in nfl_teams if rng.random() < BYE_WEEK_SHARE}
        available = [e for i, e in enumerate(entries)
                     if nfl_team_of.get(e[2]) not in on_bye
                     and rng.random() >= out_rate[i]]
        totals.append(compute_optimal_lineup_score(roster_positions, available))
    return float(np.mean(totals)), float(np.std(totals))


def project_player_strength(pid, pos, player_form, medians, floors, market_values):
    """One player's projected per-week points: his own real production (shrunk toward the
    positional median by how few real games back it) blended MARKET_BLEND with real external
    dynasty-market consensus. The market leg is a 0-100 percentile, mapped onto a points-like
    scale (2x the positional median, so a 50th-percentile player lands near median
    production) purely so the optimal-lineup solver can combine the two legs on one scale."""
    median = medians.get(pos, medians.get("_overall", 0.0))
    if pid in player_form:
        ppg, games = player_form[pid]
        k = games / (games + PLAYER_SHRINK_GAMES)
        production = k * ppg + (1 - k) * median
    else:
        production = floors.get(pos, floors.get("_overall", 0.0))
    market = (market_values.get(pid, 0.0) / 100.0) * max(median * 2.0, 1.0)
    return (1 - MARKET_BLEND) * production + MARKET_BLEND * market


@st.cache_data(ttl=1800)
def build_projection_inputs(league_id, through_week=None):
    """Everything needed to project any roster in this league, gathered once: real per-player
    production, positional baselines, market values, injury rates, and NFL teams. Shared by
    estimate_team_distributions (Power Rankings / odds / upsets) and build_league_grades (Team
    Pages), so a team's grade and its championship odds can't disagree about how good its
    roster is.

    Market values are read as of this league's own season end — for the CURRENT (in-progress)
    season that's a future date, so it resolves to the latest available snapshot, while for a
    past season it avoids pricing that season's rosters at today's values."""
    season_row = load_table("SELECT season, synced_at FROM league_seasons WHERE league_id = ?",
                             params=(league_id,))
    players_df = get_players_df()
    player_form = build_player_form_index(league_id, through_week=through_week)
    medians, floors = compute_position_baselines(player_form, players_df)
    injury_rates, injury_base = build_player_injury_rate(players_df)
    return {
        "roster_positions": get_roster_positions(league_id),
        "players_df": players_df,
        "player_form": player_form,
        "medians": medians,
        "floors": floors,
        "market_values": compute_consensus_player_values(
            season_row.iloc[0]["synced_at"] if not season_row.empty else None,
            as_of_date=season_end_as_of_date(season_row.iloc[0]["season"]) if not season_row.empty else None,
        ),
        "injury_rates": injury_rates,
        "injury_base": injury_base,
        "nfl_team_of": players_df["team"].to_dict(),
    }


def build_roster_entries(league_id, roster_id, proj):
    """(projected_points, position, player_id) for every currently-rostered player — the input
    shape the optimal-lineup solver and depth simulation both consume."""
    player_ids, _ = get_roster(league_id, roster_id)
    players_df = proj["players_df"]
    entries = []
    for pid in player_ids:
        pos = players_df.loc[pid, "position"] if pid in players_df.index else None
        if not pos:
            continue
        strength = project_player_strength(pid, pos, proj["player_form"], proj["medians"],
                                           proj["floors"], proj["market_values"])
        entries.append((strength, pos, pid))
    return entries


def project_roster_distribution(league_id, roster_id, proj, rng):
    """Current-roster-based (expected weekly points, absence-driven spread), replacing 'last
    season's team average' as estimate_team_distributions' prior: project each currently-
    rostered player (project_player_strength), then run them through the bye/injury depth
    simulation (compute_depth_adjusted_distribution) so roster DEPTH counts. Uses
    `rosters.players` -- the LIVE current roster for `league_id`'s own season, so an offseason
    trade shows up here immediately, before any game has been played under the new roster.

    Known, accepted limitation: for a PAST season's league_id this reads that season's
    last-synced (~end-of-season) roster, not a true point-in-time-at-that-week snapshot --
    only user-visible in build_upset_history's historical backtest, where an in-season trade
    that happened later in a past season leaks backward into that season's earlier-week
    upset probabilities."""
    return compute_depth_adjusted_distribution(
        build_roster_entries(league_id, roster_id, proj), proj["roster_positions"],
        proj["injury_rates"], proj["injury_base"], proj["nfl_team_of"], rng)


def estimate_team_distributions(league_id, standings, through_week=None, project_rosters=True):
    """Per-roster (mean, std) for a team's weekly score, blending this season's
    own results with a roster-composition-based prior (more prior weight early
    in the season, less as more of this season's games are in the books).
    The mean's prior leg is a projection of the CURRENT roster — each rostered
    player's real recent production blended with real external dynasty-market
    consensus (project_roster_distribution / project_player_strength), combined via the
    same optimal-lineup solver behind the Max PF stat — instead of last
    season's team-level average. A trade made before any games this season now
    moves the projection immediately, rather than waiting for the new players
    to rack up real games under this roster. The mean is recency-weighted at both the player level
    (build_player_form_index) and, for own_mean, the team level, so the
    projection reflects current form, not a flat season average. The prior's
    spread combines three real, independent sources of week-to-week variation:
    this manager's own scoring history, how hard byes/injuries swing a
    top-heavy roster (absence_std), and how uncertain any projection of a
    roster that hasn't played yet inherently is (PROJECTION_UNCERTAINTY, which
    fades as real results come in).
    `through_week` caps the window, e.g. to reconstruct last week's power
    rankings for a week-over-week movement indicator. Checked against
    `is not None`, not plain truthiness — through_week=0 (a real, meaningful
    value meaning "before week 1 has happened, use only the prior") is falsy
    in Python, and a naive `if through_week` would treat it as "no cap" and
    leak the entire season into what's supposed to be a pre-week-1
    estimate.

    `project_rosters=False` falls back to the older prior — this manager's own
    prior-season average — and skips the roster projection entirely. That's
    the right call for BACKTESTS over already-played weeks
    (build_upset_history): `rosters` only holds each season's last-synced
    roster, so projecting a past week from it would leak trades that hadn't
    happened yet into that week's pre-game odds. It also keeps those callers
    fast, since they re-estimate once per historical week."""
    prev_league_id = get_previous_league_id(league_id)
    week_filter = " AND week <= ?" if through_week is not None else ""
    week_params = (through_week,) if through_week is not None else ()
    league_scores = load_table(
        f"SELECT points FROM matchups WHERE league_id = ? AND points > 0{week_filter}",
        params=(league_id, *week_params),
    )["points"].tolist()
    fallback_mean = float(np.mean(league_scores)) if league_scores else 100.0
    fallback_std = float(np.std(league_scores)) if len(league_scores) >= 8 else 25.0

    if project_rosters:
        proj = build_projection_inputs(league_id, through_week=through_week)
        depth_rng = np.random.default_rng(DEPTH_SIM_SEED)

    distributions = {}
    for _, row in standings.iterrows():
        rid = int(row["roster_id"])
        own_rows = load_table(
            f"SELECT week, points FROM matchups WHERE league_id = ? AND roster_id = ? AND points > 0{week_filter}",
            params=(league_id, rid, *week_params),
        )
        own_scores = own_rows["points"].to_numpy()

        games = len(own_scores)
        weight = games / (games + SHRINKAGE_GAMES)

        prior_scores = get_owner_scores(prev_league_id, row["owner_id"])
        if project_rosters:
            prior_mean, absence_std = project_roster_distribution(league_id, rid, proj, depth_rng)
        else:
            prior_mean = float(np.mean(prior_scores)) if prior_scores else fallback_mean
            absence_std = 0.0
        scoring_std = float(np.std(prior_scores)) if len(prior_scores) >= 3 else fallback_std
        # Independent sources of week-to-week spread, added in quadrature: how much this
        # manager's scores have really bounced around, how much a thin roster swings when
        # byes/injuries hit (absence_std), and — while the projection is still mostly a
        # projection — how uncertain the roster estimate itself is. A top-heavy roster is
        # genuinely more volatile than its scoring history alone implies, and a roster that
        # hasn't played yet is genuinely less knowable than one that has.
        projection_doubt = (1 - weight) * PROJECTION_UNCERTAINTY if project_rosters else 0.0
        prior_std = float(np.sqrt(scoring_std ** 2 + absence_std ** 2 + projection_doubt ** 2))

        if games:
            weeks_ago = own_rows["week"].max() - own_rows["week"].to_numpy()
            recency_weights = RECENCY_DECAY ** weeks_ago
            own_mean = float(np.average(own_scores, weights=recency_weights))
            own_std = float(np.std(own_scores)) if games >= 2 else prior_std
        else:
            own_mean, own_std = 0.0, prior_std

        mean = weight * own_mean + (1 - weight) * prior_mean
        std = weight * own_std + (1 - weight) * prior_std

        distributions[rid] = (mean, max(std, 1.0))
    return distributions


FLEX_ELIGIBLE = {"FLEX": {"RB", "WR", "TE"}, "SUPER_FLEX": {"QB", "RB", "WR", "TE"}}


def get_roster_positions(league_id):
    df = load_table("SELECT roster_positions FROM league_seasons WHERE league_id = ?", params=(league_id,))
    return parse_json_field(df.iloc[0]["roster_positions"], []) if not df.empty else []


def optimal_lineup_picks(roster_positions, entries):
    """The (points, position) entries a best-possible lineup would actually start, for one
    team-week: greedily fills the most position-restrictive slots first (QB/RB/WR/TE/K), then
    FLEX (RB/WR/TE), then SUPER_FLEX (any offensive position) last. Because FLEX's eligible
    set is a subset of SUPER_FLEX's, this restrictive-first order is provably optimal here,
    not just a heuristic. Split out from compute_optimal_lineup_score (which is just its sum)
    so callers that need to know WHICH players start — e.g. the depth simulation,
    which prices what happens when one of them is out — don't have to re-derive it.

    Each entry is indexed, not unpacked, so callers may pass either a plain (points, position)
    pair or a longer tuple carrying extra fields (e.g. a player_id) straight through."""
    starting_slots = [s for s in roster_positions if s != "BN"]
    slot_order = sorted(starting_slots, key=lambda s: {"FLEX": 1, "SUPER_FLEX": 2}.get(s, 0))

    available = sorted(entries, key=lambda e: -e[0])
    used = [False] * len(available)
    picks = []

    for slot in slot_order:
        eligible = FLEX_ELIGIBLE.get(slot, {slot})
        for i, entry in enumerate(available):
            if not used[i] and entry[1] in eligible:
                used[i] = True
                picks.append(entry)
                break
    return picks


def compute_optimal_lineup_score(roster_positions, entries):
    """Total points of the best possible lineup for one team-week (see
    optimal_lineup_picks for the slot-filling logic)."""
    return float(sum(entry[0] for entry in optimal_lineup_picks(roster_positions, entries)))


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


UPSET_TRIALS = 2000  # per-matchup Monte Carlo sample size for a pre-game win probability —
# cheap (a single vectorized Gamma draw), so no need for N_TRIALS-scale precision here


def build_upset_history(cache_key):
    """Every completed regular-season matchup ever synced, with the actual
    winner's REAL pre-game win probability — not a post-hoc "how close was
    it" number, but what the season simulator's own distribution/sampling
    machinery would have predicted strictly BEFORE that week happened
    (estimate_team_distributions(..., through_week=week-1) + sample_weekly_
    scores, the exact same building blocks run_simulation uses to project
    undecided weeks, just pointed at an already-decided one). A team that
    wins with a low winner_pregame_prob was a real underdog that day."""
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
            # project_rosters=False: this is a backtest of an already-played week, and
            # `rosters` only has each season's last-synced roster -- projecting from it would
            # leak trades made later that season into this week's pre-game odds.
            distributions = estimate_team_distributions(lid, standings_local,
                                                         through_week=week - 1,
                                                         project_rosters=False)
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


@st.cache_data(ttl=1800)
def get_upset_history(cache_key):
    return build_upset_history(cache_key)


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


def build_value_table(league_id, season_year, players_df, cache_key):
    """Current roster value table — sourced from real external market
    consensus (compute_consensus_player_values) rather than the in-house
    age-curve/PPG heuristic. A player with zero external coverage at all
    (rare — deep bench/practice-squad guys nobody's ranking service tracks)
    floors to 0.0 rather than crashing."""
    rostered_ids = get_all_rostered_player_ids(league_id)
    relevant = players_df.loc[players_df.index.intersection(rostered_ids)]
    consensus = compute_consensus_player_values(cache_key)

    rows = []
    for pid, info in relevant.iterrows():
        age = compute_age(info["birth_date"], season_year)
        value = consensus.get(pid, 0.0)
        rows.append({"player_id": pid, "full_name": info["full_name"], "position": info["position"],
                      "age": age, "value": value})
    return pd.DataFrame(rows).set_index("player_id")


@st.cache_data(ttl=1800)
def get_value_table(league_id, season_year, cache_key):
    return build_value_table(league_id, season_year, get_players_df(), cache_key)


def season_end_as_of_date(season_year):
    """Representative as-of-date for grading a whole season's worth of trades
    uniformly — the archive is weekly-granular, but this table is built ONCE
    per season and reused across every trade that season for efficiency, so
    it can't give each trade its own exact date (see get_value_as_of for that
    — used by the "how this trade aged" comparison, which grades one trade
    at a time). Using the season's END still fully solves the actual
    complaint (hindsight bias from grading old trades at TODAY's price) even
    without within-season precision."""
    return f"{season_year}-12-31"


def build_historical_value_table(season_year, players_df, cache_key):
    """Value table for grading HISTORICAL picks/trades — spans every player ever
    rostered in this league, not just currently-rostered ones. Sourced from the
    real external consensus archive as of that season's end (not today's price,
    which would grade old trades through today's hindsight-colored market).

    Requires the archive to actually reach back that far — DynastyProcess's
    one-time git-history backfill (see blue_ballers_sync.py) covers 2024-01-01
    onward, which is this league's full real history, so this should never be
    an issue in production. If it's ever queried for an earlier/uncovered
    date, every player floors to 0.0 (not a crash) — a real backfill gap
    would show up as an entire season's table coming back all-zero, which is
    a visible, checkable symptom rather than a silent wrong number."""
    all_ids = get_all_ever_rostered_ids()
    relevant = players_df.loc[players_df.index.intersection(all_ids)]
    consensus = compute_consensus_player_values(cache_key, as_of_date=season_end_as_of_date(season_year))

    rows = []
    for pid, info in relevant.iterrows():
        age = compute_age(info["birth_date"], season_year)
        value = consensus.get(pid, 0.0)
        rows.append({"player_id": pid, "full_name": info["full_name"], "position": info["position"],
                      "age": age, "value": value})
    return pd.DataFrame(rows).set_index("player_id")


@st.cache_data(ttl=1800)
def get_historical_value_table(latest_season_year, cache_key):
    return build_historical_value_table(latest_season_year, get_players_df(), cache_key)


# ---------------------------------------------------------------------------
# Stock Market — every ever-rostered player's REAL market value as a genuine
# historical price series, sourced from the same external consensus archive
# as everywhere else (compute_consensus_player_values), looked up as of each
# past week's own date rather than replayed at today's price. A player only
# gets a row for weeks where they'd actually recorded real production in this
# league by then (build_player_ppg_timeline's real-week gate) — the PPG
# number itself no longer drives the VALUE, only which weeks a player shows
# up in at all, since the external consensus already reflects the market's
# own read on age/performance/opportunity. Unlike the old in-house heuristic,
# this genuinely doesn't reset at season boundaries — real dynasty value
# carries across seasons, so the timeline is continuous 2024 through today.
# ---------------------------------------------------------------------------
def build_player_ppg_timeline():
    """Cumulative PPG per player at every real (season, week) point — no longer
    used for VALUE (see build_stock_market_history), only to gate which
    (season, week, player_id) rows exist at all: a player shouldn't show up
    in a week they hadn't actually recorded real production in this league
    yet. A week only counts once its matchups have real scoring (SUM(points)
    > 0), the same bar get_last_completed_week uses elsewhere — Sleeper
    pre-populates future weeks' matchup rows with each roster's player list
    and all-zero players_points before real games happen, and without this
    gate those zero-filled weeks would masquerade as real (if quiet) games."""
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


SEASON_WEEK1_THURSDAY = {
    "2024": "2024-09-05",
    "2025": "2025-09-04",
}  # this league's real NFL season openers (Thursday Night Kickoff) — manually
# maintained per season, same pattern as CLASS_STRENGTH_MULTIPLIER. Add next
# year's date here once it's known; only affects week_end_date's precision
# for that season, nothing else.


def season_opener_date(season):
    """The season's real Thursday-night opener. An explicit SEASON_WEEK1_THURSDAY entry wins;
    otherwise it's derived as the Thursday after Labor Day (the first Monday of September),
    which reproduces every verified real opener above — so a brand-new season no longer falls
    back to a coarse Jan-1 anchor before someone remembers to add its date by hand."""
    listed = SEASON_WEEK1_THURSDAY.get(str(season))
    if listed:
        return datetime.strptime(listed, "%Y-%m-%d").date()
    labor_day = datetime(int(season), 9, 1).date()
    while labor_day.weekday() != 0:  # 0 = Monday
        labor_day += timedelta(days=1)
    return labor_day + timedelta(days=3)


def is_offseason_move(season, created_ms):
    """Whether a transaction happened before that season's first real game. Sleeper stamps
    offseason moves as week 1, indistinguishable from a genuine in-season week 1 trade by week
    number alone, so this compares the real timestamp against the real opener."""
    if created_ms is None or pd.isna(created_ms):
        return False
    try:
        opener = season_opener_date(season)
    except (TypeError, ValueError):
        return False
    moved = datetime.fromtimestamp(int(created_ms) / 1000, tz=timezone.utc).date()
    return moved < opener


def week_label(season, week, created_ms=None):
    """"Off" for an offseason move, otherwise "Wk<n>" — offseason trades showing as "Wk1" read
    as though they happened during the season's opening week."""
    if is_offseason_move(season, created_ms):
        return "Off"
    return f"Wk{week}" if week else "Off"


def week_end_date(season, week):
    """Approximate real-world date by which a fantasy week's games are fully
    played out (the Tuesday after that week's Monday Night Football game) —
    used only to pick the nearest real external-market snapshot for that
    week in Stock Market, not any actual scheduling logic, so being off by a
    day or two doesn't matter."""
    return (season_opener_date(season) + timedelta(days=5 + (int(week) - 1) * 7)).isoformat()


def build_stock_market_history(cache_key):
    ppg_timeline = build_player_ppg_timeline()
    if ppg_timeline.empty:
        return pd.DataFrame(columns=["season", "week", "player_id", "value"])

    rows = []
    for (season, week), group in ppg_timeline.groupby(["season", "week"]):
        consensus = compute_consensus_player_values(cache_key, as_of_date=week_end_date(season, week))
        for pid in group["player_id"]:
            rows.append({"season": season, "week": week, "player_id": pid, "value": consensus.get(pid, 0.0)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def get_stock_market_history(cache_key):
    return build_stock_market_history(cache_key)


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


GRADE_Z_BINS = [(1.00, "A"), (0.40, "B"), (0.00, "C+"), (-0.40, "C"), (-1.00, "D")]


def zscore_to_grade(series):
    """Grade on how far a team really sits from the league average, in standard deviations —
    not on its rank. With only 8 teams, rank percentiles are locked to 0.125/0.25/0.375/...
    no matter whether last place is a hair behind or in a different league entirely, and
    averaging several of those ranks squeezes everyone toward the middle: the genuinely worst
    roster in this league was landing a C, tied with a clearly better team. Magnitude-based
    bands let a real gap show up as a real grade gap."""
    std = series.std(ddof=0)
    z = (series - series.mean()) / std if std and not pd.isna(std) else series * 0.0

    def band(v):
        for threshold, label in GRADE_Z_BINS:
            if v >= threshold:
                return label
        return "F"

    return z.apply(band)


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

    return df.style.map(_color, subset=columns)


def build_asset_worth_curve(cache_key, as_of_date=None):
    """Sorted array of real market values (FantasyCalc's own numbers, players and picks
    pooled), used to turn a 0-100 consensus PERCENTILE back into something proportional to
    real worth before assets get added together.

    Percentiles must never be summed directly: they rank, they don't measure. On the pooled
    distribution a near-worthless 4th-round rookie pick still lands around the 60th percentile
    simply by outranking thousands of irrelevant players, so adding up a dozen of them
    manufactured a fortune out of junk. Mapping each percentile back onto the real value
    distribution fixes that using actual market data, rather than a hand-picked curve — the
    real curve is flatter at the very top and steeper through the middle than any simple
    exponent reproduces (checked against the live snapshots)."""
    player_df = get_external_player_snapshots(cache_key)
    pick_df = get_external_pick_snapshots(cache_key)
    pools, _players, _picks = _external_source_pools(player_df, pick_df, as_of_date)
    for source in ("fantasycalc", "dynastyprocess_value"):
        pool = pools.get(source)
        if pool is not None and len(pool):
            return np.asarray(pool, dtype=float)
    return None


def asset_worth(percentile, worth_curve):
    """One asset's approximate real market value, from its 0-100 consensus percentile."""
    if worth_curve is None or not len(worth_curve):
        return float(percentile)
    return float(np.quantile(worth_curve, min(max(percentile, 0.0), 100.0) / 100.0))


FUTURE_CORE_PLAYERS = 15  # how deep "future value" counts: roughly a full starting lineup plus
# immediate backups. Summing value across the WHOLE roster instead graded roster QUANTITY —
# a team carrying 31 middling players outscored a top-heavy contender and landed a C while the
# league's strongest roster got a D, because a superstar counted the same as two spare parts.
# Bench depth is graded on how much of its best lineup a roster RETAINS once byes and injuries
# are simulated, not on the raw point total of its top few reserves. Those are very different
# things: a roster can carry a pile of useful-looking backups at one position and still collapse,
# because a spare receiver cannot fill a running back slot and the drop-off behind a star is what
# actually costs points. Retention measures the insulation itself.
CONTEND_WEIGHT = 0.5  # split between win-now strength and future value in the overall grade.


def build_league_grades(league_id, value_table, standings, season_year=None, cache_key=None):
    """Team grades from two real, separately-graded halves: win-now strength (the projected
    starting lineup, depth-discounted — the same projection behind Power Rankings and
    championship odds, so a team's grade can't contradict its odds) and future value
    (youth-adjusted market value of its core PLUS the real market value of the future draft
    picks it owns, since in a dynasty league picks are a large part of what a rebuilding team
    is actually holding — grading future value off rostered players alone badly understates
    exactly the teams whose future IS their pick stash)."""
    proj = build_projection_inputs(league_id)
    rng = np.random.default_rng(DEPTH_SIM_SEED)
    worth_curve = build_asset_worth_curve(cache_key) if cache_key is not None else None

    records = []
    for _, row in standings.iterrows():
        roster_id = int(row["roster_id"])
        player_ids, starter_ids = get_roster(league_id, roster_id)
        metrics = team_overview_metrics(value_table, player_ids, starter_ids)
        metrics["roster_id"] = roster_id

        entries = build_roster_entries(league_id, roster_id, proj)
        contend, _spread = compute_depth_adjusted_distribution(
            entries, proj["roster_positions"], proj["injury_rates"], proj["injury_base"],
            proj["nfl_team_of"], rng)
        metrics["contender_score"] = contend

        healthy = compute_optimal_lineup_score(proj["roster_positions"], entries)
        metrics["starter_value"] = contend
        metrics["healthy_lineup"] = healthy
        # Share of its best lineup this roster keeps once byes/injuries are simulated. See the
        # BENCH_DEPTH note above for why this, and not a sum of the top few reserves.
        metrics["bench_value"] = (contend / healthy) if healthy else 0.0

        team_values = value_table.loc[value_table.index.intersection(player_ids)]
        youth_adjusted = team_values["value"] * (
            1 + (27 - team_values["age"].fillna(27)).clip(lower=0) * 0.03)
        metrics["core_value"] = float(sum(
            asset_worth(v, worth_curve) for v in youth_adjusted.nlargest(FUTURE_CORE_PLAYERS)))
        records.append(metrics)

    league_df = pd.DataFrame(records).set_index("roster_id")

    # Future draft capital, valued on the same 0-100 market scale as players so the two add up.
    # Each pick is priced off how strong the team it originally belongs to projects to be
    # (a rebuilding team's own first is worth far more than a contender's), using this
    # league's own projected strength rather than a second, separate power model.
    contend_pct_for_picks = league_df["contender_score"].rank(pct=True).to_dict()
    pick_capital = {rid: 0.0 for rid in league_df.index}
    if season_year is not None:
        inventory = build_pick_inventory(league_id, season_year, standings)
        for _, pick in inventory.iterrows():
            owner = int(pick["owner_roster_id"])
            if owner not in pick_capital:
                continue
            pick_capital[owner] += asset_worth(pick_value(
                int(pick["round"]),
                contend_pct_for_picks.get(int(pick["original_roster_id"]), 0.5),
                season=pick["season"], cache_key=cache_key), worth_curve)
    league_df["pick_capital"] = pd.Series(pick_capital)
    league_df["future_score"] = league_df["core_value"] + league_df["pick_capital"]

    contend_std = league_df["contender_score"].std(ddof=0)
    future_std = league_df["future_score"].std(ddof=0)
    contend_z = ((league_df["contender_score"] - league_df["contender_score"].mean()) / contend_std
                  if contend_std else league_df["contender_score"] * 0.0)
    future_z = ((league_df["future_score"] - league_df["future_score"].mean()) / future_std
                 if future_std else league_df["future_score"] * 0.0)
    league_df["contender_pct"] = league_df["contender_score"].rank(pct=True)
    league_df["future_pct"] = league_df["future_score"].rank(pct=True)

    composite = CONTEND_WEIGHT * contend_z + (1 - CONTEND_WEIGHT) * future_z
    league_df["overall_grade"] = zscore_to_grade(composite)
    league_df["starter_grade"] = zscore_to_grade(league_df["starter_value"])
    league_df["bench_grade"] = zscore_to_grade(league_df["bench_value"])
    return league_df


@st.cache_data(ttl=1800)
def get_league_grades(league_id, season_year, cache_key):
    value_table = get_value_table(league_id, season_year, cache_key)
    standings_local = get_standings(league_id)
    return build_league_grades(league_id, value_table, standings_local,
                                season_year=season_year, cache_key=cache_key)


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
    projected win-now lineup strength vs. age-adjusted value of its core — percentile
    ranks rather than the raw scores because the two are on entirely different units
    (projected points per week vs. market value), so comparing them directly would be
    meaningless. Returns "contend", "rebuild", or "balanced"."""
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


def project_team_value_curve(league_id, roster_id, season_year, players_df, cache_key, years=5):
    """Per-POSITION value trajectory, not a single aggregate line — an aggregate
    curve looks qualitatively similar for almost every roster (ages blend into a
    smooth average decline regardless of team), while per-position lines actually
    show what makes each team's aging profile distinct (e.g. a young WR corps
    holding value while an aging RB room craters in 2 years).

    Year 0 is anchored to each player's real external consensus value (the
    same number shown everywhere else). No external source prices a FUTURE
    season, so years 1+ scale that anchor by the in-house age-curve's
    *relative* decay ratio (age_curve_multiplier(age+offset) / age_curve_
    multiplier(age)) — real market data sets the level, in-house math only
    projects the shape forward, same anchor+shape pattern used for out-year
    pick values in compute_consensus_pick_value."""
    player_ids, _ = get_roster(league_id, roster_id)
    relevant = players_df.loc[players_df.index.intersection(player_ids)]
    consensus = compute_consensus_player_values(cache_key)
    base_ages = {pid: compute_age(info["birth_date"], season_year) for pid, info in relevant.iterrows()}

    trajectory = []
    for offset in range(years + 1):
        year = season_year + offset
        pos_totals = {}
        for pid, info in relevant.iterrows():
            base_age = base_ages[pid]
            anchor = consensus.get(pid)
            if base_age is None or anchor is None:
                continue
            pos = info["position"] or "Other"
            base_shape = age_curve_multiplier(info["position"], base_age)
            value = anchor * (age_curve_multiplier(info["position"], base_age + offset) / base_shape) if base_shape else 0.0
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


def rookie_draft_is_complete(league_id, season_year):
    """Whether this season's rookie draft has already been held — real Sleeper draft status,
    not a date guess."""
    df = load_table(
        "SELECT status FROM drafts WHERE league_id = ? AND season = ?",
        params=(league_id, str(season_year)),
    )
    return not df.empty and str(df.iloc[0]["status"]).lower() == "complete"


def first_unspent_pick_season(league_id, season_year):
    """The earliest season whose picks are still real, tradeable assets. Once this season's
    rookie draft is complete its picks have become players and are no longer future capital —
    listing them as draft capital double-counts the roster they already turned into."""
    return season_year + 1 if rookie_draft_is_complete(league_id, season_year) else season_year


def build_pick_inventory(league_id, season_year, standings, seasons_ahead=3):
    draft_rounds = get_draft_rounds(league_id)
    traded = get_traded_picks_map(league_id)
    roster_ids = [int(rid) for rid in standings["roster_id"].tolist()]
    first_season = first_unspent_pick_season(league_id, season_year)

    picks = []
    for offset in range(seasons_ahead):
        pick_season = str(first_season + offset)
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


CLASS_STRENGTH_MULTIPLIER = {
    "2026": 0.88,  # weak class — legit top tier (Love/Tate/Lemon/Tyson) but a steep
                    # cliff after that; thin QB/TE, shaky RB depth beyond Love
    "2027": 1.12,  # strong class — "historic" QB class (Manning/Lagway/Moore) + the
                    # deepest WR class in recent memory (Jeremiah Smith headlines it);
                    # KeepTradeCut is already pricing late 2027 1sts above mid 2026 1sts
}  # research-backed via WebSearch (2026-08-17): FFToday, DraftSharks, FantasyPros,
# Yahoo Sports, FantasyLife all converge on this same direction independently, and it
# shows up in real KTC trade pricing, not just punditry — see project memory for the
# full source list. Any season not listed (including 2028+, where commentary is still
# "too early"/pure recruiting-hype speculation) defaults to a neutral 1.0x rather than
# guessing a direction with no real consensus behind it.


def pick_value(round_no, original_roster_power_percentile, slot_distribution=None, season=None,
                cache_key=None, our_team_count=8, as_of_date=None):
    """Value of a future (undrafted) pick — prefers real external market
    consensus (compute_consensus_pick_value) and falls back to the in-house
    pick_slot_value/CLASS_STRENGTH_MULTIPLIER formula only where external
    sources don't reach yet (picks far enough out that even FantasyCalc's
    coarse round-level number doesn't cover them — in practice this is only
    ~4+ seasons out, well beyond what's normally displayed). Pass
    cache_key=None (the default) to skip the consensus lookup entirely and
    always use the in-house formula — used by callers not yet wired to a
    live/historical value source. `as_of_date=None` means "current" — pass
    an ISO date to grade a pick's value as of a historical trade instead.

    CLASS_STRENGTH_MULTIPLIER applies ONLY to the in-house fallback: real
    external prices already bake in the market's own class-strength read
    (that's WHY DynastyProcess/FantasyCalc price a 2027 pick differently
    from a 2026 one) — applying our own multiplier on top of an
    already-market-calibrated number would double-count it.

    When a real simulated `slot_distribution` is available (this league's
    immediate upcoming draft, from the Championship Odds simulator's
    `draft_slot_distribution`), values the pick as its expectation across
    that actual distribution — correct under Jensen's inequality for a
    convex per-slot curve, and a real improvement over a single
    power-percentile point estimate. Falls back to an estimated slot from
    the team's current power percentile for picks in seasons too far out to
    simulate (this is a fixed 8-team league: weakest team ≈ slot 1,
    strongest ≈ slot 8)."""
    def slot_value(slot):
        if cache_key is not None:
            consensus = compute_consensus_pick_value(season, round_no, slot, cache_key, our_team_count, as_of_date)
            if consensus is not None:
                return consensus
        return pick_slot_value(round_no, slot) * CLASS_STRENGTH_MULTIPLIER.get(str(season), 1.0)

    if slot_distribution:
        return round(sum(p * slot_value(slot) for slot, p in enumerate(slot_distribution, start=1)), 1)
    # Slot 1 is the EARLIEST, most valuable pick, and in this league the weakest teams pick
    # first (non-playoff teams take 1-4 by lowest Max PF). So a low power percentile has to map
    # to a low slot number: pct 0 -> slot 1, pct 1 -> slot 8. This previously read
    # `1 + (1 - pct) * 7`, which inverted it -- pricing a contender's future first-rounder as
    # the 1.01 and a rebuilding team's as a late pick, the exact opposite of reality.
    estimated_slot = 1 + original_roster_power_percentile * 7
    return round(slot_value(estimated_slot), 1)


def value_pick_row(row, power_pct, current_season_str, odds, cache_key=None):
    """Value one row of a pick inventory table. `current_season_str` is the season whose
    draft order the `odds` simulation actually determines — i.e. the NEXT rookie draft, since
    a draft's order comes from the season played before it. Only that season's picks have a
    real simulated slot distribution; anything further out falls back to pick_value's
    percentile-estimated slot."""
    if row["season"] == current_season_str:
        dist = odds.get(int(row["original_roster_id"]), {}).get("draft_slot_distribution")
        if dist:
            return pick_value(row["round"], power_pct.get(row["original_roster_id"], 0.5),
                               slot_distribution=dist, season=row["season"], cache_key=cache_key)
    return pick_value(row["round"], power_pct.get(row["original_roster_id"], 0.5),
                       season=row["season"], cache_key=cache_key)


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


def grade_trade(txn, value_table, power_pct, global_team_lookup, league_id, cache_key=None):
    """Grades one trade using `value_table` (players — already built as of
    that trade's season, see build_historical_value_table) and, for any
    picks involved, `cache_key` to look up each pick's own value as of that
    SAME season's end (season_end_as_of_date) — i.e. every asset in this
    trade is priced consistently at the same historical point, not a mix of
    historical players and today's pick prices. cache_key=None (the
    default) falls back to pick_value's pure in-house formula, unchanged
    from before this feature existed — used by any caller not yet passing one."""
    roster_ids = parse_json_field(txn["roster_ids"], [])
    adds = parse_json_field(txn["adds"], {})
    drops = parse_json_field(txn["drops"], {})
    draft_picks = parse_json_field(txn["draft_picks"], [])
    waiver_budget = parse_json_field(txn["waiver_budget"], [])
    # Consensus values are PERCENTILES, which rank rather than measure, so they're mapped onto
    # the real market distribution before any addition. Summing them directly made the side
    # receiving MORE PIECES win almost automatically: four assets averaging the 87th percentile
    # totalled far more than three averaging the 94th, which is how a trade sending away the
    # single best player in it came out an A.
    worth_curve = build_asset_worth_curve(cache_key, as_of_date=season_end_as_of_date(txn["season"]))         if cache_key else None

    received = {rid: 0.0 for rid in roster_ids}
    given = {rid: 0.0 for rid in roster_ids}
    future_received = {rid: 0.0 for rid in roster_ids}
    future_given = {rid: 0.0 for rid in roster_ids}

    def player_value_and_age(player_id):
        if player_id in value_table.index:
            row = value_table.loc[player_id]
            return asset_worth(float(row["value"]), worth_curve), row["age"]
        return asset_worth(20.0, worth_curve), None  # unrostered/unknown — small flat fallback

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
        val = asset_worth(
            pick_value(pick.get("round", 4), power_pct.get(pick.get("roster_id"), 0.5), season=pick.get("season"),
                        cache_key=cache_key, as_of_date=season_end_as_of_date(txn["season"]) if cache_key else None),
            worth_curve)
        new_owner, prev_owner = pick.get("owner_id"), pick.get("previous_owner_id")
        if new_owner in received:
            received[new_owner] += val
            future_received[new_owner] += val * 1.3  # picks are pure future assets
        if prev_owner in given:
            given[prev_owner] += val
            future_given[prev_owner] += val * 1.3

    for wb in waiver_budget:
        # FAAB is denominated in dollars, not percentiles, so it needs its own scaling onto the
        # worth scale rather than the old flat 0.3-per-dollar (which was tuned for percentiles).
        faab_val = asset_worth(min(wb.get("amount", 0) * 0.3, 100.0), worth_curve)
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


def trade_as_of_date(txn):
    """Exact historical date for one trade, from its real `created` timestamp
    (epoch ms) — used for the 'how this trade aged' comparison below, where
    true per-transaction precision is cheap (one trade at a time, unlike
    build_historical_value_table's season-end approximation, which trades
    precision for being reusable across every trade that season). Falls back
    to that same season-end approximation if `created` is missing (a
    handful of real transactions in this league's data lack it — see the
    Team Timeline "Offseason" grouping fix for the same underlying gap)."""
    created = txn.get("created")
    if created:
        return datetime.fromtimestamp(created / 1000, tz=timezone.utc).date().isoformat()
    return season_end_as_of_date(txn["season"])


AGING_SOURCE = "dynastyprocess_value"  # the only source with real history: FantasyCalc has no
# historical API, so its archive only begins the day this app started recording it. Comparing a
# past date against today therefore has to stay on ONE source, or "then" and "now" are read off
# two different price scales entirely.


@st.cache_data(ttl=1800)
def _aging_price_history(cache_key):
    """player_id -> (ascending dates, values) for AGING_SOURCE. Built once and bisected per
    lookup: the aging table renders inside an expander for every trade in the list, and
    Streamlit runs an expander's body whether or not it's open, so a full re-scan of the
    snapshot archive per trade made the whole page crawl."""
    df = get_external_player_snapshots(cache_key)
    df = df[df["source"] == AGING_SOURCE].sort_values("scrape_date")
    return {pid: (list(g["scrape_date"]), list(g["raw_value"]))
            for pid, g in df.groupby("sleeper_player_id")}


def _aging_player_value(history, player_id, as_of_date=None):
    """That player's AGING_SOURCE price on/before as_of_date (or his latest), else None."""
    entry = history.get(player_id)
    if not entry:
        return None
    dates, values = entry
    if as_of_date is None:
        return values[-1]
    idx = bisect.bisect_right(dates, as_of_date) - 1
    return values[idx] if idx >= 0 else None


PICK_SNAPSHOT_KEYS = ["season", "round", "granularity", "slot", "tier"]


@st.cache_data(ttl=1800)
def _aging_pick_history(cache_key):
    """One (dates, values) series per distinct AGING_SOURCE pick row, for the same reason as
    _aging_price_history: collapsing the whole pick archive once per trade was the bulk of the
    Trade Center's load time."""
    df = get_external_pick_snapshots(cache_key)
    df = df[df["source"] == AGING_SOURCE].sort_values("scrape_date")
    return {key: (list(g["scrape_date"]), list(g["raw_value"]))
            for key, g in df.groupby(PICK_SNAPSHOT_KEYS)}


def _aging_pick_snapshots(cache_key, as_of_date=None):
    """AGING_SOURCE pick rows as of a date, in the shape _pick_raw_value expects."""
    rows = []
    for key, (dates, values) in _aging_pick_history(cache_key).items():
        if as_of_date is None:
            value = values[-1]
        else:
            idx = bisect.bisect_right(dates, as_of_date) - 1
            if idx < 0:
                continue
            value = values[idx]
        rows.append(dict(zip(PICK_SNAPSHOT_KEYS, key), source=AGING_SOURCE, raw_value=value))
    return pd.DataFrame(rows, columns=PICK_SNAPSHOT_KEYS + ["source", "raw_value"])


def _aging_pick_value(pick_latest, season, round_no, power_percentile):
    """Raw AGING_SOURCE market value for one pick, priced at the slot that team's strength
    implies — the same slot estimate pick_value uses, so the two views agree on which pick
    this is."""
    if pick_latest is None or pick_latest.empty:
        return None
    estimated_slot = 1 + (1 - power_percentile) * 7
    return _pick_raw_value(pick_latest, AGING_SOURCE, season, round_no, estimated_slot, 8, True)


def build_trade_aging_detail(txn, power_pct, players_df, cache_key):
    """Per-asset value-at-the-time vs. today's value for one trade — the 'how this trade aged'
    hindsight view. Purely additive to grade_trade's at-the-time grade, never a replacement.
    Uses the trade's exact date (trade_as_of_date), not build_historical_value_table's coarser
    season-end approximation, since this is a single on-demand lookup.

    Reports each asset's real market price on ONE source's scale (AGING_SOURCE), rather than a
    consensus percentile. Percentiles saturate: a player already ranked near the top of the pool
    can gain a fifth of his real trade value and still show fractionally DOWN, because everyone
    around him moved too. Mapping each side through its own date's value distribution instead
    was worse — with only one source carrying real history, "then" and "now" ended up on
    different scales and a player who had genuinely lost two thirds of his value read as a
    small gain."""
    as_of = trade_as_of_date(txn)
    history = _aging_price_history(cache_key)
    then_picks = _aging_pick_snapshots(cache_key, as_of_date=as_of)
    now_picks = _aging_pick_snapshots(cache_key)

    adds = parse_json_field(txn["adds"], {})
    drops = parse_json_field(txn["drops"], {})
    draft_picks = parse_json_field(txn["draft_picks"], [])

    def row(label, then_val, now_val):
        if then_val is None or now_val is None:
            return {"asset": label, "then": None, "now": None, "swing": None}
        return {"asset": label, "then": round(float(then_val)), "now": round(float(now_val)),
                "swing": round(float(now_val) - float(then_val))}

    rows = []
    for player_id in set(adds) | set(drops):
        name = players_df.loc[player_id, "full_name"] if player_id in players_df.index else player_id
        rows.append(row(name, _aging_player_value(history, player_id, as_of),
                         _aging_player_value(history, player_id)))

    for pick in draft_picks:
        label = f"{pick.get('season')} Round {pick.get('round')} pick"
        pct = power_pct.get(pick.get("roster_id"), 0.5)
        rows.append(row(
            label,
            _aging_pick_value(then_picks, pick.get("season"), pick.get("round", 4), pct),
            _aging_pick_value(now_picks, pick.get("season"), pick.get("round", 4), pct)))

    # Explicit columns so an assetless trade still returns the expected shape rather than an
    # empty frame the caller then indexes into.
    return pd.DataFrame(rows, columns=["asset", "then", "now", "swing"])


TRADE_HISTORY_PAGE = 20  # trades drawn before asking, since each one is fully graded and aged


CONTEXT_FIT_SHARE = 0.15  # the wrong-direction swing has to be at least this share of the
# value a team moved in the trade before it counts as fighting that team's timeline. A share
# rather than a fixed number of points: the old absolute threshold was calibrated against
# summed percentiles and means nothing once assets are priced on the real market scale.


def trade_context_fit(state, win_now_impact, future_impact):
    """Whether a trade fits the team's own timeline (a rebuilding team chasing a marginal
    win-now upgrade doesn't make sense even at fair value) — a ⚠️ result downgrades the
    displayed letter grade one full step (see downgrade_grade).

    Only a team with an actual timeline can fight it: roughly three quarters of this league
    reads as balanced in any given season, and for those there's nothing to conflict with, so
    they say so plainly instead of claiming a check passed. The old wording marked every trade
    "fits team timeline" — including the balanced majority, where nothing had been tested —
    and it also required BOTH halves of a mismatch to clear a fixed point threshold at once,
    which in practice never happened."""
    scale = max(abs(win_now_impact), abs(future_impact), 1.0)
    if state == "rebuild" and win_now_impact > 0 and future_impact < 0:
        if abs(future_impact) >= CONTEXT_FIT_SHARE * scale:
            return "⚠️ Win-now move for a rebuilding team"
    if state == "contend" and future_impact > 0 and win_now_impact < 0:
        if abs(win_now_impact) >= CONTEXT_FIT_SHARE * scale:
            return "⚠️ Future-focused move for a team that should be contending"
    if state == "balanced":
        return "➖ No clear timeline to conflict with"
    return f"✅ Fits a {state}ing team's timeline" if state == "rebuild" else "✅ Fits a contender's timeline"


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


# ---------------------------------------------------------------------------
# Asset Trail — starting from one trade, traces every asset involved (players AND
# picks) both backward (where it came from) and forward (everywhere it went,
# including what a traded pick actually turned into once drafted), rendered as a
# chronological per-asset timeline. A pick's `roster_id` field in a trade's JSON is the ORIGINAL owner and
# stays constant across every re-trade (only owner_id/previous_owner_id change per
# hop), which is what makes tracing a single pick's full lineage tractable at all.
# ---------------------------------------------------------------------------
LINEAGE_MAX_HOPS = 15  # safety cap per asset per direction — tommy's own motivating
# example (a pick re-traded multiple times across several seasons before becoming a
# real rookie) needs real headroom; this only guards against a genuine data anomaly
# causing a runaway walk, not against legitimately long real trade chains


@st.cache_data(ttl=1800)
def get_all_pick_trade_events(cache_key):
    """Every (season, round, original-owner) pick move across every trade ever
    synced, one row per pick per trade. Lets a single pick's full lineage be found
    regardless of how many times it's been re-traded."""
    all_seasons = load_table("SELECT league_id, season FROM league_seasons")
    rows = []
    for _, srow in all_seasons.iterrows():
        lid = srow["league_id"]
        txns = load_table(
            "SELECT transaction_id, week, draft_picks, created FROM transactions WHERE league_id = ? AND type = 'trade'",
            params=(lid,),
        )
        for _, txn in txns.iterrows():
            for p in parse_json_field(txn["draft_picks"], []):
                rows.append({
                    "transaction_id": txn["transaction_id"], "league_id": lid, "created": txn["created"],
                    # when the TRADE happened -- distinct from pick_season (the draft the pick is
                    # for). Labeling a hop with the pick's own season read as though a 2027 pick
                    # had been traded in 2027, years before it actually was.
                    "trade_season": srow["season"], "trade_week": txn["week"],
                    "pick_season": p.get("season"), "pick_round": p.get("round"),
                    "original_roster_id": p.get("roster_id"),
                    "owner_id": p.get("owner_id"), "previous_owner_id": p.get("previous_owner_id"),
                })
    return pd.DataFrame(rows)


def get_exact_draft_slot_map(league_id):
    """Exact real draft slot (1-8) per roster for one season's league_id, reusing
    the verified draft-order simulator in n_trials=1 'exact replay' mode (same
    technique GM Profiles uses for exact historical placements) — once a season is
    fully complete every week is real data, so this isn't simulating anything,
    just replaying what actually happened."""
    local_standings = get_standings(league_id)
    # project_rosters=False: exact replay of real results never samples these distributions,
    # so the roster projection would be pure wasted work here.
    distributions = estimate_team_distributions(league_id, local_standings, project_rosters=False)
    result = run_simulation(league_id, local_standings, distributions, n_trials=1)
    return {rid: int(round(info["expected_draft_slot"])) for rid, info in result.items()}


@st.cache_data(ttl=1800)
def get_exact_draft_slot_map_cached(league_id, cache_key):
    return get_exact_draft_slot_map(league_id)


def resolve_pick_detail(pick_season, pick_round, original_roster_id, cache_key):
    """(player_id, drafting_roster_id, league_id) for the pick — the real player a specific
    (season, round, original-owner) pick turned into, plus who actually made the selection — None if
    that draft hasn't happened/been synced yet (a still-future pick).

    Slot comes from Sleeper's own `drafts.draft_order` (user_id -> real slot), keyed to the
    ORIGINAL owner since that's what fixes the numbered pick regardless of who ultimately used
    it. That's the authoritative order, and it needs no simulation: the previous approach
    replayed the pick season's standings to infer slots, which also meant it refused to resolve
    anything until that season was fully complete — wrong for a draft held BEFORE its season is
    played (a draft's order comes from the season before it), so picks already spent in the
    current season's completed rookie draft resolved to nothing and looked unused."""
    season_row = load_table("SELECT league_id FROM league_seasons WHERE season = ?", params=(pick_season,))
    if season_row.empty:
        return None, None, None
    lid = season_row.iloc[0]["league_id"]
    draft_row = load_table(
        "SELECT draft_id, draft_order, type FROM drafts WHERE league_id = ?", params=(lid,))
    if draft_row.empty:
        return None, None, None
    draft_id = draft_row.iloc[0]["draft_id"]
    draft_order = parse_json_field(draft_row.iloc[0]["draft_order"], {}) or {}
    if not draft_order:
        return None, None, None

    owner_row = load_table(
        "SELECT owner_id FROM rosters WHERE league_id = ? AND roster_id = ?",
        params=(lid, int(original_roster_id)))
    if owner_row.empty:
        return None, None, None
    slot = draft_order.get(str(owner_row.iloc[0]["owner_id"]))
    if slot is None:
        return None, None, None

    teams = len(draft_order)
    round_no = int(pick_round)
    # Snake drafts reverse every even round; linear ones keep the same order throughout.
    if str(draft_row.iloc[0]["type"]).lower() == "snake" and round_no % 2 == 0:
        slot_in_round = teams + 1 - int(slot)
    else:
        slot_in_round = int(slot)
    pick_no = (round_no - 1) * teams + slot_in_round

    result = load_table(
        "SELECT player_id, roster_id FROM draft_picks WHERE draft_id = ? AND pick_no = ?",
        params=(draft_id, pick_no))
    if result.empty:
        return None, None, None
    return result.iloc[0]["player_id"], int(result.iloc[0]["roster_id"]), lid


def resolve_pick_to_player(pick_season, pick_round, original_roster_id, cache_key):
    """Just the player (see resolve_pick_detail for who actually made the selection)."""
    player_id, _drafter, _lid = resolve_pick_detail(pick_season, pick_round, original_roster_id, cache_key)
    return player_id


def get_player_event_history(player_id):
    """Every event this player was involved in — draft plus every transaction where
    they were added — WITH transaction_id/draft_id, needed to label/link Sankey
    nodes back to the actual trade (build_player_ownership_history only needs
    enough detail for a plain-text list, not identifiers)."""
    events = []
    draft_rows = load_table(
        """
        SELECT d.season, dp.roster_id, d.league_id, d.draft_id
        FROM draft_picks dp JOIN drafts d ON d.draft_id = dp.draft_id
        WHERE dp.player_id = ?
        ORDER BY d.season ASC
        """,
        params=(player_id,),
    )
    for _, row in draft_rows.iterrows():
        events.append({"season": row["season"], "league_id": row["league_id"], "week": None,
                        "type": "draft", "roster_id": row["roster_id"], "transaction_id": None,
                        "draft_id": row["draft_id"], "order": (row["season"], 0)})

    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    for _, row in all_seasons.iterrows():
        lid = row["league_id"]
        txns = load_table(
            "SELECT transaction_id, week, type, adds, created FROM transactions WHERE league_id = ? ORDER BY created ASC",
            params=(lid,),
        )
        for _, txn in txns.iterrows():
            adds = parse_json_field(txn["adds"], {})
            if player_id in adds:
                events.append({"season": row["season"], "league_id": lid, "week": txn["week"],
                                "type": txn["type"], "roster_id": adds[player_id],
                                "transaction_id": txn["transaction_id"], "draft_id": None,
                                "created": txn["created"],
                                "order": (row["season"], txn["created"])})
    events.sort(key=lambda e: e["order"])
    return events


@st.cache_data(ttl=1800)
def get_player_event_history_cached(player_id, cache_key):
    return get_player_event_history(player_id)


def event_label(event, global_team_lookup, players_df):
    """Short human label for a Sankey node representing one event."""
    team = escape_markdown(global_team_lookup.get((event["league_id"], event["roster_id"]),
                                                     f"Roster {event['roster_id']}"))
    if event["type"] == "draft":
        pid = event.get("player_id")
        name = players_df.loc[pid, "full_name"] if pid is not None and pid in players_df.index else pid
        who = f" {escape_markdown(name)}" if name else ""
        return f"{event['season']} Draft{who} → {team}"
    if event["week"]:
        stamp = week_label(event["season"], event["week"], event.get("created"))
        return f"{event['season']} {stamp} {event['type'].replace('_', ' ').title()} → {team}"
    return f"{event['season']} {event['type'].replace('_', ' ').title()} → {team}"


def trace_player_chain(player_id, seed_transaction_id, seed_draft_id, direction, players_df, cache_key):
    """Walks one player's event history one direction (1 = forward, -1 = backward)
    from the seed event. hops[0] is always the closest event to the seed, hops[-1]
    the furthest — same convention in both directions, so callers don't need to
    reverse anything."""
    events = get_player_event_history_cached(player_id, cache_key)
    idx = None
    for i, e in enumerate(events):
        if seed_transaction_id is not None and e.get("transaction_id") == seed_transaction_id:
            idx = i
            break
        if seed_draft_id is not None and e.get("draft_id") == seed_draft_id:
            idx = i
            break
    if idx is None:
        return []
    player_label = escape_markdown(players_df.loc[player_id, "full_name"] if player_id in players_df.index else player_id)
    hops = []
    i = idx
    for _ in range(LINEAGE_MAX_HOPS):
        i += direction
        if i < 0 or i >= len(events):
            break
        hop_event = dict(events[i])
        hop_event["player_id"] = player_id  # disambiguates a shared draft_id — two different
        # players drafted in the same draft event must not collapse into one Sankey node
        hops.append({"event": hop_event, "asset_label": player_label})
    return hops


def trace_pick_forward(pick_season, pick_round, original_roster_id, seed_created, pick_events_df, players_df, cache_key):
    """Forward lineage of one pick: every re-trade after the seed, then — once no
    further re-trade is found — whether it's actually been drafted yet, switching
    to tracing the resulting player forward from their draft point if so."""
    hops = []
    matches = pick_events_df[
        (pick_events_df["pick_season"] == pick_season)
        & (pick_events_df["pick_round"] == pick_round)
        & (pick_events_df["original_roster_id"] == original_roster_id)
    ]
    pick_label = f"a {pick_season} {ordinal(pick_round)}"
    cursor = seed_created
    for _ in range(LINEAGE_MAX_HOPS):
        later = matches[matches["created"] > cursor].sort_values("created")
        if later.empty:
            break
        hop_row = later.iloc[0]
        fake_event = {"season": hop_row["trade_season"], "league_id": hop_row["league_id"],
                      "week": hop_row["trade_week"], "type": "trade", "created": hop_row["created"],
                      "roster_id": hop_row["owner_id"], "transaction_id": hop_row["transaction_id"], "draft_id": None}
        hops.append({"event": fake_event, "asset_label": pick_label})
        cursor = hop_row["created"]

    drafted_player = resolve_pick_to_player(pick_season, pick_round, original_roster_id, cache_key)
    if drafted_player:
        player_events = get_player_event_history_cached(drafted_player, cache_key)
        draft_event = next((e for e in player_events if e["type"] == "draft"), None)
        if draft_event:
            name = escape_markdown(players_df.loc[drafted_player, "full_name"]
                                    if drafted_player in players_df.index else drafted_player)
            draft_event = dict(draft_event)
            draft_event["player_id"] = drafted_player  # disambiguates a shared draft_id
            hops.append({"event": draft_event, "asset_label": f"{pick_label} → drafted {name}"})
            hops.extend(trace_player_chain(drafted_player, None, draft_event["draft_id"], 1, players_df, cache_key))
    return hops


def trace_pick_backward(pick_season, pick_round, original_roster_id, seed_created, pick_events_df):
    """Backward lineage of one pick: every earlier trade that moved this exact
    pick before the seed. Stops once none is found — that's the original team's
    own natural draft slot, nothing to trace further back."""
    hops = []
    matches = pick_events_df[
        (pick_events_df["pick_season"] == pick_season)
        & (pick_events_df["pick_round"] == pick_round)
        & (pick_events_df["original_roster_id"] == original_roster_id)
    ]
    pick_label = f"a {pick_season} {ordinal(pick_round)}"
    cursor = seed_created
    for _ in range(LINEAGE_MAX_HOPS):
        earlier = matches[matches["created"] < cursor].sort_values("created", ascending=False)
        if earlier.empty:
            break
        hop_row = earlier.iloc[0]
        fake_event = {"season": hop_row["trade_season"], "league_id": hop_row["league_id"],
                      "week": hop_row["trade_week"], "type": "trade", "created": hop_row["created"],
                      "roster_id": hop_row["previous_owner_id"], "transaction_id": hop_row["transaction_id"],
                      "draft_id": None}
        hops.append({"event": fake_event, "asset_label": pick_label})
        cursor = hop_row["created"]
    return hops


def build_trade_lineage(seed_txn, players_df, cache_key):
    """Full forward+backward lineage for every asset in one trade — one chain per
    asset, each independently traced (a pick's chain switches to a player's chain
    once it's actually drafted)."""
    pick_events_df = get_all_pick_trade_events(cache_key)
    seed_created = seed_txn["created"]
    seed_transaction_id = seed_txn["transaction_id"]

    adds = parse_json_field(seed_txn["adds"], {})
    picks = parse_json_field(seed_txn["draft_picks"], [])

    chains = []
    for player_id in adds:
        name = (players_df.loc[player_id, "full_name"]
                if player_id in players_df.index else str(player_id))
        chains.append({
            "asset": name,
            "backward": trace_player_chain(player_id, seed_transaction_id, None, -1, players_df, cache_key),
            "forward": trace_player_chain(player_id, seed_transaction_id, None, 1, players_df, cache_key),
        })
    for p in picks:
        pick_season, pick_round, original_roster_id = p.get("season"), p.get("round"), p.get("roster_id")
        chains.append({
            "asset": f"{pick_season} Round {pick_round} pick",
            "backward": trace_pick_backward(pick_season, pick_round, original_roster_id, seed_created, pick_events_df),
            "forward": trace_pick_forward(pick_season, pick_round, original_roster_id, seed_created,
                                           pick_events_df, players_df, cache_key),
        })
    return chains


TRADE_CONCLUSION_MAX_STEPS = 80  # safety cap on the consequence walk per side of a trade


@st.cache_data(ttl=1800)
def get_all_transactions_detail(cache_key):
    """Every transaction across every synced season with the fields needed to follow an asset
    out of a roster and see what came back in its place."""
    frames = []
    for _, srow in load_table("SELECT league_id, season FROM league_seasons").iterrows():
        df = load_table(
            "SELECT transaction_id, week, type, adds, drops, draft_picks, created "
            "FROM transactions WHERE league_id = ?", params=(srow["league_id"],))
        if df.empty:
            continue
        df["league_id"] = srow["league_id"]
        df["season"] = srow["season"]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["transaction_id", "week", "type", "adds", "drops",
                                      "draft_picks", "created", "league_id", "season"])
    return pd.concat(frames, ignore_index=True).sort_values("created")


@st.cache_data(ttl=1800)
def get_owner_roster_maps(cache_key):
    """(league_id, roster_id) -> owner_id and (owner_id, league_id) -> roster_id. A manager's
    roster_id is per-season, so following one manager's assets across seasons has to hop
    through owner_id rather than assuming a stable roster_id."""
    df = load_table("SELECT league_id, roster_id, owner_id FROM rosters")
    owner_of, roster_of = {}, {}
    for _, r in df.iterrows():
        owner_of[(r["league_id"], int(r["roster_id"]))] = r["owner_id"]
        roster_of[(r["owner_id"], r["league_id"])] = int(r["roster_id"])
    return owner_of, roster_of


def _receipts_for_roster(txn_row, roster_id):
    """(player_ids, pick_keys) one roster RECEIVED in a single transaction."""
    adds = parse_json_field(txn_row["adds"], {}) or {}
    players = [pid for pid, rid in adds.items() if int(rid) == int(roster_id)]
    picks = [(p.get("season"), p.get("round"), p.get("roster_id"))
             for p in (parse_json_field(txn_row["draft_picks"], []) or [])
             if p.get("owner_id") is not None and int(p["owner_id"]) == int(roster_id)]
    return players, picks


def _sent_by_roster(txn_row, roster_id):
    """(player_ids, pick_keys) one roster GAVE UP in a single transaction — the other half of
    the ledger, so a trade can be read as what it cost as well as what it returned."""
    drops = parse_json_field(txn_row["drops"], {}) or {}
    players = [pid for pid, rid in drops.items() if int(rid) == int(roster_id)]
    picks = [(p.get("season"), p.get("round"), p.get("roster_id"))
             for p in (parse_json_field(txn_row["draft_picks"], []) or [])
             if p.get("previous_owner_id") is not None and int(p["previous_owner_id"]) == int(roster_id)]
    return players, picks


def _trade_step_label(txn_row, my_roster_id, global_team_lookup):
    """" (from <other team>)" for a trade, so a breadcrumb reads like the way someone would
    actually recount it — "that pick became X, who I sent to Y for Z"."""
    others = set()
    for pid, rid in (parse_json_field(txn_row["adds"], {}) or {}).items():
        if int(rid) != int(my_roster_id):
            others.add(int(rid))
    for pid, rid in (parse_json_field(txn_row["drops"], {}) or {}).items():
        if int(rid) != int(my_roster_id):
            others.add(int(rid))
    names = [str(global_team_lookup.get((txn_row["league_id"], rid), "")).strip() for rid in others]
    names = [n for n in names if n]
    return f" (from {names[0]})" if len(names) == 1 else ""


def _outgoing_share(txn_row, roster_id):
    """The fraction of a later trade's return that one outgoing asset is credited with.

    When a manager sends several pieces out together, the return genuinely cannot be
    attributed to any one of them, so it's split evenly. Crediting each outgoing piece with the
    WHOLE return instead let a single branch compound through every subsequent move until it had
    absorbed most of a roster's history — which is how one traded tight end appeared to have
    turned into fourteen assets."""
    sent_players, sent_picks = _sent_by_roster(txn_row, roster_id)
    count = len(sent_players) + len(sent_picks)
    return 1.0 / count if count else 1.0


def build_trade_conclusion(seed_txn, players_df, cache_key):
    """What each side of a trade actually has to show for it TODAY.

    Follows every asset a manager received forward through their own later moves: when an asset
    was traded on, whatever came back in that trade becomes its descendant and the walk
    continues; when a pick was finally used, the drafted player takes over the chain; when an
    asset was simply dropped, that branch ends as a loss. What survives is the manager's
    current holdings that trace back to this trade.

    One honest caveat, unavoidable in any trade tree: when several assets leave in a single
    trade, the return can't be attributed to one of them in particular, so it's credited to
    each outgoing piece. Descendants are de-duplicated per side, so nothing is double-counted
    in the result itself.

    Returns {owner_id: {"received": [...], "holding": [...], "lost": [...]}}."""
    txns = get_all_transactions_detail(cache_key)
    pick_events = get_all_pick_trade_events(cache_key)
    owner_of, roster_of = get_owner_roster_maps(cache_key)
    season_league = dict(zip(
        load_table("SELECT season, league_id FROM league_seasons")["season"],
        load_table("SELECT season, league_id FROM league_seasons")["league_id"]))

    seed_lid = seed_txn["league_id"]
    seed_created = seed_txn["created"]
    seed_adds = parse_json_field(seed_txn["adds"], {}) or {}
    seed_picks = parse_json_field(seed_txn["draft_picks"], []) or []

    sides = {int(rid) for rid in seed_adds.values()}
    sides |= {int(p["owner_id"]) for p in seed_picks if p.get("owner_id") is not None}

    def label_player(pid):
        return str(players_df.loc[pid, "full_name"]) if pid in players_df.index else str(pid)

    global_team_lookup = get_global_team_lookup()

    def label_pick(key):
        # Original owner included because a manager can hold two different picks of the same
        # season and round; without it they render as one repeated line.
        season, rnd, orig = key
        lid_for_pick = season_league.get(str(season), seed_lid)
        team = global_team_lookup.get((lid_for_pick, orig)) or global_team_lookup.get((seed_lid, orig))
        whose = f" ({str(team).strip()}'s)" if team else ""
        return f"{season} {ordinal(rnd)}{whose}"

    # Hindsight valuation of what each side is left holding. Counting assets alone implied a
    # side that turned one elite player into a pile of spare parts had come out ahead; valuing
    # them says whether it actually did. Both legs land on the same 0-100 consensus scale.
    latest = load_table("SELECT league_id, season FROM league_seasons ORDER BY season DESC LIMIT 1")
    value_table, power_pct = None, {}
    if not latest.empty:
        value_table = get_value_table(latest.iloc[0]["league_id"], int(latest.iloc[0]["season"]), cache_key)
        power_pct = get_season_power_pct(latest.iloc[0]["league_id"], cache_key)
    # Percentiles are mapped back onto the real market distribution (asset_worth) BEFORE being
    # summed. Adding raw percentiles rewards quantity over quality -- a dozen fringe assets each
    # ranking in the 80s would out-total one genuinely elite player, which is exactly how a side
    # that traded away a star for spare parts looked like the winner.
    worth_curve = build_asset_worth_curve(cache_key)

    def value_of(kind, key):
        if kind == "player":
            if value_table is not None and key in value_table.index:
                return asset_worth(float(value_table.loc[key, "value"]), worth_curve)
            return 0.0
        season, rnd, orig = key
        try:
            pct = float(pick_value(rnd, power_pct.get(int(orig), 0.5),
                                    season=str(season), cache_key=cache_key))
        except Exception:
            return 0.0
        return asset_worth(pct, worth_curve)

    results = {}
    for rid in sides:
        owner = owner_of.get((seed_lid, rid))
        if owner is None:
            continue
        recv_players, recv_picks = _receipts_for_roster(seed_txn, rid)
        received = [label_player(p) for p in recv_players] + [label_pick(k) for k in recv_picks]
        sent_players, sent_picks = _sent_by_roster(seed_txn, rid)
        gave_up = [label_player(p) for p in sent_players] + [label_pick(k) for k in sent_picks]

        # Each queue item carries the breadcrumb of how it descends from this trade, so a
        # holding several moves removed can explain itself instead of appearing out of nowhere.
        queue = ([("player", p, seed_created, (label_player(p),), 1.0, "") for p in recv_players]
                 + [("pick", k, seed_created, (label_pick(k),), 1.0, "") for k in recv_picks])
        seen, holding, lost, moved_on = set(), [], [], []
        steps = 0
        while queue and steps < TRADE_CONCLUSION_MAX_STEPS:
            steps += 1
            kind, key, since, trail, share, arrived = queue.pop(0)
            if (kind, key) in seen:
                continue
            seen.add((kind, key))

            if kind == "player":
                gone = None
                for _, t in txns[txns["created"] > since].iterrows():
                    my_rid = roster_of.get((owner, t["league_id"]))
                    if my_rid is None:
                        continue
                    drops = parse_json_field(t["drops"], {}) or {}
                    if key in drops and int(drops[key]) == int(my_rid):
                        gone = t
                        break
                if gone is None:
                    holding.append(("player", key, label_player(key), trail, share, arrived))
                    continue
                if gone["type"] == "trade":
                    my_rid = roster_of.get((owner, gone["league_id"]))
                    got_players, got_picks = _receipts_for_roster(gone, my_rid)
                    step = _trade_step_label(gone, my_rid, global_team_lookup)
                    child = share * _outgoing_share(gone, my_rid)
                    queue += [("player", p, gone["created"], trail + (label_player(p),), child, step)
                              for p in got_players]
                    queue += [("pick", k, gone["created"], trail + (label_pick(k),), child, step)
                              for k in got_picks]
                else:
                    lost.append(("player", key, label_player(key), trail, share, arrived))
                continue

            season, rnd, orig = key
            mine = pick_events[(pick_events["pick_season"] == season)
                                & (pick_events["pick_round"] == rnd)
                                & (pick_events["original_roster_id"] == orig)
                                & (pick_events["created"] > since)].sort_values("created")
            traded_on = None
            for _, ev in mine.iterrows():
                my_rid = roster_of.get((owner, ev["league_id"]))
                if my_rid is not None and int(ev["previous_owner_id"]) == int(my_rid):
                    traded_on = ev
                    break
            if traded_on is not None:
                parent = txns[txns["transaction_id"] == traded_on["transaction_id"]]
                if not parent.empty:
                    row = parent.iloc[0]
                    my_rid = roster_of.get((owner, row["league_id"]))
                    got_players, got_picks = _receipts_for_roster(row, my_rid)
                    step = _trade_step_label(row, my_rid, global_team_lookup)
                    child = share * _outgoing_share(row, my_rid)
                    queue += [("player", p, row["created"], trail + (label_player(p),), child, step)
                              for p in got_players]
                    queue += [("pick", k, row["created"], trail + (label_pick(k),), child, step)
                              for k in got_picks]
                continue

            drafted, drafted_by, draft_lid = resolve_pick_detail(season, rnd, orig, cache_key)
            if drafted:
                # Only credit the player to the manager who ACTUALLY made the selection. A pick
                # this manager flipped before the draft still resolves to whoever was taken at
                # that slot, and crediting them with a player they never drafted was reporting
                # plain falsehoods ("Dalt drafted Jonah Coleman" for a pick he had traded away).
                # If the move that sent it on isn't in `transactions` we can't follow the return,
                # so the branch ends honestly as "traded away" rather than inventing a holding.
                my_rid_at_draft = roster_of.get((owner, draft_lid))
                if my_rid_at_draft is None or int(drafted_by) != int(my_rid_at_draft):
                    moved_on.append(("pick", key, label_pick(key), trail, share, arrived))
                    continue
                draft_time = since
                if draft_lid:
                    drow = load_table(
                        "SELECT last_picked, start_time FROM drafts WHERE league_id = ?",
                        params=(draft_lid,))
                    if not drow.empty:
                        stamp = drow.iloc[0]["last_picked"] or drow.iloc[0]["start_time"]
                        if pd.notna(stamp):
                            draft_time = max(since, int(stamp))
                queue.append(("player", drafted, draft_time,
                               trail + (f"drafted {label_player(drafted)}",), share, arrived))
            else:
                holding.append(("pick", key, label_pick(key), trail, share, arrived))

        # de-duplicate while preserving order
        def dedupe(items):
            out, taken = [], set()
            for kind_, key_, lbl, trail_, share_, arrived_ in items:
                if (kind_, key_) in taken:
                    continue
                taken.add((kind_, key_))
                # trail_[0] is the piece received in this trade; anything between is how one
                # became the other. The final entry repeats the asset plus who it came from, so
                # that counterparty is split off rather than left cluttering the asset's name.
                # trail_ holds plain asset names oldest-first; who each hop came from is
                # deliberately left to the full trail, so this chain stays readable. Only the
                # FINAL counterparty is kept, since that's whose hands the asset arrived from.
                out.append({"label": lbl, "from": trail_[0], "kind": kind_, "key": key_,
                             "path": list(trail_[1:-1]), "acquired": arrived_.strip(),
                             "share": share_, "full_value": value_of(kind_, key_),
                             "value": value_of(kind_, key_) * share_})
            return out

        held = dedupe(holding)
        results[owner] = {"received": received, "gave_up": gave_up,
                           "holding": held, "lost": dedupe(lost),
                           "moved_on": dedupe(moved_on),
                           "value_today": round(sum(h["value"] for h in held), 1)}
    return results


def render_trade_lineage_timeline(chains, players_df, global_team_lookup):
    """One readable chronological story per asset in the trade: where it came from, this trade,
    then everywhere it went — ending with what it ultimately became.

    Deliberately not a Sankey (which this used to be). A Sankey's whole visual language is link
    WIDTH = quantity, but every hop here is one asset moving once, so every link was the same
    width and the diagram carried no magnitude information at all — while the only thing anyone
    actually wants to read, which asset moved and what it turned into, sat in hover-only link
    labels. Provenance is a timeline, so it reads as one.

    Returns a list of (asset_label, markdown_lines, outcome) — empty if no asset had any
    history on either side of this trade."""
    blocks = []
    for chain in chains:
        backward, forward = chain["backward"], chain["forward"]
        if not backward and not forward:
            continue
        # backward hops come back closest-first; reverse so the story reads oldest -> newest
        ordered = [("before", h) for h in reversed(backward)] + [("after", h) for h in forward]
        lines = []
        last_holder = None
        for phase, hop in ordered:
            if phase == "after" and not any(l.startswith("**◆") for l in lines):
                lines.append("**◆ this trade**")
                # The seed trade is itself a change of hands, so the next event must render even
                # if it returns the asset to whoever held it before. Without this reset, an
                # asset traded away and later reacquired lost the hop back: it matched the
                # previously-shown holder and got collapsed as a no-op.
                last_holder = None
            event = hop["event"]
            holder = (event["league_id"], event["roster_id"])
            # Collapse consecutive events that leave the asset with the same team (a drop and
            # re-add by one manager, say) -- the trail is about where an asset CHANGED hands.
            if event["type"] != "draft" and holder == last_holder:
                continue
            lines.append(f"↳ {event_label(event, global_team_lookup, players_df)}")
            last_holder = holder
        if not any(l.startswith("**◆") for l in lines):
            lines.append("**◆ this trade**")

        outcome = None
        if forward:
            def team_of(event):
                return escape_markdown(global_team_lookup.get(
                    (event["league_id"], event["roster_id"]), f"Roster {event['roster_id']}"))

            # The draft is the headline for a pick — it's the moment the asset became a real
            # player — so look for it anywhere in the trail, not just at the end. A later waiver
            # or trade would otherwise bury the one thing worth reading.
            drafted = next((h["event"] for h in forward
                            if h["event"]["type"] == "draft" and h["event"].get("player_id")), None)
            last = forward[-1]["event"]
            if drafted is not None:
                pid = drafted["player_id"]
                name = (players_df.loc[pid, "full_name"]
                        if pid in players_df.index else str(pid))
                outcome = f"became **{escape_markdown(name)}**"
                if last is not drafted:
                    outcome += f", now with **{team_of(last)}**"
                else:
                    outcome += f" ({team_of(drafted)})"
            else:
                outcome = f"ended up with **{team_of(last)}**"
        blocks.append((chain.get("asset", "Asset"), lines, outcome))
    return blocks


def get_all_ever_rostered_ids():
    league_ids = load_table("SELECT league_id FROM league_seasons")["league_id"].tolist()
    ids = set()
    for lid in league_ids:
        ids |= get_all_rostered_player_ids(lid)
    return ids


# ---------------------------------------------------------------------------
# Rookie Draft Center — grades every rookie draft (excludes the 2024 startup
# draft; distinguished via Sleeper's own settings.player_type: 0 = startup/
# all players, 1 = rookies only) using the same value heuristic as Team
# Pages, plus a league-wide view of future draft pick value.
# ---------------------------------------------------------------------------
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
    """Grades each pick by where the player ranks WITHIN his own draft class against where he
    was taken: the 5th-best player going 27th is a steal, the 20th-best going 2nd is a reach.

    Ranking rather than subtracting values, because the two sides were never on the same scale.
    A player's consensus value is a percentile against the whole player pool, where any rostered
    rookie lands in the 70s or 80s, while the old expectation curve decayed to about 7 by the
    end of the draft -- so every late pick banked a large positive score automatically and the
    first pick could not win even when it was the best player in the class. Both real drafts
    graded picks 26-32 as their five best and picks 1-3 as their worst, which meant Ashton Jeanty
    and Jeremiyah Love were the worst picks of their drafts. Comparing ranks is immune to that:
    it doesn't care what scale the values are on, only their order."""
    picks = get_draft_picks_detail(draft_id)
    rows = []
    for _, p in picks.iterrows():
        pid = p["player_id"]
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
            "career_value": round(career, 1),
        })
    board = pd.DataFrame(rows)
    if board.empty:
        return pd.DataFrame(columns=["pick_no", "round", "roster_id", "team", "player",
                                      "position", "career_value", "class_rank", "delta"])
    # Ties (notably everyone with no value on record) fall back to draft order, so an unvalued
    # player is treated as having gone about where he was expected to rather than as a disaster.
    board = board.sort_values(["career_value", "pick_no"], ascending=[False, True])
    board["class_rank"] = range(1, len(board) + 1)
    board["delta"] = board["pick_no"] - board["class_rank"]
    return board.sort_values("pick_no").reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_rookie_draft_grades(draft_id, league_id, latest_season_year, cache_key):
    career_value_table = get_historical_value_table(latest_season_year, cache_key)
    return grade_rookie_draft(draft_id, league_id, career_value_table, get_global_team_lookup(), get_players_df())


def summarize_team_grades(draft_board):
    """Team draft grade from the slots each of its picks gained or lost. Banded on distance from
    the league average (zscore_to_grade) rather than rank: with eight teams a rank percentile can
    only take eight values, so a team that drafted far better than everyone landed in the same
    band as one that merely edged them out."""
    team_deltas = draft_board.groupby("team")["delta"].sum().reset_index()
    team_deltas["grade"] = zscore_to_grade(team_deltas["delta"])
    return team_deltas.sort_values("delta", ascending=False)


def draft_as_of_date(draft_row):
    """Real-world date closest to when a draft actually finished. `last_picked`
    (when the final pick was made) is the truest 'draft day' anchor; falls back
    to `start_time` (when it was scheduled to start, in case the draft finished
    async over several days), then to the season's own end-of-year anchor for
    a draft synced before these columns existed (start_time/last_picked both
    null) — same fallback pattern as trade_as_of_date."""
    for field in ("last_picked", "start_time"):
        ts = draft_row.get(field)
        # pd.notna(), not a plain truthy check: a missing value coerced into a
        # pandas Series (as get_draft_row's fallback path does) lands as NaN,
        # not None -- and NaN is truthy in Python, so `if ts:` would let it
        # through and crash datetime.fromtimestamp(nan) downstream.
        if pd.notna(ts):
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
    return season_end_as_of_date(draft_row["season"])


def get_draft_row(draft_id):
    """start_time/last_picked are new columns (see blue_ballers_sync.py's
    ensure_column) -- a deployed app whose local DB copy predates that
    sync's schema migration would hard-crash querying them directly, so this
    falls back to a query without them (draft_as_of_date's season-end
    fallback still applies) rather than taking the whole page down for
    however long it takes the next successful sync+refresh to land."""
    try:
        return load_table(
            "SELECT draft_id, season, start_time, last_picked FROM drafts WHERE draft_id = ?",
            params=(draft_id,),
        ).iloc[0]
    except Exception:
        row = load_table("SELECT draft_id, season FROM drafts WHERE draft_id = ?", params=(draft_id,)).iloc[0]
        row["start_time"], row["last_picked"] = None, None
        return row


def build_draft_day_grades(draft_id, league_id, cache_key):
    """Was this pick good AT THE TIME it was made? Compares the drafted
    player's real market value shortly after the draft against what the
    market itself said that exact slot was worth at that same moment — both
    real external-consensus numbers, unlike the existing Draft Grades
    section above (which stays on its abstract pick-position curve
    deliberately, since it also feeds Hall of Fame's Best/Worst Draft Pick
    and GM Profiles' Drafting score, and changing its meaning there wasn't
    part of this ask). `our_slot` comes from get_exact_draft_slot_map_cached
    (the same verified real-draft-order lookup the Asset Trail lineage
    feature uses) since this league's rookie draft is straight-order —
    one roster keeps the same slot in every round."""
    draft_row = get_draft_row(draft_id)
    as_of = draft_as_of_date(draft_row)
    picks = get_draft_picks_detail(draft_id)
    slot_map = get_exact_draft_slot_map_cached(league_id, cache_key)
    player_values = compute_consensus_player_values(cache_key, as_of_date=as_of)
    players_df = get_players_df()
    global_team_lookup = get_global_team_lookup()

    rows = []
    for _, p in picks.iterrows():
        pid, roster_id = p["player_id"], int(p["roster_id"])
        our_slot = slot_map.get(roster_id)
        slot_value = (compute_consensus_pick_value(draft_row["season"], int(p["round"]), our_slot,
                                                     cache_key, as_of_date=as_of)
                      if our_slot else None)
        player_value = player_values.get(pid, 0.0)
        info = players_df.loc[pid] if pid in players_df.index else None
        rows.append({
            "pick_no": int(p["pick_no"]), "round": int(p["round"]), "roster_id": roster_id,
            "team": global_team_lookup.get((league_id, roster_id), f"Roster {roster_id}"),
            "player": info["full_name"] if info is not None else pid,
            "position": info["position"] if info is not None else "",
            "slot_value": round(slot_value, 1) if slot_value is not None else None,
            "player_value": round(player_value, 1),
            "delta": round(player_value - slot_value, 1) if slot_value is not None else None,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def get_draft_day_grades(draft_id, league_id, cache_key):
    return build_draft_day_grades(draft_id, league_id, cache_key)


def build_who_knows_ball(cache_key):
    """Every rookie ever drafted in this league, ranked by how much their real
    market value has moved from draft day to today — a pure hindsight
    leaderboard. Continuously current since it's just reading live archive
    data on every call (no separate monthly refresh job needed, despite the
    original 'live, monthly-refreshed' framing — the archive itself updates
    every sync, so this is at least that fresh automatically)."""
    rookie_drafts = get_rookie_draft_list(cache_key)
    if rookie_drafts.empty:
        return pd.DataFrame(columns=["season", "pick_no", "team", "player", "position",
                                      "draft_day_value", "now_value", "swing"])

    live_values = compute_consensus_player_values(cache_key)
    players_df = get_players_df()
    global_team_lookup = get_global_team_lookup()

    rows = []
    for _, d in rookie_drafts.iterrows():
        draft_row = get_draft_row(d["draft_id"])
        as_of = draft_as_of_date(draft_row)
        draft_day_values = compute_consensus_player_values(cache_key, as_of_date=as_of)
        picks = get_draft_picks_detail(d["draft_id"])
        for _, p in picks.iterrows():
            pid, roster_id = p["player_id"], int(p["roster_id"])
            info = players_df.loc[pid] if pid in players_df.index else None
            draft_day_val = draft_day_values.get(pid, 0.0)
            now_val = live_values.get(pid, 0.0)
            rows.append({
                "season": d["season"], "pick_no": int(p["pick_no"]),
                "team": global_team_lookup.get((d["league_id"], roster_id), f"Roster {roster_id}"),
                "player": info["full_name"] if info is not None else pid,
                "position": info["position"] if info is not None else "",
                "draft_day_value": round(draft_day_val, 1), "now_value": round(now_val, 1),
                "swing": round(now_val - draft_day_val, 1),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def get_who_knows_ball(cache_key):
    return build_who_knows_ball(cache_key)


# ---------------------------------------------------------------------------
# Player Risk Score — three distinct, real signals blended into one 0-100
# per-player risk score: durability (real injury-report history — a player
# like CMC who gets hurt often), boom-bust (week-to-week scoring volatility —
# a player like Jameson Williams who's either a monster game or a dud), and
# hype (value run up on perceived opportunity rather than proven production —
# a player like a low-capital rookie whose price has outpaced what he's
# actually shown). Each signal is percentile-ranked to 0-100 independently
# before combining, same reconciliation approach as the external value
# consensus elsewhere in this file. A player missing a signal (e.g. no
# tracked injury-report history) is just averaged over whichever signals it
# does have — not penalized toward 0 for a signal that's simply unknown.
# ---------------------------------------------------------------------------
def compute_injury_designation_risk():
    """Percentile rank of how often a player has appeared on the real NFL
    injury report as Doubtful/Out (Questionable deliberately excluded — too
    common/mild on its own; most players get tagged Questionable at some
    point without missing real time) across nflverse's synced history
    (injury_reports table, blue_ballers_sync.py). Catches players who play
    through nagging issues even when they don't miss the game entirely —
    complements compute_games_missed_risk below, which catches full
    absences this weekly designation alone misses once a player goes on IR
    (see that function's docstring). A player absent from this table has no
    tracked injury-report history at all — simply absent from the returned
    dict, not assumed durable. `injury_reports` is a brand-new table
    (blue_ballers_sync.py) — a deployed app whose local DB copy predates
    that sync's schema migration would hard-crash querying it directly,
    same gotcha as get_draft_row, so this falls back to an empty dict
    rather than taking down GM Profiles/Team Pages for however long it
    takes the next successful sync+refresh to land.

    Percentile pool is restricted to players ever rostered in this league —
    nflverse tracks every NFL player, including defensive/special-teams
    players this league never rosters, and their injury-report cadence isn't
    comparable to offensive skill players'. Confirmed empirically: without
    this filter, a handful of fringe defensive backs with one real injury
    designation ranked above actual rostered players just from being
    percentile-ranked against a much broader, differently-distributed pool."""
    try:
        df = load_table(
            "SELECT sleeper_player_id FROM injury_reports WHERE report_status IN ('Doubtful', 'Out')"
        )
    except Exception:
        return {}
    if df.empty:
        return {}
    relevant_ids = get_all_ever_rostered_ids()
    df = df[df["sleeper_player_id"].isin(relevant_ids)]
    if df.empty:
        return {}
    counts = df.groupby("sleeper_player_id").size()
    return (counts.rank(pct=True) * 100).round(1).to_dict()


def compute_games_missed_risk():
    """Percentile rank of real games missed, detected from snap-count
    ABSENCE: for each real NFL season, a player missing a snap-count row for
    a week their own (most common) team otherwise has rows for (i.e. that
    team played a real game) almost certainly missed that game — IR,
    inactive, or a healthy scratch, regardless of whether the weekly injury
    report happened to still be tagging them that week. Confirmed
    empirically against a real 2024 case: the injury-report tag alone only
    flagged 2 weeks, but snap-count absence correctly showed 13 of 17
    missed games for a player who spent most of that season on IR — exactly
    the gap compute_injury_designation_risk alone can't see, since a player
    typically drops off the weekly report entirely once on IR. `snap_counts`
    is a brand-new table — same missing-table fallback as get_draft_row/
    compute_injury_designation_risk."""
    try:
        df = load_table("SELECT season, week, sleeper_player_id, team FROM snap_counts")
    except Exception:
        return {}
    if df.empty:
        return {}
    relevant_ids = get_all_ever_rostered_ids()

    missed = {}
    for season, season_df in df.groupby("season"):
        team_weeks = season_df.groupby("team")["week"].apply(set).to_dict()
        player_weeks = season_df.groupby("sleeper_player_id")["week"].apply(set)
        player_team = _most_common_team(season_df, "sleeper_player_id")
        for pid, weeks_played in player_weeks.items():
            if pid not in relevant_ids:
                continue
            real_weeks = team_weeks.get(player_team[pid], set())
            count = len(real_weeks - weeks_played)
            missed[pid] = missed.get(pid, 0) + count
    if not missed:
        return {}
    return (pd.Series(missed).rank(pct=True) * 100).round(1).to_dict()


def compute_durability_risk(cache_key):
    """Combines both real durability signals — games actually missed
    (compute_games_missed_risk, the more complete signal) and weekly
    injury-report designations (compute_injury_designation_risk, catches
    playing-through-it risk games-missed alone wouldn't show) — via simple
    mean of whichever a player has. `cache_key` isn't used directly (neither
    sub-signal is itself cached), kept for a consistent call signature with
    the other two Player Risk Score inputs."""
    games_missed = compute_games_missed_risk()
    designations = compute_injury_designation_risk()
    all_ids = set(games_missed) | set(designations)
    combined = {}
    for pid in all_ids:
        parts = [d[pid] for d in (games_missed, designations) if pid in d]
        if parts:
            combined[pid] = round(float(np.mean(parts)), 1)
    return combined


def build_player_score_volatility():
    """Percentile rank of a player's boom-bust variance: coefficient of
    variation (std/mean) of real weekly fantasy points, across every week
    they were actually in a roster's STARTERS list (any roster, any synced
    season) — isolating real on-field variance from bye/bench weeks, which
    would otherwise masquerade as "bust" games. Needs >=3 real started weeks
    to be meaningful; fewer than that has no real variance to measure."""
    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    points_by_player = {}
    for _, s in all_seasons.iterrows():
        lid = s["league_id"]
        reg_weeks = get_playoff_settings(lid)["regular_season_weeks"]
        for week in range(1, reg_weeks + 1):
            df = load_table(
                "SELECT points, starters, players_points FROM matchups WHERE league_id = ? AND week = ?",
                params=(lid, week),
            )
            if df.empty or df["points"].sum() <= 0:
                continue
            for _, row in df.iterrows():
                starter_ids = [pid for pid in parse_json_field(row["starters"], []) if pid != "0"]
                points_map = parse_json_field(row["players_points"], {})
                for pid in starter_ids:
                    points_by_player.setdefault(pid, []).append(points_map.get(pid, 0.0))

    cv = {}
    for pid, pts in points_by_player.items():
        if len(pts) < 3:
            continue
        arr = np.array(pts)
        mean = arr.mean()
        if mean > 0:
            cv[pid] = float(arr.std() / mean)
    if not cv:
        return {}
    return (pd.Series(cv).rank(pct=True) * 100).round(1).to_dict()


def compute_hype_risk(cache_key):
    """Percentile rank of the gap between a player's real external-consensus
    value percentile (compute_consensus_player_values) and their own recent
    real-production percentile (latest cumulative PPG from
    build_player_ppg_timeline, the same data Stock Market already uses) — a
    big positive gap means the market prices this player well above what
    their own recent production justifies (opportunity/story priced in,
    results not yet proven). Only meaningful for players with both a
    tracked value AND real production history; missing either, they're
    simply absent from the result."""
    value_pct = compute_consensus_player_values(cache_key)
    timeline = build_player_ppg_timeline()
    if not value_pct or timeline.empty:
        return {}
    latest_ppg = timeline.sort_values(["season", "week"]).groupby("player_id").tail(1)
    production_pct = (latest_ppg.set_index("player_id")["cum_ppg"].rank(pct=True) * 100)

    gap = {pid: vpct - production_pct.loc[pid] for pid, vpct in value_pct.items() if pid in production_pct.index}
    if not gap:
        return {}
    return (pd.Series(gap).rank(pct=True) * 100).round(1).to_dict()


def compute_player_risk_scores(cache_key):
    durability = compute_durability_risk(cache_key)
    volatility = build_player_score_volatility()
    hype = compute_hype_risk(cache_key)
    all_ids = set(durability) | set(volatility) | set(hype)
    scores = {}
    for pid in all_ids:
        parts = [d.get(pid) for d in (durability, volatility, hype) if pid in d]
        if parts:
            scores[pid] = round(float(np.mean(parts)), 1)
    return scores


# ---------------------------------------------------------------------------
# GM Profiles — seven ratings per manager, each a 0-100 percentile score
# among the league's managers, built entirely from data/logic already
# established elsewhere in this file (Rookie Draft/Trade grading, Team Page
# metrics, the season simulator's deterministic replay of completed seasons,
# and Player Risk Score above for Risk Taking).
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
    # project_rosters=False: same exact-replay reasoning as get_exact_draft_slot_map.
    distributions = estimate_team_distributions(league_id, standings_local, project_rosters=False)
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
        for side in grade_trade(txn, vt, pp, global_team_lookup_local, txn["league_id"], cache_key=cache_key):
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
    # Risk Taking: average real Player Risk Score (durability + boom-bust + hype, see above)
    # across every player on the roster.
    latest_standings = get_standings(latest_league_id)
    latest_value_table = get_value_table(latest_league_id, latest_season_year, cache_key)
    player_risk = compute_player_risk_scores(cache_key)
    roster_construction = {}
    risk_raw = {}
    for _, row in latest_standings.iterrows():
        roster_id = int(row["roster_id"])
        owner = row["owner_id"]
        player_ids, starter_ids = get_roster(latest_league_id, roster_id)
        metrics = team_overview_metrics(latest_value_table, player_ids, starter_ids)
        total = metrics["starter_value"] + metrics["bench_value"]
        roster_construction[owner] = metrics["starter_value"] / total if total else 0.5
        roster_risks = [player_risk[pid] for pid in player_ids if pid in player_risk]
        risk_raw[owner] = float(np.mean(roster_risks)) if roster_risks else 50.0

    # Neilee is Buddy's co-manager (viewing access only, not a real GM) — exclude her.
    all_owners = sorted(
        owner for owner, name in manager_name_lookup.items() if "neilee" not in name.lower()
    )

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
# already established above. "Most Injury Luck" (compute_injury_burden below)
# uses real nflverse snap-count data (see Player Risk Score above) — real
# fantasy points lost to real missed games across a roster's whole history.
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


def compute_injury_burden():
    """Per-(season, owner) real fantasy points 'lost' to missed games — for
    every player ever on a manager's roster that season, real games missed
    (snap-count absence, same signal as compute_games_missed_risk) times
    that player's own real PPG that season (their own best available
    estimate of what they'd have scored had they played), summed across
    the whole roster. Bounded to this league's real regular season
    (reg_weeks) — the games that actually decided standings, same scope as
    Luck Index. `snap_counts` is a new table — same missing-table fallback
    as get_draft_row/compute_games_missed_risk."""
    try:
        snaps = load_table("SELECT season, week, sleeper_player_id, team FROM snap_counts")
    except Exception:
        return pd.DataFrame(columns=["season", "owner_id", "injury_burden"])
    if snaps.empty:
        return pd.DataFrame(columns=["season", "owner_id", "injury_burden"])

    ppg_timeline = build_player_ppg_timeline()
    final_ppg = ppg_timeline.sort_values(["season", "week"]).groupby(["season", "player_id"]).tail(1)
    ppg_lookup = {(row["season"], row["player_id"]): row["cum_ppg"] for _, row in final_ppg.iterrows()}

    all_seasons = load_table("SELECT league_id, season FROM league_seasons ORDER BY season ASC")
    rows = []
    for _, s in all_seasons.iterrows():
        lid, season = s["league_id"], s["season"]
        reg_weeks = get_playoff_settings(lid)["regular_season_weeks"]
        season_snaps = snaps[(snaps["season"] == season) & (snaps["week"] <= reg_weeks)]
        if season_snaps.empty:
            continue
        team_weeks = season_snaps.groupby("team")["week"].apply(set).to_dict()
        player_weeks = season_snaps.groupby("sleeper_player_id")["week"].apply(set)
        player_team = _most_common_team(season_snaps, "sleeper_player_id")

        rosters_df = load_table("SELECT owner_id, players FROM rosters WHERE league_id = ?", params=(lid,))
        for _, roster_row in rosters_df.iterrows():
            owner = roster_row["owner_id"]
            burden = 0.0
            for pid in parse_json_field(roster_row["players"], []):
                if pid not in player_team.index:
                    continue
                real_weeks = team_weeks.get(player_team[pid], set())
                missed = len(real_weeks - player_weeks.get(pid, set()))
                burden += missed * ppg_lookup.get((season, pid), 0.0)
            rows.append({"season": season, "owner_id": owner, "injury_burden": round(burden, 1)})
    return pd.DataFrame(rows)


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
        for side in grade_trade(txn, vt, pp, global_team_lookup_local, txn["league_id"], cache_key=cache_key):
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
    injury_burden = compute_injury_burden()
    career_burden = (injury_burden.groupby("owner_id")["injury_burden"].sum().reset_index()
                      if not injury_burden.empty else pd.DataFrame(columns=["owner_id", "injury_burden"]))

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
    leaderboard = leaderboard.merge(career_burden, on="owner_id", how="left")
    leaderboard = leaderboard.sort_values("career_wins", ascending=False)

    biggest_choke = max(choke_candidates, key=lambda c: c["choke_score"]) if choke_candidates else None
    if biggest_choke:
        biggest_choke = {**biggest_choke, "manager": manager_name(biggest_choke["owner_id"])}
    most_unlucky = leaderboard.loc[leaderboard["luck_index"].idxmin()] if not leaderboard.empty else None
    most_injury_luck = (leaderboard.loc[leaderboard["injury_burden"].idxmax()]
                         if not leaderboard.empty and leaderboard["injury_burden"].notna().any() else None)

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
        "most_injury_luck": most_injury_luck,
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
    # Capped at this league's real final week (the same round2_weeks[1] leakage guard used
    # elsewhere, e.g. Hall of Fame's weekly-score records) -- get_last_completed_week itself
    # isn't changed since 5 other call sites rely on its current behavior, but real (if smaller)
    # NFL week-18 scoring leaks into matchups even though this league's fantasy season ends at
    # week 17, and offering "Week 18" here would show a bogus MVP/Upset for a week that isn't
    # real for this league.
    if last_week is not None:
        last_week = min(last_week, get_playoff_settings(league_id)["round2_weeks"][1])
    if last_week is None:
        st.info("No completed weeks yet this season.")
    else:
        week_choice = st.selectbox("Week", list(range(last_week, 0, -1)))
        week_matchups = get_week_matchups(league_id, week_choice)
    
        if not week_matchups.empty:
            highest = week_matchups.loc[week_matchups["points"].idxmax()]
            lowest = week_matchups.loc[week_matchups["points"].idxmin()]
            week_mvp = get_week_mvp(league_id, week_choice, players_df)

            upset_history = get_upset_history(synced_at)
            week_upsets = upset_history[(upset_history["season"] == season_choice) &
                                         (upset_history["week"] == week_choice)]
            biggest_upset = (week_upsets.loc[week_upsets["winner_pregame_prob"].idxmin()]
                              if not week_upsets.empty else None)
            if biggest_upset is not None and biggest_upset["winner_pregame_prob"] >= 0.5:
                biggest_upset = None  # nobody was actually the underdog this week -- not a real upset

            two_team_games = []
            for _, group in week_matchups.groupby("matchup_id"):
                if len(group) == 2:
                    a, b = group.iloc[0], group.iloc[1]
                    two_team_games.append({"a": a, "b": b, "margin": abs(a["points"] - b["points"])})
            closest_game = min(two_team_games, key=lambda g: g["margin"]) if two_team_games else None
            biggest_blowout = max(two_team_games, key=lambda g: g["margin"]) if two_team_games else None

            st.subheader(f"Week {week_choice} Headlines")
            c1, c2, c3, c4, c5 = st.columns(5)
            metric_block(c1, "Top Score", highest["team_name"], f"{highest['points']:.1f} pts")
            metric_block(c2, "Lowest Score", lowest["team_name"], f"{lowest['points']:.1f} pts")
            if closest_game:
                metric_block(c3, "Closest Game",
                             f"{closest_game['a']['team_name']} vs {closest_game['b']['team_name']}",
                             f"{closest_game['margin']:.1f} pt margin")
            if week_mvp:
                metric_block(c4, "Weekly MVP", f"{week_mvp['player']} ({week_mvp['position']})",
                             f"+{week_mvp['margin']:.1f} vs. position avg")
            if biggest_upset is not None:
                metric_block(c5, "Biggest Upset", f"{biggest_upset['winner']} over {biggest_upset['loser']}",
                             f"{biggest_upset['winner_pregame_prob']:.0%} pre-game win chance")
    
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

            st.subheader("League News")
            recap_article = get_league_news(league_id, week_choice)
            if recap_article:
                st.caption("AI-generated recap (Google Gemini), written from this week's real results.")
                st.markdown(recap_article)
            else:
                st.caption("No AI recap generated for this week yet.")


def page_recent_transactions():
    st.header("Recent Transactions")
    transactions = get_recent_transactions(league_id)
    if transactions.empty:
        st.info("No transactions recorded yet.")
    else:
        for _, txn in transactions.iterrows():
            st.subheader(f"{txn['type'].replace('_', ' ').title()} — "
                          f"{week_label(season_choice, txn['week'], txn.get('created'))}")
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
        "Grades and scores below use real dynasty market consensus (DynastyProcess + FantasyCalc), "
        "not just an in-house guess — the same value that powers Trade Center's grading."
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
    # Contender/Future read off grade_row, not `metrics` — build_league_grades computes these
    # two on a different (better) basis than team_overview_metrics' raw sums, and showing one
    # number next to a rank derived from the other would contradict itself.
    c3.metric("Contender Score", f"{grade_row['contender_score']:.0f}", f"#{contender_rank} of {n_teams}")
    c4.metric("Future Score", f"{grade_row['future_score']:,.0f}", f"#{future_rank} of {n_teams}")
    st.caption(
        "Contender Score is projected points per week from the best lineup this roster can "
        "field, after discounting for byes and injury risk. Future Score is real market value "
        "of the roster core plus every future draft pick it owns — picks are a real part of a "
        "dynasty team's future, and a rebuilding team's own first is worth more than a "
        "contender's. The Overall grade blends the two and is graded on how far a team sits "
        "from the league average, so a genuine gap shows up as a real grade gap."
    )

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
    curve_df = project_team_value_curve(league_id, roster_id, season_year, players_df, synced_at)
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
        lambda r: value_pick_row(r, power_pct, str(first_unspent_pick_season(league_id, season_year)),
                                  odds, cache_key=synced_at), axis=1
    )
    team_picks["Original Team"] = team_picks["original_roster_id"].map(team_lookup)
    team_picks = team_picks.sort_values(["season", "round"])
    st.dataframe(
        team_picks[["season", "round", "Original Team", "Value"]]
        .rename(columns={"season": "Season", "round": "Round"}),
        use_container_width=True, hide_index=True,
    )
    
    st.subheader("Roster")
    st.caption(
        "Risk blends three real signals into one 0-100 score: durability (real injury-report "
        "history), boom-bust (week-to-week scoring volatility), and hype (value that's outpaced "
        "actual recent production). Higher = riskier."
    )
    player_risk = compute_player_risk_scores(synced_at)
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
            "Risk": player_risk.get(pid),
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
        "Every player who's ever been rostered in this league, valued with real external "
        "market consensus (DynastyProcess + FantasyCalc) as of that exact week — not today's "
        "price replayed backwards — so this is a genuine historical price series, continuous "
        "across every synced season, not a single end-of-season snapshot. A player only shows "
        "up starting the week they first recorded real production in this league."
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
        "Every trade across every synced season, auto-graded WITHOUT hindsight — each asset is valued "
        "off real external market consensus (DynastyProcess + FantasyCalc) as of that trade's own date, "
        "not today's price, so a since-dropped player still counts instead of scoring 0, and the grade "
        "reflects what was known at the time. Grade starts from value received vs. given up per side, "
        "then gets knocked down a full letter if it doesn't fit that team's own timeline at the time "
        "(e.g. a rebuilding team giving up future assets for a marginal win-now piece drops from an A to "
        "a B, even if the raw value was fair) — see the Context Fit note under each grade. Winner/Loser "
        "is based on net value gained, not the letter grade. Want to see it with today's hindsight "
        "instead? Open \"How this trade aged\" under any trade."
    )
    
    st.subheader("Asset Trail")
    st.caption(
        "Pick any trade and follow each asset it involved: where the piece came from before this "
        "trade, everywhere it went after, and — for a traded pick — the actual player it turned "
        "into once someone finally drafted it."
    )
    tree_trades = get_all_trades(synced_at)
    if tree_trades.empty:
        st.info("No trades recorded yet.")
    else:
        tree_labels = []
        for _, t in tree_trades.iterrows():
            summary = " / ".join(describe_trade(t, players_df, t["league_id"], global_team_lookup))
            stamp = week_label(t["season"], t["week"], t.get("created"))
            tree_labels.append(f"{t['season']} {stamp}: {summary}"[:120])
        selected_tree_label = st.selectbox("Trade", tree_labels, key="trade_tree_pick")
        seed_txn = tree_trades.iloc[tree_labels.index(selected_tree_label)]

        st.markdown("##### Where it stands today")
        st.caption(
            "Each side's current holdings that trace back to this trade — following every piece "
            "through the moves that came after it, so a player later flipped for someone else "
            "shows up as whoever is on the roster now. Value is today's real market price, so "
            "this is the hindsight verdict, not the at-the-time grade above. When several pieces "
            "went out in one trade the return can't be pinned to any one of them, so it's split "
            "evenly between them and each asset shows the share it's credited with."
        )
        conclusion = build_trade_conclusion(seed_txn, players_df, synced_at)
        name_by_owner = get_manager_name_lookup()
        if not conclusion:
            st.info("Couldn't resolve the sides of this trade.")
        else:
            # Rendered as ONE markdown block on purpose. This used to be a per-side column
            # layout, but the number of columns changes with the number of teams in the trade,
            # and the browser reused DOM nodes across reruns and kept showing pieces of the
            # trade you'd just navigated away from. A single element of fixed shape has no
            # children to leave behind, so switching trades can't leave residue.
            def cell(text):
                # Table cells additionally need '|' neutralized, which escape_markdown
                # (built for inline bold/colour wrapping) doesn't handle.
                return escape_markdown(text).replace("|", "\\|")

            # Zero-sum by construction in a two-team trade: what one side is left holding IS
            # what the other side gave away, so one side's surplus is the other's shortfall.
            totals = {o: info["value_today"] for o, info in conclusion.items()}
            ranked = sorted(totals.items(), key=lambda kv: -kv[1])
            parts = []
            if len(ranked) >= 2 and round(ranked[0][1] - ranked[-1][1]) > 0:
                lead = ranked[0][1] - ranked[1][1]
                winner = escape_markdown(name_by_owner.get(ranked[0][0], str(ranked[0][0])))
                parts.append(f"### {winner} won this trade")
                parts.append(f"Ahead by **{lead:,.0f}** in market value today, counting everything "
                              f"each side's pieces have since turned into.")
            else:
                parts.append("### Dead even")
                parts.append("What each side still holds from this trade is worth the same today.")
            parts.append("")
            for owner, info in conclusion.items():
                others = [v for o, v in totals.items() if o != owner]
                net = info["value_today"] - (max(others) if others else 0.0)
                parts.append(f"##### {escape_markdown(name_by_owner.get(owner, str(owner)))} — "
                              f"{info['value_today']:,.0f} ({net:+,.0f})")
                parts.append("")
                gained = ", ".join(cell(r) for r in info["received"]) or "—"
                given = ", ".join(cell(r) for r in info["gave_up"]) or "—"
                parts.append(f"| Gave up ({len(info['gave_up'])}) | Got ({len(info['received'])}) |")
                parts.append("| :--- | :--- |")
                parts.append(f"| {given} | {gained} |")
                parts.append("")

                if info["holding"]:
                    parts.append(f"**Has today ({len(info['holding'])})**")
                    parts.append("")
                    parts.append("| Asset | Value | How it traces back |")
                    parts.append("| :--- | ---: | :--- |")
                    for h in sorted(info["holding"], key=lambda x: -x["value"]):
                        if h["path"] or h["from"] != h["label"]:
                            steps = " → ".join(cell(x) for x in [h["from"]] + h["path"])
                        else:
                            steps = "straight from this trade"
                        if h.get("acquired"):
                            steps += f" {cell(h['acquired'])}"
                        # Plain text only: st.markdown doesn't render HTML unless explicitly
                        # allowed, and enabling that for cells built from user-supplied team
                        # names isn't worth a line break.
                        worth = f"{h['value']:,.0f}"
                        if h.get("share", 1.0) < 0.99:
                            worth += f" ({h['share']:.0%} of {h['full_value']:,.0f})"
                        parts.append(f"| **{cell(h['label'])}** | {worth} | {steps} |")
                    parts.append("")
                else:
                    parts.append("**Has today (0)** — nothing left from this trade")
                    parts.append("")

                footnotes = []
                if info["lost"]:
                    footnotes.append("Dropped along the way: "
                                      + ", ".join(escape_markdown(l["label"]) for l in info["lost"]))
                if info["moved_on"]:
                    footnotes.append("Traded on before the draft: "
                                      + ", ".join(escape_markdown(m["label"]) for m in info["moved_on"]))
                for note in footnotes:
                    parts.append(f"*{note}*  ")
                parts.append("")
            st.markdown("\n".join(parts))

        with st.expander("Full asset-by-asset trail"):
            chains = build_trade_lineage(seed_txn, players_df, synced_at)
            blocks = render_trade_lineage_timeline(chains, players_df, global_team_lookup)
            if not blocks:
                st.info("Every asset in this trade was acquired here and hasn't moved since — "
                        "no trail to follow yet.")
            else:
                for asset, lines, outcome in blocks:
                    header = f"**{escape_markdown(str(asset))}**"
                    if outcome:
                        header += f" — {outcome}"
                    st.markdown(header)
                    st.markdown("  \n".join(lines))
                    st.divider()

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
        # Every trade rendered here is fully graded and aged, and Streamlit runs the body of the
        # "how this trade aged" expander whether or not it's open, so the whole history costs
        # tens of seconds to draw. Newest-first already, so the recent page is the useful one;
        # searching narrows the list before this and is unaffected.
        visible = all_trades
        if len(all_trades) > TRADE_HISTORY_PAGE:
            if st.checkbox(f"Show all {len(all_trades)} trades (slower)", value=False):
                visible = all_trades
            else:
                visible = all_trades.head(TRADE_HISTORY_PAGE)
                st.caption(f"Showing the {len(visible)} most recent — search above to find older ones.")
        for _, txn in visible.iterrows():
            trade_value_table = get_historical_value_table(int(txn["season"]), synced_at)
            trade_power_pct = get_season_power_pct(txn["league_id"], synced_at)
            sides = grade_trade(txn, trade_value_table, trade_power_pct, global_team_lookup, txn["league_id"],
                                 cache_key=synced_at)
            trade_league_grades = get_league_grades(txn["league_id"], int(txn["season"]), synced_at)

            st.write(f"**{txn['season']} — {week_label(txn['season'], txn['week'], txn.get('created'))}**")
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

            with st.expander("How this trade aged"):
                aging = build_trade_aging_detail(txn, trade_power_pct, players_df, synced_at)
                if aging.empty:
                    st.caption("No assets to compare.")
                else:
                    st.dataframe(
                        aging.rename(columns={"asset": "Asset", "then": "Value Then",
                                               "now": "Value Now", "swing": "Swing"}),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(
                        "Real market value as of this trade's own date vs. today — a pure hindsight "
                        "view, doesn't change the grade above. Shown as market value rather than a "
                        "percentile rank because a player already near the top of the pool has nowhere "
                        "left to climb: he can gain a fifth of his real trade value and still rank "
                        "fractionally lower, since everyone around him rose too. Trades from before this "
                        "feature's archive started only have DynastyProcess's own signal for their "
                        "historical side (FantasyCalc has no historical API), so \"Then\" may be less "
                        "precise than trades graded going forward."
                    )
            st.divider()

def page_rookie_draft():
    st.header("Rookie Draft Center")

    rookie_drafts = get_rookie_draft_list(synced_at)

    if rookie_drafts.empty:
        st.info("No rookie drafts found yet.")
    else:
        st.subheader("Draft Grades")
        st.caption(
            "Career Value is each player's real external market consensus value (DynastyProcess + "
            "FantasyCalc) as of the end of the latest synced season — so a player later dropped still "
            "gets credit for real market value, instead of scoring 0 just because nobody currently "
            "rosters them. Expected Value is a smooth pick-position curve, not a real consensus rookie "
            "ranking — see Draft Day Grades below for a version using real market data on both sides."
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
                    team_grades.rename(columns={"team": "Team", "delta": "Total Slots Gained", "grade": "Grade"}),
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
                draft_board[["pick_no", "round", "team", "player", "position", "class_rank",
                             "career_value", "delta"]]
                .rename(columns={"pick_no": "Pick", "round": "Round", "team": "Team", "player": "Player",
                                  "position": "Pos", "class_rank": "Class Rank", "career_value": "Value",
                                  "delta": "Slots Gained"}),
                use_container_width=True, hide_index=True,
            )

            st.subheader("Draft Day Grades")
            st.caption(
                "Was this pick good AT THE TIME it was made? Both sides here are real external market "
                "consensus (DynastyProcess + FantasyCalc) as of shortly after this draft actually "
                "happened — the player's value then, vs. what the market said that exact slot was worth "
                "then — instead of Draft Grades' abstract pick-position curve above."
            )
            day_grades = get_draft_day_grades(draft_row["draft_id"], draft_row["league_id"], synced_at)
            day_grades_valid = day_grades.dropna(subset=["delta"])
            if day_grades_valid.empty:
                st.info("No real market data reaches back to this draft yet.")
            else:
                day_team_grades = summarize_team_grades(day_grades_valid)
                st.dataframe(
                    style_grades(
                        day_team_grades.rename(columns={"team": "Team", "delta": "Total Value Over Slot", "grade": "Grade"}),
                        ["Grade"],
                    ),
                    use_container_width=True, hide_index=True,
                )
                best_day_pick = day_grades_valid.loc[day_grades_valid["delta"].idxmax()]
                worst_day_pick = day_grades_valid.loc[day_grades_valid["delta"].idxmin()]
                col_best_day, col_worst_day = st.columns(2)
                metric_block(col_best_day, "Best Pick", f"{best_day_pick['player']} ({best_day_pick['team']})",
                              f"+{best_day_pick['delta']:.0f} vs. slot value that day")
                metric_block(col_worst_day, "Worst Pick", f"{worst_day_pick['player']} ({worst_day_pick['team']})",
                              f"{worst_day_pick['delta']:.0f} vs. slot value that day")
                st.write("**Full Draft Board**")
                st.dataframe(
                    day_grades_valid[["pick_no", "round", "team", "player", "position", "slot_value",
                                       "player_value", "delta"]]
                    .rename(columns={"pick_no": "Pick", "round": "Round", "team": "Team", "player": "Player",
                                      "position": "Pos", "slot_value": "Slot Value", "player_value": "Player Value",
                                      "delta": "Delta"}),
                    use_container_width=True, hide_index=True,
                )

    st.subheader("Who Knows Ball")
    st.caption(
        "Every rookie ever drafted in this league, ranked by how much their real market value has moved "
        "since draft day — pure hindsight, and always current since it reads live market data rather "
        "than being refreshed on a schedule."
    )
    wkb = get_who_knows_ball(synced_at)
    if wkb.empty:
        st.info("No rookie drafts found yet.")
    else:
        col_wkb_up, col_wkb_down = st.columns(2)
        with col_wkb_up:
            st.write("**Biggest Risers Since Draft Day**")
            st.dataframe(
                wkb.sort_values("swing", ascending=False).head(10)
                [["season", "player", "position", "draft_day_value", "now_value", "swing"]]
                .rename(columns={"season": "Season", "player": "Player", "position": "Pos",
                                  "draft_day_value": "Draft Day", "now_value": "Now", "swing": "Swing"}),
                use_container_width=True, hide_index=True,
            )
        with col_wkb_down:
            st.write("**Biggest Fallers Since Draft Day**")
            st.dataframe(
                wkb.sort_values("swing", ascending=True).head(10)
                [["season", "player", "position", "draft_day_value", "now_value", "swing"]]
                .rename(columns={"season": "Season", "player": "Player", "position": "Pos",
                                  "draft_day_value": "Draft Day", "now_value": "Now", "swing": "Swing"}),
                use_container_width=True, hide_index=True,
            )

    st.subheader("Draft Pick Value")
    st.caption(
        "Every future pick across the league. Value comes from real market consensus (DynastyProcess + "
        "FantasyCalc), on the same 0-100 scale as player values, so picks and players are directly "
        "comparable in a trade. For this season's picks, value is the expectation over the Championship "
        "Odds simulator's actual projected draft-slot odds (below); picks further out fall back to an "
        "estimated slot from the original team's current strength, since there's nothing left to simulate "
        "that far ahead. Picks far enough out that even the external sources don't reach yet fall back to "
        "an in-house formula, scaled by real-world draft class strength consensus — 2026 is a weak class "
        "(0.88x), 2027 is a strong one (1.12x); other years are neutral until there's a real read on them."
    )
    league_pick_inventory = get_pick_inventory(league_id, int(season_choice), synced_at).copy()
    league_power_pct = dict(zip(rankings["roster_id"], rankings["power_score"].rank(pct=True)))
    league_pick_inventory["Value"] = league_pick_inventory.apply(
        lambda r: value_pick_row(r, league_power_pct,
                                  str(first_unspent_pick_season(league_id, int(season_choice))),
                                  odds, cache_key=synced_at), axis=1
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
        "Risk Taking is each roster's average real Player Risk Score (durability + boom-bust + "
        "hype — see Team Pages' Roster table for the per-player breakdown). Clutch is based purely "
        "on playoff finish (fully-completed seasons only) — regular-season performance doesn't "
        "factor in at all."
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
        "Center. Most Injury Luck is real fantasy points lost to real missed games (nflverse "
        "snap-count data) across a manager's whole roster history, not a general win/loss luck measure."
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

        if hof["most_injury_luck"] is not None:
            mil = hof["most_injury_luck"]
            metric_block(st, "Most Injury Luck", mil["manager"],
                         f"{mil['injury_burden']:.0f} pts lost to real missed games (career)")

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
