"""Updated final possession pipeline: E10 only.

Runs the established E10 integrated-control method and writes only updated_
output files. Earlier experiment pipelines and benchmark-generation code
are excluded.
"""

import argparse

import os

import sys

import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.dirname(HERE)

from games import GAME, GAMES, require, source_frames  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

FIELD_SEP = ":"

GROUP_SEP = ";"

MISSING = "N"  # first character of "NaN", which is all we need to test for

HOME, AWAY, NOBODY = 0, 1, -1

LABELS = np.array(["H", "A", "D"])

def team_of_each_track(meta_path):
    """Array over track slots: 0 if the slot belongs to the home team, 1 if away.

    The ATD metadata lists both teams and assigns every track slot to one of them. The first team
    in the file is the home side -- the FIFA/EPTS convention, and the one kloppy follows when it
    reports metadata.teams. For this match that is Barcelona, which DATA.md states independently.

    Note what these slots are NOT: there are 46 of them for 22 players on the pitch, they are
    named "Track 1".."Track 46", and their ShirtNumber is just the slot index. A slot is a
    tracklet, not a person, and it does not follow one player through the match.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)

def team_names(meta_path):
    """(home, away) club names, for printing. The submission wants a side, not a club."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))

def nearest_player_labels(raw_path, team_of_track, n_frames, pitch):
    """Per-frame team of the player closest to the ball. NOBODY where there is no answer.

    Returned as codes rather than letters because the smoothing below counts them.

    Coordinates are normalised 0-1 and the pitch is not square, so both axes are scaled to metres
    before distances are compared. Skipping that would stretch every distance along the short axis
    and quietly pick the wrong player near the touchlines.
    """
    length, width = pitch
    out = np.full(n_frames, NOBODY, dtype=np.int8)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            ball = parts[2]
            if ball[0] == MISSING:
                # No ball, no answer. 41% of graded frames land here -- see DATA.md. Leaving them
                # as NOBODY rather than guessing is what makes the D class mean anything at all.
                continue

            comma = ball.index(",")
            ball_x = float(ball[:comma]) * length
            ball_y = float(ball[comma + 1:ball.index(",", comma + 1)]) * width

            best, best_team = None, NOBODY
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                dx = float(group[:comma]) * length - ball_x
                dy = float(group[comma + 1:group.index(",", comma + 1)]) * width - ball_y
                distance = dx * dx + dy * dy
                if best is None or distance < best:
                    best, best_team = distance, team_of_track[slot]

            out[int(parts[0]) - 1] = best_team  # frame_id counts from 1, submissions from 0
    return out

def extract_proximity_features(raw_path, team_of_track, n_frames, pitch):
    """Extract interpretable per-frame proximity features from ATD tracking.

    Distances are in pitch metres. Missing distances are NaN. Nearby-player counts
    are zero when the ball is visible and NaN when it is not.

    The supplied ATD fields do not contain a calibrated ball-detector confidence.
    Therefore ball_visibility_confidence is explicitly a binary visibility proxy
    (1=ball observation present, 0=missing), not a probabilistic confidence score.
    """
    length, width = pitch
    features = np.full((n_frames, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    features[:, 0] = 0.0
    features[:, 1] = 0.0

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            try:
                frame_index = int(parts[0]) - 1
            except (ValueError, IndexError):
                continue
            if frame_index < 0 or frame_index >= n_frames:
                continue

            ball = parts[2]
            if not ball or ball[0] == MISSING:
                continue

            try:
                comma1 = ball.index(",")
                comma2 = ball.index(",", comma1 + 1)
                ball_x = float(ball[:comma1]) * length
                ball_y = float(ball[comma1 + 1:comma2]) * width
            except (ValueError, IndexError):
                continue

            if not np.isfinite(ball_x) or not np.isfinite(ball_y):
                continue

            features[frame_index, 0] = 1.0
            features[frame_index, 1] = 1.0  # binary visibility proxy

            home_distances = []
            away_distances = []
            all_distances = []

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING:
                    continue
                try:
                    comma1 = group.index(",")
                    comma2 = group.index(",", comma1 + 1)
                    px = float(group[:comma1]) * length
                    py = float(group[comma1 + 1:comma2]) * width
                except (ValueError, IndexError):
                    continue

                if not np.isfinite(px) or not np.isfinite(py):
                    continue

                distance = float(np.hypot(px - ball_x, py - ball_y))
                all_distances.append(distance)
                if slot >= len(team_of_track):
                    continue
                if int(team_of_track[slot]) == HOME:
                    home_distances.append(distance)
                else:
                    away_distances.append(distance)

            def two_smallest(values):
                values = sorted(values)
                first = values[0] if len(values) >= 1 else np.nan
                second = values[1] if len(values) >= 2 else np.nan
                return first, second

            nearest_all, second_all = two_smallest(all_distances)
            nearest_home, second_home = two_smallest(home_distances)
            nearest_away, second_away = two_smallest(away_distances)

            features[frame_index, 2:9] = (
                nearest_all, second_all,
                nearest_home, second_home,
                nearest_away, second_away,
                nearest_home - nearest_away
                if np.isfinite(nearest_home) and np.isfinite(nearest_away)
                else np.nan,
            )

            for col, radius in ((9, 2.0), (11, 5.0), (13, 10.0)):
                features[frame_index, col] = sum(d <= radius for d in home_distances)
                features[frame_index, col + 1] = sum(d <= radius for d in away_distances)

            features[frame_index, 15] = sum(d <= 2.0 for d in all_distances)
            features[frame_index, 16] = sum(d <= 5.0 for d in all_distances)
            features[frame_index, 17] = sum(d <= 10.0 for d in all_distances)
            features[frame_index, 18] = (
                features[frame_index, 11] - features[frame_index, 12]
            )

    return features

def load_ball_positions(raw_path, n_frames, pitch):
    """Return tactical-video ball positions in metres; NaN means no observation."""
    length, width = pitch
    positions = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                frame_index = int(parts[0]) - 1
                ball = parts[2]
                if frame_index < 0 or frame_index >= n_frames or not ball or ball[0] == MISSING:
                    continue
                comma1 = ball.index(",")
                comma2 = ball.index(",", comma1 + 1)
                x = float(ball[:comma1]) * length
                y = float(ball[comma1 + 1:comma2]) * width
                if np.isfinite(x) and np.isfinite(y):
                    positions[frame_index] = (x, y)
            except (ValueError, IndexError):
                continue

    return positions

def smart_tag_dead_ball_mask(smart_tagging_path, ball_positions, feature_matrix,
                             fps=25, movement_threshold_mps=1.0,
                             sustained_moving_frames=3):
    """Build a conservative dead-ball mask from Smart Tagging and tracking.

    GOALS intervals are treated as dead play until the tagged restart.
    SET PIECES intervals are candidate stoppages. Within those intervals:
      - missing ball observations count as stoppage evidence;
      - a visible, nearly stationary ball is marked dead only when tracking also
        shows at least one player within 5 metres;
      - the dead interval ends at the first sustained ball movement.

    This is an experiment, not ground truth. The XML's times are seconds on the
    tactical-video timeline, so they are converted directly at the tactical FPS.
    """
    n_frames = len(ball_positions)
    dead = np.zeros(n_frames, dtype=bool)
    visible = np.isfinite(ball_positions[:, 0]) & np.isfinite(ball_positions[:, 1])

    speed = np.full(n_frames, np.nan, dtype=np.float32)
    consecutive = visible[1:] & visible[:-1]
    deltas = np.linalg.norm(ball_positions[1:] - ball_positions[:-1], axis=1)
    speed[1:][consecutive] = deltas[consecutive] * float(fps)

    root = ET.parse(smart_tagging_path).getroot()
    goal_intervals = 0
    set_piece_intervals = 0
    movement_ends = 0

    for instance in root.findall(".//instance"):
        code = (instance.findtext("code") or "").strip().upper()
        if code not in ("GOALS", "SET PIECES"):
            continue

        try:
            start_seconds = float(instance.findtext("start"))
            end_seconds = float(instance.findtext("end"))
        except (TypeError, ValueError):
            continue

        start = max(0, int(round(start_seconds * fps)))
        end = min(n_frames - 1, int(round(end_seconds * fps)))
        if end < start:
            continue

        if code == "GOALS":
            # The GOALS annotation spans the scoring event through the restart.
            dead[start:end + 1] = True
            goal_intervals += 1
            continue

        set_piece_intervals += 1
        interval_end = end

        # Find the first run of sustained movement. The first frame in that run
        # is treated as live play, so the dead-ball mask stops immediately before it.
        run = 0
        moving_start = None
        for frame in range(max(start + 1, 1), end + 1):
            if np.isfinite(speed[frame]) and speed[frame] >= movement_threshold_mps:
                run += 1
                if run >= sustained_moving_frames:
                    moving_start = frame - sustained_moving_frames + 1
                    break
            else:
                run = 0

        if moving_start is not None:
            interval_end = max(start, moving_start - 1)
            movement_ends += 1

        # A Smart Tagging set-piece interval is the event prior, not proof that
        # every frame is dead. Use missing-ball evidence or stationary-ball plus
        # nearby-player evidence inside the candidate interval.
        for frame in range(start, interval_end + 1):
            if not visible[frame]:
                dead[frame] = True
                continue

            ball_speed = speed[frame]
            nearby_5m = feature_matrix[frame, 16]
            stationary = (not np.isfinite(ball_speed) or
                          ball_speed < movement_threshold_mps)
            players_near_ball = np.isfinite(nearby_5m) and nearby_5m >= 1
            if stationary and players_near_ball:
                dead[frame] = True

    return dead, {
        "goal_intervals": goal_intervals,
        "set_piece_intervals": set_piece_intervals,
        "set_piece_intervals_with_detected_restart_movement": movement_ends,
        "dead_ball_frames": int(np.sum(dead)),
    }

def apply_dead_ball_mask(labels, dead_ball_mask):
    """Override possession labels with D only where Experiment 3 detects a stoppage."""
    if len(labels) != len(dead_ball_mask):
        raise ValueError("labels and dead-ball mask must have the same length")
    result = np.array(labels, dtype="<U1", copy=True)
    result[dead_ball_mask] = "D"
    return result

def smooth(codes, window):
    """Majority vote over a centred window of +/- `window` frames. D where nobody has the ball.

    Computed with cumulative sums, so the cost does not grow with the window -- a 125-frame window
    over 199319 frames is the same work as a 1-frame one.

    Ties go to the home side. With an odd-sized window over two classes a tie needs the two counts
    to be exactly equal, which is rare and never worth a coin flip.
    """
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))

    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels

def write_submission(path, labels):
    """A frame,label CSV -- the whole submission format, in four lines."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path

def build(game=GAME, data_dir=None):
    """Per-frame team codes for the whole video. The expensive half, run once, smoothed many ways."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    codes = nearest_player_labels(raw_path, team_of_each_track(meta_path), n_frames,
                                  GAMES[game]["pitch"])
    seen = int(np.sum(codes != NOBODY))
    print("  ball located on {} of {} frames ({:.1f}%)\n".format(
        seen, n_frames, 100.0 * seen / n_frames))
    return codes

E10_FEATURE_NAMES = (
    "ball_visible",
    "ball_speed_mps",
    "ball_acceleration_mps2",
    "ball_direction_change_deg",
    "nearest_player_team",
    "nearest_player_distance_m",
    "nearest_player_speed_mps",
    "ball_player_direction_cosine",
    "ball_player_relative_speed_mps",
    "ball_player_distance_change_m",
    "ball_player_distance_std_5frames_m",
    "likely_touch_confidence",
    "likely_touch_event",
    "likely_touch_team",
    "time_since_last_likely_touch_sec",
    "post_touch_trajectory_continuity",
    "home_players_within_2m",
    "away_players_within_2m",
    "home_players_within_5m",
    "away_players_within_5m",
    "nearest_home_distance_m",
    "nearest_away_distance_m",
    "second_nearest_opponent_distance_m",
    "two_nearest_opponents_separation_m",
    "home_mean_player_ball_distance_m",
    "away_mean_player_ball_distance_m",
    "home_team_approach_rate_mps",
    "away_team_approach_rate_mps",
    "home_team_motion_alignment",
    "away_team_motion_alignment",
    "team_pressure_balance",
    "free_ball_indicator",
)

def extract_e10_ball_control_features(raw_path, team_of_track, n_frames, pitch,
                                      fps=25, velocity_span=2):
    """Build diagnostic E10 features from tracking; no labels or ground truth used.

    Likely touches are estimates from local ball trajectory changes near a player.
    They are not asserted to be exact touches. Track slots are only compared over
    short local windows, never treated as permanent player identities.
    """
    length, width = pitch
    n_slots = len(team_of_track)
    balls = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1
            except (ValueError, IndexError):
                continue
            if frame < 0 or frame >= n_frames:
                continue
            ball = parts[2]
            if ball and ball[0] != MISSING:
                try:
                    c1, c2 = ball.index(","), ball.index(",", ball.index(",") + 1)
                    xy = (float(ball[:c1]) * length,
                          float(ball[c1 + 1:c2]) * width)
                    if np.all(np.isfinite(xy)):
                        balls[frame] = xy
                except (ValueError, IndexError):
                    pass
            for slot, group in enumerate(parts[1].split(GROUP_SEP)[:n_slots]):
                if not group or group[0] == MISSING:
                    continue
                try:
                    c1 = group.index(",")
                    c2 = group.index(",", c1 + 1)
                    xy = (float(group[:c1]) * length,
                          float(group[c1 + 1:c2]) * width)
                    if np.all(np.isfinite(xy)):
                        players[frame, slot] = xy
                except (ValueError, IndexError):
                    continue

    out = np.full((n_frames, len(E10_FEATURE_NAMES)), np.nan, dtype=np.float32)
    visible = np.isfinite(balls[:, 0]) & np.isfinite(balls[:, 1])
    out[:, 0] = visible.astype(np.float32)

    # Per-frame team/player proximity and team-level spatial context.
    nearest_slot = np.full(n_frames, -1, dtype=np.int16)
    nearest_home = np.full(n_frames, np.nan, dtype=np.float32)
    nearest_away = np.full(n_frames, np.nan, dtype=np.float32)
    for t in np.flatnonzero(visible):
        valid = np.isfinite(players[t, :, 0]) & np.isfinite(players[t, :, 1])
        if not np.any(valid):
            continue
        slots = np.flatnonzero(valid)
        distances = np.linalg.norm(players[t, valid] - balls[t], axis=1)
        order = np.argsort(distances)
        nearest_slot[t] = int(slots[order[0]])
        nearest = float(distances[order[0]])
        out[t, 5] = nearest
        team = int(team_of_track[nearest_slot[t]])
        out[t, 4] = float(team)

        hmask = np.asarray(team_of_track[slots] == HOME)
        amask = ~hmask
        hd = distances[hmask]
        ad = distances[amask]
        if len(hd):
            nearest_home[t] = float(np.min(hd))
            out[t, 20] = float(np.min(hd))
            out[t, 16] = float(np.sum(hd <= 2.0))
            out[t, 18] = float(np.sum(hd <= 5.0))
            out[t, 24] = float(np.mean(hd))
        if len(ad):
            nearest_away[t] = float(np.min(ad))
            out[t, 21] = float(np.min(ad))
            out[t, 17] = float(np.sum(ad <= 2.0))
            out[t, 19] = float(np.sum(ad <= 5.0))
            out[t, 25] = float(np.mean(ad))

        opponent_distances = ad if team == HOME else hd
        opponent_slots = slots[amask] if team == HOME else slots[hmask]
        if len(opponent_distances):
            order_opp = np.argsort(opponent_distances)
            out[t, 22] = float(opponent_distances[order_opp[1]]) if len(order_opp) > 1 else np.nan
            if len(order_opp) > 1:
                p1 = players[t, opponent_slots[order_opp[0]]]
                p2 = players[t, opponent_slots[order_opp[1]]]
                out[t, 23] = float(np.linalg.norm(p1 - p2))

        # Balanced pressure: similar nearest-team distances imply a contest.
        if np.isfinite(nearest_home[t]) and np.isfinite(nearest_away[t]):
            out[t, 30] = 1.0 / (1.0 + abs(float(nearest_home[t] - nearest_away[t])))

    # Ball velocity, acceleration and direction changes from adjacent observations.
    ball_velocity = np.full((n_frames, 2), np.nan, dtype=np.float32)
    for t in range(1, n_frames):
        if visible[t] and visible[t - 1]:
            ball_velocity[t] = (balls[t] - balls[t - 1]) * float(fps)
    ball_speed = np.linalg.norm(ball_velocity, axis=1)
    out[:, 1] = ball_speed
    for t in range(2, n_frames):
        if np.all(np.isfinite(ball_velocity[t])) and np.all(np.isfinite(ball_velocity[t - 1])):
            out[t, 2] = float(np.linalg.norm(ball_velocity[t] - ball_velocity[t - 1]) * fps)
            v1, v2 = ball_velocity[t - 1], ball_velocity[t]
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom > 1e-5:
                cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                out[t, 3] = float(np.degrees(np.arccos(cosine)))

    # Nearest-player local movement and relationship to the ball.
    span = int(velocity_span)
    for t in range(span, n_frames - span):
        slot = int(nearest_slot[t])
        if slot < 0 or not (visible[t - span] and visible[t + span]):
            continue
        pb = players[t - span, slot]
        pa = players[t + span, slot]
        pc = players[t, slot]
        if not (np.all(np.isfinite(pb)) and np.all(np.isfinite(pa)) and np.all(np.isfinite(pc))):
            continue
        dt = (2.0 * span) / float(fps)
        bv = (balls[t + span] - balls[t - span]) / dt
        pv = (pa - pb) / dt
        bs, ps = float(np.linalg.norm(bv)), float(np.linalg.norm(pv))
        rv = float(np.linalg.norm(bv - pv))
        denom = bs * ps
        out[t, 6] = ps
        out[t, 8] = rv
        if denom > 1e-5:
            out[t, 7] = float(np.clip(np.dot(bv, pv) / denom, -1.0, 1.0))
        before_d = float(np.linalg.norm(balls[t - span] - pb))
        after_d = float(np.linalg.norm(balls[t + span] - pa))
        out[t, 9] = after_d - before_d
        local_d = []
        for k in range(t - span, t + span + 1):
            if visible[k] and np.all(np.isfinite(players[k, slot])):
                local_d.append(float(np.linalg.norm(balls[k] - players[k, slot])))
        if len(local_d) == 2 * span + 1:
            out[t, 10] = float(np.std(local_d))

    # Team closing rates and team motion alignment. Compare only short-lived slots
    # observed at both endpoints; this avoids relying on long-term track identity.
    dt = (2.0 * span) / float(fps)
    for t in range(span, n_frames - span):
        if not visible[t - span] or not visible[t + span]:
            continue
        ball_dir = balls[t + span] - balls[t - span]
        norm = float(np.linalg.norm(ball_dir))
        if norm <= 1e-5:
            continue
        ball_dir /= norm
        for team, mean_col, approach_col, align_col in (
            (HOME, 24, 26, 28), (AWAY, 25, 27, 29)
        ):
            before_slots = np.flatnonzero(
                (team_of_track == team) & np.isfinite(players[t - span, :, 0])
                & np.isfinite(players[t - span, :, 1])
            )
            after_valid = np.isfinite(players[t + span, :, 0]) & np.isfinite(players[t + span, :, 1])
            shared = before_slots[after_valid[before_slots]]
            if len(shared):
                displacement = players[t + span, shared] - players[t - span, shared]
                mean_move = np.mean(displacement, axis=0)
                out[t, align_col] = float(np.dot(mean_move / (np.linalg.norm(mean_move) + 1e-6), ball_dir))
            if visible[t - span] and visible[t + span]:
                before_d = []
                after_d = []
                for slot in range(n_slots):
                    if int(team_of_track[slot]) != team:
                        continue
                    if (np.all(np.isfinite(players[t - span, slot])) and
                            np.all(np.isfinite(players[t + span, slot]))):
                        before_d.append(float(np.linalg.norm(players[t - span, slot] - balls[t - span])))
                        after_d.append(float(np.linalg.norm(players[t + span, slot] - balls[t + span])))
                if before_d and after_d:
                    # Negative means the team's mean distance to the ball decreased.
                    out[t, approach_col] = float((np.mean(after_d) - np.mean(before_d)) / dt)

    # Likely-touch estimate: a close player plus a local change in ball motion.
    # This is intentionally a confidence proxy, not a claim of exact touch detection.
    touch_conf = np.zeros(n_frames, dtype=np.float32)
    valid_near = np.isfinite(out[:, 5])
    proximity_score = np.zeros(n_frames, dtype=np.float32)
    proximity_score[valid_near] = np.clip(1.0 - out[valid_near, 5] / 3.5, 0.0, 1.0)
    turn_score = np.zeros(n_frames, dtype=np.float32)
    valid_turn = np.isfinite(out[:, 3])
    turn_score[valid_turn] = np.clip(out[valid_turn, 3] / 65.0, 0.0, 1.0)
    accel_score = np.zeros(n_frames, dtype=np.float32)
    valid_accel = np.isfinite(out[:, 2])
    accel_score[valid_accel] = np.clip(out[valid_accel, 2] / 8.0, 0.0, 1.0)
    touch_conf = (0.45 * proximity_score + 0.30 * turn_score + 0.25 * accel_score)
    touch_conf[~visible] = 0.0
    out[:, 11] = touch_conf

    threshold_events = touch_conf >= 0.58
    # Keep local peaks rather than counting every frame of one continuous
    # trajectory disturbance as a separate touch.
    likely_event = np.zeros(n_frames, dtype=bool)
    last_event_frame = -10
    for t in np.flatnonzero(threshold_events):
        lo, hi = max(0, t - 2), min(n_frames, t + 3)
        if touch_conf[t] >= float(np.max(touch_conf[lo:hi])) and t - last_event_frame >= 5:
            likely_event[t] = True
            last_event_frame = t
    out[:, 12] = likely_event.astype(np.float32)
    out[likely_event, 13] = out[likely_event, 4]

    # Time since last likely touch; missing ball does not create a touch.
    last_touch = -1
    last_team = np.nan
    for t in range(n_frames):
        if likely_event[t]:
            last_touch = t
            last_team = out[t, 13]
        if last_touch >= 0:
            out[t, 14] = (t - last_touch) / float(fps)
        if last_touch >= 0 and np.isfinite(last_team):
            # Carry the last likely-touch team forward as context.
            if not np.isfinite(out[t, 13]):
                out[t, 13] = last_team

    # Directional continuity immediately after a frame: compare successive
    # short displacements over a three-frame horizon.
    for t in range(n_frames - 3):
        if visible[t] and visible[t + 1] and visible[t + 2] and visible[t + 3]:
            v1 = balls[t + 1] - balls[t]
            v2 = balls[t + 3] - balls[t + 2]
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom > 1e-5:
                out[t, 15] = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))

    # A fast ball well away from both teams is a candidate free-ball state.
    both_near = np.isfinite(nearest_home) & np.isfinite(nearest_away)
    free = both_near & (np.minimum(nearest_home, nearest_away) >= 2.0) & (ball_speed >= 4.5)
    out[free, 31] = 1.0
    out[~free, 31] = 0.0
    return out

def write_e10_feature_table(path, features):
    """Write E10 diagnostic features; this is not an evaluation submission."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame," + ",".join(E10_FEATURE_NAMES) + "\n")
        for frame, row in enumerate(features):
            values = []
            for value in row:
                values.append("" if not np.isfinite(value) else "{:.6f}".format(float(value)))
            handle.write(str(frame) + "," + ",".join(values) + "\n")

def e10_integrated_possession_labels(nearest_codes, proximity_features,
                                     e10_features, start_confirm=8,
                                     window_frames=15, required_strong=6,
                                     required_support=9,
                                     control_distance_m=3.0,
                                     team_advantage_m=0.25,
                                     candidate_distance_m=3.5,
                                     closing_rate_m=0.10):
    """E10: E4 control backbone, E9 motion nominations and touch/movement context.

    A candidate team needs a rolling accumulation of independent-looking control
    evidence in a short window. A likely touch alone never establishes possession.
    """
    n = len(nearest_codes)
    if len(proximity_features) != n or len(e10_features) != n:
        raise ValueError("E10 input arrays must have equal lengths")
    result = np.full(n, NOBODY, dtype=np.int8)
    current = NOBODY
    candidate = NOBODY
    strong_history = []
    support_history = []
    candidate_start = -1

    for i in range(n):
        raw_team = int(nearest_codes[i])
        nearest = float(proximity_features[i, 2])
        home_near = float(proximity_features[i, 4])
        away_near = float(proximity_features[i, 6])
        second = float(proximity_features[i, 3])
        advantage = np.nan
        if np.isfinite(home_near) and np.isfinite(away_near):
            advantage = home_near - away_near

        strong_team = NOBODY
        if raw_team in (HOME, AWAY) and all(np.isfinite(v) for v in (nearest, second, home_near, away_near)):
            team_adv = (away_near - home_near) if raw_team == HOME else (home_near - away_near)
            if nearest <= 2.5 and team_adv >= 0.5 and (second - nearest) >= 0.25:
                strong_team = raw_team

        # Candidate nomination retains E9's opponent-closing principle.
        candidate_team = strong_team
        if current in (HOME, AWAY):
            opponent = AWAY if current == HOME else HOME
            opponent_dist = float(e10_features[i, 21] if opponent == AWAY else e10_features[i, 20])
            opponent_closing = float(e10_features[i, 27] if opponent == AWAY else e10_features[i, 26])
            if (np.isfinite(opponent_dist) and opponent_dist <= candidate_distance_m
                    and np.isfinite(opponent_closing) and opponent_closing <= -closing_rate_m):
                candidate_team = opponent

        # Weaker support requires spatial advantage plus at least one ball-control cue:
        # recent likely touch by this team, coherent movement, or stable ball/player distance.
        support_team = NOBODY
        if raw_team in (HOME, AWAY) and np.isfinite(nearest) and np.isfinite(home_near) and np.isfinite(away_near):
            team_adv = (away_near - home_near) if raw_team == HOME else (home_near - away_near)
            close_enough = nearest <= control_distance_m
            touch_team = e10_features[i, 13]
            touch_age = e10_features[i, 14]
            cosine = e10_features[i, 7]
            stability = e10_features[i, 10]
            touch_support = (np.isfinite(touch_team) and int(round(touch_team)) == raw_team
                             and np.isfinite(touch_age) and touch_age <= 1.2)
            movement_support = ((np.isfinite(cosine) and cosine >= -0.05)
                                or (np.isfinite(stability) and stability <= 0.55))
            if close_enough and team_adv >= team_advantage_m and (touch_support or movement_support):
                support_team = raw_team

        # Initialise a possession with sustained evidence.
        if current == NOBODY:
            if support_team in (HOME, AWAY):
                if candidate != support_team:
                    candidate = support_team
                    candidate_start = i
                    strong_history = []
                    support_history = []
                strong_history.append(1 if strong_team == candidate else 0)
                support_history.append(1)
                strong_history = strong_history[-window_frames:]
                support_history = support_history[-window_frames:]
                initial_strong_required = max(4, required_strong - 1)
                if (sum(strong_history) >= initial_strong_required
                        and sum(support_history) >= start_confirm):
                    current = candidate
                    result[candidate_start:i + 1] = current
                    candidate = NOBODY
                    strong_history, support_history = [], []
            else:
                candidate = NOBODY
                strong_history, support_history = [], []
            result[i] = current
            continue

        if candidate_team in (HOME, AWAY) and candidate_team != current:
            if candidate != candidate_team:
                candidate = candidate_team
                candidate_start = i
                strong_history, support_history = [], []

        if candidate in (HOME, AWAY) and candidate != current:
            is_strong = strong_team == candidate
            is_support = support_team == candidate
            strong_history.append(1 if is_strong else 0)
            support_history.append(1 if is_support else 0)
            strong_history = strong_history[-window_frames:]
            support_history = support_history[-window_frames:]
            if (sum(strong_history) >= required_strong
                    and sum(support_history) >= required_support):
                current = candidate
                # Backdate to the first support frame in the active rolling window.
                first_support = max(candidate_start, i - window_frames + 1)
                result[first_support:i + 1] = current
                candidate = NOBODY
                strong_history, support_history = [], []
            elif len(support_history) >= window_frames and sum(support_history) < required_support:
                candidate = NOBODY
                strong_history, support_history = [], []
                candidate_start = -1

        # If the current team re-establishes control, cancel the pending handover.
        if strong_team == current or support_team == current:
            candidate = NOBODY
            strong_history, support_history = [], []
            candidate_start = -1

        result[i] = current

    return result



def run_updated_e10(out_dir):
    """Run E10 with its established settings and write updated_ outputs only."""
    os.makedirs(out_dir, exist_ok=True)

    nearest_codes = build()
    meta_path, atd_raw_path = require(["atd_meta", "atd_raw"], GAME)
    team_of_track = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    fps = GAMES[GAME].get("fps", 25)

    print("\nUpdated final model: E10 integrated controlled possession")
    proximity_features = extract_proximity_features(
        atd_raw_path, team_of_track, n_frames, pitch
    )
    ball_positions = load_ball_positions(atd_raw_path, n_frames, pitch)
    smart_tagging_path, = require(["smart_tagging"], GAME)
    dead_ball_mask, dead_ball_stats = smart_tag_dead_ball_mask(
        smart_tagging_path, ball_positions, proximity_features, fps=fps
    )

    e10_features = extract_e10_ball_control_features(
        atd_raw_path, team_of_track, n_frames, pitch,
        fps=fps, velocity_span=2
    )
    e10_codes = e10_integrated_possession_labels(
        nearest_codes,
        proximity_features,
        e10_features,
        start_confirm=8,
        window_frames=15,
        required_strong=6,
        required_support=9,
        control_distance_m=3.0,
        team_advantage_m=0.25,
        candidate_distance_m=3.5,
        closing_rate_m=0.10,
    )
    labels = smooth(e10_codes, 50)
    labels = apply_dead_ball_mask(labels, dead_ball_mask)

    submission_path = write_submission(
        os.path.join(out_dir, "updated_e10_integrated_control_smooth50.csv"),
        labels
    )
    feature_path = os.path.join(out_dir, "updated_e10_ball_control_features.csv")
    write_e10_feature_table(feature_path, e10_features)

    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D")}
    print("  H {:,}  A {:,}  D {:,}".format(counts["H"], counts["A"], counts["D"]))
    print("  likely-touch events: {:,}".format(int(np.sum(e10_features[:, 12] == 1.0))))
    print("  dead-ball frames: {:,}".format(dead_ball_stats["dead_ball_frames"]))
    print("  submission: {}".format(submission_path))
    print("  E10 feature diagnostics: {}".format(feature_path))
    print("Only E10 is run; no earlier experiment submissions are generated.")
    return submission_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Updated final submission pipeline using E10 only"
    )
    parser.add_argument("--out", default=RESULTS_DIR, help="output folder")
    args = parser.parse_args()
    run_updated_e10(args.out)
