# ===========================
# calibration_launcher.py
# Calibration Only (FitCorrect)
# - Calibrates and saves a profile JSON
# - No exercise loop, no reps, no ESP buzz
#
# UPDATED (logic-safe):
# ✅ Accepts BOTH:
#   A) Positional:  python calibration_launcher.py USER DIFFICULTY
#   B) Flags:       python calibration_launcher.py --user USER --difficulty Standard --cam 0
#   C) Optional:    python calibration_launcher.py profile.json
#      (reads user/difficulty/camera_index if present)
#
# Everything else is your calibration logic, unchanged.
# ===========================

import cv2
import mediapipe as mp
import math
import time
import os
import json
from statistics import median
import numpy as np

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(2)
except Exception:
    pass

# ============================================================
# CONFIG
# ============================================================
PROFILE_DIR = "calibration_profiles"
DEFAULT_USER = "DEFAULT"

VIS_THRESH = 0.55
EVAL_VIS_THRESH = 0.70

HOLD_SECONDS = 2.0
MIN_SAMPLES = 12

MAX_MAD_DEG = 18.0
MAX_MAD_TREE = 0.06

HORIZONTAL_MAX_SLOPE = 0.22

WINDOW_NAME = "FitCorrect"

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

# ============================================================
# WINDOW
# ============================================================
def make_fullscreen(window_name=WINDOW_NAME, enable=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if enable:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

def show_frame(img):
    cv2.imshow(WINDOW_NAME, img)

# ============================================================
# PROFILE IO
# ============================================================
def _ensure_profile_dir():
    os.makedirs(PROFILE_DIR, exist_ok=True)

def profile_path(user: str):
    _ensure_profile_dir()
    safe = "".join(c for c in (user or "") if c.isalnum() or c in ("-", "_")) or DEFAULT_USER
    return os.path.join(PROFILE_DIR, f"{safe}.json")

def save_profile(user: str, profile: dict):
    _ensure_profile_dir()
    with open(profile_path(user), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

# ============================================================
# GEOMETRY
# ============================================================
def lm_xyv(lm, idx: int):
    p = lm[idx]
    return (float(p.x), float(p.y)), float(getattr(p, "visibility", 1.0))

def angle_2d(a, b, c):
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    denom = max(math.hypot(bax, bay) * math.hypot(bcx, bcy), 1e-6)
    cosv = (bax * bcx + bay * bcy) / denom
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def dist_2d(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))

def mad(vals):
    if not vals:
        return 0.0
    m = median(vals)
    return float(median([abs(v - m) for v in vals]))

def slope_abs(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dy / max(dx, 1e-6)

# ============================================================
# DRAW (your lightweight subset drawing)
# ============================================================
def _to_px(img, xy):
    h, w = img.shape[:2]
    return int(xy[0] * w), int(xy[1] * h)

def draw_points(img, lm, indices, color=(0, 255, 255), r=4):
    for idx in indices:
        (xy, v) = lm_xyv(lm, idx)
        x, y = _to_px(img, xy)
        col = color if v >= VIS_THRESH else (0, 0, 255)
        cv2.circle(img, (x, y), r, col, -1)

def draw_lines(img, lm, pairs, color=(0, 255, 255), t=2):
    for a, b in pairs:
        (a_xy, a_v) = lm_xyv(lm, a)
        (b_xy, b_v) = lm_xyv(lm, b)
        ax, ay = _to_px(img, a_xy)
        bx, by = _to_px(img, b_xy)
        col = color if (a_v >= VIS_THRESH and b_v >= VIS_THRESH) else (0, 165, 255)
        cv2.line(img, (ax, ay), (bx, by), col, t)

def draw_subset(img, lm, indices, connections):
    draw_lines(img, lm, connections)
    draw_points(img, lm, indices)

# ============================================================
# LANDMARK SETS
# ============================================================
HEAD_IDX = [
    PL.NOSE.value,
    PL.LEFT_EYE_INNER.value, PL.LEFT_EYE.value, PL.LEFT_EYE_OUTER.value,
    PL.RIGHT_EYE_INNER.value, PL.RIGHT_EYE.value, PL.RIGHT_EYE_OUTER.value,
    PL.LEFT_EAR.value, PL.RIGHT_EAR.value,
    PL.MOUTH_LEFT.value, PL.MOUTH_RIGHT.value,
]

def filtered_connections_excluding(indices_to_exclude):
    ex = set(indices_to_exclude)
    conns = []
    for a, b in mp_pose.POSE_CONNECTIONS:
        if a in ex or b in ex:
            continue
        conns.append((a, b))
    return conns

TREE_CONN_NO_HEAD = filtered_connections_excluding(HEAD_IDX)
TREE_IDX_NO_HEAD = [lm.value for lm in PL if lm.value not in set(HEAD_IDX)]

SQUAT_SIDE = {
    "left":  [PL.LEFT_SHOULDER.value, PL.LEFT_HIP.value, PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value],
    "right": [PL.RIGHT_SHOULDER.value, PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value, PL.RIGHT_FOOT_INDEX.value],
}
SQUAT_CONN_LEFT = [
    (PL.LEFT_SHOULDER.value, PL.LEFT_HIP.value),
    (PL.LEFT_HIP.value, PL.LEFT_KNEE.value),
    (PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value),
    (PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value),
]
SQUAT_CONN_RIGHT = [
    (PL.RIGHT_SHOULDER.value, PL.RIGHT_HIP.value),
    (PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value),
    (PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value),
    (PL.RIGHT_ANKLE.value, PL.RIGHT_FOOT_INDEX.value),
]

LUNGE_IDX = [
    PL.LEFT_HIP.value, PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value,
    PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value, PL.RIGHT_FOOT_INDEX.value,
]
LUNGE_CONN = [
    (PL.LEFT_HIP.value, PL.LEFT_KNEE.value),
    (PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value),
    (PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value),
    (PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value),
    (PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value),
    (PL.RIGHT_ANKLE.value, PL.RIGHT_FOOT_INDEX.value),
]

PUSHUP_IDX = [
    PL.LEFT_SHOULDER.value, PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value,
    PL.LEFT_HIP.value, PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value,
]
PUSHUP_CONN = [
    (PL.LEFT_SHOULDER.value, PL.LEFT_ELBOW.value),
    (PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value),
    (PL.LEFT_SHOULDER.value, PL.LEFT_HIP.value),
    (PL.LEFT_HIP.value, PL.LEFT_KNEE.value),
    (PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value),
    (PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value),
]

SITUP_IDX = [
    PL.LEFT_SHOULDER.value, PL.LEFT_HIP.value, PL.LEFT_KNEE.value,
    PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value,
]
SITUP_CONN = [
    (PL.LEFT_SHOULDER.value, PL.LEFT_HIP.value),
    (PL.LEFT_HIP.value, PL.LEFT_KNEE.value),
    (PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value),
    (PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value),
]

# ============================================================
# VISIBILITY
# ============================================================
def _visible_all_eval(lm, indices):
    for idx in indices:
        _, v = lm_xyv(lm, idx)
        if v < EVAL_VIS_THRESH:
            return False
    return True

# ============================================================
# METRICS
# ============================================================
def squat_pick_side(lm):
    best = "left"
    best_score = -1.0
    for side, idxs in SQUAT_SIDE.items():
        score = 0.0
        ok = True
        for i in idxs:
            _, v = lm_xyv(lm, i)
            score += v
            if v < VIS_THRESH:
                ok = False
        if ok and score > best_score:
            best = side
            best_score = score
    return best

def metric_squat_knee(lm):
    side = squat_pick_side(lm)
    _, hip, knee, ankle, _ = SQUAT_SIDE[side]
    h, _ = lm_xyv(lm, hip)
    k, _ = lm_xyv(lm, knee)
    a, _ = lm_xyv(lm, ankle)
    return float(angle_2d(h, k, a))

def metric_lunge_knees(lm):
    lk = float(angle_2d(lm_xyv(lm, PL.LEFT_HIP.value)[0], lm_xyv(lm, PL.LEFT_KNEE.value)[0], lm_xyv(lm, PL.LEFT_ANKLE.value)[0]))
    rk = float(angle_2d(lm_xyv(lm, PL.RIGHT_HIP.value)[0], lm_xyv(lm, PL.RIGHT_KNEE.value)[0], lm_xyv(lm, PL.RIGHT_ANKLE.value)[0]))
    return lk, rk

def metric_pushup_elbow(lm):
    sh, _ = lm_xyv(lm, PL.LEFT_SHOULDER.value)
    el, _ = lm_xyv(lm, PL.LEFT_ELBOW.value)
    wr, _ = lm_xyv(lm, PL.LEFT_WRIST.value)
    return float(angle_2d(sh, el, wr))

def metric_pushup_body_line(lm):
    sh = PL.LEFT_SHOULDER.value
    hip = PL.LEFT_HIP.value
    ank = PL.LEFT_ANKLE.value
    return float(angle_2d(lm_xyv(lm, sh)[0], lm_xyv(lm, hip)[0], lm_xyv(lm, ank)[0]))

def metric_situp_knee(lm):
    hip = PL.LEFT_HIP.value
    knee = PL.LEFT_KNEE.value
    ankle = PL.LEFT_ANKLE.value
    return float(angle_2d(lm_xyv(lm, hip)[0], lm_xyv(lm, knee)[0], lm_xyv(lm, ankle)[0]))

def metric_tree_ankle_diff(lm):
    la, _ = lm_xyv(lm, PL.LEFT_ANKLE.value)
    ra, _ = lm_xyv(lm, PL.RIGHT_ANKLE.value)
    return float(abs(la[1] - ra[1]))

def is_body_horizontal_pushup(lm):
    sh_xy, _ = lm_xyv(lm, PL.LEFT_SHOULDER.value)
    ank_xy, _ = lm_xyv(lm, PL.LEFT_ANKLE.value)
    return slope_abs(sh_xy, ank_xy) <= HORIZONTAL_MAX_SLOPE

def is_body_horizontal_situp(lm):
    sh_xy, _ = lm_xyv(lm, PL.LEFT_SHOULDER.value)
    ank_xy, _ = lm_xyv(lm, PL.LEFT_ANKLE.value)
    return slope_abs(sh_xy, ank_xy) <= HORIZONTAL_MAX_SLOPE

# ============================================================
# PLAUSIBILITY
# ============================================================
def _limb_sane(lm, a, b, min_len=0.03):
    axy, _ = lm_xyv(lm, a)
    bxy, _ = lm_xyv(lm, b)
    return dist_2d(axy, bxy) >= min_len

def plausibility_squat(lm):
    side = squat_pick_side(lm)
    sh, hip, knee, ankle, toe = SQUAT_SIDE[side]
    if not _visible_all_eval(lm, [sh, hip, knee, ankle, toe]):
        return False, "Not visible"
    if not (_limb_sane(lm, hip, knee) and _limb_sane(lm, knee, ankle) and _limb_sane(lm, ankle, toe, min_len=0.015)):
        return False, "Bad tracking"
    return True, ""

def plausibility_lunge(lm):
    if not _visible_all_eval(lm, LUNGE_IDX):
        return False, "Not visible"
    la, _ = lm_xyv(lm, PL.LEFT_ANKLE.value)
    ra, _ = lm_xyv(lm, PL.RIGHT_ANKLE.value)
    if abs(la[0] - ra[0]) < 0.06:
        return False, "Feet too close"
    return True, ""

def plausibility_pushup(lm):
    if not _visible_all_eval(lm, PUSHUP_IDX):
        return False, "Not visible"
    if not is_body_horizontal_pushup(lm):
        return False, "Get horizontal"
    body = metric_pushup_body_line(lm)
    if body < 155:
        return False, "Body not straight"
    return True, ""

def plausibility_situp(lm):
    if not _visible_all_eval(lm, SITUP_IDX):
        return False, "Not visible"
    if not is_body_horizontal_situp(lm):
        return False, "Lie down"

    ankle_xy, _ = lm_xyv(lm, PL.LEFT_ANKLE.value)
    toe_xy, _ = lm_xyv(lm, PL.LEFT_FOOT_INDEX.value)
    knee_xy, _ = lm_xyv(lm, PL.LEFT_KNEE.value)

    if not _limb_sane(lm, PL.LEFT_ANKLE.value, PL.LEFT_FOOT_INDEX.value, min_len=0.012):
        return False, "Bad foot tracking"
    if abs(toe_xy[1] - ankle_xy[1]) > 0.06:
        return False, "Plant feet"
    if not (knee_xy[1] < ankle_xy[1] - 0.02):
        return False, "Bend knees"
    return True, ""

def plausibility_tree(lm):
    need = [
        PL.LEFT_HIP.value, PL.RIGHT_HIP.value,
        PL.LEFT_KNEE.value, PL.RIGHT_KNEE.value,
        PL.LEFT_ANKLE.value, PL.RIGHT_ANKLE.value,
    ]
    if not _visible_all_eval(lm, need):
        return False, "Not visible"

    lk = float(angle_2d(lm_xyv(lm, PL.LEFT_HIP.value)[0], lm_xyv(lm, PL.LEFT_KNEE.value)[0], lm_xyv(lm, PL.LEFT_ANKLE.value)[0]))
    rk = float(angle_2d(lm_xyv(lm, PL.RIGHT_HIP.value)[0], lm_xyv(lm, PL.RIGHT_KNEE.value)[0], lm_xyv(lm, PL.RIGHT_ANKLE.value)[0]))
    if not ((lk > 160 and rk < 160) or (rk > 160 and lk < 160)):
        return False, "Not tree shape"
    lhy = lm_xyv(lm, PL.LEFT_HIP.value)[0][1]
    rhy = lm_xyv(lm, PL.RIGHT_HIP.value)[0][1]
    if abs(lhy - rhy) > 0.08:
        return False, "Hips tilted"
    return True, ""

# ============================================================
# CALIBRATION CORE
# ============================================================
def _calibrate_one_metric(
    cap, pose, title,
    required_indices, draw_fn, metric_fn, accept_fn,
    accept_range=None, max_mad=MAX_MAD_DEG
):
    samples = []
    hold_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        img = frame.copy()

        cv2.putText(img, f"Calibration: {title}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        if res is not None and res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            draw_fn(img, lm)

            try:
                live = float(metric_fn(lm))
                cv2.putText(img, f"Metric: {live:.1f}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            except Exception:
                pass

            if _visible_all_eval(lm, required_indices):
                ok_pose, why = accept_fn(lm)
                if ok_pose:
                    cur = float(metric_fn(lm))
                    if accept_range and not (accept_range[0] <= cur <= accept_range[1]):
                        hold_start = None
                        samples.clear()
                        cv2.putText(img, "Adjust position...", (30, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                    else:
                        if hold_start is None:
                            hold_start = time.time()
                            samples.clear()
                        samples.append(cur)
                        t = time.time() - hold_start
                        cv2.putText(img, f"Hold steady... {t:.1f}s", (30, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        if t >= HOLD_SECONDS and len(samples) >= MIN_SAMPLES:
                            m = float(median(samples))
                            s = float(mad(samples))
                            if s > max_mad:
                                cv2.putText(img, "Too unstable. Try again.", (30, 190),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                hold_start = None
                                samples.clear()
                            else:
                                tol = max(8.0, 2.0 * s)
                                return m, tol
                else:
                    hold_start = None
                    samples.clear()
                    cv2.putText(img, f"Adjust: {why}", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            else:
                hold_start = None
                samples.clear()
                cv2.putText(img, "Landmarks not visible", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            hold_start = None
            samples.clear()
            cv2.putText(img, "No body detected", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        show_frame(img)
        key = cv2.waitKeyEx(1)
        if key in (81, 2424832):
            return None
        if key in (83, 2555904):
            return "SKIP"
        if key in (82, 2490368):
            try:
                cur = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                make_fullscreen(WINDOW_NAME, enable=(cur != cv2.WINDOW_FULLSCREEN))
            except Exception:
                pass

# ============================================================
# CALIBRATION FLOW
# ============================================================
def calibration_phase(cap, user=DEFAULT_USER, difficulty="Standard"):
    profile = {"user": user, "difficulty": difficulty, "timestamp": int(time.time()), "exercises": {}}

    def draw_squat(img, lm):
        side = squat_pick_side(lm)
        if side == "right":
            draw_subset(img, lm, SQUAT_SIDE["right"], SQUAT_CONN_RIGHT)
        else:
            draw_subset(img, lm, SQUAT_SIDE["left"], SQUAT_CONN_LEFT)

    def draw_lunge(img, lm):
        draw_subset(img, lm, LUNGE_IDX, LUNGE_CONN)

    def draw_pushup(img, lm):
        draw_subset(img, lm, PUSHUP_IDX, PUSHUP_CONN)

    def draw_situp(img, lm):
        draw_subset(img, lm, SITUP_IDX, SITUP_CONN)

    def draw_tree(img, lm):
        draw_subset(img, lm, TREE_IDX_NO_HEAD, TREE_CONN_NO_HEAD)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        # --- SQUAT ---
        r = _calibrate_one_metric(
            cap, pose, "Bodyweight Squat (hold your lowest point)",
            required_indices=SQUAT_SIDE["left"],  # visibility requires a full chain; side picker decides
            draw_fn=draw_squat,
            metric_fn=metric_squat_knee,
            accept_fn=plausibility_squat,
            accept_range=(50, 150),
            max_mad=MAX_MAD_DEG
        )
        if r is None:
            return None
        if r != "SKIP":
            profile["exercises"]["Bodyweight Squat"] = {"hit_deg": r[0], "tol": r[1]}

        # --- LUNGE ---
        def lunge_metric_min(lm):
            lk, rk = metric_lunge_knees(lm)
            return float(min(lk, rk))

        r = _calibrate_one_metric(
            cap, pose, "Lunge (hold your bottom lunge)",
            required_indices=LUNGE_IDX,
            draw_fn=draw_lunge,
            metric_fn=lunge_metric_min,
            accept_fn=plausibility_lunge,
            accept_range=(50, 150),
            max_mad=MAX_MAD_DEG
        )
        if r is None:
            return None
        if r != "SKIP":
            profile["exercises"]["Lunge"] = {"hit_deg": r[0], "tol": r[1]}

        # --- PUSH-UP ---
        r = _calibrate_one_metric(
            cap, pose, "Push-Up (horizontal, hold your lowest point)",
            required_indices=PUSHUP_IDX,
            draw_fn=draw_pushup,
            metric_fn=metric_pushup_elbow,
            accept_fn=plausibility_pushup,
            accept_range=(20, 160),
            max_mad=MAX_MAD_DEG
        )
        if r is None:
            return None
        if r != "SKIP":
            profile["exercises"]["Push-Up"] = {"hit_deg": r[0], "tol": r[1]}

        # --- SIT-UP ---
        r = _calibrate_one_metric(
            cap, pose, "Sit-Up (lie down, setup position and hold)",
            required_indices=SITUP_IDX,
            draw_fn=draw_situp,
            metric_fn=metric_situp_knee,
            accept_fn=plausibility_situp,
            accept_range=(50, 140),
            max_mad=MAX_MAD_DEG
        )
        if r is None:
            return None
        if r != "SKIP":
            profile["exercises"]["Sit-Up"] = {"hit_deg": r[0], "tol": r[1]}

        # --- TREE ---
        r = _calibrate_one_metric(
            cap, pose, "Tree Pose (hold tree)",
            required_indices=[
                PL.LEFT_ANKLE.value, PL.RIGHT_ANKLE.value,
                PL.LEFT_HIP.value, PL.RIGHT_HIP.value,
                PL.LEFT_KNEE.value, PL.RIGHT_KNEE.value
            ],
            draw_fn=draw_tree,
            metric_fn=metric_tree_ankle_diff,
            accept_fn=plausibility_tree,
            accept_range=(0.0, 0.30),
            max_mad=MAX_MAD_TREE
        )
        if r is None:
            return None
        if r != "SKIP":
            profile["exercises"]["Tree Pose"] = {"hit": r[0], "tol": r[1]}

    required = ["Bodyweight Squat", "Lunge", "Push-Up", "Sit-Up", "Tree Pose"]
    missing = [e for e in required if e not in profile["exercises"]]
    if missing:
        print("⚠️ Calibration incomplete. Missing:", missing)
        return None

    save_profile(user, profile)
    print(f"✅ Calibration saved for user '{user}' -> {profile_path(user)}")
    return profile

# ============================================================
# MAIN
# ============================================================
def _read_profile_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _parse_args():
    """
    Supports:
      1) positional: USER DIFFICULTY
      2) flags: --user USER --difficulty Standard --cam 0
      3) a single .json path: profile.json
    """
    import argparse

    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("maybe_profile", nargs="?", default=None, help="Optional profile.json OR user id")
    p.add_argument("maybe_difficulty", nargs="?", default=None, help="Optional difficulty when using positional user")
    p.add_argument("--user", default=None)
    p.add_argument("--difficulty", default=None, choices=["Beginner", "Standard", "Advanced"])
    p.add_argument("--cam", type=int, default=None)
    p.add_argument("--w", type=int, default=480)
    p.add_argument("--h", type=int, default=360)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dshow", action="store_true", help="Use DirectShow backend (Windows).")

    args = p.parse_args()

    user = DEFAULT_USER
    difficulty = "Standard"
    cam = 0

    # Case C: profile.json as positional
    if args.maybe_profile and str(args.maybe_profile).lower().endswith(".json") and os.path.exists(args.maybe_profile):
        data = _read_profile_json(args.maybe_profile) or {}
        user = safe = "".join(c for c in (data.get("user") or DEFAULT_USER) if c.isalnum() or c in ("-", "_")) or DEFAULT_USER
        difficulty = data.get("difficulty") or difficulty
        try:
            cam = int(data.get("camera_index", cam))
        except Exception:
            cam = cam

    # Case A: positional user + difficulty
    elif args.maybe_profile:
        user = "".join(c for c in (args.maybe_profile or "") if c.isalnum() or c in ("-", "_")) or DEFAULT_USER
        if args.maybe_difficulty in ("Beginner", "Standard", "Advanced"):
            difficulty = args.maybe_difficulty

    # Case B: flags override
    if args.user:
        user = "".join(c for c in (args.user or "") if c.isalnum() or c in ("-", "_")) or DEFAULT_USER
    if args.difficulty:
        difficulty = args.difficulty
    if args.cam is not None:
        cam = int(args.cam)

    return user, difficulty, cam, args

if __name__ == "__main__":
    user, difficulty, camera_index, args = _parse_args()

    backend = cv2.CAP_DSHOW if args.dshow else 0
    cap = cv2.VideoCapture(camera_index, backend) if backend != 0 else cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.h))
    cap.set(cv2.CAP_PROP_FPS, int(args.fps))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        print("⚠️ Camera not available.")
        raise SystemExit(1)

    make_fullscreen(WINDOW_NAME, False)

    try:
        prof = calibration_phase(cap=cap, user=user, difficulty=difficulty)
        if prof is None:
            print("⚠️ Calibration failed or incomplete.")
            raise SystemExit(1)
    finally:
        cap.release()
        cv2.destroyAllWindows()
