# ===========================
# fitcorrect_core.py
# (FULL: Original + HTTP /on /off limb buzzing, limb mapping as agreed)
# ===========================
import cv2
import mediapipe as mp
import math
import time
import os
import json
from statistics import median
from collections import deque

# NEW: HTTP client (built-in)
import urllib.request

# ============================================================
# CONFIG
# ============================================================
PROFILE_DIR = "calibration_profiles"
DEFAULT_USER = "DEFAULT"

VIS_THRESH = 0.55
EVAL_VIS_THRESH = 0.70

HOLD_SECONDS = 2.0
MIN_SAMPLES = 12

DIFFICULTY_SCALE = {
    "Beginner": 1.10,
    "Standard": 1.00,
    "Advanced": 0.85,
}

MAX_MAD_DEG = 18.0
MAX_MAD_TREE = 0.06

# Horizontal requirement (normalized image coords)
HORIZONTAL_MAX_SLOPE = 0.22  # lower = stricter

# Smoothing to reduce jitter
SMOOTH_WIN = 5

# Latching uses slightly looser tolerance so movement is allowed
HIT_TOL_MULT = {
    "Bodyweight Squat": 1.3,
    "Lunge": 1.4,
    "Push-Up": 1.4,
    "Sit-Up": 1.0,      # keep strict to avoid confusion
    "Tree Pose": 1.2,
}

# Do NOT allow deeper than calibration by more than this margin.
# Smaller angle = deeper for knees/elbows in this project.
ROM_DEEP_MARGIN_DEG = {
    "Beginner": 10.0,
    "Standard": 7.0,
    "Advanced": 5.0,
}

# ============================================================
# REQUIRED FORM CONSTRAINTS (ALWAYS ENFORCED)
# ============================================================
# Squat form constraints:
SQUAT_HIP_MIN, SQUAT_HIP_MAX = 90.0, 120.0
SQUAT_TOE_MIN, SQUAT_TOE_MAX = 15.0, 45.0

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

# ============================================================
# ESP HTTP BUZZ (Pi -> ESP-01S)
# ============================================================
LIMB_IPS = {
    "left_hand": "10.151.80.201",
    "right_hand": "10.151.80.202",
    "left_leg": "10.151.80.203",
    "right_leg": "10.151.80.204",
}

HTTP_PORT = 80
HTTP_TIMEOUT_S = 0.12
HTTP_ON_COOLDOWN_S = 0.30  # keepalive rate (per limb)

_last_http_on = {k: 0.0 for k in LIMB_IPS.keys()}

def _http_get(ip: str, path: str) -> bool:
    url = f"http://{ip}:{HTTP_PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as r:
            r.read(16)
        return True
    except Exception:
        return False

def send_on_keepalive(limbs):
    """
    Call /on periodically while ADJUST (rate-limited per limb).
    """
    now = time.time()
    for limb in limbs:
        ip = LIMB_IPS.get(limb)
        if not ip:
            continue
        if now - _last_http_on.get(limb, 0.0) < HTTP_ON_COOLDOWN_S:
            continue
        _last_http_on[limb] = now
        _http_get(ip, "/on")

def send_off(limbs):
    """
    Call /off (best effort).
    """
    for limb in limbs:
        ip = LIMB_IPS.get(limb)
        if ip:
            _http_get(ip, "/off")

def send_off_all():
    send_off(list(LIMB_IPS.keys()))

# ============================================================
# LIMB MAPPING (as agreed)
# ============================================================
def limbs_for_bad_form(title: str, lm):
    # Squat -> both legs
    if title == "Bodyweight Squat":
        return ["left_leg", "right_leg"]

    # Lunge -> whichever knee is more bent (smaller angle)
    if title == "Lunge":
        lk, rk = metric_lunge_knees(lm)
        return ["left_leg"] if lk < rk else ["right_leg"]

    # Push-up -> both hands
    if title == "Push-Up":
        return ["left_hand", "right_hand"]

    # Sit-up -> both legs
    if title == "Sit-Up":
        return ["left_leg", "right_leg"]

    # Tree -> both legs
    if title == "Tree Pose":
        return ["left_leg", "right_leg"]

    return []

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
    with open(profile_path(user), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

def load_profile(user: str):
    path = profile_path(user)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# GEOMETRY (2D)
# ============================================================
def lm_xyv(lm, idx: int):
    p = lm[idx]
    return (float(p.x), float(p.y)), float(getattr(p, "visibility", 1.0))

def angle_2d(a, b, c):
    """Angle ABC at B (2D)."""
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

def smooth_push(buf: deque, v: float):
    buf.append(float(v))
    return float(median(buf)) if buf else float(v)

# ============================================================
# DRAW HELPERS
# ============================================================
def _to_px(img, xy):
    h, w = img.shape[:2]
    return int(xy[0] * w), int(xy[1] * h)

def draw_points(img, lm, indices, color=(0, 255, 255), r=6):
    for idx in indices:
        (xy, v) = lm_xyv(lm, idx)
        x, y = _to_px(img, xy)
        col = color if v >= VIS_THRESH else (0, 0, 255)
        cv2.circle(img, (x, y), r, col, -1)

def draw_lines(img, lm, pairs, color=(0, 255, 255), t=3):
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

# Squat: one side + toe + shoulder for hip angle
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

# Lunge
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

# Push-up
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

# Sit-up
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

def metric_squat_hip(lm):
    side = squat_pick_side(lm)
    sh, hip, knee = (PL.LEFT_SHOULDER.value, PL.LEFT_HIP.value, PL.LEFT_KNEE.value) if side == "left" else \
                    (PL.RIGHT_SHOULDER.value, PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value)
    sh_xy, _ = lm_xyv(lm, sh)
    hip_xy, _ = lm_xyv(lm, hip)
    knee_xy, _ = lm_xyv(lm, knee)
    return float(angle_2d(sh_xy, hip_xy, knee_xy))

def metric_squat_toe_out(lm):
    side = squat_pick_side(lm)
    ankle = PL.LEFT_ANKLE.value if side == "left" else PL.RIGHT_ANKLE.value
    toe = PL.LEFT_FOOT_INDEX.value if side == "left" else PL.RIGHT_FOOT_INDEX.value
    a_xy, _ = lm_xyv(lm, ankle)
    t_xy, _ = lm_xyv(lm, toe)
    dx = t_xy[0] - a_xy[0]
    dy = t_xy[1] - a_xy[1]
    ang = abs(math.degrees(math.atan2(dy, dx)))
    if ang > 90:
        ang = 180 - ang
    return float(ang)

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
# CALIBRATION + EVALUATION + REP LATCHING + LOOP
# (same as your original; unchanged except buzzing hook in exercise_loop)
# ============================================================

def _calibrate_one_metric(cap, pose, title, required_indices, draw_fn, metric_fn, accept_fn,
                          accept_range=None, max_mad=MAX_MAD_DEG):
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
        cv2.putText(img, "Hold still. Keys: N=Skip  Q=Quit", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            draw_fn(img, lm)

            if _visible_all_eval(lm, required_indices):
                ok_pose, why = accept_fn(lm)
                if ok_pose:
                    cur = float(metric_fn(lm))
                    if accept_range and not (accept_range[0] <= cur <= accept_range[1]):
                        hold_start = None
                        samples.clear()
                        cv2.putText(img, "Adjust position...", (30, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                    else:
                        if hold_start is None:
                            hold_start = time.time()
                            samples.clear()
                        samples.append(cur)
                        t = time.time() - hold_start
                        cv2.putText(img, f"Hold steady... {t:.1f}s", (30, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        if t >= HOLD_SECONDS and len(samples) >= MIN_SAMPLES:
                            m = float(median(samples))
                            s = float(mad(samples))
                            if s > max_mad:
                                cv2.putText(img, "Too unstable. Try again.", (30, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                hold_start = None
                                samples.clear()
                            else:
                                tol = max(8.0, 2.0 * s)
                                return m, tol
                else:
                    hold_start = None
                    samples.clear()
                    cv2.putText(img, f"Adjust: {why}", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            else:
                hold_start = None
                samples.clear()
                cv2.putText(img, "Landmarks not visible", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            hold_start = None
            samples.clear()
            cv2.putText(img, "No body detected", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        cv2.imshow("Calibration", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if key == ord("n"):
            return "SKIP"

def calibration_phase(camera_index=0, user=DEFAULT_USER, difficulty="Standard"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("⚠️ Camera not available.")
        return None

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

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        r = _calibrate_one_metric(
            cap, pose, "Squat (go to your lowest comfortable squat and hold)",
            required_indices=SQUAT_SIDE["left"],
            draw_fn=draw_squat,
            metric_fn=metric_squat_knee,
            accept_fn=plausibility_squat,
            accept_range=(60, 150),
            max_mad=MAX_MAD_DEG
        )
        if r is None:
            cap.release(); cv2.destroyAllWindows(); return None
        if r != "SKIP":
            profile["exercises"]["Bodyweight Squat"] = {"hit_deg": r[0], "tol": r[1]}

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
            cap.release(); cv2.destroyAllWindows(); return None
        if r != "SKIP":
            profile["exercises"]["Lunge"] = {"hit_deg": r[0], "tol": r[1]}

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
            cap.release(); cv2.destroyAllWindows(); return None
        if r != "SKIP":
            profile["exercises"]["Push-Up"] = {"hit_deg": r[0], "tol": r[1]}

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
            cap.release(); cv2.destroyAllWindows(); return None
        if r != "SKIP":
            profile["exercises"]["Sit-Up"] = {"hit_deg": r[0], "tol": r[1]}

        r = _calibrate_one_metric(
            cap, pose, "Tree Pose (hold tree)",
            required_indices=[PL.LEFT_ANKLE.value, PL.RIGHT_ANKLE.value, PL.LEFT_HIP.value, PL.RIGHT_HIP.value, PL.LEFT_KNEE.value, PL.RIGHT_KNEE.value],
            draw_fn=draw_tree,
            metric_fn=metric_tree_ankle_diff,
            accept_fn=plausibility_tree,
            accept_range=(0.0, 0.30),
            max_mad=MAX_MAD_TREE
        )
        if r is None:
            cap.release(); cv2.destroyAllWindows(); return None
        if r != "SKIP":
            profile["exercises"]["Tree Pose"] = {"hit": r[0], "tol": r[1]}

    cap.release()
    cv2.destroyAllWindows()

    required = ["Bodyweight Squat", "Lunge", "Push-Up", "Sit-Up", "Tree Pose"]
    missing = [e for e in required if e not in profile["exercises"]]
    if missing:
        print("⚠️ Calibration incomplete. Missing:", missing)
        return None

    save_profile(user, profile)
    print(f"✅ Calibration saved for user '{user}' -> {profile_path(user)}")
    return profile

def within(cur, base, tol):
    return abs(cur - base) <= tol

def within_hit(cur, base, tol, ex_name, difficulty):
    mult = float(HIT_TOL_MULT.get(ex_name, 1.2))
    scale = float(DIFFICULTY_SCALE.get(difficulty, 1.0))
    return within(cur, base, tol * scale * mult)

def too_deep(cur, hit_base, difficulty):
    margin = float(ROM_DEEP_MARGIN_DEG.get(difficulty, 7.0))
    return cur < (hit_base - margin)

def evaluate_exercise_hit(title: str, lm, profile: dict, difficulty: str, smoothers: dict):
    ex = profile.get("exercises", {}).get(title)
    if not ex:
        return None, None, "Not calibrated"

    if title == "Bodyweight Squat":
        ok_pl, msg = plausibility_squat(lm)
        if not ok_pl:
            return False, False, msg

        knee = smooth_push(smoothers["squat_knee"], metric_squat_knee(lm))
        hip  = smooth_push(smoothers["squat_hip"],  metric_squat_hip(lm))
        toe  = smooth_push(smoothers["squat_toe"],  metric_squat_toe_out(lm))

        if knee > 150.0:
            return False, False, f"Not squatting (knee {knee:.0f}°)"

        ok_hip = (SQUAT_HIP_MIN <= hip <= SQUAT_HIP_MAX)
        ok_toe = (SQUAT_TOE_MIN <= toe <= SQUAT_TOE_MAX)

        hit = float(ex["hit_deg"]); tol = float(ex["tol"])
        deep_bad = too_deep(knee, hit, difficulty)
        ok_hit_knee = within_hit(knee, hit, tol, title, difficulty) and (not deep_bad)

        ok_hit = bool(ok_hit_knee and ok_hip and ok_toe)
        ok_now = ok_hit

        if deep_bad:
            info = f"Too deep vs calibration | Knee {knee:.0f}°"
        else:
            info = f"Knee {knee:.0f}° (hit {hit:.0f}±{tol:.0f}) | Hip {hip:.0f}° | Toe {toe:.0f}°"
        return ok_now, ok_hit, info

    if title == "Lunge":
        ok_pl, msg = plausibility_lunge(lm)
        if not ok_pl:
            return False, False, msg

        lk_raw, rk_raw = metric_lunge_knees(lm)
        lk = smooth_push(smoothers["lunge_lk"], lk_raw)
        rk = smooth_push(smoothers["lunge_rk"], rk_raw)
        cur_min = min(lk, rk)

        hit = float(ex["hit_deg"]); tol = float(ex["tol"])
        deep_bad = too_deep(cur_min, hit, difficulty)
        ok_hit = within_hit(cur_min, hit, tol, title, difficulty) and (not deep_bad)
        ok_now = ok_hit

        info = f"L {lk:.0f}° | R {rk:.0f}°"
        if deep_bad:
            info = "Too deep vs calibration | " + info
        return ok_now, ok_hit, info

    if title == "Push-Up":
        ok_pl, msg = plausibility_pushup(lm)
        if not ok_pl:
            return False, False, msg

        elbow = smooth_push(smoothers["push_elbow"], metric_pushup_elbow(lm))
        hit = float(ex["hit_deg"]); tol = float(ex["tol"])

        deep_bad = too_deep(elbow, hit, difficulty)
        ok_hit = within_hit(elbow, hit, tol, title, difficulty) and (not deep_bad)
        ok_now = ok_hit

        info = f"Elbow {elbow:.0f}° (hit {hit:.0f}±{tol:.0f})"
        if deep_bad:
            info = "Too deep vs calibration | " + info
        return ok_now, ok_hit, info

    if title == "Sit-Up":
        ok_pl, msg = plausibility_situp(lm)
        if not ok_pl:
            return False, False, msg

        knee = smooth_push(smoothers["sit_knee"], metric_situp_knee(lm))
        hit = float(ex["hit_deg"]); tol = float(ex["tol"])

        deep_bad = too_deep(knee, hit, difficulty)
        ok_hit = within_hit(knee, hit, tol, title, difficulty) and (not deep_bad)
        ok_now = ok_hit

        info = f"Knee {knee:.0f}° (hit {hit:.0f}±{tol:.0f})"
        if deep_bad:
            info = "Too deep vs calibration | " + info
        return ok_now, ok_hit, info

    if title == "Tree Pose":
        ok_pl, msg = plausibility_tree(lm)
        if not ok_pl:
            return False, False, msg

        d = smooth_push(smoothers["tree_diff"], metric_tree_ankle_diff(lm))
        base = float(ex["hit"]); tol = float(ex["tol"])
        ok_hit = within_hit(d, base, tol, title, difficulty)
        ok_now = ok_hit
        info = f"Ankle diff {d:.2f} (hit {base:.2f}±{tol:.2f})"
        return ok_now, ok_hit, info

    return None, None, ""

def rep_state_new():
    return {"phase": "up", "in_rep": False, "hit": False, "good": 0, "bad": 0}

def reset_rep(st):
    st["phase"] = "up"
    st["in_rep"] = False
    st["hit"] = False

def update_squat_rep(st, knee_val, hit_knee):
    down_th = min(hit_knee + 20.0, 140.0)
    up_th   = max(hit_knee + 45.0, 160.0)

    if st["phase"] == "up" and knee_val <= down_th:
        st["phase"] = "down"
        st["in_rep"] = True
        st["hit"] = False

    if st["phase"] == "down" and st["in_rep"] and knee_val >= up_th:
        if st["hit"]:
            st["good"] += 1
        else:
            st["bad"] += 1
        reset_rep(st)

def update_pushup_rep(st, elbow_val, hit_elbow):
    down_th = hit_elbow + 12.0
    up_th = min(hit_elbow + 55.0, 175.0)

    if st["phase"] == "up" and elbow_val <= down_th:
        st["phase"] = "down"
        st["in_rep"] = True
        st["hit"] = False

    if st["phase"] == "down" and st["in_rep"] and elbow_val >= up_th:
        if st["hit"]:
            st["good"] += 1
        else:
            st["bad"] += 1
        reset_rep(st)

def update_lunge_rep(st, lk, rk, hit_knee):
    down_th = hit_knee + 20.0
    up_th = min(hit_knee + 75.0, 175.0)

    in_down = (min(lk, rk) <= down_th)
    in_up = (lk >= up_th and rk >= up_th)

    if st["phase"] == "up" and in_down:
        st["phase"] = "down"
        st["in_rep"] = True
        st["hit"] = False

    if st["phase"] == "down" and st["in_rep"] and in_up:
        if st["hit"]:
            st["good"] += 1
        else:
            st["bad"] += 1
        reset_rep(st)

def exercise_loop(camera_index=0, user=DEFAULT_USER, difficulty="Standard", profile=None):
    if profile is None:
        profile = load_profile(user)
    if profile is None:
        print("⚠️ No calibration profile. Please calibrate first.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("⚠️ Camera not available.")
        return

    exercises = ["Bodyweight Squat", "Lunge", "Push-Up", "Sit-Up", "Tree Pose"]
    ex_idx = 0

    reps = {
        "Bodyweight Squat": rep_state_new(),
        "Lunge": rep_state_new(),
        "Push-Up": rep_state_new(),
    }

    smoothers = {
        "squat_knee": deque(maxlen=SMOOTH_WIN),
        "squat_hip": deque(maxlen=SMOOTH_WIN),
        "squat_toe": deque(maxlen=SMOOTH_WIN),
        "lunge_lk": deque(maxlen=SMOOTH_WIN),
        "lunge_rk": deque(maxlen=SMOOTH_WIN),
        "push_elbow": deque(maxlen=SMOOTH_WIN),
        "sit_knee": deque(maxlen=SMOOTH_WIN),
        "tree_diff": deque(maxlen=SMOOTH_WIN),
    }

    def draw_for_ex(img, lm, title):
        if title == "Bodyweight Squat":
            side = squat_pick_side(lm)
            if side == "right":
                draw_subset(img, lm, SQUAT_SIDE["right"], SQUAT_CONN_RIGHT)
            else:
                draw_subset(img, lm, SQUAT_SIDE["left"], SQUAT_CONN_LEFT)
        elif title == "Lunge":
            draw_subset(img, lm, LUNGE_IDX, LUNGE_CONN)
        elif title == "Push-Up":
            draw_subset(img, lm, PUSHUP_IDX, PUSHUP_CONN)
        elif title == "Sit-Up":
            draw_subset(img, lm, SITUP_IDX, SITUP_CONN)
        elif title == "Tree Pose":
            draw_subset(img, lm, TREE_IDX_NO_HEAD, TREE_CONN_NO_HEAD)

    last_display = {ex: None for ex in exercises}

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            img = frame.copy()

            title = exercises[ex_idx]
            cv2.putText(img, f"Exercise: {title}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(img, "Keys: N=Next  Q=Quit", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)

            display_good = None
            info = ""

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                draw_for_ex(img, lm, title)

                ok_now, ok_hit, info = evaluate_exercise_hit(title, lm, profile, difficulty, smoothers)

                if ok_now is None:
                    display_good = None
                else:
                    if title in reps:
                        st = reps[title]

                        if st["in_rep"] and ok_hit:
                            st["hit"] = True

                        if title == "Bodyweight Squat":
                            knee = smooth_push(smoothers["squat_knee"], metric_squat_knee(lm))
                            hit_knee = float(profile["exercises"]["Bodyweight Squat"]["hit_deg"])
                            update_squat_rep(st, knee, hit_knee)

                        elif title == "Push-Up":
                            elbow = smooth_push(smoothers["push_elbow"], metric_pushup_elbow(lm))
                            hit_elbow = float(profile["exercises"]["Push-Up"]["hit_deg"])
                            update_pushup_rep(st, elbow, hit_elbow)

                        elif title == "Lunge":
                            lk_raw, rk_raw = metric_lunge_knees(lm)
                            lk = smooth_push(smoothers["lunge_lk"], lk_raw)
                            rk = smooth_push(smoothers["lunge_rk"], rk_raw)
                            hit_knee = float(profile["exercises"]["Lunge"]["hit_deg"])
                            update_lunge_rep(st, lk, rk, hit_knee)

                        if st["in_rep"]:
                            display_good = True if st["hit"] else False
                        else:
                            display_good = True if ok_now else False
                    else:
                        display_good = True if ok_now else False

                if display_good is True:
                    cv2.putText(img, "GOOD", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                elif display_good is False:
                    cv2.putText(img, "ADJUST", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    cv2.putText(img, "NOT VISIBLE", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

                if info:
                    cv2.putText(img, info, (30, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

                if title in reps:
                    st = reps[title]
                    cv2.putText(img, f"Reps OK: {st['good']}", (30, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(img, f"Reps BAD: {st['bad']}", (30, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # ---- Limb buzz control ----
                limbs = limbs_for_bad_form(title, lm)

                if display_good is False:
                    send_on_keepalive(limbs)
                elif display_good is True:
                    if last_display.get(title) is not True:
                        send_off(limbs)

                last_display[title] = display_good

            else:
                cv2.putText(img, "No body detected", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                send_off_all()
                last_display[title] = None

            cv2.imshow("FitCorrect", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                send_off_all()
                break
            elif key == ord("n"):
                send_off_all()
                for dq in smoothers.values():
                    dq.clear()
                for k in reps:
                    reset_rep(reps[k])
                ex_idx = (ex_idx + 1) % len(exercises)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys as _sys

    user = DEFAULT_USER
    difficulty = "Standard"
    camera_index = 0

    args = _sys.argv[1:]
    do_calibrate = ("--calibrate" in args) or ("-c" in args)

    profile_json_path = None
    for a in args:
        if a.lower().endswith(".json") and os.path.exists(a):
            profile_json_path = a
            break

    if profile_json_path:
        try:
            with open(profile_json_path, "r", encoding="utf-8") as f:
                ui_profile = json.load(f)
            user = ui_profile.get("user", user) or user
            difficulty = ui_profile.get("difficulty", difficulty) or difficulty
            camera_index = int(ui_profile.get("camera_index", camera_index))
        except Exception as e:
            print("⚠️ Failed to read UI profile.json:", e)

    if do_calibrate:
        prof = calibration_phase(camera_index=camera_index, user=user, difficulty=difficulty)
        if prof is None:
            print("⚠️ Calibration failed or incomplete.")
            raise SystemExit(1)
        exercise_loop(camera_index=camera_index, user=user, difficulty=difficulty, profile=prof)
    else:
        exercise_loop(camera_index=camera_index, user=user, difficulty=difficulty)