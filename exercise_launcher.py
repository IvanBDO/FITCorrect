# exercise_launcher.py
#
# Runs WITHOUT calibration flow. If profile missing, it generates a fallback profile.
# Optimizations:
# - Pose inference throttling (POSE_EVERY_N)
# - Convert to RGB only when running pose.process
# - Compute metrics once per frame and reuse
# - Optional no-draw / no-buzz toggles to isolate lag
# - Non-blocking HTTP buzz (threaded fire-and-forget)
# - Optional profiler overlay
#
# ✅ FIXED TREE POSE:
# 1) plausibility_tree() now enforces "tree shape" + hips level (like calibration script)
# 2) Tree Pose tolerance is tightened by NOT applying difficulty scaling (still uses HIT_TOL_MULT)

import cv2
import mediapipe as mp
import math
import time
import os
import json
from statistics import median
from collections import deque
import urllib.request
import threading
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

DIFFICULTY_SCALE = {
    "Beginner": 1.10,
    "Standard": 1.00,
    "Advanced": 0.85,
}

HORIZONTAL_MAX_SLOPE = 0.22
SMOOTH_WIN = 5

HIT_TOL_MULT = {
    "Bodyweight Squat": 1.3,
    "Lunge": 1.4,
    "Push-Up": 1.4,
    "Sit-Up": 1.0,
    "Tree Pose": 1.2,   # kept; we tighten by removing difficulty scaling for Tree Pose
}

ROM_DEEP_MARGIN_DEG = {
    "Beginner": 10.0,
    "Standard": 7.0,
    "Advanced": 5.0,
}

SQUAT_HIP_MIN, SQUAT_HIP_MAX = 90.0, 120.0
SQUAT_TOE_MIN, SQUAT_TOE_MAX = 15.0, 45.0

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

WINDOW_NAME = "FitCorrect"

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
# OPTIONAL PROFILER
# ============================================================
PROF_WIN = 60
prof = {
    "read_ms": deque(maxlen=PROF_WIN),
    "pose_ms": deque(maxlen=PROF_WIN),
    "logic_ms": deque(maxlen=PROF_WIN),
    "ui_ms": deque(maxlen=PROF_WIN),
    "loop_ms": deque(maxlen=PROF_WIN),
}
def _ms(dt): return dt * 1000.0
def avg(dq): return (sum(dq) / len(dq)) if dq else 0.0

def draw_profiler(img, enabled):
    if not enabled:
        return
    loop_ms = avg(prof["loop_ms"])
    fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
    y = 320
    lines = [
        f"FPS: {fps:4.1f}",
        f"read : {avg(prof['read_ms']):5.1f} ms",
        f"pose : {avg(prof['pose_ms']):5.1f} ms",
        f"logic: {avg(prof['logic_ms']):5.1f} ms",
        f"ui   : {avg(prof['ui_ms']):5.1f} ms",
        f"loop : {avg(prof['loop_ms']):5.1f} ms",
    ]
    for s in lines:
        cv2.putText(img, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        y += 24

# ============================================================
# ESP HTTP BUZZ (non-blocking)
# ============================================================
LIMB_IPS = {
    "left_hand": "10.188.82.51",
    "right_hand": "10.188.82.52",
    "left_leg": "10.188.82.53",
    "right_leg": "10.188.82.54",
}

HTTP_PORT = 80
HTTP_TIMEOUT_S = 0.10
HTTP_ON_COOLDOWN_S = 0.30
_last_http_on = {k: 0.0 for k in LIMB_IPS.keys()}

def _http_get(ip: str, path: str) -> bool:
    url = f"http://{ip}:{HTTP_PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as r:
            r.read(8)
        return True
    except Exception:
        return False

def _http_get_async(ip: str, path: str):
    th = threading.Thread(target=_http_get, args=(ip, path), daemon=True)
    th.start()

def send_on_keepalive(limbs, enable_buzz=True):
    if not enable_buzz:
        return
    now = time.time()
    for limb in limbs:
        ip = LIMB_IPS.get(limb)
        if not ip:
            continue
        if now - _last_http_on.get(limb, 0.0) < HTTP_ON_COOLDOWN_S:
            continue
        _last_http_on[limb] = now
        _http_get_async(ip, "/on")

def send_off(limbs, enable_buzz=True):
    if not enable_buzz:
        return
    for limb in limbs:
        ip = LIMB_IPS.get(limb)
        if ip:
            _http_get_async(ip, "/off")

def send_off_all(enable_buzz=True):
    if not enable_buzz:
        return
    send_off(list(LIMB_IPS.keys()), enable_buzz=True)

# ============================================================
# PROFILE IO + FALLBACK
# ============================================================
def _ensure_profile_dir():
    os.makedirs(PROFILE_DIR, exist_ok=True)

def profile_path(user: str):
    _ensure_profile_dir()
    safe = "".join(c for c in (user or "") if c.isalnum() or c in ("-", "_")) or DEFAULT_USER
    return os.path.join(PROFILE_DIR, f"{safe}.json")

def load_profile(user: str):
    path = profile_path(user)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_profile(user: str, profile: dict):
    _ensure_profile_dir()
    with open(profile_path(user), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

def default_profile(user="DEFAULT", difficulty="Standard"):
    return {
        "user": user,
        "difficulty": difficulty,
        "timestamp": int(time.time()),
        "exercises": {
            "Bodyweight Squat": {"hit_deg": 95.0, "tol": 18.0},
            "Lunge":           {"hit_deg": 90.0, "tol": 20.0},
            "Push-Up":         {"hit_deg": 75.0, "tol": 18.0},
            "Sit-Up":          {"hit_deg": 95.0, "tol": 18.0},
            "Tree Pose":       {"hit": 0.08, "tol": 0.12},
        }
    }

def load_or_default_profile(user, difficulty="Standard", autosave=True):
    prof_ = load_profile(user)
    if prof_ is not None:
        return prof_

    print(f"⚠️ No calibration profile for user '{user}'. Using fallback profile.")
    prof_ = default_profile(user=user, difficulty=difficulty)
    if autosave:
        save_profile(user, prof_)
        print(f"✅ Wrote fallback profile -> {profile_path(user)}")
    return prof_

# ============================================================
# GEOMETRY / HELPERS
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

def slope_abs(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dy / max(dx, 1e-6)

def smooth_push(buf: deque, v: float):
    buf.append(float(v))
    return float(median(buf)) if buf else float(v)

# ============================================================
# DRAW (optional)
# ============================================================
def _to_px(img, xy):
    h, w = img.shape[:2]
    return int(xy[0] * w), int(xy[1] * h)

def draw_points(img, lm, indices, enabled=True, color=(0, 255, 255), r=4):
    if not enabled:
        return
    for idx in indices:
        (xy, v) = lm_xyv(lm, idx)
        x, y = _to_px(img, xy)
        col = color if v >= VIS_THRESH else (0, 0, 255)
        cv2.circle(img, (x, y), r, col, -1)

def draw_lines(img, lm, pairs, enabled=True, color=(0, 255, 255), t=2):
    if not enabled:
        return
    for a, b in pairs:
        (a_xy, a_v) = lm_xyv(lm, a)
        (b_xy, b_v) = lm_xyv(lm, b)
        ax, ay = _to_px(img, a_xy)
        bx, by = _to_px(img, b_xy)
        col = color if (a_v >= VIS_THRESH and b_v >= VIS_THRESH) else (0, 165, 255)
        cv2.line(img, (ax, ay), (bx, by), col, t)

def draw_subset(img, lm, indices, connections, enabled=True):
    draw_lines(img, lm, connections, enabled=enabled)
    draw_points(img, lm, indices, enabled=enabled)

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
# VISIBILITY + PLAUSIBILITY
# ============================================================
def _visible_all_eval(lm, indices):
    for idx in indices:
        _, v = lm_xyv(lm, idx)
        if v < EVAL_VIS_THRESH:
            return False
    return True

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

def plausibility_squat(lm, side):
    sh, hip, knee, ankle, toe = SQUAT_SIDE[side]
    if not _visible_all_eval(lm, [sh, hip, knee, ankle, toe]):
        return False, "Not visible"
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
    sh_xy, _ = lm_xyv(lm, PL.LEFT_SHOULDER.value)
    ank_xy, _ = lm_xyv(lm, PL.LEFT_ANKLE.value)
    if slope_abs(sh_xy, ank_xy) > HORIZONTAL_MAX_SLOPE:
        return False, "Get horizontal"
    return True, ""

def plausibility_situp(lm):
    if not _visible_all_eval(lm, SITUP_IDX):
        return False, "Not visible"
    sh_xy, _ = lm_xyv(lm, PL.LEFT_SHOULDER.value)
    ank_xy, _ = lm_xyv(lm, PL.LEFT_ANKLE.value)
    if slope_abs(sh_xy, ank_xy) > HORIZONTAL_MAX_SLOPE:
        return False, "Lie down"
    return True, ""

# ✅ FIXED: enforce "tree shape" + hips level (matches your calibration logic)
def plausibility_tree(lm):
    need = [
        PL.LEFT_HIP.value, PL.RIGHT_HIP.value,
        PL.LEFT_KNEE.value, PL.RIGHT_KNEE.value,
        PL.LEFT_ANKLE.value, PL.RIGHT_ANKLE.value,
    ]
    if not _visible_all_eval(lm, need):
        return False, "Not visible"

    lk = float(angle_2d(lm_xyv(lm, PL.LEFT_HIP.value)[0],
                        lm_xyv(lm, PL.LEFT_KNEE.value)[0],
                        lm_xyv(lm, PL.LEFT_ANKLE.value)[0]))
    rk = float(angle_2d(lm_xyv(lm, PL.RIGHT_HIP.value)[0],
                        lm_xyv(lm, PL.RIGHT_KNEE.value)[0],
                        lm_xyv(lm, PL.RIGHT_ANKLE.value)[0]))

    # Require one leg straight and the other bent
    if not ((lk > 160 and rk < 160) or (rk > 160 and lk < 160)):
        return False, "Not tree shape"

    lhy = lm_xyv(lm, PL.LEFT_HIP.value)[0][1]
    rhy = lm_xyv(lm, PL.RIGHT_HIP.value)[0][1]
    if abs(lhy - rhy) > 0.08:
        return False, "Hips tilted"

    return True, ""

# ============================================================
# METRICS (computed once per frame)
# ============================================================
def compute_metrics(title, lm, smoothers):
    m = {}

    if title == "Bodyweight Squat":
        side = squat_pick_side(lm)
        m["side"] = side
        _, hip, knee, ankle, _ = SQUAT_SIDE[side]
        h = lm_xyv(lm, hip)[0]
        k = lm_xyv(lm, knee)[0]
        a = lm_xyv(lm, ankle)[0]
        knee_deg = angle_2d(h, k, a)
        m["knee"] = smooth_push(smoothers["squat_knee"], knee_deg)

        sh = PL.LEFT_SHOULDER.value if side == "left" else PL.RIGHT_SHOULDER.value
        hip_i = PL.LEFT_HIP.value if side == "left" else PL.RIGHT_HIP.value
        knee_i = PL.LEFT_KNEE.value if side == "left" else PL.RIGHT_KNEE.value
        sh_xy = lm_xyv(lm, sh)[0]
        hip_xy = lm_xyv(lm, hip_i)[0]
        knee_xy = lm_xyv(lm, knee_i)[0]
        hip_deg = angle_2d(sh_xy, hip_xy, knee_xy)
        m["hip"] = smooth_push(smoothers["squat_hip"], hip_deg)

        ankle_i = PL.LEFT_ANKLE.value if side == "left" else PL.RIGHT_ANKLE.value
        toe_i   = PL.LEFT_FOOT_INDEX.value if side == "left" else PL.RIGHT_FOOT_INDEX.value
        a_xy = lm_xyv(lm, ankle_i)[0]
        t_xy = lm_xyv(lm, toe_i)[0]
        dx = t_xy[0] - a_xy[0]
        dy = t_xy[1] - a_xy[1]
        ang = abs(math.degrees(math.atan2(dy, dx)))
        if ang > 90:
            ang = 180 - ang
        m["toe"] = smooth_push(smoothers["squat_toe"], ang)

    elif title == "Lunge":
        lk = angle_2d(lm_xyv(lm, PL.LEFT_HIP.value)[0],
                      lm_xyv(lm, PL.LEFT_KNEE.value)[0],
                      lm_xyv(lm, PL.LEFT_ANKLE.value)[0])
        rk = angle_2d(lm_xyv(lm, PL.RIGHT_HIP.value)[0],
                      lm_xyv(lm, PL.RIGHT_KNEE.value)[0],
                      lm_xyv(lm, PL.RIGHT_ANKLE.value)[0])
        m["lk"] = smooth_push(smoothers["lunge_lk"], lk)
        m["rk"] = smooth_push(smoothers["lunge_rk"], rk)
        m["min_knee"] = min(m["lk"], m["rk"])

    elif title == "Push-Up":
        sh = lm_xyv(lm, PL.LEFT_SHOULDER.value)[0]
        el = lm_xyv(lm, PL.LEFT_ELBOW.value)[0]
        wr = lm_xyv(lm, PL.LEFT_WRIST.value)[0]
        elbow = angle_2d(sh, el, wr)
        m["elbow"] = smooth_push(smoothers["push_elbow"], elbow)

    elif title == "Sit-Up":
        hip = lm_xyv(lm, PL.LEFT_HIP.value)[0]
        knee = lm_xyv(lm, PL.LEFT_KNEE.value)[0]
        ankle = lm_xyv(lm, PL.LEFT_ANKLE.value)[0]
        knee_deg = angle_2d(hip, knee, ankle)
        m["knee"] = smooth_push(smoothers["sit_knee"], knee_deg)

    elif title == "Tree Pose":
        la = lm_xyv(lm, PL.LEFT_ANKLE.value)[0]
        ra = lm_xyv(lm, PL.RIGHT_ANKLE.value)[0]
        d = abs(la[1] - ra[1])
        m["diff"] = smooth_push(smoothers["tree_diff"], d)

    return m

# ============================================================
# EVAL
# ============================================================
def within(cur, base, tol):
    return abs(cur - base) <= tol

# ✅ FIXED: Tree Pose does NOT apply difficulty scaling (tightens acceptance)
def within_hit(cur, base, tol, ex_name, difficulty):
    mult = float(HIT_TOL_MULT.get(ex_name, 1.2))
    if ex_name == "Tree Pose":
        return within(cur, base, tol * mult)
    scale = float(DIFFICULTY_SCALE.get(difficulty, 1.0))
    return within(cur, base, tol * scale * mult)

def too_deep(cur, hit_base, difficulty):
    margin = float(ROM_DEEP_MARGIN_DEG.get(difficulty, 7.0))
    return cur < (hit_base - margin)

def evaluate(title, lm, profile, difficulty, metrics):
    ex = profile.get("exercises", {}).get(title)
    if not ex:
        return None, None, "Not calibrated"

    if title == "Bodyweight Squat":
        side = metrics["side"]
        ok_pl, msg = plausibility_squat(lm, side)
        if not ok_pl:
            return False, False, msg

        knee = metrics["knee"]
        hip = metrics["hip"]
        toe = metrics["toe"]

        if knee > 150.0:
            return False, False, f"Not squatting (knee {knee:.0f}°)"

        ok_hip = (SQUAT_HIP_MIN <= hip <= SQUAT_HIP_MAX)
        ok_toe = (SQUAT_TOE_MIN <= toe <= SQUAT_TOE_MAX)

        hit = float(ex["hit_deg"]); tol = float(ex["tol"])
        deep_bad = too_deep(knee, hit, difficulty)
        ok_hit_knee = within_hit(knee, hit, tol, title, difficulty) and (not deep_bad)
        ok_hit = bool(ok_hit_knee and ok_hip and ok_toe)
        info = f"Knee {knee:.0f}° (hit {hit:.0f}±{tol:.0f}) | Hip {hip:.0f}° | Toe {toe:.0f}°"
        if deep_bad:
            info = "Too deep vs calibration | " + info
        return ok_hit, ok_hit, info

    if title == "Lunge":
        ok_pl, msg = plausibility_lunge(lm)
        if not ok_pl:
            return False, False, msg
        cur_min = metrics["min_knee"]
        hit = float(ex["hit_deg"]); tol = float(ex["tol"])
        deep_bad = too_deep(cur_min, hit, difficulty)
        ok_hit = within_hit(cur_min, hit, tol, title, difficulty) and (not deep_bad)
        info = f"L {metrics['lk']:.0f}° | R {metrics['rk']:.0f}°"
        if deep_bad:
            info = "Too deep vs calibration | " + info
        return ok_hit, ok_hit, info

    if title == "Push-Up":
        ok_pl, msg = plausibility_pushup(lm)
        if not ok_pl:
            return False, False, msg
        elbow = metrics["elbow"]
        hit = float(ex["hit_deg"]); tol = float(ex["tol"])
        deep_bad = too_deep(elbow, hit, difficulty)
        ok_hit = within_hit(elbow, hit, tol, title, difficulty) and (not deep_bad)
        info = f"Elbow {elbow:.0f}° (hit {hit:.0f}±{tol:.0f})"
        if deep_bad:
            info = "Too deep vs calibration | " + info
        return ok_hit, ok_hit, info

    if title == "Sit-Up":
        ok_pl, msg = plausibility_situp(lm)
        if not ok_pl:
            return False, False, msg
        knee = metrics["knee"]
        hit = float(ex["hit_deg"]); tol = float(ex["tol"])
        deep_bad = too_deep(knee, hit, difficulty)
        ok_hit = within_hit(knee, hit, tol, title, difficulty) and (not deep_bad)
        info = f"Knee {knee:.0f}° (hit {hit:.0f}±{tol:.0f})"
        if deep_bad:
            info = "Too deep vs calibration | " + info
        return ok_hit, ok_hit, info

    if title == "Tree Pose":
        ok_pl, msg = plausibility_tree(lm)
        if not ok_pl:
            return False, False, msg
        d = metrics["diff"]
        base = float(ex["hit"]); tol = float(ex["tol"])
        ok_hit = within_hit(d, base, tol, title, difficulty)
        info = f"Ankle diff {d:.2f} (hit {base:.2f}±{tol:.2f})"
        return ok_hit, ok_hit, info

    return None, None, ""

# ============================================================
# REP LOGIC
# ============================================================
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
        st["phase"] = "down"; st["in_rep"] = True; st["hit"] = False
    if st["phase"] == "down" and st["in_rep"] and knee_val >= up_th:
        st["good"] += 1 if st["hit"] else 0
        st["bad"]  += 0 if st["hit"] else 1
        reset_rep(st)

def update_pushup_rep(st, elbow_val, hit_elbow):
    down_th = hit_elbow + 12.0
    up_th   = min(hit_elbow + 55.0, 175.0)
    if st["phase"] == "up" and elbow_val <= down_th:
        st["phase"] = "down"; st["in_rep"] = True; st["hit"] = False
    if st["phase"] == "down" and st["in_rep"] and elbow_val >= up_th:
        st["good"] += 1 if st["hit"] else 0
        st["bad"]  += 0 if st["hit"] else 1
        reset_rep(st)

def update_lunge_rep(st, lk, rk, hit_knee):
    down_th = hit_knee + 20.0
    up_th   = min(hit_knee + 75.0, 175.0)
    in_down = (min(lk, rk) <= down_th)
    in_up   = (lk >= up_th and rk >= up_th)
    if st["phase"] == "up" and in_down:
        st["phase"] = "down"; st["in_rep"] = True; st["hit"] = False
    if st["phase"] == "down" and st["in_rep"] and in_up:
        st["good"] += 1 if st["hit"] else 0
        st["bad"]  += 0 if st["hit"] else 1
        reset_rep(st)

def update_situp_rep(st, knee_val, hit_knee):
    down_th = hit_knee + 10.0
    up_th   = min(hit_knee + 35.0, 175.0)
    if st["phase"] == "up" and knee_val <= down_th:
        st["phase"] = "down"; st["in_rep"] = True; st["hit"] = False
    if st["phase"] == "down" and st["in_rep"] and knee_val >= up_th:
        st["good"] += 1 if st["hit"] else 0
        st["bad"]  += 0 if st["hit"] else 1
        reset_rep(st)

# ============================================================
# LIMB MAPPING
# ============================================================
def limbs_for_bad_form(title: str, metrics):
    if title == "Bodyweight Squat":
        return ["left_leg", "right_leg"]
    if title == "Lunge":
        return ["left_leg"] if metrics["lk"] < metrics["rk"] else ["right_leg"]
    if title == "Push-Up":
        return ["left_hand", "right_hand"]
    if title == "Sit-Up":
        return ["left_leg", "right_leg"]
    if title == "Tree Pose":
        return ["left_leg", "right_leg"]
    return []

# ============================================================
# EXERCISE LOOP
# ============================================================
def exercise_loop(cap, user, difficulty, assigned_reps, pose_every_n=2,
                  enable_draw=True, enable_buzz=True, profiler=False, autosave_profile=True):

    profile = load_or_default_profile(user, difficulty=difficulty, autosave=autosave_profile)

    exercises = ["Bodyweight Squat", "Lunge", "Push-Up", "Sit-Up", "Tree Pose"]
    ex_idx = 0

    REST_SECONDS = 10
    TREE_HOLD_SECONDS = {8: 10, 12: 15, 16: 20}
    target_reps = int(assigned_reps)
    tree_hold = int(TREE_HOLD_SECONDS.get(target_reps, 15))

    in_rest = False
    rest_end_t = 0.0

    tree_ok_start = None
    tree_done = False

    reps = {
        "Bodyweight Squat": rep_state_new(),
        "Lunge": rep_state_new(),
        "Push-Up": rep_state_new(),
        "Sit-Up": rep_state_new(),
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

    def draw_for_ex(img, lm, title, side=None):
        if not enable_draw:
            return
        if title == "Bodyweight Squat":
            if side == "right":
                draw_subset(img, lm, SQUAT_SIDE["right"], SQUAT_CONN_RIGHT, enabled=True)
            else:
                draw_subset(img, lm, SQUAT_SIDE["left"], SQUAT_CONN_LEFT, enabled=True)
        elif title == "Lunge":
            draw_subset(img, lm, LUNGE_IDX, LUNGE_CONN, enabled=True)
        elif title == "Push-Up":
            draw_subset(img, lm, PUSHUP_IDX, PUSHUP_CONN, enabled=True)
        elif title == "Sit-Up":
            draw_subset(img, lm, SITUP_IDX, SITUP_CONN, enabled=True)
        elif title == "Tree Pose":
            draw_subset(img, lm, TREE_IDX_NO_HEAD, TREE_CONN_NO_HEAD, enabled=True)

    def reset_for_next():
        nonlocal tree_ok_start, tree_done
        for dq in smoothers.values():
            dq.clear()
        for k in reps:
            reset_rep(reps[k])
        tree_ok_start = None
        tree_done = False

    last_display = {ex: None for ex in exercises}

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        frame_i = 0
        last_res = None

        while True:
            t0 = time.perf_counter()

            # CAMERA READ
            t = time.perf_counter()
            ret, frame = cap.read()
            prof["read_ms"].append(_ms(time.perf_counter() - t))
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            img = frame

            title = exercises[ex_idx]

            # REST MODE
            if in_rest:
                send_off_all(enable_buzz=enable_buzz)
                remaining = int(max(0, rest_end_t - time.time()))
                cv2.putText(img, "REST", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 3)
                cv2.putText(img, f"Next in {remaining}s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)

                if remaining <= 0:
                    in_rest = False
                    reset_for_next()
                    ex_idx = (ex_idx + 1) % len(exercises)

                t = time.perf_counter()
                draw_profiler(img, profiler)
                show_frame(img)
                key = cv2.waitKeyEx(1)
                prof["ui_ms"].append(_ms(time.perf_counter() - t))
                prof["loop_ms"].append(_ms(time.perf_counter() - t0))

                if key in (81, 2424832):   # Left arrow
                    send_off_all(enable_buzz=enable_buzz)
                    break
                if key in (83, 2555904):   # Right arrow
                    in_rest = False
                    reset_for_next()
                    ex_idx = (ex_idx + 1) % len(exercises)
                if key in (82, 2490368):   # Up arrow
                    try:
                        cur = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                        make_fullscreen(WINDOW_NAME, enable=(cur != cv2.WINDOW_FULLSCREEN))
                    except Exception:
                        pass
                continue

            # POSE PROCESS (throttled)
            t = time.perf_counter()
            frame_i += 1
            if frame_i % pose_every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_res = pose.process(rgb)
            res = last_res
            prof["pose_ms"].append(_ms(time.perf_counter() - t))

            # LOGIC + DRAW
            t = time.perf_counter()
            cv2.putText(img, f"Exercise: {title}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            if title == "Tree Pose":
                cv2.putText(img, f"Target: hold {tree_hold}s", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
            else:
                cv2.putText(img, f"Target: {target_reps} reps", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

            display_good = None
            info = ""

            if res is not None and res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                metrics = compute_metrics(title, lm, smoothers)
                side = metrics.get("side", None)

                draw_for_ex(img, lm, title, side=side)

                ok_now, ok_hit, info = evaluate(title, lm, profile, difficulty, metrics)

                if ok_now is None:
                    display_good = None
                else:
                    if title in reps:
                        st = reps[title]

                        if st["in_rep"] and ok_hit:
                            st["hit"] = True

                        if title == "Bodyweight Squat":
                            hit_knee = float(profile["exercises"]["Bodyweight Squat"]["hit_deg"])
                            update_squat_rep(st, metrics["knee"], hit_knee)

                        elif title == "Push-Up":
                            hit_elbow = float(profile["exercises"]["Push-Up"]["hit_deg"])
                            update_pushup_rep(st, metrics["elbow"], hit_elbow)

                        elif title == "Lunge":
                            hit_knee = float(profile["exercises"]["Lunge"]["hit_deg"])
                            update_lunge_rep(st, metrics["lk"], metrics["rk"], hit_knee)

                        elif title == "Sit-Up":
                            hit_knee = float(profile["exercises"]["Sit-Up"]["hit_deg"])
                            update_situp_rep(st, metrics["knee"], hit_knee)

                        display_good = True if (st["in_rep"] and st["hit"]) else (True if ok_now else False)

                        cv2.putText(img, f"Reps OK: {st['good']}/{target_reps}", (20, 230),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(img, f"Reps BAD: {st['bad']}", (20, 270),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                        if st["good"] >= target_reps:
                            send_off_all(enable_buzz=enable_buzz)
                            in_rest = True
                            rest_end_t = time.time() + REST_SECONDS
                            last_display[title] = None
                            tree_ok_start = None
                            tree_done = False

                    elif title == "Tree Pose":
                        display_good = True if ok_now else False
                        if display_good:
                            if tree_ok_start is None:
                                tree_ok_start = time.time()
                            held = time.time() - tree_ok_start
                            cv2.putText(img, f"Hold: {held:.1f}/{tree_hold}s", (20, 230),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            if held >= tree_hold and not tree_done:
                                tree_done = True
                                send_off_all(enable_buzz=enable_buzz)
                                in_rest = True
                                rest_end_t = time.time() + REST_SECONDS
                                last_display[title] = None
                        else:
                            tree_ok_start = None
                    else:
                        display_good = True if ok_now else False

                # UI status
                if display_good is True:
                    cv2.putText(img, "GOOD", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                elif display_good is False:
                    cv2.putText(img, "ADJUST", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    cv2.putText(img, "NOT VISIBLE", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

                if info:
                    cv2.putText(img, info, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)

                # BUZZ
                limbs = limbs_for_bad_form(title, metrics)
                if display_good is False:
                    send_on_keepalive(limbs, enable_buzz=enable_buzz)
                elif display_good is True:
                    if last_display.get(title) is not True:
                        send_off(limbs, enable_buzz=enable_buzz)
                else:
                    send_off(limbs, enable_buzz=enable_buzz)

                last_display[title] = display_good

            else:
                cv2.putText(img, "No body detected", (20, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                send_off_all(enable_buzz=enable_buzz)
                last_display[title] = None
                tree_ok_start = None

            prof["logic_ms"].append(_ms(time.perf_counter() - t))

            # DISPLAY
            t = time.perf_counter()
            draw_profiler(img, profiler)
            show_frame(img)
            key = cv2.waitKeyEx(1)
            prof["ui_ms"].append(_ms(time.perf_counter() - t))
            prof["loop_ms"].append(_ms(time.perf_counter() - t0))

            if key in (81, 2424832):  # Left arrow
                send_off_all(enable_buzz=enable_buzz)
                break
            elif key in (83, 2555904):  # Right arrow
                send_off_all(enable_buzz=enable_buzz)
                reset_for_next()
                ex_idx = (ex_idx + 1) % len(exercises)
            elif key in (82, 2490368):  # Up arrow
                try:
                    cur = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                    make_fullscreen(WINDOW_NAME, enable=(cur != cv2.WINDOW_FULLSCREEN))
                except Exception:
                    pass


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--difficulty", default="Standard", choices=["Beginner", "Standard", "Advanced"])
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--w", type=int, default=480)
    p.add_argument("--h", type=int, default=360)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--pose-every", type=int, default=2, help="Run pose inference every N frames (2/3/4).")
    p.add_argument("--prof", action="store_true", help="Show profiler overlay.")
    p.add_argument("--no-draw", action="store_true", help="Disable skeleton drawing overlays.")
    p.add_argument("--no-buzz", action="store_true", help="Disable ESP vibration HTTP calls.")
    p.add_argument("--no-save-fallback", action="store_true", help="Don't write fallback profile JSON to disk.")
    p.add_argument("--dshow", action="store_true", help="Use DirectShow backend (Windows).")
    args = p.parse_args()

    backend = cv2.CAP_DSHOW if args.dshow else 0
    cap = cv2.VideoCapture(args.cam, backend) if backend != 0 else cv2.VideoCapture(args.cam)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        print("⚠️ Camera not available.")
        raise SystemExit(1)

    make_fullscreen(WINDOW_NAME, False)

    try:
        exercise_loop(
            cap=cap,
            user=args.user,
            difficulty=args.difficulty,
            assigned_reps=args.reps,
            pose_every_n=max(1, int(args.pose_every)),
            enable_draw=(not args.no_draw),
            enable_buzz=(not args.no_buzz),
            profiler=args.prof,
            autosave_profile=(not args.no_save_fallback)
        )
    finally:
        send_off_all(enable_buzz=(not args.no_buzz))
        cap.release()
        cv2.destroyAllWindows()