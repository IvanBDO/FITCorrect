# ==============================
# FILE: ui_launcher.py  (FULL, updated - difficulty page tweaks)
#
# ✅ CHANGE:
# - Difficulty page subheader shows "Assigned Difficulty: <difficulty>"
# - Assigned difficulty card (icon + text) is larger
# - Difficulty is computed from BMI ranges (already in your logic)
#
# Everything else is kept as-is.
# ==============================

import os
import sys
import subprocess
import json
import time
import tkinter as tk

from PIL import Image
import customtkinter as ctk
from tkinter import messagebox

# --- CONSTANTS ---
THEME_GREEN = "#4ADE80"
THEME_DARK_TEXT = "#1F2937"
THEME_BG_WHITE = "#FFFFFF"
THEME_GREY_BTN = "#E5E7EB"

COLOR_BEGINNER = "#2A7942"
COLOR_STANDARD = "#3B82F6"
COLOR_ADVANCED = "#EF4444"
COLOR_SELECTED = "#F3F4F6"

# Fonts
THEME_FONT_TITLE = ("Milescut", 80, "bold")
THEME_FONT_SUBTITLE = ("Tilt Warp", 32)
THEME_FONT_OPTION = ("Tilt Warp", 20)
THEME_FONT_BODY = ("Arial", 20)

# ----------------------------
# DATA OPTIONS (UPDATED RANGES)
# ----------------------------
AGE_OPTIONS = [
    "Under 18 years old",
    "18 - 29 years old",
    "30 - 45 years old",
    "46 - 60 years old",
    "Over 60 years old"
]

SEX_OPTIONS = ["Male", "Female"]

# ✅ Narrower height ranges
HEIGHT_OPTIONS = [
    "Under 150cm",
    "150 - 159cm",
    "160 - 169cm",
    "170 - 179cm",
    "180 - 189cm",
    "Over 190cm"
]

# ✅ Narrower weight ranges
WEIGHT_OPTIONS = [
    "Under 50kg",
    "50 - 59kg",
    "60 - 69kg",
    "70 - 79kg",
    "80 - 89kg",
    "90 - 99kg",
    "Over 100kg"
]

EXERCISE_FREQ_OPTIONS = [
    "Sedentary (little to no exercise)",
    "Light (1 - 2 days/week)",
    "Moderate (3 - 4 days/week)",
    "Active (5+ days/week)"
]

INJURY_OPTIONS = [
    "No Injury",
    "Upper Body (Neck, Shoulders, Back)",
    "Lower Body (Hips, Knee, Ankles)"
]

# ----------------------------
# PATHS / STORAGE
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

USER_DB_DIR = os.path.join(SCRIPT_DIR, "user_profiles")
USER_INDEX_PATH = os.path.join(USER_DB_DIR, "_index.json")


def ensure_user_db():
    os.makedirs(USER_DB_DIR, exist_ok=True)
    if not os.path.exists(USER_INDEX_PATH):
        with open(USER_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump({"users": []}, f, indent=2)


def safe_id(name: str) -> str:
    return "".join(c for c in (name or "") if c.isalnum() or c in ("-", "_"))


def user_profile_path(user_id: str) -> str:
    ensure_user_db()
    uid = safe_id(user_id) or "DEFAULT"
    return os.path.join(USER_DB_DIR, f"{uid}.json")


def load_user_index():
    ensure_user_db()
    users = []
    try:
        with open(USER_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        users = data.get("users", []) or []
    except Exception:
        users = []

    try:
        for fn in os.listdir(USER_DB_DIR):
            if fn.lower().endswith(".json") and fn != "_index.json":
                base = os.path.splitext(fn)[0]
                if base not in users:
                    users.append(base)
    except Exception:
        pass

    return sorted(list(dict.fromkeys(users)))


def save_user_index(users):
    ensure_user_db()
    users = sorted(list(dict.fromkeys(users)))
    with open(USER_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"users": users}, f, indent=2)


def load_user_profile(user_id: str):
    p = user_profile_path(user_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_user_profile(profile: dict):
    ensure_user_db()
    uid = safe_id(profile.get("user") or "DEFAULT") or "DEFAULT"

    users = load_user_index()
    if uid not in users:
        users.append(uid)
        save_user_index(users)

    with open(user_profile_path(uid), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def delete_user_profile(user_id: str):
    uid = safe_id(user_id) or "DEFAULT"

    p = user_profile_path(uid)
    if os.path.exists(p):
        try:
            os.remove(p)
        except Exception:
            pass

    calib_dir = os.path.join(SCRIPT_DIR, "calibration_profiles")
    calib_path = os.path.join(calib_dir, f"{uid}.json")
    if os.path.exists(calib_path):
        try:
            os.remove(calib_path)
        except Exception:
            pass

    users = load_user_index()
    users = [u for u in users if u != uid]
    save_user_index(users)


def next_auto_user_id():
    users = load_user_index()
    mx = 0
    for u in users:
        if u.startswith("USER_"):
            try:
                n = int(u.split("_", 1)[1])
                mx = max(mx, n)
            except Exception:
                pass
    return f"USER_{mx + 1}"


def display_name(uid: str) -> str:
    return (uid or "").replace("_", " ")


# ----------------------------
# BMI FROM RANGES + DECISION
# ----------------------------
def parse_height_range(label: str):
    label = (label or "").strip()
    if label == "Under 150cm":
        return 1.20, 1.49
    if label == "150 - 159cm":
        return 1.50, 1.59
    if label == "160 - 169cm":
        return 1.60, 1.69
    if label == "170 - 179cm":
        return 1.70, 1.79
    if label == "180 - 189cm":
        return 1.80, 1.89
    if label == "Over 190cm":
        return 1.90, 2.10
    return None


def parse_weight_range(label: str):
    label = (label or "").strip()
    if label == "Under 50kg":
        return 30.0, 49.9
    if label == "50 - 59kg":
        return 50.0, 59.9
    if label == "60 - 69kg":
        return 60.0, 69.9
    if label == "70 - 79kg":
        return 70.0, 79.9
    if label == "80 - 89kg":
        return 80.0, 89.9
    if label == "90 - 99kg":
        return 90.0, 99.9
    if label == "Over 100kg":
        return 100.0, 160.0
    return None


def bmi_interval_from_ranges(height_label: str, weight_label: str):
    hr = parse_height_range(height_label)
    wr = parse_weight_range(weight_label)
    if not hr or not wr:
        return None
    h_min, h_max = hr
    w_min, w_max = wr

    # BMI = kg / m^2
    bmi_min = w_min / (h_max * h_max)
    bmi_max = w_max / (h_min * h_min)
    return round(bmi_min, 2), round(bmi_max, 2)


def bmi_class_from_midpoint(bmin: float, bmax: float) -> str:
    bmi_mid = (bmin + bmax) / 2.0
    if bmi_mid >= 30.0:
        return "Obese"
    if bmi_mid < 18.5:
        return "Underweight"
    if bmi_mid >= 25.0:
        return "Overweight"
    return "Normal"


def apply_age_modifier(difficulty: str, age_label: str) -> str:
    age_label = (age_label or "").strip()
    if age_label in ("Under 18 years old", "Over 60 years old"):
        if difficulty == "Advanced":
            return "Standard"
        if difficulty == "Standard":
            return "Beginner"
    return difficulty


def decide_difficulty_and_reps(bmi_class: str, exercise_freq: str, injuries: list):
    injury_set = set(injuries or [])
    has_injury = any(x != "No Injury" for x in injury_set) and len(injury_set) > 0
    if has_injury:
        return "Beginner", "Low reps (8–12)"

    if bmi_class == "Obese":
        return "Beginner", "Low reps (8–12)"
    if bmi_class == "Underweight":
        return "Beginner", "Low–Moderate (8–12)"
    if bmi_class == "Overweight":
        if exercise_freq in ("Moderate (3 - 4 days/week)", "Active (5+ days/week)"):
            return "Standard", "Low–Moderate (10–15)"
        return "Beginner", "Low–Moderate (10–15)"
    # Normal
    if exercise_freq == "Active (5+ days/week)":
        return "Advanced", "Moderate–High (12–20)"
    return "Standard", "Moderate–High (12–20)"


# ----------------------------
# UI APP
# ----------------------------
class FitCorrectUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("Light")
        self.title("FitCorrect")

        # ✅ Fullscreen launcher (Wayland-friendly, still uses attributes fullscreen)
        self.resizable(True, True)

        def _wayland_fullscreen_retry(n=12):
            try:
                self.attributes("-fullscreen", True)  # master method
                self.lift()
                self.focus_force()
                self.update_idletasks()
            except Exception:
                pass
            if n > 0:
                self.after(120, lambda: _wayland_fullscreen_retry(n - 1))

        self.after(0, _wayland_fullscreen_retry)

        self.configure(fg_color=THEME_BG_WHITE)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.profile = {
            "user": None,
            "is_new": False,
            "age": None,
            "sex": None,
            "height": None,
            "weight": None,
            "exercise_freq": None,
            "injuries": [],
            "bmi_est_min": None,
            "bmi_est_max": None,
            "bmi_est_class": None,
            "difficulty": "Standard",
            "rep_goal": None,
            "has_calibration": False,
            "camera_index": 0,
            "last_updated": int(time.time()),
        }

        self.page_index = 0
        self.pages = []
        self._validate_current = None

        self.nav_items = []
        self.nav_index = 0

        self.user_scroll = None
        self._user_canvas = None
        self._user_inner = None

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # --- HEADER ---
        self.header = ctk.CTkFrame(self, fg_color=THEME_BG_WHITE, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", pady=(20, 0))
        self.header.grid_columnconfigure(0, weight=1)

        self.title_lbl = ctk.CTkLabel(self.header, text="PROFILE SETUP", font=THEME_FONT_TITLE, text_color=THEME_GREEN)
        self.title_lbl.grid(row=0, column=0, pady=(0, 5))

        self.progress_bar = ctk.CTkProgressBar(self.header, width=520, height=10, progress_color=THEME_GREEN, fg_color="#E5E7EB")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, pady=(0, 15))

        self.subtitle_lbl = ctk.CTkLabel(self.header, text="", font=THEME_FONT_SUBTITLE, text_color=THEME_DARK_TEXT)
        self.subtitle_lbl.grid(row=2, column=0, pady=(0, 5))

        # --- CONTENT ---
        self.content = ctk.CTkFrame(self, fg_color=THEME_BG_WHITE, corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew", padx=80, pady=(20, 0))
        self.content.grid_columnconfigure(0, weight=1)

        # --- FOOTER ---
        self.footer = ctk.CTkFrame(self, fg_color=THEME_BG_WHITE, corner_radius=0)
        self.footer.grid(row=2, column=0, sticky="ew", padx=30, pady=(0, 30))
        self.footer.grid_columnconfigure(0, weight=1)

        self.remote_hint = ctk.CTkLabel(self.footer, text="", font=("Arial", 12), text_color="gray")
        self.remote_hint.grid(row=0, column=0, sticky="w")

        self._build_pages()
        self._bind_keys()
        self._show_page(0)

    # ---------------------------
    # small utility: find descendants
    # ---------------------------
    def _find_descendant(self, root, predicate):
        stack = [root]
        while stack:
            w = stack.pop()
            try:
                if predicate(w):
                    return w
                stack.extend(w.winfo_children())
            except Exception:
                pass
        return None

    def _resolve_scroll_parts(self):
        if self.user_scroll is None:
            return None, None

        sf = self.user_scroll

        canvas = None
        for attr in ("_parent_canvas", "_canvas", "canvas"):
            if hasattr(sf, attr):
                cand = getattr(sf, attr)
                if hasattr(cand, "yview_moveto") and hasattr(cand, "bbox"):
                    canvas = cand
                    break

        if canvas is None:
            canvas = self._find_descendant(sf, lambda w: isinstance(w, tk.Canvas))

        inner = None
        if canvas is not None:
            inner = self._find_descendant(sf, lambda w: isinstance(w, (tk.Frame, ctk.CTkFrame)) and w.master == canvas)

        return canvas, inner

    # ---------------------------
    # LIFECYCLE
    # ---------------------------
    def on_closing(self):
        # safety: if we hid the window and user tries to close
        try:
            self.destroy()
        except Exception:
            pass

    def _toggle_fullscreen(self):
        try:
            cur = bool(self.attributes("-fullscreen"))
            self.after(0, lambda: self.attributes("-fullscreen", True))
            self.after(50, lambda: (self.lift(), self.focus_force()))
        except Exception:
            pass

    def _bind_keys(self):
        self.bind("<Up>", self._nav_up)
        self.bind("<Down>", self._nav_down)
        self.bind("<Left>", self._nav_left)
        self.bind("<Right>", self._nav_right)
        self.bind("<Return>", self._nav_select)
        self.bind("<BackSpace>", lambda e: self._back())
        self.bind("<Escape>", lambda e: self._back())  # keep your original behavior
        self.bind("<F11>", lambda e: self._toggle_fullscreen())

    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    # ---------------------------
    # CORE LAUNCH (UPDATED LOGIC ONLY)
    # ---------------------------
    def _core_calibration_profile_path(self, user_id: str):
        uid = safe_id(user_id) or "DEFAULT"
        return os.path.join(SCRIPT_DIR, "calibration_profiles", f"{uid}.json")

    def _compute_bmi_and_assign(self):
        interval = bmi_interval_from_ranges(self.profile.get("height"), self.profile.get("weight"))
        if not interval:
            return False
        bmin, bmax = interval

        bcls = bmi_class_from_midpoint(bmin, bmax)

        self.profile["bmi_est_min"] = bmin
        self.profile["bmi_est_max"] = bmax
        self.profile["bmi_est_class"] = bcls

        diff, reps = decide_difficulty_and_reps(
            bmi_class=bcls,
            exercise_freq=self.profile.get("exercise_freq") or "",
            injuries=self.profile.get("injuries") or [],
        )

        diff = apply_age_modifier(diff, self.profile.get("age") or "")

        self.profile["difficulty"] = diff
        self.profile["rep_goal"] = reps
        return True

    def _run_and_return(self, cmd):
        """
        Minimal seamless transfer:
        - Launcher DOES NOT close.
        - Launcher hides (withdraw) while external script runs.
        - When script exits, launcher returns.
        """
        self.update_idletasks()
        self.withdraw()
        try:
            return subprocess.run(cmd, check=False).returncode
        finally:
            try:
                self.deiconify()
                self.lift()
                self.focus_force()
            except Exception:
                pass

    def _launch_core(self):
        """
        ✅ UPDATED FLOW (logic-only):
        - If calibration exists: run exercise_launcher.py directly
        - If not: run calibration_launcher.py first, then exercise_launcher.py
        """
        try:
            self.profile["last_updated"] = int(time.time())

            uid = safe_id(self.profile.get("user") or "DEFAULT") or "DEFAULT"
            self.profile["user"] = uid

            calib_path = self._core_calibration_profile_path(uid)
            has_cal = os.path.exists(calib_path)
            self.profile["has_calibration"] = has_cal

            # Save UI profile (unchanged behavior)
            save_user_profile(self.profile)

            # Keep writing profile.json (even if scripts don't use it yet)
            profile_json = os.path.join(SCRIPT_DIR, "profile.json")
            with open(profile_json, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2)

            # Resolve reps (do not touch other UI pages; keep logic here)
            reps_map = {"Beginner": 8, "Standard": 12, "Advanced": 16}
            difficulty = (self.profile.get("difficulty") or "Standard").strip()
            assigned_reps = self.profile.get("assigned_reps")
            try:
                assigned_reps = int(assigned_reps)
            except Exception:
                assigned_reps = int(reps_map.get(difficulty, 12))

            cam_index = self.profile.get("camera_index", 0)
            try:
                cam_index = int(cam_index)
            except Exception:
                cam_index = 0

            calib_script = os.path.join(SCRIPT_DIR, "calibration_launcher.py")
            ex_script = os.path.join(SCRIPT_DIR, "exercise_launcher.py")

            if not os.path.exists(ex_script):
                messagebox.showerror("Launch Error", f"Missing file:\n{ex_script}")
                return

            # 1) If no calibration yet -> calibrate first
            if not has_cal:
                if not os.path.exists(calib_script):
                    messagebox.showerror("Launch Error", f"Missing file:\n{calib_script}")
                    return

                # NOTE: calibration_launcher.py (as you sent) expects: user difficulty
                cmd_cal = [sys.executable, calib_script, uid, difficulty]
                rc = self._run_and_return(cmd_cal)

                # Re-check calibration file after running
                has_cal = os.path.exists(calib_path)
                self.profile["has_calibration"] = has_cal
                save_user_profile(self.profile)

                if rc != 0 or not has_cal:
                    messagebox.showerror("Calibration Error", "Calibration failed or was incomplete.")
                    return

            # 2) Run exercise
            cmd_ex = [
                sys.executable, ex_script,
                "--user", uid,
                "--difficulty", difficulty,
                "--reps", str(assigned_reps),
                "--cam", str(cam_index),
            ]
            self._run_and_return(cmd_ex)

        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch core scripts.\n\n{e}")
            return
        finally:
            # Refresh user list page
            try:
                self._show_page(1)
            except Exception:
                pass

    # ---------------------------
    # AUTO-SCROLL (PAGE 1): FOLLOW HIGHLIGHT + CENTER
    # ---------------------------
    def _scroll_nav_into_view_if_needed(self):
        if self.page_index != 1:
            return
        if self.user_scroll is None:
            return
        if not self.nav_items or not (0 <= self.nav_index < len(self.nav_items)):
            return

        w = self.nav_items[self.nav_index].get("widget")
        if w is None:
            return

        try:
            canvas = self._user_canvas
            inner = self._user_inner
            if canvas is None or inner is None:
                canvas, inner = self._resolve_scroll_parts()
                self._user_canvas, self._user_inner = canvas, inner

            if canvas is None or inner is None:
                return

            self.update_idletasks()
            canvas.update_idletasks()
            inner.update_idletasks()
            w.update_idletasks()

            target = w
            while target is not None and target.master is not None and target.master != inner:
                target = target.master
            if target is None:
                return

            bbox = canvas.bbox("all")
            if not bbox:
                return

            total_h = max(1, bbox[3] - bbox[1])
            view_h = max(1, canvas.winfo_height())

            current_top = canvas.canvasy(0)
            canvas_rooty = canvas.winfo_rooty()

            target_center_screen = target.winfo_rooty() + (target.winfo_height() / 2.0)
            target_center_canvas = current_top + (target_center_screen - canvas_rooty)

            desired_top = target_center_canvas - (view_h / 2.0)

            max_top = max(0, total_h - view_h)
            desired_top = max(0, min(max_top, desired_top))

            canvas.yview_moveto(desired_top / total_h)

        except Exception:
            pass

    # ==========================================================
    # PAGES
    # ==========================================================
    def _page_landing(self):
        self._validate_current = lambda: True
        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack(expand=True, fill="both")

        img_path = os.path.join(SCRIPT_DIR, "logo.png")
        if os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path)
                # scale logo a bit bigger for fullscreen
                sw = self.winfo_screenwidth()
                logo_w = min(1100, int(sw * 0.65))
                logo_h = int(logo_w * 0.5625)
                logo_img = ctk.CTkImage(pil_img, size=(logo_w, logo_h))
                ctk.CTkLabel(container, text="", image=logo_img).place(relx=0.5, rely=0.45, anchor="center")
            except Exception:
                pass
        else:
            ctk.CTkLabel(container, text="FIT CORRECT", font=("Milescut", 80), text_color=THEME_GREEN).place(relx=0.5, rely=0.45, anchor="center")

        ctk.CTkLabel(container, text="PRESS ENTER TO START", font=("Tilt Warp", 22), text_color="gray").place(relx=0.5, rely=0.8, anchor="center")
        self.nav_items.append({"widget": None, "action": self._next})

    def _page_user_select(self):
        self._validate_current = lambda: True
        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack(expand=True, fill="both")

        center_frame = ctk.CTkFrame(container, fg_color=THEME_BG_WHITE)
        center_frame.place(relx=0.5, rely=0.55, anchor="center")

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        scroll_w = min(950, int(sw * 0.60))
        scroll_h = min(760, int(sh * 0.62))

        self.user_scroll = ctk.CTkScrollableFrame(
            center_frame,
            fg_color=THEME_BG_WHITE,
            width=scroll_w,
            height=scroll_h,
        )
        self.user_scroll.pack()

        self._user_canvas, self._user_inner = self._resolve_scroll_parts()

        users = load_user_index()

        def select_existing(uid):
            loaded = load_user_profile(uid)
            if loaded:
                self.profile.update(loaded)
            self.profile["user"] = uid
            self.profile["is_new"] = False
            self._launch_core()

        def delete_profile(uid):
            ok = messagebox.askyesno("Delete Profile", f"Delete {display_name(uid)}?\nThis will also delete calibration.")
            if not ok:
                return
            delete_user_profile(uid)
            self._show_page(1)

        def add_user_auto():
            uid = next_auto_user_id()
            self.profile = {
                "user": uid,
                "is_new": True,
                "age": None,
                "sex": None,
                "height": None,
                "weight": None,
                "exercise_freq": None,
                "injuries": [],
                "bmi_est_min": None,
                "bmi_est_max": None,
                "bmi_est_class": None,
                "difficulty": "Standard",
                "rep_goal": None,
                "has_calibration": False,
                "camera_index": 0,
                "last_updated": int(time.time()),
            }
            self._next()

        if not users:
            ctk.CTkLabel(self.user_scroll, text="No users found.", font=("Tilt Warp", 22), text_color="gray").pack(pady=(0, 20))

        for uid in users:
            row = ctk.CTkFrame(self.user_scroll, fg_color=THEME_BG_WHITE)
            row.pack(pady=8)

            user_btn = ctk.CTkButton(
                row,
                text=display_name(uid),
                fg_color=THEME_GREY_BTN,
                text_color="#4EB977",
                font=("Tilt Warp", 28),
                height=88,
                width=int(scroll_w * 0.72),
                hover_color="#D1D5DB",
                command=lambda u=uid: select_existing(u)
            )
            user_btn.pack(side="left", padx=(0, 10))

            del_btn = ctk.CTkButton(
                row,
                text="X",
                fg_color="#EF4444",
                text_color="white",
                font=("Arial", 28, "bold"),
                height=88,
                width=100,
                hover_color="#DC2626",
                command=lambda u=uid: delete_profile(u)
            )
            del_btn.pack(side="left")

            self.nav_items.append({"widget": user_btn, "action": lambda u=uid: select_existing(u)})
            self.nav_items.append({"widget": del_btn, "action": lambda u=uid: delete_profile(u)})

        add_btn = ctk.CTkButton(
            self.user_scroll,
            text="+ ADD USER",
            fg_color=THEME_GREEN,
            text_color="white",
            font=("Tilt Warp", 28),
            height=88,
            width=int(scroll_w * 0.83),
            hover_color="#34D399",
            command=add_user_auto
        )
        add_btn.pack(pady=12)
        self.nav_items.append({"widget": add_btn, "action": add_user_auto})

        self.after(30, self._scroll_nav_into_view_if_needed)

    def _page_choice(self, key, options):
        self._validate_current = None
        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack(pady=1)

        selected_var = ctk.StringVar(value="")
        saved_val = self.profile.get(key)

        for i, opt in enumerate(options):
            wrapper = ctk.CTkFrame(container, fg_color=THEME_BG_WHITE, corner_radius=8)
            wrapper.pack(pady=10, anchor="w")

            rb = ctk.CTkRadioButton(
                wrapper, text=opt, variable=selected_var, value=opt,
                font=THEME_FONT_OPTION, text_color=THEME_DARK_TEXT, fg_color=THEME_GREEN,
                border_color=THEME_DARK_TEXT, hover_color=THEME_GREEN, bg_color=THEME_BG_WHITE
            )
            rb.pack(padx=10, pady=10)

            self.nav_items.append({"widget": wrapper, "action": self._next, "variable": selected_var, "value": opt})
            if saved_val == opt:
                self.nav_index = i

        def apply(*_):
            if selected_var.get():
                self.profile[key] = selected_var.get()

        selected_var.trace_add("write", apply)

        if saved_val in options:
            selected_var.set(saved_val)

        self._validate_current = lambda: messagebox.showwarning("Missing", "Select one.") if not selected_var.get() else True

    def _page_injuries(self):
        self._validate_current = None
        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack()

        vars_ = {opt: ctk.BooleanVar(value=False) for opt in INJURY_OPTIONS}

        def enforce_healthy():
            if vars_["No Injury"].get():
                for k in vars_:
                    if k != "No Injury":
                        vars_[k].set(False)

        def toggle(k):
            vars_[k].set(not vars_[k].get())
            enforce_healthy()

        for opt in INJURY_OPTIONS:
            wrapper = ctk.CTkFrame(container, fg_color=THEME_BG_WHITE, corner_radius=8)
            wrapper.pack(pady=10, anchor="w")

            cb = ctk.CTkCheckBox(
                wrapper, text=opt, variable=vars_[opt], command=enforce_healthy,
                font=THEME_FONT_OPTION, text_color=THEME_DARK_TEXT, fg_color=THEME_GREEN,
                border_color=THEME_DARK_TEXT, bg_color=THEME_BG_WHITE
            )
            cb.pack(padx=10, pady=10)

            self.nav_items.append({"widget": wrapper, "action": lambda o=opt: toggle(o)})

        done_btn = ctk.CTkButton(
            container,
            text=" CONFIRM SELECTION ",
            fg_color=THEME_GREEN,
            font=("Tilt Warp", 20, "bold"),
            text_color="white",
            height=60,
            width=320,
            command=self._next
        )
        done_btn.pack(pady=24)
        self.nav_items.append({"widget": done_btn, "action": self._next})

        def validate():
            selected = [k for k, v in vars_.items() if v.get()]
            if not selected:
                messagebox.showwarning("Missing", "Select at least one.")
                return False

            if "No Injury" in selected:
                selected = ["No Injury"]

            self.profile["injuries"] = selected
            return True

        self._validate_current = validate

    # ✅ UPDATED difficulty page (big assigned visuals + subtitle shows assigned)
    def _page_difficulty_display(self):
        # 1) Compute BMI -> assign difficulty/reps (existing logic)
        if not self._compute_bmi_and_assign():
            messagebox.showerror("BMI Error", "Could not compute BMI from selected height/weight ranges.")
            self._back()
            return

        self._validate_current = lambda: True

        # 2) Normalize assigned difficulty label
        raw = (self.profile.get("difficulty") or "Standard")
        key = str(raw).strip().lower()
        label_map = {"beginner": "Beginner", "standard": "Standard", "advanced": "Advanced"}
        assigned = label_map.get(key, "Standard")
        self.profile["difficulty"] = assigned

        # 3) Update header subtitle to show result
        self.page_subtitles[8] = f"Assigned Difficulty: {assigned}"

        # 4) Reps map must be in this scope (fixes your old crash)
        reps_map = {
            "Beginner": 8,
            "Standard": 12,
            "Advanced": 16,
        }
        self.profile["assigned_reps"] = int(reps_map.get(assigned, 12))

        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack(fill="both", expand=True)

        cards_frame = ctk.CTkFrame(container, fg_color=THEME_BG_WHITE)
        cards_frame.place(relx=0.5, rely=0.42, anchor="center")

        paths = {
            "Beginner": os.path.join(SCRIPT_DIR, "beginner.png"),
            "Standard": os.path.join(SCRIPT_DIR, "standard.png"),
            "Advanced": os.path.join(SCRIPT_DIR, "advanced.png")
        }

        SIZE_SMALL = (230, 130)
        SIZE_LARGE = (390, 217.5)  # much bigger for assigned

        self.imgs = {}
        images_loaded = True
        try:
            for k, v in paths.items():
                if os.path.exists(v):
                    self.imgs[k] = {
                        "small": ctk.CTkImage(Image.open(v), size=SIZE_SMALL),
                        "large": ctk.CTkImage(Image.open(v), size=SIZE_LARGE)
                    }
                else:
                    images_loaded = False
        except Exception:
            images_loaded = False

        self.card_widgets = {}

        def update_visuals(val):
            for v, w in self.card_widgets.items():
                is_sel = (v == val)
                w["frame"].configure(
                    fg_color=COLOR_SELECTED if is_sel else THEME_BG_WHITE,
                    border_color=w["color"] if is_sel else THEME_BG_WHITE,
                    border_width=6 if is_sel else 2,
                )

                if images_loaded and w["icon"] is not None:
                    new_img = self.imgs[v]["large"] if is_sel else self.imgs[v]["small"]
                    w["icon"].configure(image=new_img)

                w["label"].configure(font=("Tilt Warp", 54 if is_sel else 24, "bold"))

                if w.get("reps_label") is not None:
                    w["reps_label"].configure(
                        font=("Tilt Warp", 30 if is_sel else 18, "bold" if is_sel else "normal"),
                        text_color=w["color"]
                    )

        def build_card(col_idx, label, color):
            card = ctk.CTkFrame(
                cards_frame,
                fg_color=THEME_BG_WHITE,
                corner_radius=14,
                border_width=2,
                border_color=THEME_BG_WHITE
            )
            card.grid(row=0, column=col_idx, padx=28, sticky="nsew")

            icon_lbl = None
            if images_loaded:
                icon_lbl = ctk.CTkLabel(card, text="", image=self.imgs[label]["small"])
                icon_lbl.pack(pady=(10, 8), padx=10)
            else:
                ctk.CTkLabel(card, text="[IMG]", text_color="gray").pack(pady=(10, 8))

            text_lbl = ctk.CTkLabel(
                card,
                text=label,
                font=("Tilt Warp", 24, "bold"),
                text_color=color
            )
            text_lbl.pack(padx=18, pady=(2, 0))

            reps_lbl = ctk.CTkLabel(
                card,
                text=f"{reps_map.get(label, 12)} reps",
                font=("Tilt Warp", 18),
                text_color=color
            )
            reps_lbl.pack(pady=(4, 14))

            self.card_widgets[label] = {
                "frame": card,
                "color": color,
                "icon": icon_lbl,
                "label": text_lbl,
                "reps_label": reps_lbl,
            }

        build_card(0, "Beginner", COLOR_BEGINNER)
        build_card(1, "Standard", COLOR_STANDARD)
        build_card(2, "Advanced", COLOR_ADVANCED)

        update_visuals(assigned)
        self.after(10, lambda: update_visuals(assigned))

        ctk.CTkLabel(container, text="PRESS ENTER TO START", font=("Tilt Warp", 22), text_color="gray").place(relx=0.5, rely=0.92, anchor="center")
        self.nav_items.append({"widget": None, "action": self._next})

    # ==========================================================
    # NAVIGATION CONTROLLER
    # ==========================================================
    def _build_pages(self):
        self.pages = [
            self._page_landing,                                        # 0
            self._page_user_select,                                    # 1
            lambda: self._page_choice("age", AGE_OPTIONS),             # 2
            lambda: self._page_choice("sex", SEX_OPTIONS),             # 3
            lambda: self._page_choice("height", HEIGHT_OPTIONS),       # 4
            lambda: self._page_choice("weight", WEIGHT_OPTIONS),       # 5
            lambda: self._page_choice("exercise_freq", EXERCISE_FREQ_OPTIONS),  # 6
            self._page_injuries,                                       # 7
            self._page_difficulty_display,                             # 8
        ]

        self.page_subtitles = [
            "",
            "Who is training?",
            "Select your age range.",
            "Select your biological sex.",
            "Select your height range.",
            "Select your weight range.",
            "How often do you exercise?",
            "Do you have existing pain/injury in any of these areas?",
            "Assigned Difficulty",
        ]

    def _show_page(self, idx):
        if idx >= len(self.pages):
            self._launch_core()
            return

        self.page_index = idx
        self._clear_content()

        self.nav_items = []
        self.nav_index = 0

        self.user_scroll = None
        self._user_canvas = None
        self._user_inner = None

        self.pages[idx]()
        self._update_header()
        self._highlight_nav_item()

    def _update_header(self):
        if self.page_index == 0:
            self.header.grid_remove()
        else:
            self.header.grid()

        self.title_lbl.configure(text="PROFILE SETUP")
        try:
            self.subtitle_lbl.configure(text=self.page_subtitles[self.page_index])
        except Exception:
            self.subtitle_lbl.configure(text="")

        if self.page_index < 2:
            self.progress_bar.grid_remove()
        else:
            self.progress_bar.grid()
            total_steps = 7
            step_num = max(0, self.page_index - 1)
            self.progress_bar.set(step_num / total_steps)

    def _next(self):
        if self.page_index == len(self.pages) - 1:
            self._launch_core()
            return
        if self._validate_current and not self._validate_current():
            return
        self._show_page(self.page_index + 1)

    def _back(self):
        if self.page_index > 0:
            self._show_page(self.page_index - 1)

    def _highlight_nav_item(self):
        if not self.nav_items:
            return

        for i, item in enumerate(self.nav_items):
            try:
                w = item.get("widget")
                if w is None:
                    continue

                if isinstance(w, ctk.CTkButton):
                    w.configure(border_width=2 if i == self.nav_index else 0, border_color="black")
                elif hasattr(w, "configure"):
                    if self.page_index != 8:
                        w.configure(fg_color=COLOR_SELECTED if i == self.nav_index else THEME_BG_WHITE)

                if i == self.nav_index and "variable" in item:
                    item["variable"].set(item["value"])
            except Exception:
                pass

        if self.page_index == 1:
            self.after(10, self._scroll_nav_into_view_if_needed)

    def _nav_up(self, e=None):
        if self.page_index == 8:
            return
        if self.nav_index > 0:
            self.nav_index -= 1
            self._highlight_nav_item()
            if self.page_index == 1:
                self.after(10, self._scroll_nav_into_view_if_needed)

    def _nav_down(self, e=None):
        if self.page_index == 8:
            return
        if self.nav_index < len(self.nav_items) - 1:
            self.nav_index += 1
            self._highlight_nav_item()
            if self.page_index == 1:
                self.after(10, self._scroll_nav_into_view_if_needed)

    def _nav_left(self, e=None):
        if self.page_index == 8:
            return

    def _nav_right(self, e=None):
        if self.page_index == 8:
            return

    def _nav_select(self, e=None):
        if self.page_index == 8:
            self._next()
            return
        if 0 <= self.nav_index < len(self.nav_items):
            self.nav_items[self.nav_index]["action"]()
        else:
            self._next()


if __name__ == "__main__":
    app = FitCorrectUI()
    app.mainloop()
