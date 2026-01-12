# fitcorrect_ui.py
# Same logic as your original (profiles.json, create/delete, keyboard nav, calibration start)
# Updated DESIGN based on your provided UI theme (Light mode, colors, fonts, layout)

import os
import json
import threading
import re
import time

import customtkinter as ctk
from tkinter import messagebox

from fitcorrect_core import calibration_phase, exercise_loop, profile_path

# ============================================================
# THEME (from your design code)
# ============================================================
THEME_GREEN = "#4ADE80"
THEME_DARK_TEXT = "#1F2937"
THEME_BG_WHITE = "#FFFFFF"
THEME_GREY_BTN = "#E5E7EB"
THEME_GREY_BTN_HOVER = "#D1D5DB"

COLOR_BEGINNER = "#2A7942"
COLOR_STANDARD = "#3B82F6"
COLOR_ADVANCED = "#EF4444"
COLOR_SELECTED = "#F3F4F6"

# Fonts (if missing on the machine, Tk will fallback)
THEME_FONT_TITLE = ("Milescut", 56, "bold")
THEME_FONT_SUBTITLE = ("Tilt Warp", 22)
THEME_FONT_OPTION = ("Tilt Warp", 20)
THEME_FONT_BUTTON = ("Tilt Warp", 18, "bold")
THEME_FONT_BODY = ("Arial", 16)
THEME_FONT_HINT = ("Arial", 12)

# ============================================================
# Persistent UI profile storage (NOT calibration baseline)
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_PATH = os.path.join(APP_DIR, "profiles.json")

PROFILE_FIELDS = [
    "age",
    "sex",
    "height",
    "weight",
    "exercise_freq",
    "injuries",
    "difficulty",
    "camera_index",
]


def load_profiles() -> dict:
    if not os.path.exists(PROFILES_PATH):
        return {}
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_profiles(profiles: dict) -> None:
    tmp = PROFILES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    os.replace(tmp, PROFILES_PATH)


def next_user_name(profiles: dict) -> str:
    """Generate next label: User 1, User 2, ..."""
    nums = []
    pat = re.compile(r"^User\s+(\d+)$", re.IGNORECASE)
    for k in profiles.keys():
        if isinstance(k, str):
            m = pat.match(k.strip())
            if m:
                try:
                    nums.append(int(m.group(1)))
                except Exception:
                    pass
    n = (max(nums) + 1) if nums else 1
    return f"User {n}"


def sort_users(users: list[str]) -> list[str]:
    """Sort by numeric suffix when possible."""
    def key(s: str):
        m = re.search(r"(\d+)", s)
        return (int(m.group(1)) if m else 10**9, s)
    return sorted(users, key=key)


# ============================================================
# Options (same as your original)
# ============================================================
AGE_OPTIONS = [
    "Under 18 years old",
    "18 - 29 years old",
    "30 - 45 years old",
    "46 - 60 years old",
    "Over 60 years old",
]
SEX_OPTIONS = ["Male", "Female"]
HEIGHT_OPTIONS = ["Under 150 cm", "150 - 165 cm", "166 - 180 cm", "Over 180 cm"]
WEIGHT_OPTIONS = ["Under 50 kg", "50 - 65 kg", "66 - 80 kg", "81 - 100 kg", "Over 100 kg"]
EXERCISE_FREQ_OPTIONS = [
    "Sedentary (little to no exercise)",
    "Light (1 - 2 days/week)",
    "Moderate (3 - 4 days/week)",
    "Active (5+ days/week)",
]
INJURY_OPTIONS = [
    "No Injury (I am healthy)",
    "Upper Body (Neck, Shoulders, Back)",
    "Lower Body (Hips, Knees, Ankles)",
]
DIFFICULTY_OPTIONS = ["Beginner", "Standard", "Advanced"]


# ============================================================
# UI
# ============================================================
class FitCorrectUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("green")

        self.title("FitCorrect")
        self.geometry("1000x700")
        self.resizable(False, False)
        self.configure(fg_color=THEME_BG_WHITE)

        self.profile = {
            "user": None,
            "age": None,
            "sex": None,
            "height": None,
            "weight": None,
            "exercise_freq": None,
            "injuries": [],
            "difficulty": "Standard",
            "camera_index": 0,
        }

        self.page_index = 0
        self.pages = []
        self.page_titles = []
        self._validate_current = None

        # User page nav (kept)
        self._user_items = []
        self._user_buttons = []
        self._user_sel = 0
        self._kbd_bound = False

        self._last_enter_ts = 0.0
        self._enter_debounce_s = 0.30

        # Layout grid
        self.grid_rowconfigure(0, weight=0)  # header
        self.grid_rowconfigure(1, weight=1)  # content
        self.grid_rowconfigure(2, weight=0)  # footer
        self.grid_columnconfigure(0, weight=1)

        # ======================================================
        # HEADER (New design)
        # ======================================================
        self.header = ctk.CTkFrame(self, fg_color=THEME_BG_WHITE, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", pady=(20, 0))
        self.header.grid_columnconfigure(0, weight=1)

        self.title_lbl = ctk.CTkLabel(
            self.header,
            text="PROFILE SETUP",
            font=THEME_FONT_TITLE,
            text_color=THEME_GREEN
        )
        self.title_lbl.grid(row=0, column=0, pady=(0, 6))

        self.progress = ctk.CTkProgressBar(
            self.header,
            width=420,
            height=10,
            progress_color=THEME_GREEN,
            fg_color=THEME_GREY_BTN
        )
        self.progress.set(0.0)
        self.progress.grid(row=1, column=0, pady=(0, 14))

        self.subtitle_lbl = ctk.CTkLabel(
            self.header,
            text="",
            font=THEME_FONT_SUBTITLE,
            text_color=THEME_DARK_TEXT
        )
        self.subtitle_lbl.grid(row=2, column=0, pady=(0, 8))

        # ======================================================
        # CONTENT (kept scroll, restyled)
        # ======================================================
        self.content = ctk.CTkScrollableFrame(self, fg_color=THEME_BG_WHITE, corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew", padx=50, pady=(10, 0))
        self.content.grid_columnconfigure(0, weight=1)

        # ======================================================
        # FOOTER (New design)
        # ======================================================
        self.footer = ctk.CTkFrame(self, fg_color=THEME_BG_WHITE, corner_radius=0)
        self.footer.grid(row=2, column=0, sticky="ew", padx=30, pady=(0, 30))
        self.footer.grid_columnconfigure(0, weight=1)

        self.remote_hint = ctk.CTkLabel(self.footer, text="", font=THEME_FONT_HINT, text_color="gray")
        self.remote_hint.grid(row=0, column=0, sticky="w")

        btn_row = ctk.CTkFrame(self.footer, fg_color=THEME_BG_WHITE)
        btn_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        self.back_btn = ctk.CTkButton(
            btn_row,
            text="Back",
            width=180,
            height=48,
            fg_color=THEME_GREY_BTN,
            hover_color=THEME_GREY_BTN_HOVER,
            text_color=THEME_DARK_TEXT,
            font=THEME_FONT_BUTTON,
            command=self._back
        )
        self.back_btn.grid(row=0, column=0, sticky="w")

        self.next_btn = ctk.CTkButton(
            btn_row,
            text="Next",
            width=220,
            height=48,
            fg_color=THEME_GREEN,
            hover_color="#34D399",
            text_color="white",
            font=THEME_FONT_BUTTON,
            command=self._next
        )
        self.next_btn.grid(row=0, column=1, sticky="e")

        self._build_pages()
        self._show_page(0)

    # ============================================================
    # Helpers (restyled “card”)
    # ============================================================
    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _card(self, title, subtitle=None):
        # Flat, white container like your design
        card = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE, corner_radius=0)
        card.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text=title,
            font=("Tilt Warp", 28, "bold"),
            text_color=THEME_DARK_TEXT
        ).grid(row=0, column=0, sticky="w", pady=(10, 8))

        if subtitle:
            ctk.CTkLabel(
                card,
                text=subtitle,
                font=THEME_FONT_BODY,
                text_color="gray"
            ).grid(row=1, column=0, sticky="w", pady=(0, 14))

        body = ctk.CTkFrame(card, fg_color=THEME_BG_WHITE, corner_radius=0)
        body.grid(row=2, column=0, sticky="ew")
        body.grid_columnconfigure(0, weight=1)
        return body

    def _persist_profile(self):
        user = (self.profile.get("user") or "").strip()
        if not user:
            return
        profiles = load_profiles()
        profiles[user] = {k: self.profile.get(k) for k in PROFILE_FIELDS}
        save_profiles(profiles)

    def _load_profile_from_store(self, username: str):
        profiles = load_profiles()
        saved = profiles.get(username, {})
        if not isinstance(saved, dict):
            saved = {}
        self.profile["user"] = username
        for k in PROFILE_FIELDS:
            if k in saved:
                self.profile[k] = saved[k]
        self.profile.setdefault("difficulty", "Standard")
        self.profile.setdefault("camera_index", 0)
        self.profile.setdefault("injuries", [])

    def _profile_complete(self) -> bool:
        required = ["age", "sex", "height", "weight", "exercise_freq", "injuries", "difficulty"]
        for k in required:
            v = self.profile.get(k)
            if v is None:
                return False
            if k == "injuries" and (not isinstance(v, list) or len(v) == 0):
                return False
            if isinstance(v, str) and not v.strip():
                return False
        return True

    # ============================================================
    # Navigation
    # ============================================================
    def _build_pages(self):
        self.pages = [
            self._page_user_select,
            lambda: self._page_choice("Age", "Select your age range.", "age", AGE_OPTIONS),
            lambda: self._page_choice("Sex", "Select your biological sex.", "sex", SEX_OPTIONS),
            lambda: self._page_choice("Height", "Select your height range.", "height", HEIGHT_OPTIONS),
            lambda: self._page_choice("Weight", "Select your weight range.", "weight", WEIGHT_OPTIONS),
            lambda: self._page_choice("Activity", "How often do you exercise?", "exercise_freq", EXERCISE_FREQ_OPTIONS),
            self._page_injuries,
            self._page_difficulty,
            self._page_ready,
        ]

        self.page_titles = [
            ("Who is training?", "Use ↑/↓ then Enter to select"),
            ("Select your age range.", ""),
            ("Select your biological sex.", ""),
            ("Select your height range.", ""),
            ("Select your weight range.", ""),
            ("How often do you exercise?", ""),
            ("Do you have existing pain/injury?", "Select all that apply (or choose No Injury)."),
            ("Select Difficulty", ""),
            ("Calibration Phase", "Ready to start calibration + training"),
        ]

    def _update_header(self):
        total = len(self.pages)
        idx = self.page_index + 1
        title, sub = self.page_titles[self.page_index]

        # Title behavior like your second UI:
        # - Survey pages say PROFILE SETUP
        # - Final page says CALIBRATION PHASE
        if self.page_index >= total - 1:
            self.title_lbl.configure(text="CALIBRATION PHASE")
        else:
            self.title_lbl.configure(text="PROFILE SETUP")

        self.subtitle_lbl.configure(text=title if title else "")
        self.remote_hint.configure(text=sub if sub else "")

        # Progress behavior: show only for questionnaire pages (Age..Difficulty)
        if 1 <= self.page_index <= 7:
            self.progress.grid()
            # Map pages 1..7 onto 0..1
            q_total = 7
            q_pos = self.page_index / q_total
            self.progress.set(max(0.0, min(1.0, q_pos)))
        else:
            self.progress.grid_remove()

        self.back_btn.configure(state="normal" if self.page_index > 0 else "disabled")
        self.next_btn.configure(text="I'm Ready" if self.page_index == total - 1 else "Next")

    def _show_page(self, idx: int):
        if self.page_index == 0 and idx != 0:
            self._unbind_user_kbd()

        self.page_index = idx
        self._clear_content()
        self.pages[idx]()
        self._update_header()

        try:
            self.content._parent_canvas.yview_moveto(0)
        except Exception:
            pass

        if idx == 0:
            self._bind_user_kbd()

        try:
            self.focus_force()
        except Exception:
            pass

    def _back(self):
        if self.page_index > 0:
            self._show_page(self.page_index - 1)

    def _next(self):
        if self._validate_current and not self._validate_current():
            return
        if self.page_index == len(self.pages) - 1:
            self._start_program(immediate=False)
            return
        self._show_page(self.page_index + 1)

    # ============================================================
    # Keyboard navigation (User page only) - kept
    # ============================================================
    def _bind_user_kbd(self):
        if self._kbd_bound:
            return
        self.bind("<Up>", self._user_up)
        self.bind("<Down>", self._user_down)
        self.bind("<Return>", self._user_enter)
        self.bind("<KP_Enter>", self._user_enter)
        self._kbd_bound = True

    def _unbind_user_kbd(self):
        if not self._kbd_bound:
            return
        self.unbind("<Up>")
        self.unbind("<Down>")
        self.unbind("<Return>")
        self.unbind("<KP_Enter>")
        self._kbd_bound = False

    def _user_up(self, _evt=None):
        if self.page_index != 0 or not self._user_items:
            return
        self._user_sel = (self._user_sel - 1) % len(self._user_items)
        self._user_render_selection()

    def _user_down(self, _evt=None):
        if self.page_index != 0 or not self._user_items:
            return
        self._user_sel = (self._user_sel + 1) % len(self._user_items)
        self._user_render_selection()

    def _user_enter(self, _evt=None):
        if self.page_index != 0 or not self._user_items:
            return

        now = time.time()
        if now - self._last_enter_ts < self._enter_debounce_s:
            return
        self._last_enter_ts = now

        item = self._user_items[self._user_sel]
        kind = item["kind"]

        if kind == "user":
            self._load_profile_from_store(item["user"])
            if self._profile_complete():
                self._start_program(immediate=True)
            else:
                messagebox.showinfo("Setup", "This user needs setup questions first.")
                self._show_page(1)
            return

        if kind == "create":
            self._create_next_user()
            return

        if kind == "delete":
            self._delete_selected_user()
            return

    def _user_render_selection(self):
        for i, btn in enumerate(self._user_buttons):
            if i == self._user_sel:
                btn.configure(fg_color=COLOR_SELECTED, border_width=2, border_color="black")
            else:
                btn.configure(fg_color=THEME_GREY_BTN, border_width=0)

    # ============================================================
    # Pages (restyled)
    # ============================================================
    def _page_user_select(self):
        self._validate_current = None
        body = self._card("Select Profile", "Use ↑/↓ then Enter. Users auto-start if setup is complete.")

        profiles = load_profiles()
        users = sort_users([u for u in profiles.keys() if isinstance(u, str) and u.strip()])

        items = []
        for u in users:
            items.append({"kind": "user", "label": u, "user": u})

        items.append({"kind": "create", "label": "➕ Create Next User", "user": None})
        if users:
            items.append({"kind": "delete", "label": "🗑️ Delete Selected User", "user": None})

        self._user_items = items
        self._user_buttons = []
        self._user_sel = 0

        for i, it in enumerate(items):
            is_primary = (it["kind"] == "create")
            btn = ctk.CTkButton(
                body,
                text=it["label"],
                height=72,
                width=520,
                fg_color=THEME_GREEN if is_primary else THEME_GREY_BTN,
                hover_color="#34D399" if is_primary else THEME_GREY_BTN_HOVER,
                text_color="white" if is_primary else "#4EB977",
                font=("Tilt Warp", 26, "bold"),
                command=lambda idx=i: self._click_user_item(idx),
            )
            btn.grid(row=i, column=0, sticky="ew", pady=10)
            self._user_buttons.append(btn)

        self._user_render_selection()
        self._validate_current = lambda: True

    def _click_user_item(self, idx: int):
        if self.page_index != 0:
            return
        self._user_sel = idx
        self._user_render_selection()
        self._user_enter()

    def _create_next_user(self):
        profiles = load_profiles()
        u = next_user_name(profiles)

        self.profile = {
            "user": u,
            "age": None,
            "sex": None,
            "height": None,
            "weight": None,
            "exercise_freq": None,
            "injuries": [],
            "difficulty": "Standard",
            "camera_index": 0,
        }
        self._persist_profile()
        messagebox.showinfo("Created", f"Created: {u}\nProceeding to setup questions.")
        self._show_page(1)

    def _delete_selected_user(self):
        profiles = load_profiles()
        users = sort_users([u for u in profiles.keys() if isinstance(u, str) and u.strip()])
        if not users:
            messagebox.showwarning("No users", "No users to delete.")
            return

        highlighted = self._user_items[self._user_sel]
        target = highlighted["user"] if highlighted["kind"] == "user" else users[0]

        if not messagebox.askyesno("Delete", f"Delete profile '{target}'?"):
            return

        profiles.pop(target, None)
        save_profiles(profiles)

        # delete calibration baseline too
        try:
            cal_path = profile_path(target)
            if os.path.exists(cal_path):
                os.remove(cal_path)
        except Exception:
            pass

        if self.profile.get("user") == target:
            self.profile["user"] = None

        messagebox.showinfo("Deleted", f"Deleted: {target}")
        self._show_page(0)

    def _page_choice(self, title, subtitle, key, options):
        self._validate_current = None
        body = self._card(title, subtitle)

        selected_var = ctk.StringVar(value=self.profile.get(key) or "")

        wrappers = {}

        def apply(val: str):
            self.profile[key] = val
            selected_var.set(val)
            for opt, wrap in wrappers.items():
                wrap.configure(fg_color=COLOR_SELECTED if opt == val else THEME_BG_WHITE)

        for i, opt in enumerate(options):
            wrap = ctk.CTkFrame(body, fg_color=THEME_BG_WHITE, corner_radius=10)
            wrap.grid(row=i, column=0, sticky="ew", pady=8)
            wrap.grid_columnconfigure(0, weight=1)

            btn = ctk.CTkButton(
                wrap,
                text=opt,
                height=56,
                fg_color=THEME_BG_WHITE,
                hover_color=COLOR_SELECTED,
                text_color=THEME_DARK_TEXT,
                font=THEME_FONT_OPTION,
                anchor="w",
                command=lambda v=opt: apply(v),
            )
            btn.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

            wrappers[opt] = wrap

        if selected_var.get():
            apply(selected_var.get())

        def validate():
            if not self.profile.get(key):
                messagebox.showwarning("Missing", "Please select one option to continue.")
                return False
            return True

        self._validate_current = validate

    def _page_injuries(self):
        self._validate_current = None
        body = self._card("Injuries", "Select all that apply (or choose “No Injury”).")

        saved = self.profile.get("injuries") or []
        vars_ = {opt: ctk.BooleanVar(value=(opt in saved)) for opt in INJURY_OPTIONS}

        def enforce(opt=None):
            if opt and opt != "No Injury (I am healthy)" and vars_[opt].get():
                vars_["No Injury (I am healthy)"].set(False)
            if vars_["No Injury (I am healthy)"].get():
                for o in INJURY_OPTIONS:
                    if o != "No Injury (I am healthy)":
                        vars_[o].set(False)

        for i, opt in enumerate(INJURY_OPTIONS):
            wrap = ctk.CTkFrame(body, fg_color=THEME_BG_WHITE, corner_radius=10)
            wrap.grid(row=i, column=0, sticky="ew", pady=8)
            wrap.grid_columnconfigure(0, weight=1)

            cb = ctk.CTkCheckBox(
                wrap,
                text=opt,
                variable=vars_[opt],
                command=lambda o=opt: enforce(o),
                fg_color=THEME_GREEN,
                hover_color=THEME_GREEN,
                text_color=THEME_DARK_TEXT,
                font=THEME_FONT_OPTION,
            )
            cb.grid(row=0, column=0, sticky="w", padx=14, pady=14)

        enforce("No Injury (I am healthy)" if vars_["No Injury (I am healthy)"].get() else None)

        confirm = ctk.CTkButton(
            body,
            text="CONFIRM SELECTION",
            fg_color=THEME_GREEN,
            hover_color="#34D399",
            text_color="white",
            font=("Tilt Warp", 20, "bold"),
            height=54,
            command=self._next,
        )
        confirm.grid(row=len(INJURY_OPTIONS), column=0, sticky="w", pady=(18, 0))

        def validate():
            selected = [opt for opt, v in vars_.items() if v.get()]
            if not selected:
                messagebox.showwarning("Missing", "Select at least one option (or 'No Injury').")
                return False
            if "No Injury (I am healthy)" in selected:
                selected = ["No Injury (I am healthy)"]
            self.profile["injuries"] = selected
            return True

        self._validate_current = validate

    def _page_difficulty(self):
        self._validate_current = lambda: True
        body = self._card("Difficulty", "Choose how strict the checker should be.")

        # Card row
        row = ctk.CTkFrame(body, fg_color=THEME_BG_WHITE)
        row.grid(row=0, column=0, sticky="ew", pady=10)
        row.grid_columnconfigure((0, 1, 2), weight=1)

        cards = {}

        def color_for(opt: str):
            return COLOR_BEGINNER if opt == "Beginner" else COLOR_STANDARD if opt == "Standard" else COLOR_ADVANCED

        def apply(value: str):
            self.profile["difficulty"] = value
            for opt, frame in cards.items():
                is_sel = (opt == value)
                frame.configure(
                    fg_color=COLOR_SELECTED if is_sel else THEME_BG_WHITE,
                    border_width=4 if is_sel else 2,
                    border_color=color_for(opt) if is_sel else THEME_BG_WHITE,
                )
                cards[opt].label.configure(
                    text_color=color_for(opt),
                    font=("Tilt Warp", 32 if is_sel else 24, "bold"),
                )

        for j, opt in enumerate(DIFFICULTY_OPTIONS):
            frame = ctk.CTkFrame(
                row,
                fg_color=THEME_BG_WHITE,
                corner_radius=12,
                border_width=2,
                border_color=THEME_BG_WHITE,
            )
            frame.grid(row=0, column=j, padx=10, sticky="nsew")
            frame.grid_columnconfigure(0, weight=1)

            lbl = ctk.CTkLabel(
                frame,
                text=opt,
                font=("Tilt Warp", 24, "bold"),
                text_color=color_for(opt)
            )
            lbl.grid(row=0, column=0, padx=16, pady=30)

            def click_factory(v=opt):
                return lambda _e=None: apply(v)

            frame.bind("<Button-1>", click_factory(opt))
            lbl.bind("<Button-1>", click_factory(opt))

            frame.label = lbl
            cards[opt] = frame

        apply(self.profile.get("difficulty", "Standard") or "Standard")
        self._validate_current = lambda: True

    def _page_ready(self):
        self._validate_current = lambda: True
        body = self._card("Ready to start", "Calibration will open the camera and build your personal baseline.")

        user = (self.profile.get("user") or "").strip()

        tips = [
            f"User: {user}",
            "Make sure your full body is visible (as required per exercise).",
            "Press N to next / skip (but skipping will cancel now).",
            "Press Q to quit the camera window.",
            "IMPORTANT: Click the camera window before pressing keys.",
        ]
        for i, t in enumerate(tips):
            ctk.CTkLabel(body, text=f"• {t}", font=THEME_FONT_BODY, text_color=THEME_DARK_TEXT).grid(
                row=i, column=0, sticky="w", pady=6
            )

        cam_box = ctk.CTkFrame(body, fg_color=THEME_BG_WHITE, corner_radius=10)
        cam_box.grid(row=len(tips), column=0, sticky="ew", pady=(18, 6))
        cam_box.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(cam_box, text="Camera Index", font=("Tilt Warp", 18, "bold"), text_color=THEME_DARK_TEXT).grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 6)
        )
        self.camera_entry = ctk.CTkEntry(cam_box, placeholder_text="0")
        self.camera_entry.insert(0, str(self.profile.get("camera_index", 0)))
        self.camera_entry.grid(row=0, column=1, sticky="ew", padx=12, pady=(12, 6))

    # ============================================================
    # Start program (unchanged)
    # ============================================================
    def _start_program(self, immediate: bool):
        user = (self.profile.get("user") or "").strip()
        if not user:
            messagebox.showwarning("Missing", "Please select/create a user first.")
            self._show_page(0)
            return

        if immediate or not hasattr(self, "camera_entry"):
            try:
                cam_idx = int(self.profile.get("camera_index", 0) or 0)
            except Exception:
                cam_idx = 0
        else:
            try:
                cam_idx = int((self.camera_entry.get() or "0").strip())
            except ValueError:
                messagebox.showwarning("Invalid", "Camera index must be a number (e.g., 0).")
                return

        self.profile["camera_index"] = cam_idx
        difficulty = self.profile.get("difficulty", "Standard") or "Standard"
        self._persist_profile()

        if not messagebox.askyesno("Start", f"Start calibration now for '{user}'?\n(Camera index: {cam_idx})"):
            return

        self.next_btn.configure(state="disabled")
        self.back_btn.configure(state="disabled")

        def runner():
            try:
                prof = calibration_phase(camera_index=cam_idx, user=user, difficulty=difficulty)
                if prof is None:
                    self.after(0, lambda: messagebox.showinfo("Stopped", "Calibration cancelled or incomplete."))
                    return
                exercise_loop(camera_index=cam_idx, user=user, difficulty=difficulty, profile=prof)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Runtime Error", str(e)))
            finally:
                self.after(0, lambda: (
                    self.next_btn.configure(state="normal"),
                    self.back_btn.configure(state="normal")
                ))

        threading.Thread(target=runner, daemon=True).start()
        messagebox.showinfo("Running", "Camera window opened.\nPress Q to quit.\nPress N for next.")


if __name__ == "__main__":
    app = FitCorrectUI()
    app.mainloop()