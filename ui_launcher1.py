# ==============================
# FILE: fitcorrect_setup_ui.py
# (Questions preserved, calibration cut -> core handles calibration)
# ==============================
import os
import sys
import subprocess
import json

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

# Data (PRESERVED)
AGE_OPTIONS = ["Under 18 years old", "18 - 29 years old", "30 - 45 years old", "46 - 60 years old", "Over 60 years old"]
SEX_OPTIONS = ["Male", "Female"]
HEIGHT_OPTIONS = ["Under 150cm", "150 - 165cm", "166 - 180cm", "Over 180cm"]
WEIGHT_OPTIONS = ["Under 50kg", "50 - 65kg", "66 - 80kg", "81 - 100kg", "Over 100kg"]
EXERCISE_FREQ_OPTIONS = ["Sedentary (little to no exercise)", "Light (1 - 2 days/week)", "Moderate (3 - 4 days/week)", "Active (5+ days/week)"]
INJURY_OPTIONS = ["No Injury", "Upper Body (Neck, Shoulders, Back)", "Lower Body (Hips, Knee, Ankles)"]

# Mock Database of Users
EXISTING_USERS = ["USER 1", "USER 2", "USER 3"]


class FitCorrectUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("Light")
        self.title("FitCorrect")
        self.geometry("1000x700")
        self.resizable(False, False)
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
            "difficulty": "Standard",
            # optional camera_index if you ever want to store it
            "camera_index": 0,
        }

        self.page_index = 0
        self.pages = []
        self._validate_current = None

        # --- REMOTE NAVIGATION STATE ---
        self.nav_items = []
        self.nav_index = 0

        # Grid Layout
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Content
        self.grid_rowconfigure(2, weight=0)  # Footer
        self.grid_columnconfigure(0, weight=1)

        # --- HEADER ---
        self.header = ctk.CTkFrame(self, fg_color=THEME_BG_WHITE, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", pady=(20, 0))
        self.header.grid_columnconfigure(0, weight=1)

        self.title_lbl = ctk.CTkLabel(self.header, text="PROFILE SETUP", font=THEME_FONT_TITLE, text_color=THEME_GREEN)
        self.title_lbl.grid(row=0, column=0, pady=(0, 5))

        self.progress_bar = ctk.CTkProgressBar(self.header, width=400, height=10, progress_color=THEME_GREEN, fg_color="#E5E7EB")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, pady=(0, 15))

        self.subtitle_lbl = ctk.CTkLabel(self.header, text="", font=THEME_FONT_SUBTITLE, text_color=THEME_DARK_TEXT)
        self.subtitle_lbl.grid(row=2, column=0, pady=(0, 5))

        # --- CONTENT ---
        self.content = ctk.CTkFrame(self, fg_color=THEME_BG_WHITE, corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew", padx=50, pady=(20, 0))
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
    # LIFECYCLE
    # ---------------------------
    def on_closing(self):
        self.destroy()

    def _bind_keys(self):
        self.bind("<Up>", self._nav_up)
        self.bind("<Down>", self._nav_down)
        self.bind("<Left>", self._nav_left)
        self.bind("<Right>", self._nav_right)
        self.bind("<Return>", self._nav_select)
        self.bind("<BackSpace>", lambda e: self._back())
        self.bind("<Escape>", lambda e: self._back())

    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    # ---------------------------
    # CORE LAUNCH
    # ---------------------------
    def _safe_user_for_core(self, user: str):
        return "".join(c for c in (user or "") if c.isalnum() or c in ("-", "_")) or "DEFAULT"

    def _core_calibration_profile_path(self, user: str):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        safe = self._safe_user_for_core(user)
        return os.path.join(script_dir, "calibration_profiles", f"{safe}.json")

    def _launch_core(self):
        """
        Save UI profile.json then launch core.
        If calibration profile exists => normal run
        else => run with --calibrate
        """
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            profile_json = os.path.join(script_dir, "profile.json")
            core_path = os.path.join(script_dir, "fitcorrect_core.py")

            with open(profile_json, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2)

            user = self.profile.get("user") or "DEFAULT"
            calib_path = self._core_calibration_profile_path(user)
            has_calib = os.path.exists(calib_path)

            if has_calib:
                subprocess.Popen([sys.executable, core_path, profile_json])
            else:
                subprocess.Popen([sys.executable, core_path, "--calibrate", profile_json])

        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to open fitcorrect_core.py\n\n{e}")
            return

        self.on_closing()

    # ==========================================================
    # PAGES
    # ==========================================================
    def _page_landing(self):
        self._validate_current = lambda: True
        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack(expand=True, fill="both")

        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
        if os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path)
                logo_img = ctk.CTkImage(pil_img, size=(1000, 562.5))
                ctk.CTkLabel(container, text="", image=logo_img).place(relx=0.5, rely=0.45, anchor="center")
            except:
                pass
        else:
            ctk.CTkLabel(container, text="FIT CORRECT", font=("Milescut", 80), text_color=THEME_GREEN).place(relx=0.5, rely=0.45, anchor="center")

        ctk.CTkLabel(container, text="PRESS ENTER TO START", font=("Tilt Warp", 20), text_color="gray").place(relx=0.5, rely=0.8, anchor="center")
        self.nav_items.append({"widget": None, "action": self._next})

    def _page_user_select(self):
        self._validate_current = lambda: True
        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack(expand=True, fill="both")

        center_frame = ctk.CTkFrame(container, fg_color=THEME_BG_WHITE)
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        # ✅ FIX: DO NOT jump to calibration/core here.
        # Keep the same questions flow.
        def select_existing(u):
            self.profile["user"] = u
            self.profile["is_new"] = False
            self._next()  # go to AGE question

        def select_new():
            self.profile["user"] = "New User"
            self.profile["is_new"] = True
            self._next()  # go to AGE question

        for user in EXISTING_USERS:
            btn = ctk.CTkButton(
                center_frame,
                text=user,
                fg_color=THEME_GREY_BTN,
                text_color="#4EB977",
                font=("Tilt Warp", 26),
                height=80,
                width=490,
                hover_color="#D1D5DB",
                command=lambda u=user: select_existing(u)
            )
            btn.pack(pady=10)
            self.nav_items.append({"widget": btn, "action": lambda u=user: select_existing(u)})

        add_btn = ctk.CTkButton(
            center_frame,
            text="+ ADD USER",
            fg_color=THEME_GREEN,
            text_color="white",
            font=("Tilt Warp", 26),
            height=80,
            width=490,
            hover_color="#34D399",
            command=select_new
        )
        add_btn.pack(pady=10)
        self.nav_items.append({"widget": add_btn, "action": select_new})

    def _page_choice(self, key, options):
        self._validate_current = None
        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack(pady=1)

        selected_var = ctk.StringVar(value="")
        saved_val = self.profile.get(key)

        for i, opt in enumerate(options):
            wrapper = ctk.CTkFrame(container, fg_color=THEME_BG_WHITE, corner_radius=8)
            wrapper.pack(pady=8, anchor="w")

            rb = ctk.CTkRadioButton(
                wrapper, text=opt, variable=selected_var, value=opt,
                font=THEME_FONT_OPTION, text_color=THEME_DARK_TEXT, fg_color=THEME_GREEN,
                border_color=THEME_DARK_TEXT, hover_color=THEME_GREEN, bg_color=THEME_BG_WHITE
            )
            rb.pack(padx=10, pady=10)

            self.nav_items.append({"widget": wrapper, "action": self._next, "variable": selected_var, "value": opt})
            if saved_val == opt:
                self.nav_index = i

        def apply(*args):
            if selected_var.get():
                self.profile[key] = selected_var.get()

        selected_var.trace_add("write", apply)
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
            wrapper.pack(pady=8, anchor="w")

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
            height=50,
            width=260,
            command=self._next
        )
        done_btn.pack(pady=20)
        self.nav_items.append({"widget": done_btn, "action": self._next})

        def validate():
            selected = [k for k, v in vars_.items() if v.get()]
            if not selected:
                messagebox.showwarning("Missing", "Select at least one.")
                return False
            self.profile["injuries"] = selected
            return True

        self._validate_current = validate

    def _page_difficulty(self):
        self._validate_current = lambda: True
        container = ctk.CTkFrame(self.content, fg_color=THEME_BG_WHITE)
        container.pack(fill="both", expand=True, pady=20)

        cards_frame = ctk.CTkFrame(container, fg_color=THEME_BG_WHITE)
        cards_frame.place(relx=0.5, rely=0.5, anchor="center")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        paths = {
            "Beginner": os.path.join(script_dir, "beginner.png"),
            "Standard": os.path.join(script_dir, "standard.png"),
            "Advanced": os.path.join(script_dir, "advanced.png")
        }

        SIZE_SMALL = (219, 120)
        SIZE_LARGE = (302, 165)

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
        except:
            images_loaded = False

        self.card_widgets = {}

        def update_visuals(val):
            self.profile["difficulty"] = val
            for v, w in self.card_widgets.items():
                is_sel = (v == val)
                w["frame"].configure(fg_color=COLOR_SELECTED if is_sel else THEME_BG_WHITE)
                w["frame"].configure(border_color=w["color"] if is_sel else THEME_BG_WHITE)
                w["frame"].configure(border_width=4 if is_sel else 2)

                if images_loaded:
                    new_img = self.imgs[v]["large"] if is_sel else self.imgs[v]["small"]
                    w["icon"].configure(image=new_img)

                new_font = ("Tilt Warp", 32 if is_sel else 24, "bold")
                w["label"].configure(font=new_font)

        def build_card(col_idx, label, color):
            card = ctk.CTkFrame(cards_frame, fg_color=THEME_BG_WHITE, corner_radius=8, border_width=2, border_color=THEME_BG_WHITE)
            card.grid(row=0, column=col_idx, padx=10, sticky="nsew")

            def on_click(e=None):
                update_visuals(label)

            card.bind("<Button-1>", on_click)

            icon_lbl = None
            if images_loaded:
                icon_lbl = ctk.CTkLabel(card, text="", image=self.imgs[label]["small"])
                icon_lbl.pack(pady=(15, 5), padx=5)
                icon_lbl.bind("<Button-1>", on_click)
            else:
                ctk.CTkLabel(card, text="[IMG]", text_color="gray").pack(pady=(15, 5))

            text_lbl = ctk.CTkLabel(card, text=label, font=("Tilt Warp", 24, "bold"), text_color=color)
            text_lbl.pack(padx=15, pady=5)
            text_lbl.bind("<Button-1>", on_click)

            self.card_widgets[label] = {"frame": card, "color": color, "icon": icon_lbl, "label": text_lbl}
            self.nav_items.append({"widget": card, "action": lambda: update_visuals(label)})

        build_card(0, "Beginner", COLOR_BEGINNER)
        build_card(1, "Standard", COLOR_STANDARD)
        build_card(2, "Advanced", COLOR_ADVANCED)

        self.nav_index = 1
        update_visuals("Standard")

        # NOTE: On ENTER in difficulty page, we proceed to core
        self.nav_items.append({"widget": None, "action": self._next})

    # ==========================================================
    # NAVIGATION CONTROLLER
    # ==========================================================
    def _build_pages(self):
        self.pages = [
            self._page_landing,                 # 0
            self._page_user_select,             # 1
            lambda: self._page_choice("age", AGE_OPTIONS),
            lambda: self._page_choice("sex", SEX_OPTIONS),
            lambda: self._page_choice("height", HEIGHT_OPTIONS),
            lambda: self._page_choice("weight", WEIGHT_OPTIONS),
            lambda: self._page_choice("exercise_freq", EXERCISE_FREQ_OPTIONS),
            self._page_injuries,                # 7
            self._page_difficulty,              # 8
            # 9+ calibration is CUT -> core handles
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
            "Select Difficulty",
        ]

    def _show_page(self, idx):
        # CUT at calibration: after difficulty page (8), next would be 9 => launch core
        if idx >= 9:
            self._launch_core()
            return

        self.page_index = idx
        self._clear_content()
        self.nav_items = []
        self.nav_index = 0
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
        except:
            self.subtitle_lbl.configure(text="")

        # progress bar: show only on question pages
        if self.page_index < 2:
            self.progress_bar.grid_remove()
        else:
            self.progress_bar.grid()
            # pages 2..8 inclusive => 7 steps
            total_steps = 7
            step_num = max(0, self.page_index - 1)  # age=1 step
            self.progress_bar.set(step_num / total_steps)

    def _next(self):
        if self.page_index < len(self.pages) - 1:
            if self._validate_current and not self._validate_current():
                return
            self._show_page(self.page_index + 1)
        else:
            # after difficulty => calibration cut => core
            self._show_page(9)

    def _back(self):
        if self.page_index > 0:
            self._show_page(self.page_index - 1)

    def _highlight_nav_item(self):
        if not self.nav_items:
            return
        for i, item in enumerate(self.nav_items):
            try:
                color = COLOR_SELECTED if i == self.nav_index else THEME_BG_WHITE
                if isinstance(item["widget"], ctk.CTkButton):
                    item["widget"].configure(border_width=2 if i == self.nav_index else 0, border_color="black")
                elif hasattr(item["widget"], "configure"):
                    if self.page_index != 8:
                        item["widget"].configure(fg_color=color)
                if i == self.nav_index and "variable" in item:
                    item["variable"].set(item["value"])
            except:
                pass

    def _nav_up(self, e=None):
        if self.page_index == 8:
            return
        if self.nav_index > 0:
            self.nav_index -= 1
            self._highlight_nav_item()

    def _nav_down(self, e=None):
        if self.page_index == 8:
            return
        if self.nav_index < len(self.nav_items) - 1:
            self.nav_index += 1
            self._highlight_nav_item()

    def _nav_left(self, e=None):
        if self.page_index == 8 and self.nav_index > 0:
            self.nav_index -= 1
            self._highlight_nav_item()
            self.nav_items[self.nav_index]["action"]()

    def _nav_right(self, e=None):
        if self.page_index == 8 and self.nav_index < len(self.nav_items) - 1:
            self.nav_index += 1
            self._highlight_nav_item()
            self.nav_items[self.nav_index]["action"]()

    def _nav_select(self, e=None):
        if self.page_index == 8:
            # pressing enter on difficulty proceeds (cut -> core)
            self._next()
            return
        if 0 <= self.nav_index < len(self.nav_items):
            self.nav_items[self.nav_index]["action"]()


if __name__ == "__main__":
    app = FitCorrectUI()
    app.mainloop()
