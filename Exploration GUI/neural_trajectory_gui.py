"""
Neural Trajectory Viewer  (Interactive GUI)
════════════════════════════════════════════
Animates the 3D PCA neural-state trajectory (global or per brain-area)
side-by-side with the animated mouse skeleton (DLC tracking).

Matches the PCA approach in 03_pca_3d.py — uses sklearn + temporal Gaussian
smoothing (σ=5 frames ≈ 166 ms) — so trajectories look identical to your
static plots, just live.

Controls
────────
  Space       Play / Pause
  ← / →       Step 1 frame
  Shift+←/→   Step 10 frames
  Scrubber    Jump to any frame
  Session     Dropdown — reload
  Area        Dropdown — switch between Global PCA and per-brain-area PCA
  Speed       Frames advanced per timer tick

Usage
─────
  python 10_neural_trajectory_gui.py [SESSION]
  e.g.  python 10_neural_trajectory_gui.py M7_Female_1

Daniel Katz — Neural Data Science Project 5
"""

import os, sys, re
import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d, median_filter
from sklearn.decomposition import PCA
import cv2

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3d projection
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────────
def _find_data_dir() -> str:
    """Locate the .mat data folder.

    Resolution order:
      1. NEURAL_DATA_DIR environment variable  (set this on any machine)
      2. ../../Data  relative to this script   (works when the whole repo is cloned)
      3. ./Data      next to this script        (simple flat layout)
    """
    env = os.environ.get("NEURAL_DATA_DIR")
    if env and os.path.isdir(env):
        return os.path.normpath(env)
    # repo layout: <root>/Daniel/data_exploration/script.py  →  <root>/Data
    rel = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data"))
    if os.path.isdir(rel):
        return rel
    # flat layout: Data/ next to the script
    flat = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    if os.path.isdir(flat):
        return flat
    raise FileNotFoundError(
        "Cannot find the session data folder.\n"
        f"  Tried (relative):  {rel}\n"
        f"  Tried (flat):      {flat}\n"
        "Set the NEURAL_DATA_DIR environment variable to the folder "
        "that contains the .mat files and (optionally) the .mp4 videos."
    )

DATA_DIR = _find_data_dir()

# ── DLC body-part layout (8 parts × 3 cols per animal, 2 animals = 48 cols) ──
PART_NAMES = ["Ear_L","Ear_R","Nose","Centre","Lat_L","Lat_R","Tail_base","Tail_end"]
SKELETON   = [(2,3),(0,3),(1,3),(4,3),(5,3),(3,6),(6,7)]  # (from_idx, to_idx)
ANIMAL_COLORS = ["#4ea8de", "#ff6b6b"]

# ── Tuning ─────────────────────────────────────────────────────────────────────
SMOOTH_SIGMA     = 5     # σ for pre-PCA Gaussian smoothing (frames, ~166 ms)
PCA_N_COMPONENTS = 3
MIN_CELLS_FOR_AREA = 5   # minimum cells for an area to appear in the dropdown
TRAIL_LEN        = 180   # frames of fading trail in PCA space
GHOST_LEN        = 350   # frames of ghost path in arena
VOC_WINDOW_SEC   = 0.25  # seconds either side of voc to show flash
VOC_FLASH_RADIUS = 22    # pixels, arena flash circle
ANIM_INTERVAL_MS = 40    # timer period (≈25 fps display)

# ── Dark theme ─────────────────────────────────────────────────────────────────
BG_DARK  = "#0f0f1a"
BG_MID   = "#16213e"
BG_PANEL = "#1a1a2e"
C_TICK   = "#8888aa"
C_AXIS   = "#aaaacc"
C_TITLE  = "#ffffff"
C_TRAIL  = "#5599ff"
C_DOT    = "#ffffff"
C_VOC    = "#ff4444"

# ── Per-area colour palette (for Global → By-area overlay mode) ────────────────
AREA_PALETTE = [
    "#4ecdc4",   # teal
    "#ffe66d",   # yellow
    "#ff6b6b",   # coral
    "#a8e6cf",   # mint
    "#c77dff",   # violet
    "#ffd6a5",   # peach
    "#b5ead7",   # seafoam
    "#ff8b94",   # salmon
    "#b9d5ff",   # sky-blue
    "#c7f2a4",   # lime
]


# ══════════════════════════════════════════════════════════════════════════════
class NeuralViewer:

    def __init__(self, root: tk.Tk, default: str | None = None):
        self.root = root
        self.root.title("Neural Trajectory Viewer")
        self.root.configure(bg=BG_PANEL)

        self.sessions = sorted(f[:-4] for f in os.listdir(DATA_DIR)
                               if f.endswith(".mat"))
        if not self.sessions:
            raise FileNotFoundError(f"No .mat files in {DATA_DIR}")

        initial = (default if default and default in self.sessions
                   else ("M7_Female_1" if "M7_Female_1" in self.sessions
                         else self.sessions[0]))

        self.sel_session = tk.StringVar(value=initial)
        self.sel_area    = tk.StringVar(value="Global")
        self.speed       = tk.IntVar(value=3)
        self.frame_idx   = 0
        self.playing     = False
        self._timer_id   = None

        # data populated by load_session / compute_pca
        self.dlc_t = self.dlc_X = self.voc_t = None
        self.pca_cache: dict[str, np.ndarray] = {}   # area → scores (T,3)
        self.area_var:  dict[str, float]      = {}   # area → var_explained sum
        self.pca_traj   = None
        self.n_frames   = 0

        # toggle state: show full-session background vs. live window only
        self.show_full  = tk.BooleanVar(value=True)
        # references to static background artists (set in _init_artists)
        self._bg_traj     = None
        self._voc_scatter = None
        self._arena_bg    = []

        # Global → by-area overlay mode
        self.color_mode    = tk.StringVar(value="all")   # "all" | "by_area"
        self.partial_cache: dict[str, np.ndarray] = {}   # area → partial scores (T, 3)
        self._by_area_mode = False
        self._area_trails: list = []
        self._area_dots:   list = []

        # DLC-derived time-series signals (session-level, set in load_session)
        self.nose_dist: np.ndarray | None = None   # (n_frames,) px
        self.nose_dot:  np.ndarray | None = None   # (n_frames,) −1..+1
        self.nose_tail: np.ndarray | None = None   # (n_frames,) −1..+1 nose→other's tail
        # artist-level neural speed (depends on current pca_traj, set in _init_artists)
        self._neural_speed_arr: np.ndarray | None = None
        # correlation scatter animated dots (None until _init_artists)
        self._c12_dot = self._c13_dot = self._c23_dot = None
        self._ntail_vline = self._ntail_dot = self._ntail_ptr = None

        # video playback
        self.video_cap  = None   # cv2.VideoCapture, or None
        self.video_fps  = 30.0
        self.video_dur  = 0.0    # seconds
        self._vid_img   = None   # imshow artist in ax_skel

        # user-controllable trace length (in frames; ghost is 2×)
        self.trail_len  = tk.IntVar(value=TRAIL_LEN)

        self._build_ui()
        self._bind_keys()
        self.load_session(initial)

    # ══════════════════════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG_MID, pady=7)
        top.pack(fill=tk.X, side=tk.TOP)

        def lbl(parent, text):
            return tk.Label(parent, text=text, bg=BG_MID, fg=C_AXIS,
                            font=("Helvetica", 10))

        lbl(top, "Session:").pack(side=tk.LEFT, padx=(12, 3))
        self.sess_cb = ttk.Combobox(top, textvariable=self.sel_session,
                                     values=self.sessions, width=22,
                                     state="readonly")
        self.sess_cb.pack(side=tk.LEFT, padx=(0, 16))
        self.sess_cb.bind("<<ComboboxSelected>>",
                           lambda _: self.load_session(self.sel_session.get()))

        lbl(top, "Area PCA:").pack(side=tk.LEFT, padx=(0, 3))
        self.area_cb = ttk.Combobox(top, textvariable=self.sel_area,
                                     values=["Global"], width=10,
                                     state="readonly")
        self.area_cb.pack(side=tk.LEFT, padx=(0, 8))
        self.area_cb.bind("<<ComboboxSelected>>",
                           lambda _: self._switch_area(self.sel_area.get()))

        # by-area radiobuttons — always in layout, children shown only for Global
        self._byarea_frame = tk.Frame(top, bg=BG_MID)
        self._byarea_frame.pack(side=tk.LEFT, padx=(0, 12))
        self._rb_all = tk.Radiobutton(
            self._byarea_frame, text="All cells",
            variable=self.color_mode, value="all",
            command=self._on_area_mode,
            bg=BG_MID, fg=C_AXIS, selectcolor=BG_DARK,
            activebackground=BG_MID, activeforeground=C_AXIS,
            font=("Helvetica", 9)
        )
        self._rb_byarea = tk.Radiobutton(
            self._byarea_frame, text="By area",
            variable=self.color_mode, value="by_area",
            command=self._on_area_mode,
            bg=BG_MID, fg=C_AXIS, selectcolor=BG_DARK,
            activebackground=BG_MID, activeforeground=C_AXIS,
            font=("Helvetica", 9)
        )
        # packed/forgotten by _switch_area depending on selected area

        lbl(top, "Speed:").pack(side=tk.LEFT, padx=(0, 3))
        tk.Scale(top, variable=self.speed, from_=1, to=30,
                 orient=tk.HORIZONTAL, length=100, bg=BG_MID, fg=C_AXIS,
                 highlightthickness=0, troughcolor="#0f3460").pack(side=tk.LEFT)
        lbl(top, "×").pack(side=tk.LEFT, padx=(2, 16))

        self.play_btn = tk.Button(top, text="▶  Play", command=self._toggle_play,
                                   bg="#e94560", fg=C_TITLE,
                                   font=("Helvetica", 11, "bold"),
                                   relief=tk.FLAT, padx=14, pady=4)
        self.play_btn.pack(side=tk.LEFT, padx=4)

        # toggle: full session background vs. live window only
        ttk.Checkbutton(top, text="Full session", variable=self.show_full,
                        command=self._toggle_view).pack(side=tk.LEFT, padx=(10, 4))

        self.time_lbl = tk.Label(top, text="t = 0.000 s", bg=BG_MID,
                                  fg="#a8dadc", font=("Courier", 11, "bold"))
        self.time_lbl.pack(side=tk.LEFT, padx=18)

        self.voc_lbl = tk.Label(top, text="", bg=BG_MID,
                                 fg="#ffd166", font=("Helvetica", 10, "bold"))
        self.voc_lbl.pack(side=tk.LEFT, padx=6)

        self.var_lbl = tk.Label(top, text="", bg=BG_MID,
                                 fg=C_TICK, font=("Courier", 9))
        self.var_lbl.pack(side=tk.RIGHT, padx=12)

        # ── matplotlib canvas ─────────────────────────────────────────────────
        self.fig = plt.Figure(figsize=(15.5, 12.5), facecolor=BG_DARK)
        # 3-row layout:
        #   row 0 = 3D PCA (6 cols) | skeleton/video (6 cols)
        #   row 1 = neural speed | nose distance | nose-nose align | nose-tail align
        #   row 2 = 3 pairwise correlation scatter plots
        gs = gridspec.GridSpec(
            4, 12, figure=self.fig,
            height_ratios=[2.6, 1.0, 1.0, 0.30],
            hspace=0.72, wspace=0.55,
            left=0.05, right=0.97, top=0.95, bottom=0.05,
        )
        self.ax_pca   = self.fig.add_subplot(gs[0, :6], projection="3d")
        self.ax_skel  = self.fig.add_subplot(gs[0, 6:])
        self.ax_speed = self.fig.add_subplot(gs[1, 0:3])
        self.ax_ndist = self.fig.add_subplot(gs[1, 3:6])
        self.ax_ndot  = self.fig.add_subplot(gs[1, 6:9])
        self.ax_ntail = self.fig.add_subplot(gs[1, 9:12])
        self.ax_c12   = self.fig.add_subplot(gs[2, 0:4])   # speed ↔ nose dist
        self.ax_c13   = self.fig.add_subplot(gs[2, 4:8])   # speed ↔ nose-nose align
        self.ax_c23   = self.fig.add_subplot(gs[2, 8:12])  # nose dist ↔ nose-nose align
        self.ax_etho  = self.fig.add_subplot(gs[3, :])     # B-SOiD ethogram timeline
        self._style_3d(self.ax_pca)
        for _ax in [self.ax_skel, self.ax_speed, self.ax_ndist, self.ax_ndot,
                    self.ax_ntail, self.ax_c12, self.ax_c13, self.ax_c23, self.ax_etho]:
            self._style_2d(_ax)

        # ── bottom scrubber (packed BEFORE canvas so it always stays visible) ──
        bot = tk.Frame(self.root, bg=BG_MID, pady=5)
        bot.pack(fill=tk.X, side=tk.BOTTOM)

        # canvas fills whatever is left between top bar and bottom bar
        cf = tk.Frame(self.root, bg=BG_DARK)
        cf.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=cf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── bottom scrubber contents ──────────────────────────────────────────
        tk.Label(bot, text="t:", bg=BG_MID, fg=C_AXIS,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(12, 4))
        self.scrubber = tk.Scale(bot, from_=0, to=1000, orient=tk.HORIZONTAL,
                                  bg=BG_MID, fg=C_AXIS, highlightthickness=0,
                                  troughcolor="#0f3460",
                                  command=self._on_scrub)
        self.scrubber.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 16))

        # trace-length slider (right side of bottom bar)
        tk.Label(bot, text="Trace:", bg=BG_MID, fg=C_AXIS,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 3))
        self.trail_lbl = tk.Label(bot, text=f"{TRAIL_LEN} fr", bg=BG_MID,
                                   fg="#a8dadc", font=("Courier", 10), width=7)
        self.trail_lbl.pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(bot, variable=self.trail_len,
                 from_=10, to=3000, resolution=10,
                 orient=tk.HORIZONTAL, length=160,
                 bg=BG_MID, fg=C_AXIS, highlightthickness=0,
                 troughcolor="#0f3460", showvalue=False,
                 command=self._on_trail_change).pack(side=tk.LEFT, padx=(0, 12))

    def _style_3d(self, ax):
        ax.set_facecolor(BG_DARK)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_edgecolor("#2a2a4a")
        ax.tick_params(colors=C_TICK, labelsize=7)
        ax.xaxis.label.set_color(C_AXIS)
        ax.yaxis.label.set_color(C_AXIS)
        ax.zaxis.label.set_color(C_AXIS)
        ax.title.set_color(C_TITLE)

    def _style_2d(self, ax):
        ax.set_facecolor(BG_DARK)
        for sp in ax.spines.values():
            sp.set_color("#2a2a4a")
        ax.tick_params(colors=C_TICK, labelsize=8)
        ax.xaxis.label.set_color(C_AXIS)
        ax.yaxis.label.set_color(C_AXIS)
        ax.title.set_color(C_TITLE)

    def _bind_keys(self):
        self.root.bind("<space>",       lambda _: self._toggle_play())
        self.root.bind("<Right>",       lambda _: self._step(+1))
        self.root.bind("<Left>",        lambda _: self._step(-1))
        self.root.bind("<Shift-Right>", lambda _: self._step(+10))
        self.root.bind("<Shift-Left>",  lambda _: self._step(-10))

    # ══════════════════════════════════════════════════════════════════════════
    # Video helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _find_video(self, session_name: str) -> str | None:
        """Return path to the best-matching .mp4 for session_name, or None."""
        best = None
        for fname in os.listdir(DATA_DIR):
            if not fname.lower().endswith(".mp4"):
                continue
            m = re.search(r'(M\d+).*?(Female|Male)(\d+)', fname, re.IGNORECASE)
            if not m:
                continue
            key = f"{m.group(1)}_{m.group(2).capitalize()}_{m.group(3)}"
            if key == session_name:
                full = os.path.join(DATA_DIR, fname)
                if best is None or os.path.getsize(full) > os.path.getsize(best):
                    best = full
        return best

    @staticmethod
    def _smooth_dlc(dlc_X: np.ndarray) -> np.ndarray:
        """3-point median filter on every x/y column to remove single-frame
        tracking glitches (jumps to a wrong location and back).
        Likelihood columns (every 3rd starting at col 2) are left untouched.
        NaN gaps are interpolated before filtering and restored afterwards."""
        out = dlc_X.copy()
        for c in range(dlc_X.shape[1]):
            if c % 3 == 2:          # likelihood column — skip
                continue
            series = out[:, c]
            valid = np.isfinite(series)
            if valid.sum() < 3:
                continue
            # fill NaN so the median filter doesn't eat edge values
            idx = np.arange(len(series))
            filled = np.where(valid, series,
                              np.interp(idx, idx[valid], series[valid]))
            smoothed = median_filter(filled, size=3)
            smoothed[~valid] = np.nan
            out[:, c] = smoothed
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Data loading & PCA
    # ══════════════════════════════════════════════════════════════════════════
    def load_session(self, name: str):
        # stop playback
        self.playing = False
        self.play_btn.config(text="▶  Play")
        if self._timer_id:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None

        self.time_lbl.config(text="Loading…")
        self.voc_lbl.config(text="")
        self.root.update_idletasks()

        path = os.path.join(DATA_DIR, f"{name}.mat")
        mat  = sio.loadmat(path, simplify_cells=True)
        d    = mat["data"]
        self._sp = mat["sessionParams"]

        self.dlc_t    = d["dlcData"]["t"]
        self.dlc_X    = self._smooth_dlc(d["dlcData"]["X"])   # 3-pt median, kills jumps
        voc_raw       = np.atleast_1d(d["vocalizationTimes"]).astype(float)
        self.voc_t    = voc_raw[np.isfinite(voc_raw)]          # drop NaN padding
        self.n_frames = len(self.dlc_t)
        self._brain_areas = np.array(d["cells"]["brainArea"])

        # ── B-SOiD ethogram motif labels for this session (precomputed) ───────
        self.ethogram = None; self.n_motifs = 0
        try:
            if not hasattr(self, "_etho_npz"):
                _ep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ethogram_labels.npz")
                self._etho_npz = np.load(_ep) if os.path.exists(_ep) else None
            if self._etho_npz is not None and name in self._etho_npz.files:
                self.n_motifs = int(self._etho_npz["__n_motifs__"][0])
                self.etho_names = (list(self._etho_npz["__names__"])
                                   if "__names__" in self._etho_npz.files else None)
                em = np.asarray(self._etho_npz[name], dtype=int)
                # align length to dlc_t
                self.ethogram = em[:self.n_frames] if len(em) >= self.n_frames else \
                    np.concatenate([em, np.full(self.n_frames-len(em), em[-1])])
        except Exception as _e:
            self.ethogram = None

        # ── compute PCA for every available area + global ─────────────────────
        sc = d["spikeCounts100ms"].astype(float)   # (n_cells, T)
        T  = min(sc.shape[1], self.n_frames)
        self.pca_cache.clear()
        self.area_var.clear()
        self.partial_cache.clear()
        _pca_objects: dict = {}   # temporary — for partial projection computation

        def _run_pca(X_cells_T: np.ndarray, label: str):
            """X_cells_T: (n_cells, T)  — returns scores (T, 3)"""
            X  = gaussian_filter1d(X_cells_T[:, :T], sigma=SMOOTH_SIGMA, axis=1).T  # (T, nc)
            nc = X.shape[1]
            if nc < PCA_N_COMPONENTS:
                pad = np.zeros((T, PCA_N_COMPONENTS - nc))
                pca = PCA(n_components=nc).fit(X)
                scores = np.hstack([pca.transform(X), pad])
                var    = pca.explained_variance_ratio_.sum()
            else:
                pca    = PCA(n_components=PCA_N_COMPONENTS)
                scores = pca.fit_transform(X)
                var    = pca.explained_variance_ratio_.sum()
            self.pca_cache[label] = scores          # (T, 3)
            self.area_var[label]  = var
            _pca_objects[label]   = pca             # saved for partial projections

        _run_pca(sc, "Global")
        areas, counts = np.unique(self._brain_areas, return_counts=True)
        for area, cnt in zip(areas, counts):
            if cnt >= MIN_CELLS_FOR_AREA:
                mask = self._brain_areas == area
                _run_pca(sc[mask], area)

        # ── partial projections: each area's contribution through global PCA ──
        # Uses:  X_area @ W_global[:, area_mask].T  →  (T, 3) in global PC space
        # All areas share the same coordinate axes → directly comparable trajectories
        if "Global" in _pca_objects:
            g_pca    = _pca_objects["Global"]   # components_: (3, n_cells)
            X_all_sm = gaussian_filter1d(sc[:, :T], sigma=SMOOTH_SIGMA, axis=1).T  # (T, n_cells)
            for area, cnt in zip(areas, counts):
                if cnt >= MIN_CELLS_FOR_AREA:
                    mask  = self._brain_areas == area
                    W_a   = g_pca.components_[:, mask]   # (3, n_area_cells)
                    X_a   = X_all_sm[:, mask]             # (T, n_area_cells)
                    self.partial_cache[area] = X_a @ W_a.T   # (T, 3)

        # ── populate area dropdown ────────────────────────────────────────────
        area_list  = ["Global"] + [a for a in areas
                                    if a in self.pca_cache and a != "Global"]
        self.area_cb.config(values=area_list)
        cur = self.sel_area.get()
        self.sel_area.set(cur if cur in area_list else "Global")

        # ── position limits ───────────────────────────────────────────────────
        all_x = np.concatenate([self.dlc_X[:, 9],  self.dlc_X[:, 33]])
        all_y = np.concatenate([self.dlc_X[:, 10], self.dlc_X[:, 34]])
        mg = 25
        self.arena_xlim = (np.nanmin(all_x)-mg, np.nanmax(all_x)+mg)
        self.arena_ylim = (np.nanmin(all_y)-mg, np.nanmax(all_y)+mg)

        # ── DLC-derived time-series signals (bottom panels) ──────────────────
        _NOSE   = PART_NAMES.index("Nose")    # 2
        _CENTRE = PART_NAMES.index("Centre")  # 3
        _o0, _o1 = 0, 24   # per-animal column offsets

        _n0x  = self.dlc_X[:, _o0 + _NOSE*3]
        _n0y  = self.dlc_X[:, _o0 + _NOSE*3 + 1]
        _lkn0 = self.dlc_X[:, _o0 + _NOSE*3 + 2]
        _n1x  = self.dlc_X[:, _o1 + _NOSE*3]
        _n1y  = self.dlc_X[:, _o1 + _NOSE*3 + 1]
        _lkn1 = self.dlc_X[:, _o1 + _NOSE*3 + 2]
        _c0x  = self.dlc_X[:, _o0 + _CENTRE*3]
        _c0y  = self.dlc_X[:, _o0 + _CENTRE*3 + 1]
        _lkc0 = self.dlc_X[:, _o0 + _CENTRE*3 + 2]
        _c1x  = self.dlc_X[:, _o1 + _CENTRE*3]
        _c1y  = self.dlc_X[:, _o1 + _CENTRE*3 + 1]
        _lkc1 = self.dlc_X[:, _o1 + _CENTRE*3 + 2]

        # Nose-nose Euclidean distance (px)
        _dist = np.hypot(_n0x - _n1x, _n0y - _n1y).astype(float)
        _dist[(_lkn0 < 0.3) | (_lkn1 < 0.3)] = np.nan
        self.nose_dist = _dist                            # (n_frames,)

        # Nose-direction dot product  (unit vector Centre→Nose per animal)
        _v0x = _n0x - _c0x;  _v0y = _n0y - _c0y
        _v1x = _n1x - _c1x;  _v1y = _n1y - _c1y
        _len0 = np.hypot(_v0x, _v0y)
        _len1 = np.hypot(_v1x, _v1y)
        with np.errstate(invalid="ignore", divide="ignore"):
            _dot = (_v0x*_v1x + _v0y*_v1y) / (_len0 * _len1)
        _bad = ((_lkn0 < 0.3) | (_lkn1 < 0.3) |
                (_lkc0 < 0.3) | (_lkc1 < 0.3) |
                (_len0 < 5)   | (_len1 < 5))
        _dot = _dot.astype(float);  _dot[_bad] = np.nan
        self.nose_dot = _dot                              # (n_frames,)

        # ── Nose-to-tail alignment ─────────────────────────────────────────────
        # +1 = each animal's nose heading toward other's tail  |  −1 = away from tail
        _TAIL_IDX = PART_NAMES.index("Tail_base")        # 6
        _tb0x  = self.dlc_X[:, _o0 + _TAIL_IDX * 3]
        _tb0y  = self.dlc_X[:, _o0 + _TAIL_IDX * 3 + 1]
        _lktb0 = self.dlc_X[:, _o0 + _TAIL_IDX * 3 + 2]
        _tb1x  = self.dlc_X[:, _o1 + _TAIL_IDX * 3]
        _tb1y  = self.dlc_X[:, _o1 + _TAIL_IDX * 3 + 1]
        _lktb1 = self.dlc_X[:, _o1 + _TAIL_IDX * 3 + 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            _u0x = _v0x / _len0;  _u0y = _v0y / _len0   # unit headings
            _u1x = _v1x / _len1;  _u1y = _v1y / _len1
            # unit vec: animal 0 nose → animal 1 tail base
            _d01x = _tb1x - _n0x;  _d01y = _tb1y - _n0y
            _d01l = np.hypot(_d01x, _d01y)
            _ud01x = _d01x / _d01l;  _ud01y = _d01y / _d01l
            # unit vec: animal 1 nose → animal 0 tail base
            _d10x = _tb0x - _n1x;  _d10y = _tb0y - _n1y
            _d10l = np.hypot(_d10x, _d10y)
            _ud10x = _d10x / _d10l;  _ud10y = _d10y / _d10l
        _nt01 = _u0x * _ud01x + _u0y * _ud01y   # animal 0 heading → 1's tail
        _nt10 = _u1x * _ud10x + _u1y * _ud10y   # animal 1 heading → 0's tail
        _nt   = (_nt01 + _nt10) / 2.0
        _bad_nt = _bad | (_lktb0 < 0.3) | (_lktb1 < 0.3)
        _nt = _nt.astype(float);  _nt[_bad_nt] = np.nan
        self.nose_tail = _nt

        # ── Interpolate over low-confidence gaps (tracking drop-outs) ─────────
        for _sig_attr in ("nose_dot", "nose_tail"):
            _arr = getattr(self, _sig_attr).copy()
            _ok  = np.isfinite(_arr)
            if _ok.sum() > 2:
                _ii = np.arange(len(_arr))
                setattr(self, _sig_attr, np.interp(_ii, _ii[_ok], _arr[_ok]))

        self.frame_idx = 0
        self.scrubber.config(to=self.n_frames - 1)
        self.scrubber.set(0)

        # ── open video if available ───────────────────────────────────────────
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.video_dur = 0.0
        vid_path = self._find_video(name)
        if vid_path:
            cap  = cv2.VideoCapture(vid_path)
            fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
            tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_cap = cap
            self.video_fps = fps
            self.video_dur = tot / fps

        self._session_name = name
        self._switch_area(self.sel_area.get(), init=True)

    def _switch_area(self, area: str, init: bool = False):
        self.pca_traj = self.pca_cache.get(area, self.pca_cache["Global"])
        v = self.area_var.get(area, 0.0)
        self.var_lbl.config(
            text=f"{area}  PC1+2+3 = {v*100:.1f}% var"
        )
        # show radiobuttons only when Global is selected
        if area == "Global":
            self._rb_all.pack(side=tk.LEFT, padx=2)
            self._rb_byarea.pack(side=tk.LEFT, padx=2)
        else:
            self._rb_all.pack_forget()
            self._rb_byarea.pack_forget()
            self.color_mode.set("all")   # reset to single-trail mode
        self._init_artists(self._session_name, area)
        self._draw_frame(self.frame_idx if not init else 0)

    def _on_area_mode(self):
        """Called when the All cells / By area radiobutton changes."""
        self._init_artists(self._session_name, self.sel_area.get())
        self._draw_frame(self.frame_idx)

    def _toggle_view(self):
        """Show/hide full-session background elements."""
        show = self.show_full.get()
        if self._bg_traj:
            self._bg_traj.set_visible(show)
        if self._voc_scatter:
            self._voc_scatter.set_visible(show)
        for sc in self._arena_bg:
            sc.set_visible(show)
        self.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # Artist initialisation (once per session × area change)
    # ══════════════════════════════════════════════════════════════════════════
    def _init_artists(self, name: str, area: str):
        self.ax_pca.clear()
        self.ax_skel.clear()
        self._style_3d(self.ax_pca)
        self._style_2d(self.ax_skel)

        traj = self.pca_traj   # (T, 3)
        sp   = self._sp

        # ── 3D PCA panel ──────────────────────────────────────────────────────
        n_cells_area = (int((self._brain_areas == area).sum())
                        if area != "Global" else len(self._brain_areas))
        v = self.area_var.get(area, 0.0)

        self.ax_pca.set_title(
            f"PCA — {area}  ({n_cells_area} cells)  var={v*100:.1f}%",
            fontsize=10, pad=8
        )
        self.ax_pca.set_xlabel("PC 1", fontsize=8, labelpad=4)
        self.ax_pca.set_ylabel("PC 2", fontsize=8, labelpad=4)
        self.ax_pca.set_zlabel("PC 3", fontsize=8, labelpad=4)

        # dim full-session background (reference saved for toggle)
        self._bg_traj, = self.ax_pca.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                                           color="#222244", linewidth=0.4,
                                           alpha=0.6, zorder=1)

        # static all-session vocalization markers (reference saved for toggle)
        if len(self.voc_t) > 0:
            vf = np.searchsorted(self.dlc_t, self.voc_t).clip(0, self.n_frames - 1).astype(int)
            vp = self.pca_traj[vf]
            self._voc_scatter = self.ax_pca.scatter(
                vp[:, 0], vp[:, 1], vp[:, 2],
                c=C_VOC, s=8, alpha=0.40, marker="D",
                depthshade=False, zorder=2,
                label=f"USVs ({len(self.voc_t)})")
            self.ax_pca.legend(fontsize=7, loc="upper right",
                               facecolor=BG_MID, edgecolor="#333366",
                               labelcolor=C_AXIS)
        else:
            self._voc_scatter = None
        # apply current toggle state
        show = self.show_full.get()
        self._bg_traj.set_visible(show)
        if self._voc_scatter:
            self._voc_scatter.set_visible(show)

        # animated line objects (3D line supports set_data + set_3d_properties)
        self._trail3d, = self.ax_pca.plot([], [], [], "-",
                                            color=C_TRAIL, linewidth=1.5,
                                            alpha=0.85, zorder=3)
        self._dot3d,   = self.ax_pca.plot([], [], [], "o",
                                            color=C_DOT, markersize=9,
                                            markeredgecolor=C_TRAIL,
                                            markeredgewidth=2, zorder=5)
        self._vring3d, = self.ax_pca.plot([], [], [], "o",
                                            color=C_VOC, markersize=14,
                                            markerfacecolor="none",
                                            markeredgewidth=2.5,
                                            alpha=0.0, zorder=4)

        # ── By-area overlay (Global + "By area" radiobutton) ──────────────────
        self._area_trails = []
        self._area_dots   = []
        self._by_area_mode = (
            area == "Global" and
            self.color_mode.get() == "by_area" and
            len(self.partial_cache) > 0
        )

        if self._by_area_mode:
            # hide the single-colour trail/dot — per-area ones take over
            self._trail3d.set_visible(False)
            self._dot3d.set_visible(False)

            area_names = list(self.partial_cache.keys())
            all_pts    = np.vstack(list(self.partial_cache.values()))

            for i, aname in enumerate(area_names):
                c     = AREA_PALETTE[i % len(AREA_PALETTE)]
                line, = self.ax_pca.plot([], [], [], "-",
                                          color=c, linewidth=1.5,
                                          alpha=0.85, zorder=3)
                dot,  = self.ax_pca.plot([], [], [], "o",
                                          color="white", markersize=8, zorder=5,
                                          markeredgecolor=c, markeredgewidth=2.0)
                self._area_trails.append(line)
                self._area_dots.append(dot)

            # legend showing area colours + cell counts
            leg_handles = [
                Line2D([0], [0], color=AREA_PALETTE[i % len(AREA_PALETTE)],
                       lw=2,
                       label=f"{aname}  (n={int((self._brain_areas == aname).sum())})")
                for i, aname in enumerate(area_names)
            ]
            self.ax_pca.legend(
                handles=leg_handles, fontsize=7, loc="upper right",
                facecolor=BG_MID, edgecolor="#333366", labelcolor=C_AXIS,
                ncol=(2 if len(area_names) > 5 else 1)
            )

            # update title to reflect partial-projection mode
            n_all = len(self._brain_areas)
            self.ax_pca.set_title(
                f"PCA — Global partial projection by area  "
                f"({n_all} cells total)  var={v*100:.1f}%",
                fontsize=9, pad=8
            )

            # view limits from all partial projections combined
            p = 2
            for axis_idx, setter in enumerate(
                    [self.ax_pca.set_xlim, self.ax_pca.set_ylim, self.ax_pca.set_zlim]):
                data = all_pts[:, axis_idx]
                data = data[np.isfinite(data)]
                lo, hi = np.percentile(data, p), np.percentile(data, 100 - p)
                pad = (hi - lo) * 0.08
                setter(lo - pad, hi + pad)

        if not self._by_area_mode:
            # standard single-area view limits
            p = 2
            for getter, setter in [
                (traj[:, 0], self.ax_pca.set_xlim),
                (traj[:, 1], self.ax_pca.set_ylim),
                (traj[:, 2], self.ax_pca.set_zlim),
            ]:
                lo, hi = np.percentile(getter, p), np.percentile(getter, 100-p)
                pad = (hi - lo) * 0.06
                setter(lo - pad, hi + pad)

        # ── Arena / video panel ───────────────────────────────────────────────
        vid_label = ""
        self._vid_img = None
        self._arena_bg = []

        if self.video_cap is not None:
            # ── VIDEO MODE: show actual video frame as background ─────────────
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame0 = self.video_cap.read()
            if ret:
                rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                self._vid_img = self.ax_skel.imshow(
                    rgb0, origin="upper", aspect="auto", zorder=0)
                vid_h, vid_w = rgb0.shape[:2]
                self.ax_skel.set_xlim(0, vid_w)
                self.ax_skel.set_ylim(vid_h, 0)   # y-axis flipped for images
                self.ax_skel.axis("off")
                vid_label = f"  [video {self.video_dur:.0f}s]"
            else:
                # fallback to arena if first frame unreadable
                self.video_cap.release(); self.video_cap = None

        if self.video_cap is None:
            # ── ARENA MODE: DLC-coordinate plot ──────────────────────────────
            self.ax_skel.set_xlabel("x (px)", fontsize=9)
            self.ax_skel.set_ylabel("y (px)", fontsize=9)
            self.ax_skel.set_xlim(*self.arena_xlim)
            self.ax_skel.set_ylim(*self.arena_ylim)
            self.ax_skel.set_aspect("equal")
            # faint full-session position scatter for both mice
            for col_x, col_y, hexcol in [(9, 10, ANIMAL_COLORS[0]),
                                          (33, 34, ANIMAL_COLORS[1])]:
                xs_all = self.dlc_X[::8, col_x]
                ys_all = self.dlc_X[::8, col_y]
                ok = np.isfinite(xs_all) & np.isfinite(ys_all)
                sc = self.ax_skel.scatter(xs_all[ok], ys_all[ok],
                                           c=hexcol, s=0.8, alpha=0.12,
                                           zorder=0, linewidths=0)
                sc.set_visible(self.show_full.get())
                self._arena_bg.append(sc)

        self.ax_skel.set_title(
            f"{name}  mouse={sp['mouse']}  stim={sp['stim']}"
            f"  vocs={len(self.voc_t)}{vid_label}",
            fontsize=10
        )

        # ── Shared overlay artists (work in both pixel-coord systems) ─────────
        ghost_alpha = 0.55 if self.video_cap is not None else 0.45
        self._ghost = [
            self.ax_skel.plot([], [], "-", color=ANIMAL_COLORS[a],
                               linewidth=1.2, alpha=ghost_alpha, zorder=2)[0]
            for a in range(2)
        ]
        NOSE_IDX = PART_NAMES.index("Nose")   # = 2
        self._bones = []
        self._jdots = []
        self._nose_dots = []
        for a in range(2):
            bs = [self.ax_skel.plot([], [], "-", color=ANIMAL_COLORS[a],
                                     linewidth=2.5, alpha=0.90,
                                     solid_capstyle="round", zorder=4)[0]
                  for _ in SKELETON]
            # regular joints (everything except the nose)
            dots, = self.ax_skel.plot([], [], "o", color=ANIMAL_COLORS[a],
                                       markersize=5, alpha=0.95, zorder=5)
            # nose: white filled, animal-colour ring — visually pops
            nose, = self.ax_skel.plot([], [], "o",
                                       color="white",
                                       markeredgecolor=ANIMAL_COLORS[a],
                                       markeredgewidth=1.8,
                                       markersize=8, alpha=1.0, zorder=6)
            self._bones.append(bs)
            self._jdots.append(dots)
            self._nose_dots.append(nose)
        self._NOSE_IDX = NOSE_IDX

        self._flash = Circle((0, 0), radius=VOC_FLASH_RADIUS,
                              color="#ffd166", alpha=0.0, zorder=6)
        self.ax_skel.add_patch(self._flash)

        if self.video_cap is None:   # legend only in arena mode (no room on video)
            self.ax_skel.legend(
                [Line2D([0],[0], color=ANIMAL_COLORS[0], lw=2),
                 Line2D([0],[0], color=ANIMAL_COLORS[1], lw=2)],
                ["Focal mouse", "Stimulus"],
                loc="upper right", fontsize=8,
                facecolor=BG_MID, edgecolor="#333366", labelcolor=C_AXIS
            )

        # ── Bottom time-series panels ─────────────────────────────────────────
        # Neural speed computed fresh from current pca_traj (area-dependent)
        _g   = self.pca_traj                                  # (T_pca, 3)
        _spd = np.linalg.norm(np.diff(_g, axis=0), axis=1)   # (T_pca-1,)
        _spd = gaussian_filter1d(_spd, sigma=2)               # smooth for display
        _spd_full = np.full(self.n_frames, np.nan)
        _spd_full[1 : 1 + len(_spd)] = _spd
        _spd_full[0] = _spd[0] if len(_spd) > 0 else 0.0
        self._neural_speed_arr = _spd_full                    # (n_frames,)

        _t  = self.dlc_t        # time axis
        _vt = self.voc_t        # voc event times

        for _ax in [self.ax_speed, self.ax_ndist, self.ax_ndot, self.ax_ntail,
                    self.ax_c12, self.ax_c13, self.ax_c23]:
            _ax.clear()
            self._style_2d(_ax)

        def _bottom_panel(bax, signal, color, title, ylabel, ylim=None):
            """Draw static signal + voc markers.
            Returns (vline, dot, ptr) — all animated in _draw_frame."""
            bax.plot(_t, signal, color=color, lw=0.7, alpha=0.75)
            if len(_vt) > 0:
                bax.vlines(_vt, 7/8, 1,
                           transform=bax.get_xaxis_transform(),
                           color=C_VOC, lw=1.2, alpha=0.60, zorder=2)
            bax.set_title(title, fontsize=8, pad=3)
            bax.set_ylabel(ylabel, fontsize=7)
            bax.set_xlabel("time (s)", fontsize=7)
            bax.set_xlim(_t[0], _t[-1])   # initial full view; window follows cursor
            if ylim is not None:
                bax.set_ylim(*ylim)
            else:
                _fin = signal[np.isfinite(signal)]
                if len(_fin) > 0:
                    _lo = np.percentile(_fin, 1)
                    _hi = np.percentile(_fin, 99)
                    _rng = (_hi - _lo) or 1.0
                    bax.set_ylim(_lo - _rng * 0.08, _hi + _rng * 0.15)
                else:
                    bax.set_ylim(0, 1)
            bax.tick_params(labelsize=7)
            bax.set_autoscalex_on(False)   # prevent draw_idle from resetting xlim
            # bright vertical cursor line
            _vline = bax.axvline(x=_t[0], color="#ffd166",
                                 lw=1.5, alpha=0.85, zorder=5)
            # current-value dot on the signal
            _dot, = bax.plot([], [], "o", color=color, markersize=5,
                             zorder=6, markeredgecolor="#ffffff",
                             markeredgewidth=0.6)
            # downward-pointing triangle pointer at the top edge
            _ptr, = bax.plot([_t[0]], [1.0], "v",
                             color="#ffd166", markersize=9,
                             transform=bax.get_xaxis_transform(),
                             clip_on=False, zorder=8,
                             markeredgecolor=BG_DARK, markeredgewidth=0.5)
            return _vline, _dot, _ptr

        (self._spd_vline,   self._spd_dot,   self._spd_ptr)   = _bottom_panel(
            self.ax_speed, self._neural_speed_arr, C_TRAIL,
            f"Neural speed  ‖ΔPC/Δt‖  ({area})", "a.u.")

        (self._ndist_vline, self._ndist_dot, self._ndist_ptr) = _bottom_panel(
            self.ax_ndist, self.nose_dist, ANIMAL_COLORS[0],
            "Nose-to-nose distance", "px")

        (self._ndot_vline,  self._ndot_dot,  self._ndot_ptr)  = _bottom_panel(
            self.ax_ndot, self.nose_dot, "#ffd166",
            "Nose direction alignment (dot product)", "dot product",
            ylim=(-1.05, 1.05))
        self.ax_ndot.axhline(0, color="#445566", lw=0.7, ls="--", zorder=1)

        (self._ntail_vline, self._ntail_dot, self._ntail_ptr) = _bottom_panel(
            self.ax_ntail, self.nose_tail, "#c77dff",
            "Nose-to-tail align", "dot product", ylim=(-1.05, 1.05))
        self.ax_ntail.axhline(0, color="#445566", lw=0.7, ls="--", zorder=1)

        # ── Pairwise correlation scatter plots (row 2) ────────────────────────
        _spd_s = self._neural_speed_arr
        _dst_s = self.nose_dist
        _dot_s = self.nose_dot

        def _corr_panel(cax, sig_x, sig_y, xlabel, ylabel, title):
            """Static full-session scatter + Pearson r. Returns current-frame dot."""
            ok = np.isfinite(sig_x) & np.isfinite(sig_y)
            if ok.sum() > 2:
                _x, _y = sig_x[ok], sig_y[ok]
                cax.scatter(_x[::4], _y[::4], s=2, color="#445588",
                            alpha=0.22, linewidths=0, zorder=1)
                r = np.corrcoef(_x, _y)[0, 1]
                cax.set_title(f"{title}   r = {r:.3f}", fontsize=8, pad=3)
                for _arr, _setlim in [(_x, cax.set_xlim), (_y, cax.set_ylim)]:
                    _lo = np.percentile(_arr, 1);  _hi = np.percentile(_arr, 99)
                    _rng = (_hi - _lo) or 1.0
                    _setlim(_lo - _rng * 0.06, _hi + _rng * 0.06)
            else:
                cax.set_title(f"{title}   (no data)", fontsize=8, pad=3)
            cax.set_xlabel(xlabel, fontsize=7)
            cax.set_ylabel(ylabel, fontsize=7)
            cax.tick_params(labelsize=7)
            _cdot, = cax.plot([], [], "o", color="#ffd166", markersize=8,
                              zorder=6, markeredgecolor="#ffffff",
                              markeredgewidth=0.8)
            return _cdot

        self._c12_dot = _corr_panel(
            self.ax_c12, _spd_s, _dst_s,
            "Neural speed (a.u.)", "Nose distance (px)",
            "Speed  ↔  Nose dist")
        self._c13_dot = _corr_panel(
            self.ax_c13, _spd_s, _dot_s,
            "Neural speed (a.u.)", "Nose-nose align",
            "Speed  ↔  Nose-nose align")
        self._c23_dot = _corr_panel(
            self.ax_c23, _dst_s, _dot_s,
            "Nose distance (px)", "Nose-nose align",
            "Nose dist  ↔  Nose-nose align")

        # ── B-SOiD ethogram timeline strip (full session, with moving pointer) ─
        self.ax_etho.clear(); self._style_2d(self.ax_etho)
        self._etho_ptr = None; self._etho_colors = None
        if self.ethogram is not None and self.n_motifs > 0:
            from matplotlib.colors import ListedColormap
            _ecol = ["#4ecdc4", "#ffe66d", "#ff6b6b", "#c77dff", "#a8e6cf",
                     "#ffd6a5", "#b9d5ff", "#ff8b94", "#c7f2a4", "#f7a072"][:max(self.n_motifs, 1)]
            self._etho_colors = _ecol
            _cmap = ListedColormap(_ecol)
            self.ax_etho.imshow(self.ethogram[None, :], aspect="auto", cmap=_cmap,
                                vmin=0, vmax=self.n_motifs - 1, interpolation="nearest",
                                extent=[0, float(self.dlc_t[-1]), 0, 1], origin="lower")
            self.ax_etho.set_yticks([])
            self.ax_etho.set_xlabel("time (s)", fontsize=7)
            self.ax_etho.set_title(f"Behaviour ethogram (B-SOiD, {self.n_motifs} motifs)",
                                   fontsize=8, pad=2)
            self.ax_etho.set_xlim(0, float(self.dlc_t[-1]))
            # legend chips for the motifs
            _enames = getattr(self, "etho_names", None)
            self.ax_etho.legend(
                handles=[Line2D([0], [0], color=_ecol[i], lw=6,
                                label=(f"{i}:{_enames[i]}" if _enames else f"motif {i}"))
                         for i in range(self.n_motifs)],
                fontsize=6, ncol=self.n_motifs, loc="upper right",
                facecolor=BG_MID, edgecolor="#333366", labelcolor=C_AXIS,
                bbox_to_anchor=(1.0, 1.55))
            self._etho_ptr = self.ax_etho.axvline(0, color="#ffffff", lw=1.6, zorder=6)
        else:
            self.ax_etho.text(0.5, 0.5, "ethogram_labels.npz not found — "
                              "run build_ethogram_labels.py", ha="center", va="center",
                              color=C_TICK, fontsize=8, transform=self.ax_etho.transAxes)
            self.ax_etho.set_xticks([]); self.ax_etho.set_yticks([])

        self.canvas.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # Per-frame update
    # ══════════════════════════════════════════════════════════════════════════
    def _draw_frame(self, fi: int):
        fi = int(np.clip(fi, 0, self.n_frames - 1))
        self.frame_idx = fi
        t_now = self.dlc_t[fi]

        # ── 3D PCA trail ──────────────────────────────────────────────────────
        trail = self.trail_len.get()
        t0    = max(0, fi - trail)

        near_voc = (len(self.voc_t) > 0 and
                    np.any(np.abs(self.voc_t - t_now) < VOC_WINDOW_SEC))

        if self._by_area_mode:
            # update one coloured trail + dot per brain area
            for aname, line, dot in zip(
                    self.partial_cache.keys(),
                    self._area_trails,
                    self._area_dots):
                traj_a = self.partial_cache[aname]
                tr = traj_a[t0:fi+1]
                line.set_data(tr[:, 0], tr[:, 1])
                line.set_3d_properties(tr[:, 2])
                dot.set_data([traj_a[fi, 0]], [traj_a[fi, 1]])
                dot.set_3d_properties([traj_a[fi, 2]])
            self._vring3d.set_alpha(0.0)   # vring not used in by-area mode
        else:
            # single-colour global / per-area trail
            tx = self.pca_traj[t0:fi+1, 0]
            ty = self.pca_traj[t0:fi+1, 1]
            tz = self.pca_traj[t0:fi+1, 2]
            self._trail3d.set_data(tx, ty)
            self._trail3d.set_3d_properties(tz)
            self._dot3d.set_data([self.pca_traj[fi, 0]], [self.pca_traj[fi, 1]])
            self._dot3d.set_3d_properties([self.pca_traj[fi, 2]])
            if near_voc:
                self._vring3d.set_data([self.pca_traj[fi, 0]], [self.pca_traj[fi, 1]])
                self._vring3d.set_3d_properties([self.pca_traj[fi, 2]])
                self._vring3d.set_alpha(0.75)
            else:
                self._vring3d.set_alpha(0.0)

        # voc label (both modes)
        if near_voc:
            self.voc_lbl.config(text="🔊 VOCAL")
        else:
            self.voc_lbl.config(text="")

        # ── Video frame update ────────────────────────────────────────────────
        if self.video_cap is not None and self._vid_img is not None:
            vf_idx = int(t_now * self.video_fps)
            vf_idx = max(0, min(vf_idx, int(self.video_cap.get(
                cv2.CAP_PROP_FRAME_COUNT)) - 1))
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, vf_idx)
            ret, frame = self.video_cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if near_voc:
                    # red border flash for USV events on video
                    h, w = rgb.shape[:2]
                    rgb = rgb.copy()
                    rgb[:6, :] = rgb[-6:, :] = [220, 50, 50]
                    rgb[:, :6] = rgb[:, -6:] = [220, 50, 50]
                self._vid_img.set_data(rgb)

        # current behaviour-motif colour (focal mouse coloured by what it's doing)
        _mcol = None
        if (getattr(self, "ethogram", None) is not None and
                getattr(self, "_etho_colors", None) and 0 <= fi < len(self.ethogram)):
            _mcol = self._etho_colors[int(self.ethogram[fi])]

        # ── Skeleton overlay (pixel coords — same in video & arena modes) ─────
        for a in range(2):
            off = a * 24
            xs  = self.dlc_X[fi, off     :off+24:3]    # 8 × x
            ys  = self.dlc_X[fi, off+1   :off+25:3]    # 8 × y
            lk  = self.dlc_X[fi, off+2   :off+26:3]    # 8 × likelihood
            # focal mouse (a==0) skeleton recoloured by current behaviour motif
            _bc = _mcol if (a == 0 and _mcol) else ANIMAL_COLORS[a]

            for bone, (pf, pt) in zip(self._bones[a], SKELETON):
                bone.set_color(_bc)
                if lk[pf] > 0.3 and lk[pt] > 0.3:
                    bone.set_data([xs[pf], xs[pt]], [ys[pf], ys[pt]])
                    bone.set_alpha(0.90 if min(lk[pf], lk[pt]) > 0.5 else 0.50)
                else:
                    bone.set_data([], [])
                    bone.set_alpha(0.90)

            ni = self._NOSE_IDX
            good = lk > 0.3
            # regular joints (all except nose)
            non_nose = good.copy(); non_nose[ni] = False
            self._jdots[a].set_color(_bc); self._jdots[a].set_data(xs[non_nose], ys[non_nose])
            # nose — white dot, edge tinted by motif for the focal mouse
            if a == 0 and _mcol:
                self._nose_dots[a].set_markeredgecolor(_mcol)
            if lk[ni] > 0.3:
                self._nose_dots[a].set_data([xs[ni]], [ys[ni]])
            else:
                self._nose_dots[a].set_data([], [])

        g0 = max(0, fi - trail * 2)    # ghost is 2× the PCA trail
        self._ghost[0].set_data(self.dlc_X[g0:fi, 9],  self.dlc_X[g0:fi, 10])
        self._ghost[1].set_data(self.dlc_X[g0:fi, 33], self.dlc_X[g0:fi, 34])

        cx, cy = self.dlc_X[fi, 9], self.dlc_X[fi, 10]
        if near_voc and np.isfinite(cx) and np.isfinite(cy):
            self._flash.set_center((cx, cy))
            self._flash.set_alpha(0.45)
        else:
            self._flash.set_alpha(0.0)

        # ── Ethogram timeline playhead (full-session strip) ──────────────────
        if getattr(self, "_etho_ptr", None) is not None:
            self._etho_ptr.set_xdata([t_now, t_now])

        # ── Bottom panel cursors + moving window ─────────────────────────────
        if self._neural_speed_arr is not None:
            # window = trail_len frames wide on the left, ~15% padding on right
            _dt  = float(self.dlc_t[-1]) / max(self.n_frames - 1, 1)
            _win = self.trail_len.get() * _dt          # seconds of history to show
            _t_l = max(float(self.dlc_t[0]),  t_now - _win)
            _t_r = min(float(self.dlc_t[-1]), t_now + _win * 0.18)
            _t_r = max(_t_r, _t_l + _win * 0.05)      # guard against zero-width
            for _ax in [self.ax_speed, self.ax_ndist, self.ax_ndot, self.ax_ntail]:
                _ax.set_xlim(_t_l, _t_r)

            # vertical cursor + triangle pointer
            for _vl in [self._spd_vline, self._ndist_vline,
                        self._ndot_vline, self._ntail_vline]:
                _vl.set_xdata([t_now, t_now])
            for _ptr in [self._spd_ptr, self._ndist_ptr,
                         self._ndot_ptr, self._ntail_ptr]:
                _ptr.set_data([t_now], [1.0])

            # current-value dots on time-series panels
            _v_spd  = self._neural_speed_arr[fi]
            _v_dist = self.nose_dist[fi]
            _v_dot  = self.nose_dot[fi]
            _v_tail = self.nose_tail[fi] if self.nose_tail is not None else np.nan

            self._spd_dot.set_data(
                [t_now] if np.isfinite(_v_spd)  else [],
                [_v_spd]  if np.isfinite(_v_spd)  else [])
            self._ndist_dot.set_data(
                [t_now] if np.isfinite(_v_dist) else [],
                [_v_dist] if np.isfinite(_v_dist) else [])
            self._ndot_dot.set_data(
                [t_now] if np.isfinite(_v_dot)  else [],
                [_v_dot]  if np.isfinite(_v_dot)  else [])
            self._ntail_dot.set_data(
                [t_now] if np.isfinite(_v_tail) else [],
                [_v_tail] if np.isfinite(_v_tail) else [])

            # current-frame dot on correlation scatter plots
            if self._c12_dot is not None:
                _ok12 = np.isfinite(_v_spd)  and np.isfinite(_v_dist)
                _ok13 = np.isfinite(_v_spd)  and np.isfinite(_v_dot)
                _ok23 = np.isfinite(_v_dist) and np.isfinite(_v_dot)
                self._c12_dot.set_data(
                    [_v_spd]  if _ok12 else [], [_v_dist] if _ok12 else [])
                self._c13_dot.set_data(
                    [_v_spd]  if _ok13 else [], [_v_dot]  if _ok13 else [])
                self._c23_dot.set_data(
                    [_v_dist] if _ok23 else [], [_v_dot]  if _ok23 else [])

        # ── Status bar ────────────────────────────────────────────────────────
        self.time_lbl.config(text=f"t = {t_now:7.3f} s")
        self.scrubber.set(fi)
        self.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # Playback
    # ══════════════════════════════════════════════════════════════════════════
    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="⏸  Pause")
            self._tick()
        else:
            self.play_btn.config(text="▶  Play")
            if self._timer_id:
                self.root.after_cancel(self._timer_id)
                self._timer_id = None

    def _tick(self):
        if not self.playing:
            return
        nxt = self.frame_idx + self.speed.get()
        if nxt >= self.n_frames:
            nxt = 0
        self._draw_frame(nxt)
        self._timer_id = self.root.after(ANIM_INTERVAL_MS, self._tick)

    def _step(self, delta: int):
        self._draw_frame(self.frame_idx + delta)

    def _on_scrub(self, val):
        self._draw_frame(int(float(val)))

    def _on_trail_change(self, val):
        """Update the trace-length label and redraw the current frame."""
        frames = int(float(val))
        if self.dlc_t is not None and self.n_frames > 1:
            dt   = float(self.dlc_t[-1]) / max(self.n_frames - 1, 1)  # s/frame
            secs = frames * dt
            self.trail_lbl.config(text=f"{secs:.1f} s")
        else:
            self.trail_lbl.config(text=f"{frames} fr")
        self._draw_frame(self.frame_idx)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    default = sys.argv[1] if len(sys.argv) > 1 else "M7_Female_1"
    root = tk.Tk()
    root.geometry("1540x1040")   # sensible fallback before maximize
    root.resizable(True, True)
    root.update_idletasks()
    try:
        root.state("zoomed")    # Windows: maximize to fill screen minus taskbar
    except Exception:
        pass
    NeuralViewer(root, default=default)
    root.mainloop()
