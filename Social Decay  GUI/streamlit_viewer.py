"""
11_streamlit_viewer.py  —  Neural Trajectory Viewer (Streamlit)
═══════════════════════════════════════════════════════════════
Panels
  ┌─ 3D PCA ──────────────┬─ Video / Skeleton ─────────────┐
  │  neural state trajectory │  video frame + DLC overlay   │
  │  + vocalization markers  │  (or Plotly arena if no vid) │
  ├─ PC1 vs PC2 ──────────┼─ PC2 vs PC3 ───┬ PC3 vs PC1 ──┤
  │                          │                │               │
  └──────────────────────────┴────────────────┴───────────────┘
  Vocalization timeline strip (full session, all events marked)

Run
  streamlit run Daniel/data_exploration/11_streamlit_viewer.py

Daniel Katz — Neural Data Science Project 5
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, re, time

# ── third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# Paths & constants
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = "G:/Technion-970405-Course-Final-Project/Data"

PART_NAMES   = ["Ear_L","Ear_R","Nose","Centre","Lat_L","Lat_R","Tail_base","Tail_end"]
SKELETON     = [(2,3),(0,3),(1,3),(4,3),(5,3),(3,6),(6,7)]   # (from_part, to_part)

FOCAL_HEX    = "#4ea8de"    # focal mouse  (blue)
STIM_HEX     = "#ff6b6b"    # stimulus     (red)
FOCAL_BGR    = (222, 168, 78)   # OpenCV BGR (gold) — visible on most backgrounds
STIM_BGR     = (107, 107, 255)  # OpenCV BGR (purple-ish)

SMOOTH_SIGMA = 5        # Gaussian smoothing σ before PCA  (~166 ms)
N_PCS        = 3
MIN_CELLS    = 5        # minimum cells for a brain area to appear in dropdown
VOC_WIN_SEC  = 0.25     # ±s around a vocalization to show flash/ring
VOC_COLOR    = "#ff4444"
VOC_FLASH_R  = 22       # arena flash ring radius (px)


# ══════════════════════════════════════════════════════════════════════════════
# Page config  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Neural Trajectory Viewer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
div[data-testid="stMetric"] label { font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Video inventory  (run once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def scan_videos() -> dict[str, str]:
    """Return {session_name: path_to_best_video}."""
    result: dict[str, str] = {}
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".mp4"):
            continue
        m = re.search(r'(M\d+).*?(Female|Male)(\d+)', fname, re.IGNORECASE)
        if not m:
            continue
        key  = f"{m.group(1)}_{m.group(2).capitalize()}_{m.group(3)}"
        full = os.path.join(DATA_DIR, fname)
        # prefer higher-resolution / larger file
        if key not in result or os.path.getsize(full) > os.path.getsize(result[key]):
            result[key] = full
    return result

VIDEO_MAP    = scan_videos()
ALL_SESSIONS = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".mat"))

def session_label(name: str) -> str:
    return f"📹 {name}" if name in VIDEO_MAP else name


# ══════════════════════════════════════════════════════════════════════════════
# Session loading & PCA  (cached per session name)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading session & computing PCA…")
def load_session(name: str) -> dict:
    mat = sio.loadmat(f"{DATA_DIR}/{name}.mat", simplify_cells=True)
    d   = mat["data"]
    sp  = mat["sessionParams"]

    dlc_t       = d["dlcData"]["t"]
    dlc_X       = d["dlcData"]["X"]
    voc_t_raw   = np.atleast_1d(d["vocalizationTimes"]).astype(float)
    voc_t       = voc_t_raw[np.isfinite(voc_t_raw)]     # drop NaN
    brain_areas = np.array(d["cells"]["brainArea"])
    T           = len(dlc_t)

    sc = d["spikeCounts100ms"].astype(float)[:, :T]      # (n_cells, T)

    pca_cache: dict[str, np.ndarray]  = {}
    var_cache:  dict[str, np.ndarray] = {}

    def _fit(X_nc_T: np.ndarray, label: str):
        X  = gaussian_filter1d(X_nc_T, sigma=SMOOTH_SIGMA, axis=1).T  # (T, nc)
        k  = min(N_PCS, X.shape[1])
        p  = PCA(n_components=k).fit(X)
        s  = p.transform(X)                               # (T, k)
        if k < N_PCS:                                     # pad with zeros if nc < 3
            s = np.hstack([s, np.zeros((T, N_PCS - k))])
        pca_cache[label] = s
        var_cache[label] = np.pad(p.explained_variance_ratio_,
                                   (0, N_PCS - k))        # always length 3

    _fit(sc, "Global")
    areas, counts = np.unique(brain_areas, return_counts=True)
    for area, cnt in zip(areas, counts):
        if cnt >= MIN_CELLS:
            _fit(sc[brain_areas == area], area)

    return dict(
        name=name, sp=sp,
        dlc_t=dlc_t, dlc_X=dlc_X,
        voc_t=voc_t, brain_areas=brain_areas,
        n_frames=T,
        pca_cache=pca_cache, var_cache=var_cache,
        areas=list(pca_cache.keys()),
    )


@st.cache_resource
def open_video(path: str):
    """Keep VideoCapture open across reruns."""
    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total / fps
    return cap, fps, total, dur


# ══════════════════════════════════════════════════════════════════════════════
# Video helpers
# ══════════════════════════════════════════════════════════════════════════════
def read_video_frame(cap, fps: float, total: int, t_now: float) -> np.ndarray | None:
    """Read the video frame that best matches t_now (proportional mapping)."""
    fi = int(t_now * fps)
    fi = max(0, min(fi, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None


def overlay_skeleton(img_rgb: np.ndarray, dlc_X: np.ndarray, fi: int) -> np.ndarray:
    """Draw both mice skeletons onto a copy of the frame. Returns RGB uint8."""
    img  = img_rgb.copy()
    h, w = img.shape[:2]

    for a_idx, color_bgr in enumerate([FOCAL_BGR, STIM_BGR]):
        off = a_idx * 24
        xs  = dlc_X[fi, off  :off+24:3].astype(float)   # 8 × x
        ys  = dlc_X[fi, off+1:off+25:3].astype(float)   # 8 × y
        lk  = dlc_X[fi, off+2:off+26:3]                 # 8 × likelihood

        for pf, pt in SKELETON:
            if lk[pf] > 0.5 and lk[pt] > 0.5:
                p1 = (int(xs[pf]), int(ys[pf]))
                p2 = (int(xs[pt]), int(ys[pt]))
                if (0 <= p1[0] < w and 0 <= p1[1] < h and
                        0 <= p2[0] < w and 0 <= p2[1] < h):
                    cv2.line(img, p1, p2, color_bgr, 2, cv2.LINE_AA)

        for j in range(len(PART_NAMES)):
            if lk[j] > 0.5:
                pt = (int(xs[j]), int(ys[j]))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(img, pt, 4, color_bgr, -1, cv2.LINE_AA)

    return img


def add_voc_banner(img_rgb: np.ndarray) -> np.ndarray:
    """Draw a red border + 'USV' label when a vocalization is active."""
    img = img_rgb.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w-1, h-1), (255, 60, 60), 5)
    cv2.putText(img, "USV", (16, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 80, 80), 2, cv2.LINE_AA)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Plotly figure builders
# ══════════════════════════════════════════════════════════════════════════════
_DARK = dict(
    paper_bgcolor="#0f0f1a",
    plot_bgcolor ="#0f0f1a",
    font         =dict(color="#aaaacc"),
)

def _voc_frames(dlc_t: np.ndarray, voc_t: np.ndarray) -> np.ndarray:
    return np.searchsorted(dlc_t, voc_t).clip(0, len(dlc_t)-1)


def fig_3d(traj, fi, dlc_t, voc_t, trail_len, var):
    """3D PCA: dim background + fading trail + current dot + voc diamonds."""
    fi   = int(np.clip(fi, 0, len(traj)-1))
    t0   = max(0, fi - trail_len)
    tr   = traj[t0:fi+1]
    n    = len(tr)

    f = go.Figure()

    # Dim full-session skeleton (every 4th frame for speed)
    f.add_trace(go.Scatter3d(
        x=traj[::4, 0], y=traj[::4, 1], z=traj[::4, 2],
        mode="lines",
        line=dict(color="#161630", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # Fading trail (color = age within trail)
    f.add_trace(go.Scatter3d(
        x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
        mode="lines+markers",
        line=dict(color="#5599ff", width=2.5),
        marker=dict(size=2, color=list(range(n)), colorscale="Blues",
                    cmin=0, cmax=n, opacity=0.75, showscale=False),
        showlegend=False, hoverinfo="skip",
    ))

    # Current position
    f.add_trace(go.Scatter3d(
        x=[traj[fi, 0]], y=[traj[fi, 1]], z=[traj[fi, 2]],
        mode="markers",
        marker=dict(size=10, color="white",
                    line=dict(color="#5599ff", width=2.5)),
        name="Current position",
    ))

    # Vocalization events
    if len(voc_t):
        vf = _voc_frames(dlc_t, voc_t)
        f.add_trace(go.Scatter3d(
            x=traj[vf, 0], y=traj[vf, 1], z=traj[vf, 2],
            mode="markers",
            marker=dict(size=5, color=VOC_COLOR, symbol="diamond",
                        opacity=0.75, line=dict(color="#ff8888", width=0.5)),
            name=f"Vocalizations ({len(voc_t)})",
        ))

    v = var
    _dk3d = {**_DARK, "margin": dict(l=0, r=0, t=36, b=0)}
    f.update_layout(
        **_dk3d,
        height=430,
        scene=dict(
            xaxis=dict(title=f"PC1  {v[0]*100:.1f}%",
                       backgroundcolor="#0a0a14", gridcolor="#1a2040"),
            yaxis=dict(title=f"PC2  {v[1]*100:.1f}%",
                       backgroundcolor="#0a0a14", gridcolor="#1a2040"),
            zaxis=dict(title=f"PC3  {v[2]*100:.1f}%",
                       backgroundcolor="#0a0a14", gridcolor="#1a2040"),
            bgcolor="#0f0f1a",
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0.4)"),
    )
    return f


def fig_2d(traj, fi, dlc_t, voc_t, trail_len, var, xi, yi, pc_i, pc_j):
    """2D projection: PCi vs PCj."""
    fi  = int(np.clip(fi, 0, len(traj)-1))
    t0  = max(0, fi - trail_len)
    tr  = traj[t0:fi+1]
    n   = len(tr)

    f = go.Figure()

    # Background
    f.add_trace(go.Scatter(
        x=traj[::4, xi], y=traj[::4, yi],
        mode="lines",
        line=dict(color="#161630", width=0.8),
        showlegend=False, hoverinfo="skip",
    ))

    # Trail
    f.add_trace(go.Scatter(
        x=tr[:, xi], y=tr[:, yi],
        mode="lines+markers",
        line=dict(color="#5599ff", width=1.8),
        marker=dict(size=2, color=list(range(n)), colorscale="Blues",
                    cmin=0, cmax=n, opacity=0.75, showscale=False),
        showlegend=False, hoverinfo="skip",
    ))

    # Current
    f.add_trace(go.Scatter(
        x=[traj[fi, xi]], y=[traj[fi, yi]],
        mode="markers",
        marker=dict(size=10, color="white",
                    line=dict(color="#5599ff", width=2)),
        showlegend=False,
    ))

    # Vocalizations
    if len(voc_t):
        vf = _voc_frames(dlc_t, voc_t)
        f.add_trace(go.Scatter(
            x=traj[vf, xi], y=traj[vf, yi],
            mode="markers",
            marker=dict(size=5, color=VOC_COLOR, symbol="diamond", opacity=0.65),
            showlegend=False, hoverinfo="skip",
        ))

    vi = var[xi]*100; vj = var[yi]*100
    _dk2d = {**_DARK, "margin": dict(l=45, r=10, t=38, b=42)}
    f.update_layout(
        **_dk2d,
        height=310,
        xaxis=dict(title=f"PC{pc_i}  {vi:.1f}%",
                   gridcolor="#1a2040", zerolinecolor="#2a2a4a"),
        yaxis=dict(title=f"PC{pc_j}  {vj:.1f}%",
                   gridcolor="#1a2040", zerolinecolor="#2a2a4a"),
        showlegend=False,
    )
    return f


def fig_arena(dlc_X, fi, voc_t, dlc_t, trail_len):
    """Plotly arena: both mice as stick-figures + ghost trails + voc ring."""
    fi   = int(np.clip(fi, 0, dlc_X.shape[0]-1))
    g0   = max(0, fi - trail_len)
    t_now = dlc_t[fi]
    near_voc = len(voc_t) > 0 and np.any(np.abs(voc_t - t_now) < VOC_WIN_SEC)

    f = go.Figure()

    # Ghost trails
    for a, (cx, cy, hexcol) in enumerate([(9, 10, FOCAL_HEX), (33, 34, STIM_HEX)]):
        f.add_trace(go.Scatter(
            x=dlc_X[g0:fi, cx], y=dlc_X[g0:fi, cy],
            mode="lines",
            line=dict(color=hexcol, width=0.8),
            opacity=0.22, showlegend=False, hoverinfo="skip",
        ))

    # Skeletons
    for a_idx, (hexcol, lbl) in enumerate(
            zip([FOCAL_HEX, STIM_HEX], ["Focal mouse", "Stimulus"])):
        off = a_idx * 24
        xs  = dlc_X[fi, off  :off+24:3]
        ys  = dlc_X[fi, off+1:off+25:3]
        lk  = dlc_X[fi, off+2:off+26:3]
        first_bone = True
        for pf, pt in SKELETON:
            if lk[pf] > 0.5 and lk[pt] > 0.5:
                f.add_trace(go.Scatter(
                    x=[xs[pf], xs[pt]], y=[ys[pf], ys[pt]],
                    mode="lines",
                    line=dict(color=hexcol, width=2.5),
                    showlegend=first_bone, name=lbl,
                    legendgroup=f"a{a_idx}",
                ))
                first_bone = False
        good = lk > 0.5
        f.add_trace(go.Scatter(
            x=xs[good], y=ys[good],
            mode="markers",
            marker=dict(size=6, color=hexcol),
            showlegend=False, legendgroup=f"a{a_idx}",
        ))

    # Vocalization ring on focal mouse
    if near_voc:
        cx_pos, cy_pos = dlc_X[fi, 9], dlc_X[fi, 10]
        if np.isfinite(cx_pos) and np.isfinite(cy_pos):
            theta = np.linspace(0, 2*np.pi, 64)
            f.add_trace(go.Scatter(
                x=cx_pos + VOC_FLASH_R * np.cos(theta),
                y=cy_pos + VOC_FLASH_R * np.sin(theta),
                mode="lines",
                line=dict(color="#ffd166", width=2.5),
                showlegend=False,
            ))

    all_x = np.concatenate([dlc_X[:, 9],  dlc_X[:, 33]])
    all_y = np.concatenate([dlc_X[:, 10], dlc_X[:, 34]])
    mg = 25
    _dkar = {**_DARK, "margin": dict(l=0, r=0, t=36, b=0)}
    f.update_layout(
        **_dkar,
        height=430,
        xaxis=dict(range=[np.nanmin(all_x)-mg, np.nanmax(all_x)+mg],
                   gridcolor="#1a2040", zerolinecolor="#2a2a4a",
                   scaleanchor="y"),
        yaxis=dict(range=[np.nanmin(all_y)-mg, np.nanmax(all_y)+mg],
                   gridcolor="#1a2040", zerolinecolor="#2a2a4a"),
        legend=dict(x=0.01, y=0.99, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0.4)"),
    )
    return f, near_voc


def fig_timeline(dlc_t, voc_t, frame):
    """Compact timeline bar: red ticks = vocalizations, white line = now."""
    t_now = dlc_t[int(np.clip(frame, 0, len(dlc_t)-1))]
    t_max = dlc_t[-1]

    f = go.Figure()

    # Session background bands (light / dark alternating per 60 s)
    for i in range(0, int(t_max), 120):
        f.add_shape(type="rect", x0=i, x1=min(i+60, t_max), y0=0, y1=1,
                    fillcolor="rgba(255,255,255,0.02)", line_width=0)

    # Vocalization ticks
    for vt in voc_t:
        f.add_shape(type="line", x0=vt, x1=vt, y0=0, y1=1,
                    line=dict(color=VOC_COLOR, width=1), opacity=0.75)

    # Current time cursor
    f.add_shape(type="line", x0=t_now, x1=t_now, y0=0, y1=1,
                line=dict(color="white", width=2))

    _dark_tl = {**_DARK, "plot_bgcolor": "#12124a"}   # override bg for timeline
    f.update_layout(
        **_dark_tl,
        height=52,
        margin=dict(l=40, r=10, t=6, b=22),
        xaxis=dict(range=[0, t_max], tickfont=dict(size=8),
                   gridcolor="#1a2040"),
        yaxis=dict(showticklabels=False, range=[0, 1]),
        showlegend=False,
    )
    return f


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [("frame", 0), ("playing", False), ("speed", 3),
             ("trail_len", 200), ("_session", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠  Neural Trajectory")
    st.caption("Project 5 — Decode position from neural activity")

    # Session selector
    display  = [session_label(s) for s in ALL_SESSIONS]
    def_idx  = (ALL_SESSIONS.index("M7_Female_1")
                if "M7_Female_1" in ALL_SESSIONS else 0)
    chosen_d = st.selectbox("Session", display, index=def_idx)
    session  = ALL_SESSIONS[display.index(chosen_d)]

    if st.session_state._session != session:
        st.session_state.frame   = 0
        st.session_state.playing = False
        st.session_state._session = session

    # Load (cached)
    data     = load_session(session)
    n_frames = data["n_frames"]
    dlc_t    = data["dlc_t"]
    dlc_X    = data["dlc_X"]
    voc_t    = data["voc_t"]
    sp       = data["sp"]

    # Brain area selector
    area     = st.selectbox("PCA area", data["areas"])
    traj     = data["pca_cache"][area]
    var      = data["var_cache"][area]
    v_sum    = var.sum() * 100
    st.caption(f"PC1+PC2+PC3 explains **{v_sum:.1f}%** variance")

    st.divider()

    # Playback
    c1, c2 = st.columns(2)
    with c1:
        play_lbl = "⏸ Pause" if st.session_state.playing else "▶ Play"
        if st.button(play_lbl, use_container_width=True):
            st.session_state.playing = not st.session_state.playing
    with c2:
        if st.button("⏮ Reset", use_container_width=True):
            st.session_state.frame   = 0
            st.session_state.playing = False

    speed = st.slider("Speed  (frames/step)", 1, 30, st.session_state.speed)
    st.session_state.speed = speed

    trail_len = st.slider("Trail length  (frames)", 50, 500,
                           st.session_state.trail_len, step=25)
    st.session_state.trail_len = trail_len

    st.divider()

    # Frame scrubber (synced with session_state.frame)
    frame = st.slider("Frame", 0, n_frames - 1, st.session_state.frame)
    st.session_state.frame = frame

    t_now    = dlc_t[frame]
    near_voc = len(voc_t) > 0 and np.any(np.abs(voc_t - t_now) < VOC_WIN_SEC)

    m1, m2 = st.columns(2)
    m1.metric("Time", f"{t_now:.2f} s")
    m2.metric("Vocs", str(len(voc_t)))
    if near_voc:
        st.error("🔊  VOCALIZATION")

    st.divider()

    # Session info
    st.markdown(f"""
**Mouse:** {sp['mouse']}  &nbsp;|&nbsp;  **Stim:** {sp['stim']}
**Cells:** {len(data['brain_areas'])}  &nbsp;|&nbsp;  **Duration:** {dlc_t[-1]:.0f} s
""")
    if session in VIDEO_MAP:
        st.success("📹  Video clip available")


# ══════════════════════════════════════════════════════════════════════════════
# Main layout
# ══════════════════════════════════════════════════════════════════════════════
fi        = st.session_state.frame
TRAIL     = st.session_state.trail_len
CFG_NOBAR = {"displayModeBar": False}

# ── Row 1: 3D PCA  |  Video + Skeleton ───────────────────────────────────────
col_l, col_r = st.columns(2, gap="small")

with col_l:
    st.markdown(f"**3D Neural Trajectory** — {area}")
    st.plotly_chart(fig_3d(traj, fi, dlc_t, voc_t, TRAIL, var),
                    use_container_width=True, config=CFG_NOBAR)

with col_r:
    has_video = session in VIDEO_MAP
    if has_video:
        cap, fps, total_vf, vid_dur = open_video(VIDEO_MAP[session])
        if t_now <= vid_dur:
            frame_rgb = read_video_frame(cap, fps, total_vf, t_now)
            if frame_rgb is not None:
                frame_rgb = overlay_skeleton(frame_rgb, dlc_X, fi)
                if near_voc:
                    frame_rgb = add_voc_banner(frame_rgb)
                voc_badge = "  🔊" if near_voc else ""
                st.markdown(f"**Video + Skeleton** (t ≤ {vid_dur:.0f} s){voc_badge}")
                st.image(frame_rgb, use_container_width=True)
            else:
                st.warning("Video frame unreadable.")
                has_video = False      # fall through to arena
        else:
            has_video = False          # video clip ended, show arena

    if not has_video:
        voc_note = "  🔊" if near_voc else ""
        st.markdown(f"**Arena + Skeleton**{voc_note}")
        f_ar, _ = fig_arena(dlc_X, fi, voc_t, dlc_t, TRAIL)
        st.plotly_chart(f_ar, use_container_width=True, config=CFG_NOBAR)

# ── Vocalization timeline ─────────────────────────────────────────────────────
st.markdown("<small>Vocalization timeline  |  🔴 = events  |  ─── = now</small>",
            unsafe_allow_html=True)
st.plotly_chart(fig_timeline(dlc_t, voc_t, fi),
                use_container_width=True, config=CFG_NOBAR)

# ── Row 2: 2D projections ─────────────────────────────────────────────────────
st.markdown("**2D PCA Projections**")
c1, c2, c3 = st.columns(3, gap="small")
for col, (xi, yi, pi, pj) in zip(
        [c1, c2, c3],
        [(0, 1, 1, 2), (1, 2, 2, 3), (2, 0, 3, 1)]):
    with col:
        st.plotly_chart(fig_2d(traj, fi, dlc_t, voc_t, TRAIL, var, xi, yi, pi, pj),
                        use_container_width=True, config=CFG_NOBAR)

# ══════════════════════════════════════════════════════════════════════════════
# Auto-advance  (must be last — triggers st.rerun)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.playing:
    time.sleep(0.05)          # ≈20 display fps
    st.session_state.frame = (st.session_state.frame + speed) % n_frames
    st.rerun()
