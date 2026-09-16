"""
build_ethogram_labels.py — social-aware B-SOiD-style ethogram, saved per-session for the GUI.

v3: now uses PARTNER-RELATIVE features too (focal->partner distance, nose->partner distance =
investigation, approach velocity, egocentric bearing, partner-present), so the ethogram can
discover SOCIAL motifs (approach / investigate / near-partner) and not just the focal mouse's
own posture. Pipeline: rich features -> UMAP -> KMeans(k=10) -> RF assign -> 0.7s bout smoothing.
Saves ethogram_labels.npz {session: motif(T,), __n_motifs__, __names__} and a UMAP figure.
"""
import os, numpy as np, scipy.io as sio
from itertools import combinations
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import umap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "G:/Technion-970405-Course-Final-Project/Data"
SESS = sorted(f[:-4] for f in os.listdir(DATA) if f.endswith(".mat"))
NP = 8; CENTRE, NOSE, TAILB = 3, 2, 6
PCX, PCY, PLK = 33, 34, 35                 # partner (stimulus) centre x,y,likelihood
K = 10; SENT = 25.0                        # motifs; sentinel "partner far/absent" (body lengths)
rng = np.random.default_rng(0)

def features(X):
    P = [(X[:, p*3], X[:, p*3+1]) for p in range(NP)]
    def dist(i, j): return np.hypot(P[i][0]-P[j][0], P[i][1]-P[j][1])
    body = dist(CENTRE, TAILB) + 1e-3
    F = [dist(i, j)/body for i, j in combinations(range(NP), 2)]          # 28 shape distances
    def ang(a, b, c):
        v1 = np.stack([P[a][0]-P[b][0], P[a][1]-P[b][1]]); v2 = np.stack([P[c][0]-P[b][0], P[c][1]-P[b][1]])
        return np.arccos(np.clip((v1*v2).sum(0)/(np.hypot(*v1)*np.hypot(*v2)+1e-9), -1, 1))
    F += [ang(NOSE, CENTRE, TAILB), ang(0, 1, 2)]
    S = gaussian_filter1d(np.nan_to_num(np.column_stack(F)), 3, axis=0)
    cx, cy = P[CENTRE]; nx, ny = P[NOSE]
    speed = gaussian_filter1d(np.hypot(np.gradient(cx), np.gradient(cy)), 3)
    nose_sp = gaussian_filter1d(np.hypot(np.gradient(nx), np.gradient(ny)), 3)
    dyn = np.abs(np.gradient(S, axis=0))
    # ── partner-relative (social) features ──
    pcx, pcy, pl = X[:, PCX], X[:, PCY], X[:, PLK]; present = (pl > 0.5).astype(float)
    d_cc = np.clip(np.hypot(cx-pcx, cy-pcy)/body, 0, SENT); d_nc = np.clip(np.hypot(nx-pcx, ny-pcy)/body, 0, SENT)
    d_cc = np.where(present > 0, d_cc, SENT); d_nc = np.where(present > 0, d_nc, SENT)
    appr = np.where(present > 0, -np.gradient(gaussian_filter1d(d_cc, 5)), 0.0)
    head = np.arctan2(gaussian_filter1d(np.gradient(cy), 5), gaussian_filter1d(np.gradient(cx), 5))
    bear = np.where(present > 0, (np.arctan2(pcy-cy, pcx-cx)-head+np.pi) % (2*np.pi)-np.pi, 0.0)
    base = np.column_stack([S, dyn, speed/(np.nanmedian(speed)+1e-3), nose_sp/(np.nanmedian(nose_sp)+1e-3)])
    partner = np.column_stack([d_cc, d_nc, appr, np.cos(bear), np.sin(bear), present])
    return np.nan_to_num(np.column_stack([base, partner])), np.nan_to_num(np.column_stack([speed, nose_sp, d_nc, present]))

print("features (focal pose + partner-relative) for all sessions...")
allF, allRaw, sidx = [], [], []
for n in SESS:
    d = sio.loadmat(f"{DATA}/{n}.mat", simplify_cells=True)["data"]
    F, raw = features(np.asarray(d["dlcData"]["X"], float)); allF.append(F); allRaw.append(raw); sidx += [n]*len(F)
allF = np.vstack(allF); allRaw = np.vstack(allRaw); sidx = np.array(sidx)
print(f"  feature matrix {allF.shape} (last 6 cols = partner-relative)")
Z = StandardScaler().fit_transform(allF)
sub = rng.choice(len(Z), min(50000, len(Z)), replace=False)
print("UMAP..."); emb = umap.UMAP(n_components=3, n_neighbors=25, min_dist=0.0, random_state=42, n_jobs=20).fit_transform(Z[sub])
print(f"KMeans k={K}..."); km = KMeans(K, n_init=10, random_state=0).fit(emb)
rf = RandomForestClassifier(n_estimators=150, n_jobs=20, random_state=0).fit(Z[sub], km.labels_)
motif = rf.predict(Z)

# ── name motifs by movement + partner signature ──
sp, ns, dnc, pres = allRaw[:, 0], allRaw[:, 1], allRaw[:, 2], allRaw[:, 3]
msp = np.array([sp[motif == m].mean() for m in range(K)]); mns = np.array([ns[motif == m].mean() for m in range(K)])
mpr = np.array([pres[motif == m].mean() for m in range(K)]); mdn = np.array([dnc[motif == m].mean() for m in range(K)])
sp_lo, sp_hi = np.percentile(msp, [33, 66]); ns_hi = np.median(mns)
names = []
for m in range(K):
    base = "fast move" if msp[m] > sp_hi else ("still" if msp[m] < sp_lo else "slow move")
    if mns[m] > ns_hi: base += "+sniff"
    if mpr[m] > 0.6:                                   # mostly during social
        base += " | investigate" if mdn[m] < 6 else " | social"
    names.append(base)
print("motif names:"); [print(f"  {m}: {names[m]}  (spd{msp[m]:.1f} nose{mns[m]:.1f} partner-present{mpr[m]:.2f} nose-dist{mdn[m]:.1f})") for m in range(K)]

def smooth(lab):  # 0.7s majority-vote bouts
    return uniform_filter1d(np.eye(K)[lab], size=21, axis=0, mode="nearest").argmax(1)
out = {n: smooth(motif[sidx == n]).astype(np.int8) for n in SESS}
out["__n_motifs__"] = np.array([K]); out["__names__"] = np.array(names)
np.savez_compressed(f"{HERE}/ethogram_labels.npz", **out)
print(f"saved ethogram_labels.npz ({K} motifs)")

# ── UMAP figure: clusters + what drives them ──
ECOL = ["#4ecdc4","#ffe66d","#ff6b6b","#c77dff","#a8e6cf","#ffd6a5","#b9d5ff","#ff8b94","#c7f2a4","#f7a072"]
lab_sub = km.labels_
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), facecolor="#0f0f1a")
for a in ax: a.set_facecolor("#0f0f1a"); a.tick_params(colors="#888"); a.title.set_color("#fff")
for m in range(K):
    mk = lab_sub == m
    ax[0].scatter(emb[mk, 0], emb[mk, 1], s=3, color=ECOL[m], alpha=.5, label=f"{m}:{names[m]}")
ax[0].set_title(f"UMAP of behaviour features, coloured by motif (k={K})")
ax[0].legend(fontsize=6, markerscale=3, loc="upper right", facecolor="#16213e", labelcolor="#aaa")
sc1 = ax[1].scatter(emb[:, 0], emb[:, 1], s=3, c=sp[sub], cmap="viridis", alpha=.5, vmax=np.percentile(sp[sub], 98))
ax[1].set_title("coloured by SPEED (locomotion axis)"); fig.colorbar(sc1, ax=ax[1], shrink=.6)
sc2 = ax[2].scatter(emb[:, 0], emb[:, 1], s=3, c=pres[sub], cmap="coolwarm", alpha=.5)
ax[2].set_title("coloured by PARTNER PRESENT (social axis)"); fig.colorbar(sc2, ax=ax[2], shrink=.6)
plt.tight_layout(); fig.savefig(f"{HERE}/fig_ethogram_umap.png", dpi=140, facecolor="#0f0f1a")
print("saved fig_ethogram_umap.png")
