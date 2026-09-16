# Exploration GUI

A Tkinter desktop app for browsing the raw recording sessions in depth: neural population trajectories, tracked mouse pose, vocalizations, and behavioral motifs, all synced to one shared playhead. This is the tool we used ourselves to get a feel for each session before writing any of the analysis in the report.

## What it does

- **3D PCA trajectory** of neural population activity, with vocalization markers
- **2D PCA projections** (PC1 vs PC2, PC2 vs PC3, PC3 vs PC1)
- **Mouse skeleton overlay** from DLC tracking (or video frame if available)
- **Behaviour ethogram** (B-SOiD-style motif labels) as a full-session timeline strip, synced to the same playhead as the trajectory and video
- **Vocalization timeline** strip across the full session
- Session selector to switch between all 33 sessions

https://github.com/user-attachments/assets/71e59fa5-a4c5-466b-a86b-e0d092cc49cf

## How to run

```bash
pip install -r requirements.txt
python neural_trajectory_gui.py
```

The ethogram labels ship precomputed (`ethogram_labels.npz`), so you don't need to regenerate them to use the GUI. If you want to rebuild them from scratch (or run this on new sessions), `build_ethogram_labels.py` is included; it needs `umap-learn` in addition to the packages above.
