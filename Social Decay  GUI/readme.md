# Social Decay GUI

A Streamlit viewer built specifically around the social-residue result in the report: pick any session, watch the neural-state trajectory move through PRE, SOCIAL, and POST alongside the synced video and DLC skeleton, and see exactly where the "fading" happens in time. This is the tool we used to sanity-check individual sessions while writing the report, including the M2_Female_1 walkthrough.

## What it does

- **3D PCA neural trajectory** for the selected session, colored by phase, with vocalization events marked
- **Synced video + DLC skeleton overlay** panel, playhead-linked to the trajectory
- **PC1 vs PC2 / PC2 vs PC3 / PC3 vs PC1** projections, updated live as you scrub through the session
- **Full-session vocalization timeline** strip

https://github.com/user-attachments/assets/fe37da07-3d25-4948-a2d8-f48abd12a084

## How to run

```bash
pip install -r requirements.txt
streamlit run streamlit_viewer.py
```
