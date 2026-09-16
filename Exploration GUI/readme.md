# Exploration GUI

Interactive neural trajectory viewer for browsing recording sessions.

## What it does

- **3D PCA trajectory** of neural population activity, with vocalization markers
- **2D PCA projections** (PC1 vs PC2, PC2 vs PC3, PC3 vs PC1)
- **Mouse skeleton overlay** from DLC tracking (or video frame if available)
- **Vocalization timeline** strip across the full session
- Session selector to switch between all 33 sessions

https://github.com/user-attachments/assets/71e59fa5-a4c5-466b-a86b-e0d092cc49cf


## How to run

```bash
pip install -r requirements.txt
streamlit run streamlit_viewer.py
```


