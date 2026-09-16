
# Social Decay GUI

Desktop application for running and exploring the social-residue analysis described in the project report.

## What it does

- **Run experiment**: train a PRE-vs-SOCIAL classifier (mean firing rate or covariance-geometry features) on any session, with configurable windowing and neuron-subset options; view the POST decay curve in real time
- **Browse history**: browse all saved classifier runs, compare sessions and feature types side by side
- **Covariance explorer**: build correlation matrices and diffusion-map embeddings for any session, visualize how population covariance structure moves through PRE, SOCIAL, and POST phases

https://github.com/user-attachments/assets/fe37da07-3d25-4948-a2d8-f48abd12a084

## How to run

```bash
pip install -r requirements.txt
python -m social_decay_gui
```

A window will open with three tabs: *Run experiment*, *Browse history*, and *Covariance explorer*.
