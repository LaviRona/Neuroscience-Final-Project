# Appendix: full results, all 33 sessions

Supplementary material for `final.md`. The main report walks through one session, M2_Female_1, in depth because session-to-session variability makes a pooled average misleading on its own; everything here is the fuller picture that claim is built on: every session individually, every mouse split by stimulus sex, the pooled group averages, and the robustness/confound checks referenced but not shown in the main text.

---

## 1. The pooled picture, all 33 sessions

The group-level average the main report deliberately does not lead with: PRE/SOCIAL reference separation and POST decay pooled across all 33 sessions, split by stimulus sex, for each feature representation. Useful as a sanity check on the overall direction of the effect, but flattens the session-to-session variability that motivates walking through one session individually in the main report.

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![pooled mean rate](figures/fig2_classifier_mean_rate.png) | ![pooled covariance](figures/fig4_classifier_covariance.png) |

---

## 2. All 33 sessions in one plot, per classifier

Every session as its own small panel, so the decay (or its absence) can be checked session by session rather than averaged away. Panels bordered in gold are the sessions where the mean-rate and covariance-geometry decay slopes disagree the most, the six sessions with the largest gap between the two representations.

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![fig2 grid](figures/fig2_per_session_grid.png) | ![fig4 grid](figures/fig4_per_session_grid.png) |

---

## 3. Per mouse: male-stimulus vs. female-stimulus sessions

Each mouse's own sessions pooled together and split by stimulus sex, one classifier per column. This is the level the Bonus section's stimulus-sex question is actually answered at, mouse by mouse rather than as one group average; M2 is the one example walked through in the main report, the other four are here for comparison.

### M2

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M2 mean rate](figures/fig2_classifier_mean_rate_M2.png) | ![M2 covariance](figures/fig4_classifier_covariance_M2.png) |

### M3

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3 mean rate](figures/fig2_classifier_mean_rate_M3.png) | ![M3 covariance](figures/fig4_classifier_covariance_M3.png) |

### M6

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M6 mean rate](figures/fig2_classifier_mean_rate_M6.png) | ![M6 covariance](figures/fig4_classifier_covariance_M6.png) |

### M7

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M7 mean rate](figures/fig2_classifier_mean_rate_M7.png) | ![M7 covariance](figures/fig4_classifier_covariance_M7.png) |

### M8

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8 mean rate](figures/fig2_classifier_mean_rate_M8.png) | ![M8 covariance](figures/fig4_classifier_covariance_M8.png) |

---

## 4. Per mouse, per session: classifiers and clustering side by side

Every session individually, grouped by mouse. For each session: the mean-rate and covariance-geometry classifier results on top, and the unsupervised PCA and diffusion-map views of the same session below, so the supervised and unsupervised story can be checked against each other session by session (this is the same 2x2 layout as the main report's Figure 2, for every other session).

### Mouse M2

<details>
<summary>M2_Female_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M2_Female_1 rate](figures/M2_Female_1_rate.png) | ![M2_Female_1 covariance](figures/M2_Female_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M2_Female_1](figures/fig_pca_M2_Female_1.png) | ![Diffusion map M2_Female_1](figures/fig_diffmap_M2_Female_1.png) |
</details>

<details>
<summary>M2_Female_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M2_Female_2 rate](figures/M2_Female_2_rate.png) | ![M2_Female_2 covariance](figures/M2_Female_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M2_Female_2](figures/fig_pca_M2_Female_2.png) | ![Diffusion map M2_Female_2](figures/fig_diffmap_M2_Female_2.png) |
</details>

<details>
<summary>M2_Female_3</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M2_Female_3 rate](figures/M2_Female_3_rate.png) | ![M2_Female_3 covariance](figures/M2_Female_3_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M2_Female_3](figures/fig_pca_M2_Female_3.png) | ![Diffusion map M2_Female_3](figures/fig_diffmap_M2_Female_3.png) |
</details>

<details>
<summary>M2_Female_4</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M2_Female_4 rate](figures/M2_Female_4_rate.png) | ![M2_Female_4 covariance](figures/M2_Female_4_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M2_Female_4](figures/fig_pca_M2_Female_4.png) | ![Diffusion map M2_Female_4](figures/fig_diffmap_M2_Female_4.png) |
</details>

<details>
<summary>M2_Male_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M2_Male_1 rate](figures/M2_Male_1_rate.png) | ![M2_Male_1 covariance](figures/M2_Male_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M2_Male_1](figures/fig_pca_M2_Male_1.png) | ![Diffusion map M2_Male_1](figures/fig_diffmap_M2_Male_1.png) |
</details>

<details>
<summary>M2_Male_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M2_Male_2 rate](figures/M2_Male_2_rate.png) | ![M2_Male_2 covariance](figures/M2_Male_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M2_Male_2](figures/fig_pca_M2_Male_2.png) | ![Diffusion map M2_Male_2](figures/fig_diffmap_M2_Male_2.png) |
</details>

<details>
<summary>M2_Male_3</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M2_Male_3 rate](figures/M2_Male_3_rate.png) | ![M2_Male_3 covariance](figures/M2_Male_3_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M2_Male_3](figures/fig_pca_M2_Male_3.png) | ![Diffusion map M2_Male_3](figures/fig_diffmap_M2_Male_3.png) |
</details>

---

### Mouse M3

<details>
<summary>M3_Female_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3_Female_1 rate](figures/M3_Female_1_rate.png) | ![M3_Female_1 covariance](figures/M3_Female_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M3_Female_1](figures/fig_pca_M3_Female_1.png) | ![Diffusion map M3_Female_1](figures/fig_diffmap_M3_Female_1.png) |
</details>

<details>
<summary>M3_Female_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3_Female_2 rate](figures/M3_Female_2_rate.png) | ![M3_Female_2 covariance](figures/M3_Female_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M3_Female_2](figures/fig_pca_M3_Female_2.png) | ![Diffusion map M3_Female_2](figures/fig_diffmap_M3_Female_2.png) |
</details>

<details>
<summary>M3_Female_3</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3_Female_3 rate](figures/M3_Female_3_rate.png) | ![M3_Female_3 covariance](figures/M3_Female_3_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M3_Female_3](figures/fig_pca_M3_Female_3.png) | ![Diffusion map M3_Female_3](figures/fig_diffmap_M3_Female_3.png) |
</details>

<details>
<summary>M3_Female_4</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3_Female_4 rate](figures/M3_Female_4_rate.png) | ![M3_Female_4 covariance](figures/M3_Female_4_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M3_Female_4](figures/fig_pca_M3_Female_4.png) | ![Diffusion map M3_Female_4](figures/fig_diffmap_M3_Female_4.png) |
</details>

<details>
<summary>M3_Male_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3_Male_1 rate](figures/M3_Male_1_rate.png) | ![M3_Male_1 covariance](figures/M3_Male_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M3_Male_1](figures/fig_pca_M3_Male_1.png) | ![Diffusion map M3_Male_1](figures/fig_diffmap_M3_Male_1.png) |
</details>

<details>
<summary>M3_Male_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3_Male_2 rate](figures/M3_Male_2_rate.png) | ![M3_Male_2 covariance](figures/M3_Male_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M3_Male_2](figures/fig_pca_M3_Male_2.png) | ![Diffusion map M3_Male_2](figures/fig_diffmap_M3_Male_2.png) |
</details>

<details>
<summary>M3_Male_3</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3_Male_3 rate](figures/M3_Male_3_rate.png) | ![M3_Male_3 covariance](figures/M3_Male_3_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M3_Male_3](figures/fig_pca_M3_Male_3.png) | ![Diffusion map M3_Male_3](figures/fig_diffmap_M3_Male_3.png) |
</details>

<details>
<summary>M3_Male_4</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M3_Male_4 rate](figures/M3_Male_4_rate.png) | ![M3_Male_4 covariance](figures/M3_Male_4_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M3_Male_4](figures/fig_pca_M3_Male_4.png) | ![Diffusion map M3_Male_4](figures/fig_diffmap_M3_Male_4.png) |
</details>

---

### Mouse M6

<details>
<summary>M6_Female_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M6_Female_1 rate](figures/M6_Female_1_rate.png) | ![M6_Female_1 covariance](figures/M6_Female_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M6_Female_1](figures/fig_pca_M6_Female_1.png) | ![Diffusion map M6_Female_1](figures/fig_diffmap_M6_Female_1.png) |
</details>

<details>
<summary>M6_Male_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M6_Male_1 rate](figures/M6_Male_1_rate.png) | ![M6_Male_1 covariance](figures/M6_Male_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M6_Male_1](figures/fig_pca_M6_Male_1.png) | ![Diffusion map M6_Male_1](figures/fig_diffmap_M6_Male_1.png) |
</details>

<details>
<summary>M6_Male_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M6_Male_2 rate](figures/M6_Male_2_rate.png) | ![M6_Male_2 covariance](figures/M6_Male_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M6_Male_2](figures/fig_pca_M6_Male_2.png) | ![Diffusion map M6_Male_2](figures/fig_diffmap_M6_Male_2.png) |
</details>

---

### Mouse M7

<details>
<summary>M7_Female_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M7_Female_1 rate](figures/M7_Female_1_rate.png) | ![M7_Female_1 covariance](figures/M7_Female_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M7_Female_1](figures/fig_pca_M7_Female_1.png) | ![Diffusion map M7_Female_1](figures/fig_diffmap_M7_Female_1.png) |
</details>

<details>
<summary>M7_Female_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M7_Female_2 rate](figures/M7_Female_2_rate.png) | ![M7_Female_2 covariance](figures/M7_Female_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M7_Female_2](figures/fig_pca_M7_Female_2.png) | ![Diffusion map M7_Female_2](figures/fig_diffmap_M7_Female_2.png) |
</details>

<details>
<summary>M7_Female_3</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M7_Female_3 rate](figures/M7_Female_3_rate.png) | ![M7_Female_3 covariance](figures/M7_Female_3_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M7_Female_3](figures/fig_pca_M7_Female_3.png) | ![Diffusion map M7_Female_3](figures/fig_diffmap_M7_Female_3.png) |
</details>

<details>
<summary>M7_Female_4</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M7_Female_4 rate](figures/M7_Female_4_rate.png) | ![M7_Female_4 covariance](figures/M7_Female_4_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M7_Female_4](figures/fig_pca_M7_Female_4.png) | ![Diffusion map M7_Female_4](figures/fig_diffmap_M7_Female_4.png) |
</details>

<details>
<summary>M7_Male_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M7_Male_2 rate](figures/M7_Male_2_rate.png) | ![M7_Male_2 covariance](figures/M7_Male_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M7_Male_2](figures/fig_pca_M7_Male_2.png) | ![Diffusion map M7_Male_2](figures/fig_diffmap_M7_Male_2.png) |
</details>

<details>
<summary>M7_Male_3</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M7_Male_3 rate](figures/M7_Male_3_rate.png) | ![M7_Male_3 covariance](figures/M7_Male_3_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M7_Male_3](figures/fig_pca_M7_Male_3.png) | ![Diffusion map M7_Male_3](figures/fig_diffmap_M7_Male_3.png) |
</details>

<details>
<summary>M7_Male_4</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M7_Male_4 rate](figures/M7_Male_4_rate.png) | ![M7_Male_4 covariance](figures/M7_Male_4_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M7_Male_4](figures/fig_pca_M7_Male_4.png) | ![Diffusion map M7_Male_4](figures/fig_diffmap_M7_Male_4.png) |
</details>

---

### Mouse M8

<details>
<summary>M8_Female_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8_Female_1 rate](figures/M8_Female_1_rate.png) | ![M8_Female_1 covariance](figures/M8_Female_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M8_Female_1](figures/fig_pca_M8_Female_1.png) | ![Diffusion map M8_Female_1](figures/fig_diffmap_M8_Female_1.png) |
</details>

<details>
<summary>M8_Female_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8_Female_2 rate](figures/M8_Female_2_rate.png) | ![M8_Female_2 covariance](figures/M8_Female_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M8_Female_2](figures/fig_pca_M8_Female_2.png) | ![Diffusion map M8_Female_2](figures/fig_diffmap_M8_Female_2.png) |
</details>

<details>
<summary>M8_Female_3</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8_Female_3 rate](figures/M8_Female_3_rate.png) | ![M8_Female_3 covariance](figures/M8_Female_3_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M8_Female_3](figures/fig_pca_M8_Female_3.png) | ![Diffusion map M8_Female_3](figures/fig_diffmap_M8_Female_3.png) |
</details>

<details>
<summary>M8_Female_4</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8_Female_4 rate](figures/M8_Female_4_rate.png) | ![M8_Female_4 covariance](figures/M8_Female_4_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M8_Female_4](figures/fig_pca_M8_Female_4.png) | ![Diffusion map M8_Female_4](figures/fig_diffmap_M8_Female_4.png) |
</details>

<details>
<summary>M8_Male_1</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8_Male_1 rate](figures/M8_Male_1_rate.png) | ![M8_Male_1 covariance](figures/M8_Male_1_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M8_Male_1](figures/fig_pca_M8_Male_1.png) | ![Diffusion map M8_Male_1](figures/fig_diffmap_M8_Male_1.png) |
</details>

<details>
<summary>M8_Male_2</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8_Male_2 rate](figures/M8_Male_2_rate.png) | ![M8_Male_2 covariance](figures/M8_Male_2_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M8_Male_2](figures/fig_pca_M8_Male_2.png) | ![Diffusion map M8_Male_2](figures/fig_diffmap_M8_Male_2.png) |
</details>

<details>
<summary>M8_Male_3</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8_Male_3 rate](figures/M8_Male_3_rate.png) | ![M8_Male_3 covariance](figures/M8_Male_3_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M8_Male_3](figures/fig_pca_M8_Male_3.png) | ![Diffusion map M8_Male_3](figures/fig_diffmap_M8_Male_3.png) |
</details>

<details>
<summary>M8_Male_4</summary>

| Mean-rate classifier | Covariance-geometry classifier |
|---|---|
| ![M8_Male_4 rate](figures/M8_Male_4_rate.png) | ![M8_Male_4 covariance](figures/M8_Male_4_covariance.png) |

| PCA (mean rate) | Diffusion map (covariance geometry) |
|---|---|
| ![PCA M8_Male_4](figures/fig_pca_M8_Male_4.png) | ![Diffusion map M8_Male_4](figures/fig_diffmap_M8_Male_4.png) |
</details>

---

## 5. Regularization: does the classifier just look unsure at 0.5?

Worked examples referenced in the main report's "What regularization does" and "Limitations" sections. Each panel compares the full-data, default-regularization fit against stronger L2 regularization and smaller training-data fractions, to check whether a weak or flat POST slope is a real absence of residue or an artifact of the classifier defaulting to 0.5 on unfamiliar input. M2_Female_1 already had a clean slope (a check that the signal survives, not a rescue); the rest are the sessions with the weakest full-data slope, where the check actually matters.

![M2_Female_1 regularization check](figures/fig_M2_Female_1_regularization_check.png)
![M3_Male_3 regularization check](figures/fig_M3_Male_3_regularization_check.png)
![M8_Male_2 regularization check](figures/fig_M8_Male_2_regularization_check.png)
![M8_Male_3 regularization check](figures/fig_M8_Male_3_regularization_check.png)
![M3_Female_4 regularization check](figures/fig_M3_Female_4_regularization_check.png)
![M8_Female_4 regularization check](figures/fig_M8_Female_4_regularization_check.png)
![M6_Female_1 regularization check](figures/fig_M6_Female_1_regularization_check.png)

The main report's "What regularization does" section is built on mean-rate features only (M2_Female_1 and M8_Male_2). Here is the same comparison for both feature types side by side: mean rate (top) and covariance geometry (bottom), both regularization approaches, both sessions. Mean rate behaves as described in the main text; covariance geometry does not, less data or stronger regularization weakens its slope in both sessions rather than sharpening it, so for covariance geometry the full-data fit is the one to trust.

![Regularization grid: mean rate vs. covariance geometry, both sessions, both regularization types](figures/fig_regularization_grid_combined.png)

---

## 6. Searching for confounders: does drift explain the pattern?

One candidate explanation for why some sessions show a clean POST decay and others look flat: within-session representational drift, population activity changing over time for reasons unrelated to PRE/SOCIAL/POST, could be masking a real decay in some sessions or manufacturing an artifactual one in others. We tested this by training a plain regressor with y = elapsed session time on the mean-rate features, and using its cross-validated R2 as a measure of drift strength, for three groups of sessions (clean decay at full data, decay only after extra regularization, weak decay throughout). If drift explained the group differences, drift strength should separate the groups. It does not.

![Drift strength by session group, no clean separation](figures/fig_drift_r2_by_group.png)
