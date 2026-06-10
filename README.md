# RehabGraph-RL Perception Experiment


Comparative Analysis of 8 Methods for Upper-Limb Motion Regression

## Research Question

How can a spatio-temporal graph transformer be designed to effectively model structured upper-limb joint movements during rehabilitation exercises?

---

## Dataset

| Item                 | Description                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| Dataset              | Toronto Rehab Stroke Pose Dataset                                                                      |
| Total Participants   | 19 (10 Healthy + 9 Stroke Survivors)                                                                   |
| Current Experiment   | Participant P07 (Stroke Survivor)                                                                      |
| Frames Used          | 2,261 frames after preprocessing                                                                       |
| Data Collection      | Microsoft Kinect sensor during seated robot-assisted upper-limb rehabilitation exercises               |
| Information Included | 3D joint positions (25 joints × 3 coordinates per frame) plus expert labels for compensatory movements |
| Why Chosen           | Clinically realistic dataset capturing compensatory movement patterns during stroke rehabilitation     |

---

## Methodology

| Component        | Configuration                                  |
| ---------------- | ---------------------------------------------- |
| Train/Test Split | 75% Training / 25% Testing (`random_state=42`) |
| Cross-Validation | 5-Fold Cross-Validation on training set        |
| Environment      | Jupyter Notebook + Anaconda                    |

---

## Methods Compared

### Group 1: Baseline Methods

- Ridge Regression
- LSTM
- GCN

### Group 2: Local Experimental Methods

- TCN (Temporal Convolutional Network)
- ST-GCN (Spatio-Temporal Graph Convolutional Network)

### Group 3: Recent Literature Methods (2024–2025)

- Advanced Skeleton-Graph Transformer (Li et al., 2025)
- Adaptive Trajectory Prediction Model (IEEE Transactions on Robotics)

### Group 4: Proposed Method

- **Graph-Temporal Fusion Network (GTFN)** – Adapted and optimised for compensatory movement assessment in upper-limb stroke rehabilitation

---

## Experimental Results

| Model                                | RMSE ↓ | MAE ↓ | R² ↑ | Inference Time (ms) |
| ------------------------------------ | ------ | ----- | ---- | ------------------- |
| Ridge Regression                     | 0.142  | 0.098 | 0.812 | 3.2                 |
| LSTM                                 | 0.128  | 0.089 | 0.835 | 12.5                |
| GCN                                  | 0.115  | 0.078 | 0.872 | 8.3                 |
| TCN (Local)                          | 0.102  | 0.071 | 0.891 | 9.8                 |
| ST-GCN (Local)                       | 0.095  | 0.066 | 0.905 | 14.2                |
| Advanced Skeleton-Graph Transformer  | 0.091  | 0.063 | 0.912 | 22.0                |
| Adaptive Trajectory Prediction Model | 0.088  | 0.060 | 0.918 | 24.5                |
| **GTFN (Proposed)**                  | **0.079** | **0.054** | **0.942** | **18.5**            |

---

## Key Findings

- **GTFN** achieves the best overall performance among all 8 methods.
- 16.8% lower RMSE than ST-GCN.
- 4.1% higher R² than the best literature method.
- Inference time of 18.5 ms, satisfying real-time HRI requirements (<200 ms).

---

## What is GTFN and Why is it Useful for Stroke Rehabilitation?

This work **does not** claim to invent new neural network components (GCN, Transformer, multi-scale convolution, learnable fusion are all established techniques). Instead, the contribution is the **systematic adaptation and optimisation** of a graph-temporal fusion framework for the specific, clinically important task of **compensatory movement assessment in upper-limb stroke rehabilitation**.

**Key adaptations for this task:**

- Anatomical graph tailored to the shoulder–elbow–wrist hierarchy, where compensatory movements commonly occur.
- Temporal window size (10 frames) optimised for stroke patient movement speeds.
- Learnable fusion gate that dynamically balances spatial and temporal features, adapted to the variability of stroke patient motion.
- Lightweight design (18.5 ms inference) meeting real-time human-robot interaction requirements.

**Demonstrated improvement:**

- Outperforms seven baselines on participant P07 from the Toronto Rehab Stroke Pose Dataset.
- Provides a reproducible, openly available implementation for other researchers to build upon.

---

## Limitations

Current evaluation is limited to a single participant (**P07**). Cross-subject generalisation across all 19 participants (healthy and stroke survivors) is required before clinical deployment.

---

## Repository Structure

```text
RehabGraph-RL-Perception-Experiment/
├── data/
│   ├── P07_processed.npy
│   └── preprocess.py
├── experiments/
│   ├── TCN/
│   │   └── tcn_model.py
│   ├── STGCN/
│   │   └── stgcn_model.py
│   └── GTFN/
│       └── gtfn_model.py
├── notebooks/
│   └── experiment_upper_limb_regression.ipynb
├── README.md
└── requirements.txt
