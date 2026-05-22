# RehabGraph-RL Perception Experiment
**Comparative Analysis of 7 Methods for Upper-Limb Motion Regression**

This repository contains the implementation and experimental analysis for **Stage (i)** of my PhD research on **RehabGraph-RL** (Rehabilitation Graph Transformer with Reinforcement Learning).

## Research Question
**How can a spatio-temporal graph transformer be designed to effectively model structured upper-limb joint movements during rehabilitation exercises?**

## Dataset
- **Full Dataset:** Toronto Rehab Stroke Pose Dataset
- **Total Participants:** 19 (10 Healthy + 9 Stroke Survivors)
- **Current Experiment:** Participant **P07** (Stroke survivor)
- **Frames Used:** 2,261 frames after preprocessing
- **Data Collection:** Microsoft Kinect sensor during seated robot-assisted upper-limb rehabilitation exercises
- **Information Included:** 3D joint positions (25 joints × 3 coordinates per frame) + expert labels for compensatory movements
- **Why Chosen?** Clinically realistic data capturing real compensatory patterns in stroke rehabilitation

## Methodology
- **Train/Test Split:** 75% Training / 25% Testing (random split with `random_state=42`)
- **Cross-Validation:** 5-Fold Cross-Validation on the training set
- **Environments:** Jupyter Notebook + Anaconda (for local experiments and data augmentation)

## 7 Methods Compared

**Group 1: Core Python Methods (Existing)**
- Ridge Regression
- LSTM
- GCN

**Group 2: New Local Methods (Anaconda Experiments)**
- TCN (Temporal Convolutional Network)
- ST-GCN (Spatio-Temporal Graph Convolutional Network)

**Group 3: Recent Literature Methods (2024-2025)**
- Advanced Skeleton-Graph Transformer
- Adaptive Trajectory Prediction Model

**Proposed Original Method (My Contribution)**
- **Graph-Temporal Fusion Network (GTFN)**

## Experimental Results (Summary)

| Model                                      | RMSE   | MAE    | R²     | Accuracy | F1-score | Inference Time (ms) |
|--------------------------------------------|--------|--------|--------|----------|----------|---------------------|
| Ridge Regression                           | 0.142  | 0.098  | 0.812  | 0.72     | 0.71     | 3.2                 |
| LSTM                                       | 0.128  | 0.089  | 0.835  | 0.76     | 0.75     | 12.5                |
| GCN                                        | 0.115  | 0.078  | 0.872  | 0.81     | 0.80     | 8.3                 |
| TCN (New)                                  | 0.102  | 0.071  | 0.891  | 0.84     | 0.83     | 9.8                 |
| ST-GCN (New)                               | 0.095  | 0.066  | 0.905  | 0.86     | 0.85     | 14.2                |
| Literature Method                          | 0.091  | 0.063  | 0.912  | 0.87     | 0.86     | 22.0                |
| **Proposed GTFN (Original)**               | **0.082** | **0.057** | **0.935** | **0.90** | **0.89** | **19.5**            |

## Key Findings
- The proposed GTFN achieves the best overall performance by intelligently fusing spatial and temporal modeling.
- All methods meet real-time requirements (< 200 ms per frame).
- Strong temporal sensitivity and movement quality classification are key advantages.

## Limitations
Current evaluation is limited to a single participant (P07). Cross-subject generalization will be addressed in future work.

## Repository Structure
- `data/` — Raw and processed P07 data
- `data/preprocess.py` — Data loading and reshaping
- `experiments/` — New methods (TCN, ST-GCN, GTFN)
- `notebooks/experiment_upper_limb_regression.ipynb` — Main comparative experiment

## How to Reproduce
1. Clone the repository
2. Run `python data/preprocess.py`
3. Open `notebooks/experiment_upper_limb_regression.ipynb`

**Author:** Aybars Oztuna (PhD Candidate)  
**Date:** April 2026

This perception module establishes a strong foundation for **Stage (ii)** — integration with reinforcement learning for adaptive robotic assistance.
