# RehabGraph-RL Perception Experiment
**Comparative Evaluation of Spatio-Temporal Models for Upper-Limb Motion Regression**

This repository contains the implementation and experimental analysis for **Stage (i)** of my PhD research on **RehabGraph-RL** (Rehabilitation Graph Transformer with Reinforcement Learning).

## Research Question
**How can a spatio-temporal graph transformer be designed to effectively model structured upper-limb joint movements during rehabilitation exercises?**

## Dataset
- **Toronto Rehab Stroke Pose Dataset** (Participant P07)
- Real 3D Kinect joint positions recorded during robot-assisted upper-limb rehabilitation exercises
- After preprocessing: 2,261 frames with 25 joints × 3 coordinates
- Focus: Shoulder, elbow, and wrist joints

## Methods Compared
Four methods were evaluated for upper-limb motion regression and movement quality assessment:
- Ridge Regression (simple baseline)
- LSTM (temporal baseline)
- Graph Convolutional Network (GCN - spatial baseline)
- **Proposed Spatio-Temporal Graph Transformer** (main contribution)

## Experimental Results

| Model                                      | RMSE   | MAE    | R²     | Accuracy | F1-score | Inference Time (ms) |
|--------------------------------------------|--------|--------|--------|----------|----------|---------------------|
| Ridge Regression                           | 0.142  | 0.098  | 0.812  | 0.72     | 0.71     | 3.2                 |
| LSTM (Temporal)                            | 0.128  | 0.089  | 0.835  | 0.76     | 0.75     | 12.5                |
| GCN (Spatial)                              | 0.115  | 0.078  | 0.872  | 0.81     | 0.80     | 8.3                 |
| **Proposed Spatio-Temporal Graph Transformer** | **0.087** | **0.061** | **0.921** | **0.89** | **0.88** | **18.7**            |

## Key Findings
- The proposed Spatio-Temporal Graph Transformer outperforms all baselines in regression accuracy, movement quality classification, and temporal sensitivity.
- All models meet real-time requirements (< 200 ms per frame), making them suitable for human-robot interaction in rehabilitation.
- The integration of anatomical graph modeling and transformer attention provides clear advantages for detecting subtle deviations in upper-limb movements.

## Limitations
Current evaluation is limited to a single participant (P07). Cross-subject generalization across healthy and stroke survivors will be addressed in future experiments.

## Repository Structure
- `data/` — Raw and processed P07 data
- `data/preprocess.py` — Data loading and reshaping
- `notebooks/experiment_upper_limb_regression.ipynb` — Main comparative experiment
- `models/` — Model implementations

## How to Reproduce
1. Clone the repository
2. Run `python data/preprocess.py`
3. Open `notebooks/experiment_upper_limb_regression.ipynb`

**Author:** Aybars Oztuna (PhD Candidate)  
**Date:** April 2026

This perception module establishes a strong foundation for **Stage (ii)** — integration with reinforcement learning for adaptive robotic assistance in the full RehabGraph-RL framework.
