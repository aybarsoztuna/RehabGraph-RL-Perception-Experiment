# RehabGraph-RL Perception Experiment  
**Spatio-Temporal Graph Transformer for Upper-Limb Rehabilitation**

This repository contains the implementation and experiments for **Stage 1** of my PhD research on RehabGraph-RL (Rehabilitation Graph Transformer with Reinforcement Learning).

## Research Question
**How can a spatio-temporal graph transformer be designed to effectively model structured upper-limb joint movements during rehabilitation exercises?**

## Dataset
Toronto Rehab Stroke Pose Dataset (P07 participant)  
- Real 3D Kinect joint positions from robot-assisted upper-limb exercises  
- 2,261 frames after preprocessing (25 joints × 3 coordinates)  
- Includes compensatory movements from stroke rehabilitation

## Key Files
- `data/preprocess.py` — Loads and reshapes stacked Joint_Positions.csv files
- `notebooks/experiment_upper_limb_regression.ipynb` — Main experiment with results
- `data/P07_processed.npy` — Processed motion data

## Experimental Results

| Model                                      | RMSE   | MAE    | R²     |
|--------------------------------------------|--------|--------|--------|
| LSTM / GCN Baselines                       | 0.119–0.142 | 0.082–0.098 | 0.812–0.867 |
| **Spatio-Temporal Graph Transformer (Proposed)** | **0.087** | **0.061** | **0.921** |

The proposed model significantly reduces posture regression error by integrating anatomical graph modeling with temporal transformer attention.

## How to Reproduce
1. Clone the repository
2. Run `python data/preprocess.py`
3. Open `notebooks/experiment_upper_limb_regression.ipynb`

**Author:** Aybars Oztuna (PhD Candidate)  
**Date:** April 2026

This perception module forms the foundation for integrating reinforcement learning-based adaptive robotic assistance in the full RehabGraph-RL framework.
