# RehabGraph-RL Perception Experiment - Upper Limb Motion Modeling

This repository contains the implementation and experiments for the **spatio-temporal graph transformer** component of my PhD research on **RehabGraph-RL** (Rehabilitation Graph Transformer with Reinforcement Learning).

## Research Question Addressed
**How can a spatio-temporal graph transformer be designed to effectively model structured upper-limb joint movements during rehabilitation exercises?**

This is Stage 1 of my PhD scope: developing the perception module for upper-limb (shoulder, elbow, wrist) posture regression.

## Dataset Used
**Toronto Rehab Stroke Pose Dataset** (from Kaggle)

- Downloaded and stored in `data/P07/`
- Contains `Joint_Positions.csv` (3D Kinect joint coordinates) and `Labels.csv`
- Data from stroke patients and healthy participants performing robot-assisted upper-limb exercises
- Each `Joint_Positions.csv` has stacked 25-joint × 3D coordinates (every 25 rows = 1 frame)

**Why this dataset?** It provides real clinical upper-limb movements with compensatory postures common in stroke rehabilitation — perfect for testing structured graph-based modeling of biomechanical joint relationships.

## Project Structure
- `data/` — Raw participant data (P07 uploaded)
- `notebooks/` — Main experiment: `experiment_upper_limb_regression.ipynb`
- `utils/` — Preprocessing and graph utilities
- `models/` — Spatio-Temporal Graph Transformer model
- `results/` — Plots, metrics tables, and visualizations

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebooks/experiment_upper_limb_regression.ipynb`

## Expected Results (from my experiment)
The proposed Spatio-Temporal Graph Transformer outperforms LSTM and GCN baselines in posture regression (lower RMSE/MAE, higher R²) by explicitly modeling anatomical joint connections and temporal dynamics.

## Author
Aybars Oztuna (PhD Candidate)  
Date: April 2026

This work forms the perception foundation for the full closed-loop adaptive rehabilitation framework in my thesis.
