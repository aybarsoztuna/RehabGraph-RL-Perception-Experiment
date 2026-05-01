# RehabGraph-RL Perception Experiment
**Comparative Analysis of Spatio-Temporal Models for Upper-Limb Motion Regression**

This repository contains the implementation and comparative experiments for **Stage (i)** of my PhD research on **RehabGraph-RL** (Rehabilitation Graph Transformer with Reinforcement Learning).

## Research Question
**How can a spatio-temporal graph transformer be designed to effectively model structured upper-limb joint movements during rehabilitation exercises?**

## Dataset
Toronto Rehab Stroke Pose Dataset (Participant P07)  
- Real 3D Kinect joint positions recorded during robot-assisted upper-limb rehabilitation  
- 2,261 frames after preprocessing (25 joints × 3 coordinates)  
- Focus on shoulder, elbow, and wrist joints

## Methods Compared
Four methods were evaluated:
- Ridge Regression (simple baseline)
- LSTM (temporal baseline)
- Graph Convolutional Network (GCN - spatial baseline)
- **Proposed Spatio-Temporal Graph Transformer** (main contribution)

## Experimental Results

| Model                                      | RMSE   | MAE    | R²     | Accuracy | F1-score | Inference Time (ms) |
|--------------------------------------------|--------|--------|--------|----------|----------|---------------------|
| Ridge Regression                           | 0.142  | 0.098  | 0.812  | 0.72     | 0.71     | ~3                  |
| LSTM (Temporal)                            | 0.128  | 0.089  | 0.835  | 0.76     | 0.75     | ~12.5               |
| GCN (Spatial)                              | 0.115  | 0.078  | 0.872  | 0.81     | 0.80     | ~8.3                |
| **Proposed Spatio-Temporal Graph Transformer** | **0.087** | **0.061** | **0.921** | **0.89** | **0.88** | **18.7**            |

## Key Findings
- The proposed model achieves the best regression and classification performance.
- All models satisfy real-time constraints (< 200ms per frame).
- Strong temporal sensitivity and anatomical modeling give the Graph Transformer a clear advantage.

## Limitations
Evaluation is currently limited to a single participant (P07). Cross-subject generalization will be addressed in future work.

**Author:** Aybars Oztuna (PhD Candidate)  
**Date:** April 2026  

This perception module provides the foundation for Stage (ii): integration with reinforcement learning for adaptive robotic assistance.
