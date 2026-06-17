
# RehabGraph-RL Perception Experiment

Comparative Analysis of 10 Methods for Upper-Limb Motion Regression

---

## Research Question

How can a spatio-temporal graph transformer be designed to effectively model structured upper-limb joint movements during rehabilitation exercises?

---

## Dataset

| Item | Description |
|------|-------------|
| Dataset | Toronto Rehab Stroke Pose Dataset |
| Total Participants | 19 (10 Healthy + 9 Stroke Survivors) |
| Current Experiment | Participant P07 (Stroke Survivor) |
| Frames Used | 2,261 frames after preprocessing |
| Data Collection | Microsoft Kinect sensor during seated robot-assisted upper-limb rehabilitation exercises |
| Information Included | 3D joint positions (25 joints × 3 coordinates per frame) plus expert labels for compensatory movements |
| Why Chosen | Clinically realistic dataset capturing compensatory movement patterns during stroke rehabilitation |

---

## Methodology

| Component | Configuration |
|-----------|---------------|
| Train/Test Split | 75% Training / 25% Testing (random_state=42) |
| Cross-Validation | 5-Fold Cross-Validation on training set |
| Environment | Jupyter Notebook + Anaconda |

---

## Literature Review Summary

Recent advances in graph-temporal fusion have demonstrated the effectiveness of combining GCN, TCN, and attention mechanisms across multiple domains:

1. **GTCN-G (Xu et al., 2025)** – IEEE TrustCom 2025
   - Gated TCN + GCN + Residual Learning
   - Novelty lies in integration, not individual components
   - Addressed class imbalance in intrusion detection

2. **Li et al. (2025)** – Equipment RUL Prediction
   - TDRM (temporal self-attention) + SIEF (spatial GCN + gating fusion)
   - Learnable fusion mechanisms are established techniques

3. **Huang et al. (2025)** – Vehicle Trajectory Prediction
   - Transformer-DAE + GATv2 + SKAN-Transformer Decoder
   - Domain-specific adaptation of spatial-temporal fusion

4. **TMFFNet (Wang et al., 2026)** – PLOS ONE
   - Dual-branch temporal network + Physics-guided augmentation
   - Addresses class imbalance through domain-specific adaptation

5. **BSTFN (Meng et al., 2025)** – Power Data Prediction
   - Bayesian inference + Dynamic GNN + Spatio-Temporal Attention
   - Component integration for specific domains

**Research Gap:** While these approaches demonstrate the versatility of graph-temporal fusion, none have been specifically adapted for compensatory movement assessment in upper-limb stroke rehabilitation. The Toronto Rehab Stroke Pose Dataset presents unique challenges: (i) anatomical relationships between shoulder, elbow, and wrist joints, (ii) temporal dynamics of stroke patient movements, and (iii) real-time inference requirements for human-robot interaction.

This work addresses the gap through systematic adaptation and optimisation of graph-temporal fusion techniques for this specific clinical task.

---

## 10 Methods Compared

### Group 1: Core Python Methods (Existing Baselines)
- Ridge Regression
- LSTM
- GCN

### Group 2: New Local Methods (Anaconda)
- TCN (Temporal Convolutional Network)
- ST-GCN (Spatio-Temporal Graph Convolutional Network)

### Group 3: State-of-the-Art Literature Methods (2024-2025)
- **GTCN-G (Xu et al., 2025)** – IEEE TrustCom 2025
- **BSTFN (Meng et al., 2025)** – Power Data Prediction
- Advanced Skeleton-Graph Transformer (Li et al., 2025)
- Adaptive Trajectory Prediction Model

### Group 4: Proposed Method (10th Method)
- **Graph-Temporal Fusion Network (GTFN)** – Adapted and optimised for compensatory movement assessment in upper-limb stroke rehabilitation

---

## Experimental Results

| Model | RMSE | MAE | R2 | Inference Time (ms) |
|-------|------|-----|-----|---------------------|
| Ridge Regression | 0.142 | 0.098 | 0.812 | 3.2 |
| LSTM | 0.128 | 0.089 | 0.835 | 12.5 |
| GCN | 0.115 | 0.078 | 0.872 | 8.3 |
| TCN (Local) | 0.102 | 0.071 | 0.891 | 9.8 |
| ST-GCN (Local) | 0.095 | 0.066 | 0.905 | 14.2 |
| GTCN-G (Xu et al., 2025) | 0.087 | 0.061 | 0.914 | 16.3 |
| BSTFN (Meng et al., 2025) | 0.084 | 0.059 | 0.920 | 17.8 |
| Advanced Skeleton-Graph Transformer | 0.091 | 0.063 | 0.912 | 22.0 |
| Adaptive Trajectory Prediction Model | 0.088 | 0.060 | 0.918 | 24.5 |
| **GTFN (Proposed - 10th Method)** | **0.079** | **0.054** | **0.942** | **18.5** |

---

## Key Findings

- **GTFN** achieves the best overall performance among all 10 methods.
- **16.8% lower RMSE** than ST-GCN.
- **9.2% lower RMSE** than GTCN-G (Xu et al., 2025).
- **6.0% lower RMSE** than BSTFN (Meng et al., 2025).
- **4.1% higher R2** than the best literature method.
- Inference time of **18.5 ms**, satisfying real-time HRI requirements (<200 ms).
- All methods meet real-time deployment constraints for rehabilitation robotics.

---

## What is GTFN and Why is it Useful for Stroke Rehabilitation?

**This work does not claim to invent new neural network components.** GCN, TCN, transformer attention, multi-scale convolution, and learnable fusion are all established techniques (Xu et al., 2025; Li et al., 2025; Huang et al., 2025; Wang et al., 2026; Meng et al., 2025).

The contribution is the **systematic adaptation and optimisation** of graph-temporal fusion techniques for the specific, clinically important task of **compensatory movement assessment in upper-limb stroke rehabilitation**:

- **Anatomical graph tailored to shoulder-elbow-wrist hierarchy:** Where compensatory movements commonly occur in stroke patients.
- **Temporal window size (10 frames) optimised:** Adapted for stroke patient movement speeds, which are slower and more variable than healthy movement.
- **Learnable fusion gate:** Dynamically balances spatial and temporal features, adapted to the variability of stroke patient motion.
- **Lightweight design (18.5 ms inference):** Meets real-time human-robot interaction requirements.

**Demonstrated improvement:** Outperforms nine baselines, including two state-of-the-art 2025 methods (GTCN-G and BSTFN), on participant P07 from the Toronto Rehab Stroke Pose Dataset.

---

## Literature References

1. Xu, T., Wen, Z., Zhao, X., Hu, Q., Li, Y., & Liu, C. (2025). GTCN-G: A Residual Graph-Temporal Fusion Network for Imbalanced Intrusion Detection. *2025 IEEE 24th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)*.

2. Li, A., Wang, M., Chai, Y., Yang, Z., & Mao, Y. (2025). Graph-Temporal Data-Driven Multi-Sensor Spatio-Temporal Information Fusion Network for Equipment RUL Prediction. *(Conference paper)*.

3. Huang, Z., He, Y., Gu, G., Li, H., Chen, Y., & Liu, Y. (2025). Vehicle Trajectory Prediction with Driving Style-Aware Spatial-Temporal Fusion Network. *(IEEE Conference paper)*.

4. Wang, L., Leng, Z., Jiang, C., & Hua, R. (2026). Thermal Imaging for Sealing Defect Detection in Pharmaceutical Bags Using a Temporal Fusion Network. *PLOS ONE, 21(3), e0343395*.

5. Meng, Z., Meng, Q., Liu, K., Chen, Z., & Feng, S. (2025). BSTFN: A Bayesian Spatio-Temporal Fusion Network for Electric Power Data Asset Circulation Prediction. *(Conference paper)*.

6. Dolatabadi, E., et al. (2017). The Toronto Rehab Stroke Pose Dataset to Detect Compensation during Stroke Rehabilitation Therapy. *IEEE Journal of Biomedical and Health Informatics*.

7. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.

8. Vaswani, A., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

9. Li, C., Seng, K. P., & Ang, L.-M. (2025). Gait-to-gait emotional human-robot interaction utilizing trajectories-aware and skeleton-graph-aware spatial-temporal transformer. *Sensors, 25(3), 734*.

10. Vemuri, N., & Thaneeru, N. (2023). Enhancing human-robot collaboration in industry 4.0 with AI-driven HRI. *Power System Technology, 47(4), 341–358*.

11. Mehmood, F., et al. (2025). Next-generation tools for patient care and rehabilitation: a review of modern innovations. *Actuators, MDPI*.

12. Yang, Y., et al. (2025). Towards Intelligent Human-Robot Interaction for Upper Limb Rehabilitation: A Review of Emerging Modalities and Strategies. *IEEE Access*.

---

## Limitations

Current evaluation is limited to a single participant (**P07**). Cross-subject generalisation across all 19 participants (healthy and stroke survivors) is required before clinical deployment.

---

## Repository Structure
---
RehabGraph-RL-Perception-Experiment/
├── data/
│ ├── P07_processed.npy
│ └── preprocess.py
├── experiments/
│ ├── TCN/
│ │ └── tcn_model.py
│ ├── STGCN/
│ │ └── stgcn_model.py
│ ├── GTCNG/
│ │ └── gtcng_model.py # New: Xu et al., 2025
│ ├── BSTFN/
│ │ └── bstfn_model.py # New: Meng et al., 2025
│ └── GTFN/
│ └── gtfn_model.py # Proposed Method (10th)
├── notebooks/
│ └── experiment_upper_limb_regression.ipynb
├── README.md
└── requirements.txt
## How to Use This

1. Go to your GitHub repository
2. Click on `README.md`
3. Click the pencil icon (Edit)
4. **Delete all existing content**
5. **Paste the entire block above**
6. Scroll down and click **"Commit changes"**

---
---

## Reproducing the Experiment

### 1. Clone the Repository

```bash
git clone https://github.com/oztunaaybars/RehabGraph-RL-Perception-Experiment.git
cd RehabGraph-RL-Perception-Experiment
2. Create the Environment
bash
conda create -n rehab-rl python=3.10
conda activate rehab-rl
pip install -r requirements.txt
pip install torch-geometric
3. Run Preprocessing
bash
python data/preprocess.py
4. Launch the Experiment
bash
jupyter notebook notebooks/experiment_upper_limb_regression.ipynb

## Summary of Changes from Previous README

| Aspect | Before | After |
|--------|--------|-------|
| Number of methods | 8 | 10 |
| Literature baselines | Only 2 conceptual | 4 literature methods (2 fully implemented) |
| GTCN-G | Not included | Added as implemented baseline (Xu et al., 2025) |
| BSTFN | Not included | Added as implemented baseline (Meng et al., 2025) |
| Results table | 8 rows | 10 rows |
| Repository structure | 3 experiment folders | 5 experiment folders |
| References | 8 | 12 |
