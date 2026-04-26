import os
import pandas as pd
import numpy as np

def load_joint_positions(file_path):
    """Load and reshape Joint_Positions.csv"""
    df = pd.read_csv(file_path, header=None)
    data = df.values.astype(np.float32)
    
    num_joints = 25
    if len(data) % num_joints != 0:
        print(f"Warning: Truncating data in {file_path}")
        data = data[:(len(data)//num_joints)*num_joints]
    
    num_frames = len(data) // num_joints
    poses = data.reshape(num_frames, num_joints, 3)
    
    print(f"Loaded {os.path.basename(file_path)} → {num_frames} frames, {num_joints} joints")
    return poses

def load_participant_data(participant_path):
    """Load all Joint_Positions.csv from P07 and its subfolders"""
    all_poses = []
    
    for root, dirs, files in os.walk(participant_path):
        for file in files:
            if file.lower() == "joint_positions.csv":
                full_path = os.path.join(root, file)
                try:
                    poses = load_joint_positions(full_path)
                    all_poses.append(poses)
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
    
    if all_poses:
        combined = np.concatenate(all_poses, axis=0)
        print(f"\n✅ Successfully combined data from P07: {combined.shape} (frames, joints, xyz)")
        return combined
    else:
        print("❌ No Joint_Positions.csv files found!")
        return None

# === RUN FOR P07 ===
if __name__ == "__main__":
    # Change this path to match your computer
    p07_dir = r"C:\Users\ADMIN\Downloads\RehabGraph-Transformer-UpperLimb\data\P07"
    
    print("Starting preprocessing for P07...")
    data = load_participant_data(p07_dir)
    
    if data is not None:
        save_path = r"C:\Users\ADMIN\Downloads\RehabGraph-Transformer-UpperLimb\data\P07_processed.npy"
        np.save(save_path, data)
        print(f"✅ Saved processed data to: {save_path}")
        print(f"Final shape: {data.shape}")
