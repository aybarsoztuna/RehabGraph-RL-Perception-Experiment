import os
import pandas as pd
import numpy as np

def load_joint_positions(file_path):
    """Load Joint_Positions.csv and reshape to (num_frames, 25 joints, 3 coordinates)"""
    df = pd.read_csv(file_path, header=None)
    data = df.values.astype(np.float32)  # shape: (num_rows, 3)
    
    num_joints = 25
    if len(data) % num_joints != 0:
        print(f"Warning: Truncating data in {file_path}")
        data = data[: (len(data) // num_joints) * num_joints]
    
    num_frames = len(data) // num_joints
    # Reshape to (frames, joints, xyz)
    poses = data.reshape(num_frames, num_joints, 3)
    
    print(f"Loaded {os.path.basename(file_path)}: {num_frames} frames, {num_joints} joints")
    return poses

def load_participant_data(participant_path):
    """Load all Joint_Positions.csv from a participant (including subfolders)"""
    all_poses = []
    
    for root, _, files in os.walk(participant_path):
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
        print(f"Total combined shape for participant: {combined.shape} (frames, joints, 3)")
        return combined
    else:
        print("No Joint_Positions.csv found!")
        return None

# Test with P07
if __name__ == "__main__":
    p07_dir = "data/P07"   # Adjust if your path is different
    data = load_participant_data(p07_dir)
    
    if data is not None:
        # Save processed data
        save_path = "data/P07_processed.npy"
        np.save(save_path, data)
        print(f"✅ Saved processed data to {save_path}")
        print(f"Shape: {data.shape}")
