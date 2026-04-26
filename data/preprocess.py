import os
import pandas as pd
import numpy as np

def load_joint_positions(file_path):
    """Load and reshape Joint_Positions.csv from Toronto Rehab dataset"""
    df = pd.read_csv(file_path, header=None)
    
    # The file has 3 columns: X, Y, Z stacked
    # Every 25 rows = 1 frame with 25 joints
    num_joints = 25
    data = df.values  # shape: (num_frames * 25, 3)
    
    num_frames = len(data) // num_joints
    if len(data) % num_joints != 0:
        print(f"Warning: Data length not divisible by 25 in {file_path}")
        data = data[:num_frames * num_joints]
    
    # Reshape to (num_frames, num_joints, 3)
    reshaped = data.reshape(num_frames, num_joints, 3)
    print(f"Loaded {file_path}: {num_frames} frames, {num_joints} joints")
    return reshaped

def preprocess_participant(participant_dir):
    """Load all Joint_Positions.csv from a participant subfolders"""
    all_data = []
    
    for root, dirs, files in os.walk(participant_dir):
        for file in files:
            if file == "Joint_Positions.csv":
                file_path = os.path.join(root, file)
                try:
                    poses = load_joint_positions(file_path)
                    all_data.append(poses)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    if all_data:
        combined = np.concatenate(all_data, axis=0)
        print(f"Combined data shape: {combined.shape} (frames, joints, 3)")
        return combined
    return None

# Example usage for P07
if __name__ == "__main__":
    p07_path = "data/P07"   # change if your path is different
    data = preprocess_participant(p07_path)
    if data is not None:
        np.save("data/P07_processed.npy", data)
        print("Saved processed data as P07_processed.npy")
