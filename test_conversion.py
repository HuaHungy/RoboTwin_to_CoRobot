
import os
import sys
import pandas as pd
import numpy as np
import shutil
import subprocess

def test_conversion():
    # Define paths
    base_dir = "/home/huahungy/RoboTwin_to_CoRobot"
    source_dir = os.path.join(base_dir, "RoboTwin_dataset/aloha_adjustbottle_left_source")
    corobot_dir = os.path.join(base_dir, "test_output_corobot")
    robotwin_restored_dir = os.path.join(base_dir, "test_output_robotwin_restored")

    # Clean up previous runs
    if os.path.exists(corobot_dir):
        shutil.rmtree(corobot_dir)
    if os.path.exists(robotwin_restored_dir):
        shutil.rmtree(robotwin_restored_dir)

    print("--- Step 1: Running RoboTwin -> CoRobot Conversion ---")
    cmd1 = [
        "python3", "robotwin_to_corobot.py",
        "--source", source_dir,
        "--target", corobot_dir
    ]
    subprocess.check_call(cmd1)

    # Verification 1
    print("\n--- Verifying CoRobot Output ---")
    video_path = os.path.join(corobot_dir, "videos/chunk-000/observation.images.cam_high_rgb/episode_000000.mp4")
    parquet_path = os.path.join(corobot_dir, "data/chunk-000/episode_000000.parquet")
    
    if not os.path.exists(video_path):
        print(f"FAILED: Video file not found: {video_path}")
        return
    if not os.path.exists(parquet_path):
        print(f"FAILED: Parquet file not found: {parquet_path}")
        return
    
    # Check parquet content (should NOT have images)
    df = pd.read_parquet(parquet_path)
    if 'observation.images.cam_high_rgb' in df.columns:
        print("FAILED: Image column found in CoRobot parquet (should be removed)")
        return
    if 'observation.images.cam_high' in df.columns:
        print("FAILED: Old image column found in CoRobot parquet")
        return
        
    # Check if other meta files exist
    meta_files = ['episodes.jsonl', 'tasks.jsonl']
    for mf in meta_files:
        if not os.path.exists(os.path.join(corobot_dir, 'meta', mf)):
             print(f"FAILED: Meta file {mf} not found in CoRobot output")
             return
             
    print("SUCCESS: CoRobot structure looks correct.")

    print("\n--- Step 2: Running CoRobot -> RoboTwin Conversion ---")
    cmd2 = [
        "python3", "corobot_to_robotwin.py",
        "--source", corobot_dir,
        "--target", robotwin_restored_dir
    ]
    subprocess.check_call(cmd2)

    # Verification 2
    print("\n--- Verifying RoboTwin Restored Output ---")
    restored_parquet_path = os.path.join(robotwin_restored_dir, "data/chunk-000/episode_000000.parquet")
    
    if not os.path.exists(restored_parquet_path):
        print(f"FAILED: Restored parquet not found: {restored_parquet_path}")
        return
    
    df_restored = pd.read_parquet(restored_parquet_path)
    
    # Check if image column exists and is struct
    if 'observation.images.cam_high' not in df_restored.columns:
        print("FAILED: Image column missing in restored parquet")
        return
    
    # Check sample
    sample = df_restored['observation.images.cam_high'].iloc[0]
    if not isinstance(sample, dict) or 'bytes' not in sample:
        print(f"FAILED: Image column content is not a dict with bytes: {type(sample)}")
        print(sample)
        return
        
    # Check if other meta files exist in restored
    meta_files = ['episodes.jsonl', 'tasks.jsonl']
    for mf in meta_files:
        if not os.path.exists(os.path.join(robotwin_restored_dir, 'meta', mf)):
             print(f"FAILED: Meta file {mf} not found in Restored output")
             return
    
    print("SUCCESS: RoboTwin structure restored successfully.")
    print("Test Completed.")

if __name__ == "__main__":
    test_conversion()
