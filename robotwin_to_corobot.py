
import os
import sys
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import io
import json
import shutil
from tqdm import tqdm

def process_episode(source_path, target_data_dir, target_video_base, episode_name):
    # Read source parquet
    try:
        df = pd.read_parquet(source_path)
    except Exception as e:
        print(f"Error reading {source_path}: {e}")
        return

    # Columns mapping (Source -> Target Video Folder Name)
    # Note: Target parquet will NOT contain these columns
    image_cols = {
        'observation.images.cam_high': 'observation.images.cam_high_rgb',
        'observation.images.cam_left_wrist': 'observation.images.cam_left_wrist_rgb',
        'observation.images.cam_right_wrist': 'observation.images.cam_right_wrist_rgb'
    }

    # Ensure video directories exist
    for target_col in image_cols.values():
        os.makedirs(os.path.join(target_video_base, target_col), exist_ok=True)

    # Process each camera
    for source_col, target_col in image_cols.items():
        if source_col not in df.columns:
            print(f"Warning: Column {source_col} not found in {source_path}")
            continue

        # Extract images
        frames = []
        for item in df[source_col]:
            # Item is a struct/dict: {'bytes': b'...', 'path': '...'}
            if isinstance(item, dict) and 'bytes' in item:
                img_bytes = item['bytes']
                img = Image.open(io.BytesIO(img_bytes))
                frames.append(np.array(img))
            else:
                # Handle potential PyArrow struct scalar if not converted to dict automatically
                # But pandas usually converts struct to dict
                print(f"Unexpected data type in {source_col}: {type(item)}")
                return

        # Write video using lossless encoding to ensure zero pixel difference
        video_path = os.path.join(target_video_base, target_col, episode_name.replace('.parquet', '.mp4'))
        imageio.mimwrite(video_path, frames, fps=30, codec='libx264rgb', ffmpeg_params=['-crf', '0'])

    # Create target dataframe (drop image columns)
    df_target = df.drop(columns=list(image_cols.keys()))
    
    # Save target parquet
    target_parquet_path = os.path.join(target_data_dir, episode_name)
    df_target.to_parquet(target_parquet_path, index=False)

def convert_robotwin_to_corobot(source_dir, target_dir):
    print(f"Converting RoboTwin -> CoRobot")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    # Setup directories
    source_data_dir = os.path.join(source_dir, 'data', 'chunk-000')
    target_data_dir = os.path.join(target_dir, 'data', 'chunk-000')
    target_video_dir = os.path.join(target_dir, 'videos', 'chunk-000')
    target_meta_dir = os.path.join(target_dir, 'meta')

    os.makedirs(target_data_dir, exist_ok=True)
    os.makedirs(target_video_dir, exist_ok=True)
    os.makedirs(target_meta_dir, exist_ok=True)

    # List parquet files
    if not os.path.exists(source_data_dir):
        print(f"Source data directory not found: {source_data_dir}")
        return

    files = sorted([f for f in os.listdir(source_data_dir) if f.endswith('.parquet')])
    
    for f in tqdm(files, desc="Processing episodes"):
        source_path = os.path.join(source_data_dir, f)
        process_episode(source_path, target_data_dir, target_video_dir, f)

    # Copy other files and folders (recursively)
    # Exclude 'data' (processed above) and potentially 'videos' if it existed in source (unlikely for RoboTwin format but safe to check)
    # For 'meta/info.json', we will handle it specifically, but other meta files should be copied.
    
    print("Copying other files...")
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(target_dir, item)
        
        if item == 'data':
            continue
            
        if os.path.isdir(s):
            # If directory, copy recursively but handle meta separately if needed
            if item == 'meta':
                if not os.path.exists(d):
                    os.makedirs(d)
                # Copy content of meta
                for meta_item in os.listdir(s):
                    if meta_item == 'info.json':
                        continue # Handle info.json later
                    meta_s = os.path.join(s, meta_item)
                    meta_d = os.path.join(d, meta_item)
                    if os.path.isdir(meta_s):
                        if os.path.exists(meta_d):
                            shutil.rmtree(meta_d)
                        shutil.copytree(meta_s, meta_d)
                    else:
                        shutil.copy2(meta_s, meta_d)
            else:
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # Generate/Update metadata (info.json)
    source_info_path = os.path.join(source_dir, 'meta', 'info.json')
    info = {}
    
    if os.path.exists(source_info_path):
        with open(source_info_path, 'r') as f:
            info = json.load(f)
            
        # Update features to reflect video format
        if "features" in info:
            # Remove old image keys and add new ones
            image_cols = {
                'observation.images.cam_high': 'observation.images.cam_high_rgb',
                'observation.images.cam_left_wrist': 'observation.images.cam_left_wrist_rgb',
                'observation.images.cam_right_wrist': 'observation.images.cam_right_wrist_rgb'
            }
            
            for old_key, new_key in image_cols.items():
                if old_key in info['features']:
                    # Copy old info but update dtype to video
                    feat_info = info['features'][old_key]
                    # It might be struct in source, change to video
                    # Or if it's missing detailed info, create default
                    
                    # Create new video entry
                    info['features'][new_key] = {
                        "dtype": "video",
                        "shape": [480, 640, 3], # Default assumption if not present
                        "names": ["height", "width", "channels"]
                    }
                    
                    # Remove old key
                    del info['features'][old_key]
    else:
        # Fallback to creating minimal info if source doesn't exist
        info = {
            "codebase_version": "v2.1",
            "robot_type": "agilex_cobot_decoupled_magic",
            "total_episodes": len(files),
            "total_frames": 0, 
            "fps": 30,
            "features": {
                "observation.images.cam_high_rgb": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"]
                },
                "observation.images.cam_left_wrist_rgb": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"]
                },
                "observation.images.cam_right_wrist_rgb": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"]
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [16], 
                    "names": [] 
                },
                "action": {
                    "dtype": "float32",
                    "shape": [16], 
                    "names": []
                }
            }
        }
    
    with open(os.path.join(target_meta_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboTwin dataset to CoRobot format")
    parser.add_argument("--source", default="/home/huahungy/RoboTwin_to_CoRobot/RoboTwin_dataset/aloha_adjustbottle_left_source", help="Path to source RoboTwin dataset")
    parser.add_argument("--target", default="/home/huahungy/RoboTwin_to_CoRobot/RoboTwin_converted", help="Path to target CoRobot dataset")
    
    args = parser.parse_args()
    convert_robotwin_to_corobot(args.source, args.target)
