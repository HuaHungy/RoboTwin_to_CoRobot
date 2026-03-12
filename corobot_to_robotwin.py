
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

# Standard RoboTwin columns in order
ROBOTWIN_COLUMNS = [
    'observation.state',
    'action',
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
    'timestamp',
    'frame_index',
    'episode_index',
    'index',
    'task_index'
]

# Mapping from source video column (in info.json or folder name) to target parquet column
IMAGE_COLS_MAP = {
    'observation.images.cam_high_rgb': 'observation.images.cam_high',
    'observation.images.cam_left_wrist_rgb': 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist_rgb': 'observation.images.cam_right_wrist'
}

def process_episode(source_parquet_path, source_video_base, target_data_dir, episode_name):
    # Read source parquet (state only)
    try:
        df = pd.read_parquet(source_parquet_path)
    except Exception as e:
        print(f"Error reading {source_parquet_path}: {e}")
        return

    # Iterate over cameras
    for source_vid_col, target_col in IMAGE_COLS_MAP.items():
        # Construct video path
        # videos/chunk-000/observation.images.cam_high_rgb/episode_000000.mp4
        video_path = os.path.join(source_video_base, source_vid_col, episode_name.replace('.parquet', '.mp4'))
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found at {video_path}")
            # Fill with None or empty struct? Better to skip or fill empty
            df[target_col] = None
            continue

        # Read video frames
        try:
            reader = imageio.get_reader(video_path, 'ffmpeg')
            frames = []
            for i, frame in enumerate(reader):
                # Convert frame (numpy array) to PNG bytes
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                img_bytes = buf.getvalue()
                
                # Create struct: {'bytes': ..., 'path': ...}
                # Path format in RoboTwin: frame_000000.png
                frame_path = f"frame_{i:06d}.png"
                frames.append({'bytes': img_bytes, 'path': frame_path})
            
            # Check length match
            if len(frames) != len(df):
                print(f"Warning: Frame count mismatch for {video_path}. Video: {len(frames)}, DF: {len(df)}")
                # Truncate or pad? 
                if len(frames) > len(df):
                    frames = frames[:len(df)]
                else:
                    # Pad with last frame or None
                    frames += [frames[-1]] * (len(df) - len(frames))

            df[target_col] = frames

        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            df[target_col] = None

    # Filter and reorder columns
    final_cols = []
    for col in ROBOTWIN_COLUMNS:
        if col in df.columns:
            final_cols.append(col)
        else:
            print(f"Warning: Missing column {col} in {episode_name}")
            # Optional: Add missing column with default values if critical?
            # For now, we skip it but this might cause schema issues if inconsistent across files.
            # If standard requires it, we should probably add it.
            if col == 'index':
                 df[col] = df.index
                 final_cols.append(col)
            elif col == 'task_index':
                 # Default to 0 if missing
                 df[col] = 0
                 final_cols.append(col)
            else:
                 pass # Skip other missing columns

    # Select only the relevant columns
    df_final = df[final_cols]

    # Detect dimensions for schema
    if len(df_final) > 0:
        state_dim = len(df_final['observation.state'].iloc[0]) if 'observation.state' in df_final.columns else 0
        action_dim = len(df_final['action'].iloc[0]) if 'action' in df_final.columns else 0
    else:
        state_dim = 0
        action_dim = 0
    
    # Define PyArrow schema to enforce fixed_size_list and specific struct for images
    fields = []
    # Only include columns that are actually in df_final
    for col in df_final.columns:
        if col == 'observation.state':
            fields.append((col, pa.list_(pa.float32(), state_dim)))
        elif col == 'action':
            fields.append((col, pa.list_(pa.float32(), action_dim)))
        elif col.startswith('observation.images.'):
            fields.append((col, pa.struct([
                ('bytes', pa.binary()),
                ('path', pa.string())
            ])))
        elif col == 'timestamp':
            fields.append((col, pa.float32()))
        elif col in ['frame_index', 'episode_index', 'index', 'task_index']:
            fields.append((col, pa.int64()))
            
    schema = pa.schema(fields)

    # Convert to PyArrow Table with schema
    table = pa.Table.from_pandas(df_final, schema=schema, preserve_index=False)

    # Save target parquet
    target_parquet_path = os.path.join(target_data_dir, episode_name)
    pq.write_table(table, target_parquet_path)

def convert_corobot_to_robotwin(source_dir, target_dir):
    print(f"Converting CoRobot -> RoboTwin")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    # Setup directories
    source_data_dir = os.path.join(source_dir, 'data', 'chunk-000')
    source_video_dir = os.path.join(source_dir, 'videos', 'chunk-000')
    
    target_data_dir = os.path.join(target_dir, 'data', 'chunk-000')
    target_meta_dir = os.path.join(target_dir, 'meta')

    os.makedirs(target_data_dir, exist_ok=True)
    os.makedirs(target_meta_dir, exist_ok=True)

    # List parquet files
    if not os.path.exists(source_data_dir):
        print(f"Source data directory not found: {source_data_dir}")
        return

    files = sorted([f for f in os.listdir(source_data_dir) if f.endswith('.parquet')])
    
    for f in tqdm(files, desc="Processing episodes"):
        source_path = os.path.join(source_data_dir, f)
        process_episode(source_path, source_video_dir, target_data_dir, f)

    # Copy other files and folders (recursively)
    print("Copying other files...")
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(target_dir, item)
        
        # Skip directories we processed manually
        if item in ['data', 'videos']:
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

    # Copy/Generate metadata
    source_info_path = os.path.join(source_dir, 'meta', 'info.json')
    if os.path.exists(source_info_path):
        with open(source_info_path, 'r') as f:
            info = json.load(f)
        
        # Filter features to only keep ROBOTWIN_COLUMNS
        new_features = {}
        
        if "features" in info:
            # First, handle non-image columns
            for col in ROBOTWIN_COLUMNS:
                if col in info['features']:
                    new_features[col] = info['features'][col]
                
                # Check mapping for image columns
                # Iterate through map to find if 'col' is a target
                for src_key, tgt_key in IMAGE_COLS_MAP.items():
                    if tgt_key == col and src_key in info['features']:
                        # Create new image feature definition matching RoboTwin standard
                        # Standard RoboTwin uses dtype: image, shape: [3, 480, 640], names: [channels, height, width]
                        new_features[col] = {
                            "dtype": "image",
                            "shape": [3, 480, 640],
                            "names": ["channels", "height", "width"]
                        }
                        # We do NOT copy the video info as it is now struct of images
                        break
        
        info['features'] = new_features
        
        with open(os.path.join(target_meta_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4)
    else:
        print("Warning: Source info.json not found. Metadata not generated.")

    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CoRobot dataset to RoboTwin format")
    parser.add_argument("--source", default="/home/huahungy/RoboTwin_to_CoRobot/RoboTwin_converted", help="Path to source CoRobot dataset")
    parser.add_argument("--target", default="/home/huahungy/RoboTwin_to_CoRobot/RoboTwin_restored", help="Path to target RoboTwin dataset")
    
    args = parser.parse_args()
    convert_corobot_to_robotwin(args.source, args.target)
