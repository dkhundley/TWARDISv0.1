import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pandas as pd
from scipy.ndimage import distance_transform_edt
from pathlib import Path

#region [functions]

def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        nb_frames = 0
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            nb_frames += 1
    
    print(f"{nb_frames} frames loaded from {filename}")
    return cleaned_segments

def get_random_unprocessed_video(cleaned_aligned_segments_dir, final_data_dir):
    all_videos = [
        os.path.splitext(d)[0]
        for d in os.listdir(cleaned_aligned_segments_dir)
        if d.endswith("_riasegmentation.h5")
    ]
    if not all_videos:
        raise ValueError(
            f"No RIA segmentation files found in {cleaned_aligned_segments_dir}. Run 3_autoprompted_RIAsegmentation.py first."
        )

    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(final_data_dir, video + ".csv"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All segmentation files already have brightness CSV outputs.")
    
    return os.path.join(cleaned_aligned_segments_dir, random.choice(unprocessed_videos) + ".h5")

### Brightness extraction
def extract_top_percent_brightness(aligned_images, masks_dict, object_id, percent):
    """
    Extracts mean brightness of the top X% brightest pixels from masked regions.
    """
    if not 0 < percent <= 100:
        raise ValueError("Percentage must be between 0 and 100")
        
    data = []
    for frame_idx, image in enumerate(aligned_images):
        if frame_idx in masks_dict and object_id in masks_dict[frame_idx]:
            mask = masks_dict[frame_idx][object_id][0]
            masked_pixels = image[mask]
            n_pixels = int(round(len(masked_pixels) * (percent / 100)))
            n_pixels = max(1, n_pixels)
            top_n_pixels = np.sort(masked_pixels)[-n_pixels:]
            mean_top_percent = np.mean(top_n_pixels)
            
            data.append({
                'frame': frame_idx,
                'mean_top_percent_brightness': mean_top_percent,
                'n_pixels_used': n_pixels,
                'total_pixels': len(masked_pixels),
                'percent_used': percent
            })
    
    return pd.DataFrame(data)

def get_background_sample(frame_masks, image_shape, num_samples=100, min_distance=40):
    combined_mask = np.zeros(image_shape[1:], dtype=bool)
    for mask in frame_masks.values():
        combined_mask |= mask.squeeze()
    
    distance_map = distance_transform_edt(~combined_mask)
    valid_bg = (distance_map >= min_distance)
    valid_coords = np.column_stack(np.where(valid_bg))
    
    if len(valid_coords) < num_samples:
        print(f"Warning: Only {len(valid_coords)} valid background pixels found. Sampling all of them.")
        return valid_coords
    else:
        sampled_indices = random.sample(range(len(valid_coords)), num_samples)
        return valid_coords[sampled_indices]

def load_image(frame_idx):
    """
    Load a single frame image corresponding to the currently selected video.
    """
    video_basename = os.path.splitext(os.path.basename(filename))[0]
    if video_basename.endswith("_riasegmentation"):
        video_basename = video_basename[: -len("_riasegmentation")]
    image_folder = os.path.join(video_dir, video_basename)
    image_path = os.path.join(image_folder, f"{frame_idx:06d}.jpg")
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def count_mask_pixels(masks):
    pixel_counts = {}
    for obj_id, mask in masks.items():
        pixel_counts[obj_id] = np.sum(mask)
    return pixel_counts

def calculate_mean_values_and_pixel_counts(image, masks, background_coordinates):
    mean_values = {}
    pixel_counts = count_mask_pixels(masks)
    
    bg_pixel_values = image[background_coordinates[:, 0], background_coordinates[:, 1]]
    mean_values['background'] = np.mean(bg_pixel_values)
    
    for obj_id, mask in masks.items():
        mask_pixel_values = image[mask.squeeze()]
        mean_values[obj_id] = np.mean(mask_pixel_values)
    
    return mean_values, pixel_counts

def create_wide_format_table_with_bg_correction_and_pixel_count(mean_values, pixel_counts):
    data = {'frame': []}
    
    all_objects = set()
    for frame_data in mean_values.values():
        all_objects.update(frame_data.keys())
    all_objects.remove('background')
    
    for obj in all_objects:
        data[obj] = []
        data[f"{obj}_bg_corrected"] = []
        data[f"{obj}_pixel_count"] = []
    
    for frame_idx, frame_data in mean_values.items():
        data['frame'].append(frame_idx)
        bg_value = frame_data['background']
        frame_pixel_counts = pixel_counts[frame_idx]
        for obj in all_objects:
            obj_value = frame_data.get(obj, np.nan)
            data[obj].append(obj_value)
            
            if pd.notnull(obj_value):
                bg_corrected = obj_value - bg_value
            else:
                bg_corrected = np.nan
            data[f"{obj}_bg_corrected"].append(bg_corrected)
            
            data[f"{obj}_pixel_count"].append(frame_pixel_counts.get(obj, 0))
    
    df = pd.DataFrame(data)
    
    return df

def process_cleaned_segments(cleaned_segments):
    first_frame = next(iter(cleaned_segments.values()))
    first_mask = next(iter(first_frame.values()))
    image_shape = first_mask.shape

    mean_values = {}
    pixel_counts = {}

    for frame_idx, frame_masks in tqdm(cleaned_segments.items(), desc="Processing frames"):
        bg_coordinates = get_background_sample(frame_masks, image_shape)
        image = load_image(frame_idx)
        
        if image is None:
            print(f"Warning: Could not load image for frame {frame_idx}")
            continue
        
        frame_mean_values, frame_pixel_counts = calculate_mean_values_and_pixel_counts(image, frame_masks, bg_coordinates)
        mean_values[frame_idx] = frame_mean_values
        pixel_counts[frame_idx] = frame_pixel_counts

    df_wide_bg_corrected = create_wide_format_table_with_bg_correction_and_pixel_count(mean_values, pixel_counts)
    df_wide_bg_corrected.columns = df_wide_bg_corrected.columns.astype(str)

    if 'background' not in df_wide_bg_corrected.columns:
        background_values = [frame_data['background'] for frame_data in mean_values.values()]
        df_wide_bg_corrected['background'] = background_values

    background_column = ['background']
    original_columns = [col for col in df_wide_bg_corrected.columns if not col.endswith('_bg_corrected') and not col.endswith('_pixel_count') and col != 'frame']
    original_columns = [col for col in original_columns if col != 'background']
    bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_bg_corrected')]
    pixel_count_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_pixel_count')]

    all_columns = ['frame'] + background_column + original_columns + bg_corrected_columns + pixel_count_columns

    print(df_wide_bg_corrected[all_columns].describe())

    return df_wide_bg_corrected

### Side extraction
def get_centroid(mask):
    y_indices, x_indices = np.where(mask[0])
    if len(x_indices) == 0:
        return None
    
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)
    return (centroid_x, centroid_y)

def get_relative_position(first_frame):
    """
    Get left/right orientation from available object masks in the first frame.
    Uses object IDs 2 and 4 when available; otherwise falls back to the two
    lowest numeric object IDs that have non-empty centroids.
    """
    numeric_keys = [key for key in first_frame.keys() if isinstance(key, (int, np.integer))]

    preferred_pair = (2, 4)
    if all(key in first_frame for key in preferred_pair):
        left_ref_id, right_ref_id = preferred_pair
    else:
        if len(numeric_keys) < 2:
            return "unknown"
        sorted_keys = sorted(numeric_keys)
        left_ref_id, right_ref_id = sorted_keys[0], sorted_keys[1]

    centroid_left_ref = get_centroid(first_frame[left_ref_id])
    centroid_right_ref = get_centroid(first_frame[right_ref_id])

    if centroid_left_ref is None or centroid_right_ref is None:
        return "unknown"

    if centroid_right_ref[0] < centroid_left_ref[0]:
        return "left"
    else:
        return "right"
    
def save_brightness_and_side_data(df_wide_brightness_and_background, cleaned_segments, filename, final_data_dir):
    position = get_relative_position(next(iter(cleaned_segments.values())))
    df_wide_brightness_and_background['side_position'] = position

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_filename = os.path.join(final_data_dir, base_filename + ".csv")

    df_wide_brightness_and_background.to_csv(output_filename, index=False)

    print(df_wide_brightness_and_background.describe())
    print("side_position unique values:", df_wide_brightness_and_background['side_position'].unique())
    print(df_wide_brightness_and_background['side_position'].value_counts())
    print(f"Data saved to: {output_filename}")

    return df_wide_brightness_and_background

#endregion [functions]

project_root = Path(__file__).resolve().parents[1]
ria_base_dir = project_root / "data/processed_files/RIA_calcium_imaging"
segments_dir = ria_base_dir / "segmentation_outputs"
final_data_dir = ria_base_dir / "final_data"
video_dir = ria_base_dir / "crop_outputs"

for required_dir in [segments_dir, video_dir]:
    if not required_dir.exists():
        raise FileNotFoundError(f"Required directory not found: {required_dir}")
final_data_dir.mkdir(parents=True, exist_ok=True)

segments_dir = str(segments_dir)
final_data_dir = str(final_data_dir)
video_dir = str(video_dir)

filename = get_random_unprocessed_video(segments_dir, final_data_dir)
cleaned_segments = load_cleaned_segments_from_h5(filename)

df_wide_brightness_and_background = process_cleaned_segments(cleaned_segments)	

df_wide_brightness_and_side = save_brightness_and_side_data(df_wide_brightness_and_background, cleaned_segments, filename, final_data_dir)