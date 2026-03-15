"""
Unified multi-worm feature extraction pipeline runner.

This script mirrors the UX style of RIA's run_ria_pipeline.py:
- one CLI entrypoint
- deterministic output directory layout
- per-image logs
- run manifest JSON with result/status metadata
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import skimage
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from scipy.ndimage import convolve
from skimage.measure import label
from torchvision import transforms


@dataclass
class ImageResult:
    image_path: str
    status: str
    worm_count: int
    log_file: str
    message: str


@dataclass
class ModelContext:
    mask_generator: object
    classifier_model: Any
    classifier_device: torch.device
    class_names: List[str]
    data_transform: Optional[transforms.Compose]
    classifier_preprocessor: Optional[Any]
    classifier_backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-worm cutout extraction + metrics pipeline from one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        default="data/processed_files/multiworm_feature_extraction/converted_images",
        help="Folder containing converted image files to process.",
    )
    parser.add_argument(
        "--output-base",
        default="data/processed_files/multiworm_feature_extraction",
        help="Base folder where outputs and run artifacts are written.",
    )
    parser.add_argument(
        "--sam2-repo",
        default="sam2",
        help="Path to local SAM2 repository root (contains sam2 package).",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="models/sam2_hiera_large.pt",
        help="Path to SAM2 checkpoint file.",
    )
    parser.add_argument(
        "--sam2-config",
        default="sam2_hiera_l.yaml",
        help="SAM2 config name (preferred) or path to SAM2 config yaml.",
    )
    parser.add_argument(
        "--classifier-weights",
        default=os.environ.get("WORM_CLASSIFIER_WEIGHTS", ""),
        help="Path to worm-vs-not-worm classifier .pth weights (required for torchvision backend).",
    )
    parser.add_argument(
        "--classifier-source",
        choices=["huggingface", "torchvision"],
        default="huggingface",
        help="Classifier backend: direct Hugging Face model or local torchvision .pth state dict.",
    )
    parser.add_argument(
        "--classifier-hf-repo",
        default="lillyguisnet/celegans-classifier-vit-h-14-finetuned",
        help="Hugging Face repo ID for classifier when --classifier-source=huggingface.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="How many cycles to run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining images/cycles when an image fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have a metrics pickle in output.",
    )
    parser.add_argument(
        "--valid-margin",
        type=int,
        default=5,
        help="Erosion margin for valid imaging area detection.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.95,
        help="Overlap threshold for merging worm masks.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=25,
        help="Minimum mask area to keep after cleaning.",
    )
    parser.add_argument(
        "--area-filter-threshold",
        type=float,
        default=0.75,
        help="Relative area threshold used to keep detected worms.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Execution device for SAM2 + classifier.",
    )
    return parser.parse_args()


def discover_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def ensure_directories(project_root: Path, output_base: Path) -> Dict[str, Path]:
    dirs = {
        "project_root": project_root,
        "base": output_base,
        "temp_cutouts": output_base / "temp_cutouts",
        "final_cutouts": output_base / "final_cutouts",
        "metrics": output_base / "metrics",
        "pipeline_runs": output_base / "pipeline_runs",
    }
    for key in ["temp_cutouts", "final_cutouts", "metrics", "pipeline_runs"]:
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def normalize_sam2_config_name(sam2_repo: Path, sam2_config: Path) -> str:
    config_candidate = Path(sam2_config)
    package_root = (sam2_repo / "sam2").resolve()
    legacy_config_dir = package_root / "configs" / "sam2"

    def map_legacy_name(name: str) -> str:
        normalized_name = Path(name).name
        source_candidates = [
            package_root / normalized_name,
            sam2_repo.parent / "models" / normalized_name,
            sam2_repo / "models" / normalized_name,
        ]
        source_path = next((candidate for candidate in source_candidates if candidate.exists()), None)
        if source_path is not None:
            legacy_config_dir.mkdir(parents=True, exist_ok=True)
            target = legacy_config_dir / normalized_name
            if not target.exists():
                shutil.copy2(source_path, target)
            return f"configs/sam2/{normalized_name}"
        return normalized_name

    if config_candidate.exists():
        resolved = config_candidate.resolve()
        try:
            relative = resolved.relative_to(package_root).as_posix()
            if "/" not in relative and relative.endswith(".yaml"):
                return map_legacy_name(relative)
            return relative
        except ValueError:
            return map_legacy_name(resolved.name)

    config_text = str(sam2_config).strip()
    if not config_text:
        raise ValueError("SAM2 config value is empty.")

    if "/" in config_text or "\\" in config_text:
        return map_legacy_name(config_text)

    return map_legacy_name(config_text)


def configure_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device=cuda, but CUDA is not available.")
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda:0")

    if requested_device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Requested --device=mps, but MPS is not available.")
        return torch.device("mps")

    if requested_device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device selection: {requested_device}")


def build_model_context(
    sam2_repo: Path,
    sam2_config: Path,
    sam2_checkpoint: Path,
    classifier_weights: Optional[Path],
    classifier_source: str,
    classifier_hf_repo: str,
    requested_device: str,
) -> ModelContext:
    sam2_repo = sam2_repo.resolve()
    if not sam2_repo.exists():
        raise FileNotFoundError(f"SAM2 repo path not found: {sam2_repo}")
    if not sam2_checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")
    if classifier_source == "torchvision":
        if classifier_weights is None:
            raise ValueError("--classifier-weights is required for --classifier-source=torchvision")
        if not classifier_weights.exists():
            raise FileNotFoundError(f"Classifier weights not found: {classifier_weights}")

    sys.path.insert(0, str(sam2_repo))
    sam2_automatic_module = importlib.import_module("sam2.automatic_mask_generator")
    sam2_build_module = importlib.import_module("sam2.build_sam")
    SAM2AutomaticMaskGenerator = sam2_automatic_module.SAM2AutomaticMaskGenerator
    build_sam2 = sam2_build_module.build_sam2

    classifier_device = configure_device(requested_device)
    sam2_device = "cuda" if classifier_device.type == "cuda" else classifier_device.type
    sam2_config_name = normalize_sam2_config_name(sam2_repo, sam2_config)
    config_candidates = [sam2_config_name]
    config_basename = Path(sam2_config_name).name
    legacy_candidate = f"configs/sam2/{config_basename}"
    if legacy_candidate not in config_candidates:
        config_candidates.append(legacy_candidate)

    sam2_model = None
    selected_config_name = None
    last_error: Optional[Exception] = None
    for candidate in config_candidates:
        try:
            sam2_model = build_sam2(
                candidate,
                str(sam2_checkpoint),
                device=sam2_device,
                apply_postprocessing=False,
            )
            selected_config_name = candidate
            break
        except Exception as exc:
            if "Cannot find primary config" in str(exc):
                last_error = exc
                continue
            if sam2_device == "mps":
                try:
                    sam2_model = build_sam2(
                        candidate,
                        str(sam2_checkpoint),
                        device="cpu",
                        apply_postprocessing=False,
                    )
                    print(f"Warning: SAM2 failed on MPS ({exc}). Falling back to CPU.")
                    classifier_device = torch.device("cpu")
                    selected_config_name = candidate
                    break
                except Exception as cpu_exc:
                    last_error = cpu_exc
                    continue
            last_error = exc
            continue

    if sam2_model is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to initialize SAM2 with any config candidate.")

    print(f"Using SAM2 config: {selected_config_name}")

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.85,
        stability_score_offset=0.85,
    )

    classifier_preprocessor = None
    if classifier_source == "huggingface":
        try:
            transformers_module = importlib.import_module("transformers")
            AutoImageProcessor = transformers_module.AutoImageProcessor
            AutoModelForImageClassification = transformers_module.AutoModelForImageClassification
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for --classifier-source=huggingface. "
                "Install with: pip install transformers"
            ) from exc

        classifier_preprocessor = AutoImageProcessor.from_pretrained(classifier_hf_repo)
        classifier_model = AutoModelForImageClassification.from_pretrained(classifier_hf_repo)
        classifier_model = classifier_model.to(classifier_device)
        classifier_model.eval()

        id2label = getattr(classifier_model.config, "id2label", {}) or {}
        ordered_labels = [id2label.get(index, f"LABEL_{index}") for index in range(classifier_model.config.num_labels)]
        class_names: List[str] = []
        for index, label_name in enumerate(ordered_labels):
            normalized = str(label_name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            if "worm" in normalized and ("not" in normalized or "noworm" in normalized or "non" in normalized):
                class_names.append("notworm")
            elif "worm" in normalized:
                class_names.append("worm_any")
            elif index == 1:
                class_names.append("worm_any")
            else:
                class_names.append("notworm")

        if len(class_names) < 2:
            raise RuntimeError("Hugging Face classifier must expose at least 2 labels for worm classification.")

        data_transform = None
    else:
        classif_weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        classifier_model = torchvision.models.vit_h_14(weights=classif_weights)
        num_features = classifier_model.heads.head.in_features
        classifier_model.heads.head = nn.Linear(num_features, 2)
        classifier_model = classifier_model.to(classifier_device)
        classifier_model.load_state_dict(torch.load(str(classifier_weights), map_location=classifier_device))
        classifier_model.eval()

        data_transform = transforms.Compose(
            [
                transforms.Resize(518),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        class_names = ["notworm", "worm_any"]

    return ModelContext(
        mask_generator=mask_generator,
        classifier_model=classifier_model,
        classifier_device=classifier_device,
        class_names=class_names,
        data_transform=data_transform,
        classifier_preprocessor=classifier_preprocessor,
        classifier_backend=classifier_source,
    )

def show_anns(anns, borders=True):
    import matplotlib.pyplot as plt

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

def is_on_edge(x, y, w, h, img_width, img_height):
    # Check left edge
    if x <= 0:
        return True
    # Check top edge
    if y <= 0:
        return True
    # Check right edge
    if (x + w) >= img_width - 1:
        return True
    # Check bottom edge
    if (y + h) >= img_height - 1:
        return True
    return False

def get_valid_imaging_area(image, margin=5, max_iterations=100):
    """
    Find the actual microscope field of view in the image.
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No valid imaging area found")
        return np.ones_like(gray, dtype=bool), False
    
    # Find the largest contour that's not the entire image
    valid_contours = [cnt for cnt in contours 
                     if 0.1 < cv2.contourArea(cnt) / (gray.shape[0] * gray.shape[1]) < 0.99]
    
    if not valid_contours:
        print("Warning: No valid contours found within acceptable size range")
        return np.ones_like(gray, dtype=bool), False
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Create mask of valid area
    valid_area_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(valid_area_mask, [largest_contour], -1, 255, -1)
    
    # Erode the mask by margin pixels with iteration limit
    if margin > 0:
        kernel = np.ones((3, 3), np.uint8)  # Using smaller kernel for more controlled erosion
        eroded_mask = valid_area_mask.copy()
        for _ in range(min(margin, max_iterations)):
            temp_mask = cv2.erode(eroded_mask, kernel)
            if np.sum(temp_mask) < 1000:
                break
            eroded_mask = temp_mask
        valid_area_mask = eroded_mask
    
    return valid_area_mask > 0, True

def get_nonedge_masks(img_path: Path, mask_generator, margin: int):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Unable to read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_height, img_width = image.shape[:2]

    # Generate masks
    masks2 = mask_generator.generate(image)

    valid_area, success = get_valid_imaging_area(image, margin=margin)
    
    nonedge_masks = []

    if success:
        # Use valid area method
        for mask in masks2:
            segmentation = mask['segmentation']
            if np.all(segmentation * valid_area == segmentation):
                nonedge_masks.append(segmentation)
    else:
        # Fall back to simple edge detection
        print(f"Falling back to simple edge detection for {img_path}")
        for mask in masks2:
            segmentation = mask['segmentation']
            coords = np.where(segmentation)
            y1, x1 = np.min(coords[0]), np.min(coords[1])
            y2, x2 = np.max(coords[0]), np.max(coords[1])
            h, w = (y2 - y1 + 1), (x2 - x1 + 1)
            
            if not is_on_edge(x1, y1, w, h, img_width, img_height):
                nonedge_masks.append(segmentation)

    return image, img_height, img_width, nonedge_masks


def save_mask_cutouts(image, nonedge_masks, output_dir: Path):
    """
    Save cutouts of the masks from the image to the specified directory.
    Refreshes the output directory each time.
    """
    # Refresh temp directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(nonedge_masks)} non-edge cutouts to")
    
    for i, mask in enumerate(nonedge_masks):
        # Get bounding box coordinates of the mask
        coords = np.where(mask)
        y1, x1 = np.min(coords[0]), np.min(coords[1])
        y2, x2 = np.max(coords[0]), np.max(coords[1])
        
        # Create a 3D mask by repeating the 2D mask for each color channel
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # Apply mask to original image
        cutout = image * mask_3d
        
        # Crop to bounding box
        cutout = cutout[y1:y2+1, x1:x2+1]
        
        # Save the cutout as jpg
        cutout_path = output_dir / f"{i}.jpg"
        cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))


def classify_cutouts(nonedge_masks, model_ctx: ModelContext, cutouts_dir: Path):
    """
    Classify each cutout image as either 'worm' or 'not worm' using the pre-trained classifier.
    """
    classifications = []
    for i in range(len(nonedge_masks)):
        cutout_path = cutouts_dir / f"{i}.jpg"
        imgg = Image.open(cutout_path)
        if model_ctx.classifier_backend == "huggingface":
            if model_ctx.classifier_preprocessor is None:
                raise RuntimeError("Hugging Face classifier preprocessor is not initialized.")

            inputs = model_ctx.classifier_preprocessor(images=imgg.convert("RGB"), return_tensors="pt")
            inputs = {key: value.to(model_ctx.classifier_device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = model_ctx.classifier_model(**inputs).logits
                pred_index = int(torch.argmax(logits, dim=1).item())
        else:
            if model_ctx.data_transform is None:
                raise RuntimeError("Torchvision classifier transform is not initialized.")
            imgg = model_ctx.data_transform(imgg)
            imgg = imgg.unsqueeze(0)
            imgg = imgg.to(model_ctx.classifier_device)

            with torch.no_grad():
                outputs = model_ctx.classifier_model(imgg)
                _, preds = torch.max(outputs, 1)
                pred_index = int(preds.item())

        if pred_index >= len(model_ctx.class_names):
            pred_index = 1 if pred_index > 0 else 0
        classifications.append(model_ctx.class_names[pred_index])

    return classifications


def merge_and_clean_worm_masks(classifications, nonedge_masks, overlap_threshold=0.95, min_area=25):
    """
    Merge overlapping worm masks and clean the results by removing small regions and keeping only the largest connected component.
    Also checks for and removes masks with holes.
    """
    worm_masks = []
    for i, classification in enumerate(classifications):
        if classification == "worm_any":
            worm_masks.append(nonedge_masks[i])

    if worm_masks:
        # Initialize list to track which masks have been merged
        merged_masks = []
        final_masks = []
        
        # Compare each mask with every other mask
        for i in range(len(worm_masks)):
            if i in merged_masks:
                continue
                
            current_mask = worm_masks[i]
            current_area = np.sum(current_mask)
            for j in range(i + 1, len(worm_masks)):
                if j in merged_masks:
                    continue
                    
                other_mask = worm_masks[j]
                # Calculate overlap
                overlap = np.sum(current_mask & other_mask)
                overlap_ratio = overlap / min(current_area, np.sum(other_mask))
                
                # If overlap is more than threshold, merge the masks
                if overlap_ratio > overlap_threshold:
                    current_mask = current_mask | other_mask
                    current_area = np.sum(current_mask)
                    merged_masks.append(j)
            
            final_masks.append(current_mask)
        
        # Clean final masks - remove regions smaller than min_area pixels and handle discontinuous segments
        worm_masks = []
        for i, mask in enumerate(final_masks):
            if np.sum(mask) >= min_area:
                # Check for holes using contour hierarchy
                contours, hierarchy = cv2.findContours((mask * 255).astype(np.uint8), 
                                                     cv2.RETR_TREE, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                
                has_holes = False
                if hierarchy is not None:
                    hierarchy = hierarchy[0]  # Get the first dimension
                    for h in hierarchy:
                        if h[3] >= 0:  # If has parent, it's a hole
                            has_holes = True
                            print(f"Skipping mask {i} due to holes in the mask")
                            break
                
                if not has_holes:
                    # Find connected components in the mask
                    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
                    
                    if num_labels > 2:  # More than one segment (label 0 is background)
                        # Get sizes of each segment
                        unique_labels, label_counts = np.unique(labels[labels != 0], return_counts=True)
                        # Keep only the largest segment
                        largest_label = unique_labels[np.argmax(label_counts)]
                        mask = (labels == largest_label).astype(np.uint8)
                    
                    worm_masks.append(mask)
        
        num_distinct_worms = len(worm_masks)
        print(f"Number of distinct worm regions: {num_distinct_worms}")
    else:
        num_distinct_worms = 0

    return worm_masks, num_distinct_worms


def filter_worms(allworms_metrics, threshold):
    if not allworms_metrics:
        return []

    area_values = [worm["area"] for worm in allworms_metrics]
    area_mean = float(np.mean(area_values)) if area_values else 0.0
    if area_mean <= 0:
        return []

    filtered_metrics = []
    for worm in allworms_metrics:
        if worm["area"] > threshold * area_mean:
            filtered_metrics.append(worm)
    return filtered_metrics

def extract_worm_metrics(worm_masks, img_path, img_height, img_width, threshold=0.75):
    """
    Extract metrics for each worm mask including area, perimeter, medial axis measurements, etc.
    """
    # Get image ID (filename without extension)
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    
    allworms_metrics = []
    for i, npmask in enumerate(worm_masks):
        print(f"Processing worm {i}")

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((npmask * 255).astype(np.uint8), connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component_mask = (labels == largest_label).astype(np.uint8)

        area = np.sum(largest_component_mask)

        contours, hierarchy = cv2.findContours((largest_component_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        perimeter = cv2.arcLength(contours[0], True)

        # Get medial axis and distance transform
        medial_axis, distance = skimage.morphology.medial_axis(largest_component_mask > 0, return_distance=True)   
        structuring_element = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        neighbours = convolve(medial_axis.astype(np.uint8), structuring_element, mode='constant', cval=0)
        end_points = np.where(neighbours == 11, 1, 0)
        branch_points = np.where(neighbours > 12, 1, 0)
        labeled_branches = label(branch_points, connectivity=2)
        branch_indices = np.argwhere(labeled_branches > 0)
        end_indices = np.argwhere(end_points > 0)
        indices = np.concatenate((branch_indices, end_indices), axis=0)
        
        # Find longest path through medial axis
        if len(indices) == 0:
            continue

        paths = []
        for start in range(len(indices)):
            for end in range(len(indices)):
                startid = tuple(indices[start])
                endid = tuple(indices[end])
                try:
                    route, weight = skimage.graph.route_through_array(np.invert(medial_axis), startid, endid)
                except Exception:
                    continue
                length = len(route)
                paths.append([startid, endid, length, route, weight])

        if not paths:
            continue

        longest_length = max(paths, key=lambda x: x[2])
        pruned_mediala = np.zeros((img_height, img_width), dtype=np.uint8)
        for coord in range(len(longest_length[3])):
            pruned_mediala[longest_length[3][coord]] = 1
            
        # Get measurements along medial axis
        medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
        medialaxis_length_list = 0 + np.arange(0, len(medial_axis_distances_sorted))
        pruned_medialaxis_length = np.sum(pruned_mediala)
        mean_wormwidth = np.mean(medial_axis_distances_sorted)
        mid_length = medial_axis_distances_sorted[int(len(medial_axis_distances_sorted)/2)]

        worm_metrics = {
            "img_id": img_id, 
            "worm_id": i,
            "area": area, 
            "perimeter": perimeter, 
            "medial_axis_distances_sorted": medial_axis_distances_sorted, 
            "medialaxis_length_list": np.ndarray.tolist(medialaxis_length_list), 
            "pruned_medialaxis_length": pruned_medialaxis_length, 
            "mean_wormwidth": mean_wormwidth, 
            "mid_length_width": mid_length,
            "mask": largest_component_mask
        }
        allworms_metrics.append(worm_metrics)
    
    return filter_worms(allworms_metrics, threshold = threshold)


def save_worms(allworms_metrics, original_image=None, cutouts_dir: Path = Path("."), metrics_dir: Path = Path(".")):
    """
    Filter worms by area and save the filtered cutouts and metrics.
    """
    if not allworms_metrics:
        print("No worm metrics provided")
        return []
          
    img_id = allworms_metrics[0]["img_id"]
    
    # Save cutouts of filtered worms for visualization
    for i, worm in enumerate(allworms_metrics):
        cutout_path = cutouts_dir / f"{img_id}_worm_{i}.png"
        cutout_name = f'{img_id}_worm_{i}'  # The name that will appear on the image
        
        if original_image is not None:
            overlay = original_image.copy()
            overlay[worm["mask"] > 0] = [0, 255, 0]  # Green color
            
            alpha = 0.4  # Back to original 40% transparency
            blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(blended, cutout_name, (10, 30), font, 1, (255, 255, 255), 2)
            
            cv2.imwrite(str(cutout_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        else:
            # Fall back to saving just the mask if original image not provided
            cv2.imwrite(str(cutout_path), (worm["mask"] * 255).astype(np.uint8))
    
    # Save metrics as pickle using img_id as filename
    metrics_path = metrics_dir / f"{img_id}.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(allworms_metrics, f)
    print(f"Saved filtered metrics to {metrics_path}")
    
    return allworms_metrics

def image_to_log_name(image_path: Path) -> str:
    return image_path.as_posix().replace("/", "__").replace(":", "_") + ".log"


def process_image(
    image_path: Path,
    model_ctx: ModelContext,
    dirs: Dict[str, Path],
    log_dir: Path,
    valid_margin: int,
    overlap_threshold: float,
    min_area: int,
    area_filter_threshold: float,
    noworms_file: Path,
) -> ImageResult:
    log_path = log_dir / image_to_log_name(image_path)

    try:
        image, img_height, img_width, nonedge_masks = get_nonedge_masks(
            str(image_path), model_ctx.mask_generator, margin=valid_margin
        )

        if len(nonedge_masks) == 0:
            with open(noworms_file, "a", encoding="utf-8") as file_handle:
                file_handle.write(f"{image_path}\n")
            message = "No valid masks found."
            log_path.write_text(message + "\n", encoding="utf-8")
            return ImageResult(str(image_path), "noop", 0, str(log_path), message)

        save_mask_cutouts(image, nonedge_masks, dirs["temp_cutouts"])
        classifications = classify_cutouts(nonedge_masks, model_ctx, dirs["temp_cutouts"])

        worm_masks, num_distinct_worms = merge_and_clean_worm_masks(
            classifications,
            nonedge_masks,
            overlap_threshold=overlap_threshold,
            min_area=min_area,
        )
        if num_distinct_worms == 0:
            with open(noworms_file, "a", encoding="utf-8") as file_handle:
                file_handle.write(f"{image_path}\n")
            message = "No worm masks after merge/cleanup."
            log_path.write_text(message + "\n", encoding="utf-8")
            return ImageResult(str(image_path), "noop", 0, str(log_path), message)

        worm_metrics = extract_worm_metrics(
            worm_masks,
            str(image_path),
            img_height,
            img_width,
            threshold=area_filter_threshold,
        )
        if not worm_metrics:
            with open(noworms_file, "a", encoding="utf-8") as file_handle:
                file_handle.write(f"{image_path}\n")
            message = "All worms filtered out by threshold."
            log_path.write_text(message + "\n", encoding="utf-8")
            return ImageResult(str(image_path), "noop", 0, str(log_path), message)

        save_worms(
            worm_metrics,
            original_image=image,
            cutouts_dir=dirs["final_cutouts"],
            metrics_dir=dirs["metrics"],
        )
        message = f"Processed successfully with {len(worm_metrics)} worms."
        log_path.write_text(message + "\n", encoding="utf-8")
        return ImageResult(str(image_path), "success", len(worm_metrics), str(log_path), message)
    except Exception as exc:
        details = traceback.format_exc()
        log_path.write_text(details, encoding="utf-8")
        return ImageResult(str(image_path), "failed", 0, str(log_path), f"{type(exc).__name__}: {exc}")


def collect_images(input_folder: Path) -> List[Path]:
    image_suffixes = {".jpg", ".jpeg", ".png"}
    return sorted([path for path in input_folder.rglob("*") if path.is_file() and path.suffix.lower() in image_suffixes])


def write_manifest(
    run_dir: Path,
    project_root: Path,
    args: argparse.Namespace,
    results: List[ImageResult],
    output_counts: Dict[str, int],
) -> None:
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "args": vars(args),
        "results": [asdict(result) for result in results],
        "output_counts": output_counts,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def summarize_outputs(dirs: Dict[str, Path], noworms_file: Path) -> Dict[str, int]:
    metrics_files = list(dirs["metrics"].glob("*.pkl"))
    cutout_files = list(dirs["final_cutouts"].glob("*.png"))
    if noworms_file.exists():
        lines = [line for line in noworms_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        no_worm_rows = max(len(lines) - 1, 0)
    else:
        no_worm_rows = 0
    return {
        "metrics_pickle_files": len(metrics_files),
        "final_cutout_png_files": len(cutout_files),
        "noworm_rows": no_worm_rows,
    }


def print_user_summary(
    run_dir: Path,
    requested_device: str,
    input_folder: Path,
    results: List[ImageResult],
    output_counts: Dict[str, int],
) -> None:
    print("\n" + "=" * 72)
    print("Multi-Worm Pipeline Run Summary")
    print("=" * 72)
    print(f"Run folder: {run_dir}")
    print(f"Requested device: {requested_device}")
    print(f"Input folder: {input_folder}")

    print("\nImage status:")
    for result in results:
        print(f"- {Path(result.image_path).name:40s} -> {result.status:7s} (worms={result.worm_count})")
        print(f"  log: {result.log_file}")
        print(f"  note: {result.message}")

    print("\nCurrent output counts:")
    print(f"- metrics pickle files: {output_counts['metrics_pickle_files']}")
    print(f"- final cutout PNG files: {output_counts['final_cutout_png_files']}")
    print(f"- noworm CSV rows: {output_counts['noworm_rows']}")

    print("\nOutput layout:")
    print("- <output-base>/")
    print("  - temp_cutouts/")
    print("  - final_cutouts/")
    print("  - metrics/")
    print("  - noworms.csv")
    print("  - pipeline_runs/<timestamp>/")
    print("    - logs/")
    print("    - run_manifest.json")


def main() -> int:
    args = parse_args()
    if args.cycles < 1:
        print("ERROR: --cycles must be >= 1")
        return 2
    if args.classifier_source == "torchvision" and not args.classifier_weights:
        print("ERROR: --classifier-weights is required when --classifier-source=torchvision.")
        return 2

    project_root = discover_project_root()
    input_folder = resolve_path(project_root, args.input_folder)
    output_base = resolve_path(project_root, args.output_base)
    sam2_repo = resolve_path(project_root, args.sam2_repo)
    sam2_checkpoint = resolve_path(project_root, args.sam2_checkpoint)
    sam2_config_text = str(args.sam2_config).strip()
    if not sam2_config_text:
        print("ERROR: --sam2-config cannot be empty")
        return 2
    if Path(sam2_config_text).is_absolute() or "/" in sam2_config_text or "\\" in sam2_config_text:
        sam2_config = resolve_path(project_root, sam2_config_text)
    else:
        sam2_config = Path(sam2_config_text)
    classifier_weights = resolve_path(project_root, args.classifier_weights) if args.classifier_weights else None

    if not input_folder.exists():
        print(f"ERROR: Input folder does not exist: {input_folder}")
        return 2

    dirs = ensure_directories(project_root, output_base)
    noworms_file = dirs["base"] / "noworms.csv"
    if not noworms_file.exists():
        noworms_file.write_text("image_path\n", encoding="utf-8")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dirs["pipeline_runs"] / run_stamp
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Starting unified multi-worm pipeline...")
    print(f"Project root: {project_root}")
    print(f"Input folder: {input_folder}")
    print(f"Output base: {output_base}")
    print(f"Run folder: {run_dir}")

    try:
        model_ctx = build_model_context(
            sam2_repo,
            sam2_config,
            sam2_checkpoint,
            classifier_weights,
            classifier_source=args.classifier_source,
            classifier_hf_repo=args.classifier_hf_repo,
            requested_device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: Model initialization failed: {exc}")
        return 2

    all_results: List[ImageResult] = []
    for cycle in range(1, args.cycles + 1):
        print(f"\n--- Cycle {cycle}/{args.cycles} ---")
        images = collect_images(input_folder)
        if args.skip_existing:
            images = [image for image in images if not (dirs["metrics"] / f"{image.stem}.pkl").exists()]

        if not images:
            print("No images found for processing in this cycle.")
            continue

        print(f"Images queued: {len(images)}")
        for image_path in images:
            print(f"Processing image: {image_path.name}")
            result = process_image(
                image_path=image_path,
                model_ctx=model_ctx,
                dirs=dirs,
                log_dir=log_dir,
                valid_margin=args.valid_margin,
                overlap_threshold=args.overlap_threshold,
                min_area=args.min_area,
                area_filter_threshold=args.area_filter_threshold,
                noworms_file=noworms_file,
            )
            all_results.append(result)
            print(f"  -> {result.status} (worms={result.worm_count})")
            if result.status == "failed" and not args.continue_on_error:
                print("Stopping early due to image failure. Use --continue-on-error to keep going.")
                output_counts = summarize_outputs(dirs, noworms_file)
                write_manifest(run_dir, project_root, args, all_results, output_counts)
                print_user_summary(run_dir, args.device, input_folder, all_results, output_counts)
                return 1

    output_counts = summarize_outputs(dirs, noworms_file)
    write_manifest(run_dir, project_root, args, all_results, output_counts)
    print_user_summary(run_dir, args.device, input_folder, all_results, output_counts)

    failed = [result for result in all_results if result.status == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())