"""
Utility functions for image I/O and mask filtering.
Uses mounted Snowflake stage volumes for file access.
"""

import os
import logging
from typing import List, Dict, Any
from io import BytesIO

import numpy as np
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


def read_image_from_stage(stage_url: str) -> Image.Image:
    """
    Read an image from a Snowflake stage.
    Supports both mounted volumes and fsspec stage URLs.
    
    Args:
        stage_url: Stage path (e.g., '@AD_INPUT_STAGE/demo/ad.jpg') or local path
    
    Returns:
        PIL Image in RGB format
    """
    logger.info(f"Reading image from: {stage_url}")
    
    # Convert stage URL to mounted path if needed
    if stage_url.startswith("@"):
        # Parse @DATABASE.SCHEMA.STAGE/path format
        # Example: @SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE/demo/1.png
        # Mounted at: /input/demo/1.png
        parts = stage_url.split("/", 1)
        if len(parts) == 2:
            file_path = f"/input/{parts[1]}"
        else:
            file_path = "/input/"
        
        logger.info(f"Using mounted path: {file_path}")
        
        with open(file_path, "rb") as f:
            image = Image.open(f).convert("RGB")
    else:
        # Try as direct file path
        with open(stage_url, "rb") as f:
            image = Image.open(f).convert("RGB")
    
    return image


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((OSError, IOError)),
    reraise=True
)
def _write_file_with_retry(file_path: str, data: bytes) -> None:
    """
    Write file data with retry logic for transient failures.
    Uses atomic write pattern: write to temp, then rename.
    
    Args:
        file_path: Destination file path (must be flat, no subdirectories)
        data: File data as bytes
    """
    # Atomic write: write to temp file, then rename
    # NO mkdir() calls - flat structure only
    temp_path = f"{file_path}.tmp"
    try:
        with open(temp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic rename
        os.replace(temp_path, file_path)
        logger.debug(f"Successfully wrote {len(data)} bytes to {file_path}")
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e


def upload_crop_to_stage(image: Image.Image, stage_url: str) -> None:
    """
    Upload a cropped image to a Snowflake stage.
    Supports both mounted volumes and fsspec stage URLs.
    
    Args:
        image: PIL Image to upload
        stage_url: Destination stage path (flat structure)
    """
    logger.debug(f"Uploading crop to: {stage_url}")
    
    # Convert stage URL to mounted path if needed
    if stage_url.startswith("@"):
        # Parse @DATABASE.SCHEMA.STAGE/path format
        # Output mounted at: /output/
        parts = stage_url.split("/", 1)
        if len(parts) == 2:
            file_path = f"/output/{parts[1]}"
        else:
            file_path = "/output/"
        
        logger.debug(f"Using mounted path: {file_path}")
        
        # Write image to BytesIO buffer first (in-memory operation)
        # This decouples PIL operations from filesystem I/O
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False)  # optimize=False for speed
        image_data = buffer.getvalue()
        buffer.close()
        
        # Write to file with retry logic (no artificial delays needed)
        _write_file_with_retry(file_path, image_data)
        
    else:
        # Try as direct file path
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        image_data = buffer.getvalue()
        buffer.close()
        
        _write_file_with_retry(stage_url, image_data)


def filter_masks_minimal(
    masks: List[Dict[str, Any]],
    image_shape: tuple,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.70,
    min_aspect_ratio: float = 1/6,
    max_aspect_ratio: float = 6.0
) -> List[Dict[str, Any]]:
    """
    Filter masks using product-detection heuristics for ad images.
    
    Args:
        masks: List of SAM mask dictionaries
        image_shape: (height, width, channels) of original image
        min_area_ratio: Minimum mask area fraction (default 1%)
        max_area_ratio: Maximum mask area fraction (default 70%)
        min_aspect_ratio: Minimum aspect ratio (default 1:6)
        max_aspect_ratio: Maximum aspect ratio (default 6:1)
    
    Returns:
        Filtered list of product-likely masks with duplicates removed
    """
    height, width = image_shape[:2]
    image_area = height * width
    
    filtered = []
    
    for mask in masks:
        bbox = mask["bbox"]  # [x, y, w, h]
        area = mask["area"]
        area_ratio = area / image_area
        
        # Filter 1: Area constraints (drop tiny specks and huge backgrounds)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        
        # Filter 2: Aspect ratio bounds (drop ribbons/edges)
        bbox_w, bbox_h = bbox[2], bbox[3]
        if bbox_h == 0 or bbox_w == 0:
            continue
        
        aspect = bbox_w / bbox_h
        if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
            continue
        
        # Filter 3: Border-touch filter
        x, y = bbox[0], bbox[1]
        
        # Check if touching borders (within 10 pixels of edge)
        touches_left = x < 10
        touches_top = y < 10
        touches_right = (x + bbox_w) > (width - 10)
        touches_bottom = (y + bbox_h) > (height - 10)
        border_touches = sum([touches_left, touches_top, touches_right, touches_bottom])
        
        if border_touches >= 2:  # Touching 2+ edges = likely border/background
            continue
        
        # Filter 4: Quality thresholds
        pred_iou = mask.get("predicted_iou", 0.0)
        stability = mask.get("stability_score", 0.0)
        if pred_iou < 0.80 or stability < 0.90:
            continue
        
        filtered.append(mask)
    
    # Sort by area (largest first)
    filtered.sort(key=lambda m: m["area"], reverse=True)
    
    # Deduplication
    final_masks = []
    for mask in filtered:
        bbox = mask["bbox"]
        x1, y1, w1, h1 = bbox
        
        # Check overlap with already selected masks
        overlaps = False
        for selected in final_masks:
            bbox2 = selected["bbox"]
            x2, y2, w2, h2 = bbox2
            
            # Calculate intersection area
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            
            # Consider duplicate if >25% overlap
            min_area = min(mask["area"], selected["area"])
            if overlap_area / min_area > 0.25:
                overlaps = True
                break
        
        if not overlaps:
            final_masks.append(mask)
    
    # Limit to top 3 candidates (SAM's job: provide candidates, Cortex's job: pick winners)
    final_masks = final_masks[:3]
    
    return final_masks


def create_transparent_crop(
    image: np.ndarray,
    mask: Dict[str, Any]
) -> Image.Image:
    """
    Create a bounding box crop of the product region.
    
    For embeddings use case: Returns simple rectangular crop without transparency.
    This preserves all product pixels (important for dark/black products).
    
    Args:
        image: Original image as numpy array (H, W, 3)
        mask: SAM mask dictionary with 'bbox'
    
    Returns:
        PIL Image in RGB format (bounding box crop, no transparency)
    """
    bbox = mask["bbox"]  # [x, y, w, h]
    x, y, w, h = [int(v) for v in bbox]
    
    # Simple bounding box crop (no alpha masking)
    # This preserves all pixels within the product region
    cropped_rgb = image[y:y+h, x:x+w]
    
    return Image.fromarray(cropped_rgb, mode="RGB")

