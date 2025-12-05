"""
SAM 2 Product Detector - Clean HuggingFace Implementation.

This service ONLY detects products and returns bounding boxes.
Cropping is done separately by a simple Python UDF.

Configuration can be loaded from a JSON file for tuning without Docker rebuilds.
"""

import json
import logging
import os
from typing import Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

logger = logging.getLogger(__name__)

# Default detection parameters - can be overridden via config file
DEFAULT_CONFIG = {
    # Grid sampling
    "points_per_side": 24,           # Grid density (24x24 = 576 points)
    
    # Quality thresholds
    "pred_iou_thresh": 0.92,         # Minimum IoU score from SAM2
    "min_fill_ratio": 0.25,          # Minimum mask/bbox fill ratio (filters text/logos)
    
    # Area filters (as ratio of image area)
    "min_area_ratio": 0.02,          # Minimum 2% of image (was 1%)
    "max_area_ratio": 0.70,          # Maximum 70% of image
    
    # Shape filters
    "min_aspect_ratio": 0.15,        # Minimum width/height ratio (1:6.67)
    "max_aspect_ratio": 6.0,         # Maximum width/height ratio (6:1)
    
    # Border touch filter
    "border_margin_px": 10,          # Pixels from edge to consider "touching"
    "max_border_touches": 1,         # Skip if touching more than N borders
    
    # NMS and limits
    "nms_threshold": 0.5,            # Non-max suppression IoU threshold
    "max_products_per_image": 10     # Maximum products to return
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load detection configuration from JSON file.
    Falls back to defaults for any missing parameters.
    
    Args:
        config_path: Path to JSON config file (optional)
        
    Returns:
        Merged configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Try loading from provided path
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            config.update(user_config)
            logger.info(f"✓ Loaded config from {config_path}")
            logger.info(f"  Overrides: {list(user_config.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return config


class SAM2Detector:
    """
    SAM 2 Product Detector using HuggingFace Transformers.
    
    Returns bounding boxes only - no cropping.
    Configuration loaded from JSON file for easy tuning.
    """
    
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", 
                 config_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading SAM 2 from HuggingFace: {model_id}")
        logger.info(f"Device: {self.device}")
        
        # Load HuggingFace model with float16 for memory efficiency
        self.model = Sam2Model.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(self.device)
        
        self.processor = Sam2Processor.from_pretrained(
            model_id,
            local_files_only=True
        )
        
        # Load detection parameters from config
        self.config = load_config(config_path)
        self._log_config()
        
        logger.info("✓ SAM 2 Detector ready")
    
    def _log_config(self):
        """Log current configuration for debugging."""
        logger.info("Detection parameters:")
        logger.info(f"  points_per_side: {self.config['points_per_side']}")
        logger.info(f"  pred_iou_thresh: {self.config['pred_iou_thresh']}")
        logger.info(f"  min_fill_ratio: {self.config['min_fill_ratio']}")
        logger.info(f"  min_area_ratio: {self.config['min_area_ratio']}")
        logger.info(f"  max_area_ratio: {self.config['max_area_ratio']}")
    
    def reload_config(self, config_path: str):
        """Reload configuration from file (for runtime tuning)."""
        self.config = load_config(config_path)
        self._log_config()
        logger.info("✓ Configuration reloaded")
    
    def detect_products(self, image_path: str) -> Dict[str, Any]:
        """
        Detect products in an image and return bounding boxes.
        
        Args:
            image_path: Path to image file
            
        Returns:
            {
                "image_path": str,
                "width": int,
                "height": int,
                "products": [
                    {"bbox": [x, y, w, h], "confidence": float, "area": int, "fill_ratio": float},
                    ...
                ],
                "config": {current config values}
            }
        """
        # Extract config values for cleaner code
        cfg = self.config
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Generate point grid
        points_per_side = cfg['points_per_side']
        points_1d = np.linspace(0, 1, points_per_side)
        xv, yv = np.meshgrid(points_1d, points_1d)
        points = np.stack([xv.flatten() * width, yv.flatten() * height], axis=-1)
        
        # Reshape for HuggingFace API: [batch, num_points, points_per_object, 2]
        input_points = points.reshape(1, -1, 1, 2)
        input_labels = np.ones((1, input_points.shape[1], 1), dtype=np.int32)
        
        # Run inference with autocast for mixed precision
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            inputs = self.processor(
                image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Post-process masks
            masks = self.processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"]
            )[0]  # (N, 3, H, W)
            
            iou_scores = outputs.iou_scores[0]  # (N, 3)
        
        # Flatten multimask outputs
        masks = masks.flatten(0, 1)  # (N*3, H, W)
        scores = iou_scores.flatten()  # (N*3,)
        
        # Filter by IoU threshold
        keep = scores > cfg['pred_iou_thresh']
        masks = masks[keep]
        scores = scores[keep]
        
        if masks.shape[0] == 0:
            image.close()
            return self._empty_result(image_path, width, height)
        
        # Convert masks to bounding boxes
        from torchvision.ops import masks_to_boxes, nms
        
        boxes = masks_to_boxes(masks)  # (N, 4) [x1, y1, x2, y2]
        mask_areas = masks.sum(dim=(1, 2))  # Actual mask pixel count
        
        # Calculate bounding box areas for fill ratio
        bbox_widths = boxes[:, 2] - boxes[:, 0]
        bbox_heights = boxes[:, 3] - boxes[:, 1]
        bbox_areas = bbox_widths * bbox_heights
        
        # Calculate fill ratio: how much of the bbox is filled by the mask
        # Low fill ratio = likely text/logo (sparse), High = likely solid product
        fill_ratios = mask_areas.float() / (bbox_areas.float() + 1e-6)
        
        # Filter by area ratio (relative to image)
        image_area = width * height
        area_ratios = mask_areas.float() / image_area
        keep_area = (
            (area_ratios > cfg['min_area_ratio']) & 
            (area_ratios < cfg['max_area_ratio'])
        )
        
        # Filter by fill ratio (removes text/logos with sparse masks)
        keep_fill = fill_ratios > cfg['min_fill_ratio']
        
        # Combine filters
        keep_combined = keep_area & keep_fill
        
        boxes = boxes[keep_combined]
        scores = scores[keep_combined]
        mask_areas = mask_areas[keep_combined]
        fill_ratios = fill_ratios[keep_combined]
        
        if boxes.shape[0] == 0:
            image.close()
            return self._empty_result(image_path, width, height)
        
        # NMS to remove duplicates
        keep_nms = nms(boxes.float(), scores.float(), cfg['nms_threshold'])
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        mask_areas = mask_areas[keep_nms]
        fill_ratios = fill_ratios[keep_nms]
        
        # Convert to product list with additional filtering
        products = []
        max_products = cfg['max_products_per_image']
        border_margin = cfg['border_margin_px']
        max_touches = cfg['max_border_touches']
        min_aspect = cfg['min_aspect_ratio']
        max_aspect = cfg['max_aspect_ratio']
        
        for i in range(min(boxes.shape[0], max_products * 2)):  # Check more, filter down
            box = boxes[i].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Convert to [x, y, w, h] format
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            
            # Filter by aspect ratio
            w, h = bbox[2], bbox[3]
            if h == 0 or w == 0:
                continue
            aspect = w / h
            if aspect < min_aspect or aspect > max_aspect:
                continue
            
            # Filter by border touch
            touches = sum([
                bbox[0] < border_margin,
                bbox[1] < border_margin,
                (bbox[0] + bbox[2]) > (width - border_margin),
                (bbox[1] + bbox[3]) > (height - border_margin)
            ])
            if touches > max_touches:
                continue
            
            products.append({
                "bbox": bbox,
                "confidence": round(float(scores[i].cpu()), 3),
                "area": int(mask_areas[i].cpu()),
                "fill_ratio": round(float(fill_ratios[i].cpu()), 3)
            })
            
            if len(products) >= max_products:
                break
        
        # Sort by bounding box area (largest first)
        products.sort(key=lambda p: p["bbox"][2] * p["bbox"][3], reverse=True)
        
        image.close()
        
        # Clear GPU memory
        del masks, boxes, scores, mask_areas, fill_ratios
        torch.cuda.empty_cache()
        
        return {
            "image_path": image_path,
            "width": width,
            "height": height,
            "products": products,
            "config_used": {
                "min_area_ratio": cfg['min_area_ratio'],
                "max_area_ratio": cfg['max_area_ratio'],
                "min_fill_ratio": cfg['min_fill_ratio'],
                "pred_iou_thresh": cfg['pred_iou_thresh']
            }
        }
    
    def _empty_result(self, image_path: str, width: int, height: int) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "image_path": image_path,
            "width": width,
            "height": height,
            "products": [],
            "config_used": {
                "min_area_ratio": self.config['min_area_ratio'],
                "max_area_ratio": self.config['max_area_ratio'],
                "min_fill_ratio": self.config['min_fill_ratio'],
                "pred_iou_thresh": self.config['pred_iou_thresh']
            }
        }
