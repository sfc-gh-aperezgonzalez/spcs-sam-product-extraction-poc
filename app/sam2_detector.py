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

# Default detection parameters - tuned for retail ad product extraction
# These match config/detection_config.json and are used as fallback
DEFAULT_CONFIG = {
    "points_per_side": 24,
    "pred_iou_thresh": 0.88,
    "min_fill_ratio": 0.20,
    "min_area_ratio": 0.015,
    "max_area_ratio": 0.75,
    "min_aspect_ratio": 0.20,
    "max_aspect_ratio": 5.0,
    "border_margin_px": 15,
    "max_border_touches": 1,
    "nms_threshold": 0.5,
    "max_products_per_image": 10,
    "batch_size": 4,
    "num_workers": 2
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
        logger.info(f"  batch_size: {self.config.get('batch_size', 4)}")
        logger.info(f"  num_workers: {self.config.get('num_workers', 2)}")
    
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
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
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
    
    def _load_image(self, image_path: str) -> tuple:
        """Load and preprocess a single image. Used by batch loader."""
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            return (image_path, image, width, height, None)
        except Exception as e:
            return (image_path, None, 0, 0, str(e))
    
    def _process_single_image_fast(self, image_data: tuple) -> Dict[str, Any]:
        """
        Process a single pre-loaded image. Optimized version that reuses point grid.
        
        Args:
            image_data: (image_path, PIL.Image, width, height, error)
        """
        image_path, image, width, height, error = image_data
        
        if error:
            logger.warning(f"Skipping {image_path}: {error}")
            return self._empty_result(image_path, 0, 0)
        
        cfg = self.config
        points_per_side = cfg['points_per_side']
        
        # Generate point grid for this image size
        points_1d = np.linspace(0, 1, points_per_side)
        xv, yv = np.meshgrid(points_1d, points_1d)
        points = np.stack([xv.flatten() * width, yv.flatten() * height], axis=-1)
        
        input_points = points.reshape(1, -1, 1, 2)
        input_labels = np.ones((1, input_points.shape[1], 1), dtype=np.int32)
        
        # Run inference
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            inputs = self.processor(
                image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            masks = self.processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"]
            )[0]
            
            iou_scores = outputs.iou_scores[0]
        
        # Close image immediately after processing
        image.close()
        
        # Post-process on GPU
        return self._post_process_masks(
            masks, iou_scores, image_path, width, height, cfg
        )
    
    def _post_process_masks(self, masks, iou_scores, image_path: str, 
                            width: int, height: int, cfg: Dict) -> Dict[str, Any]:
        """Post-process masks to extract products. Extracted for clarity."""
        from torchvision.ops import masks_to_boxes, nms
        
        # Flatten multimask outputs
        masks = masks.flatten(0, 1)
        scores = iou_scores.flatten()
        
        # Filter by IoU threshold
        keep = scores > cfg['pred_iou_thresh']
        masks = masks[keep]
        scores = scores[keep]
        
        if masks.shape[0] == 0:
            return self._empty_result(image_path, width, height)
        
        boxes = masks_to_boxes(masks)
        mask_areas = masks.sum(dim=(1, 2))
        
        bbox_widths = boxes[:, 2] - boxes[:, 0]
        bbox_heights = boxes[:, 3] - boxes[:, 1]
        bbox_areas = bbox_widths * bbox_heights
        
        fill_ratios = mask_areas.float() / (bbox_areas.float() + 1e-6)
        
        image_area = width * height
        area_ratios = mask_areas.float() / image_area
        keep_area = (area_ratios > cfg['min_area_ratio']) & (area_ratios < cfg['max_area_ratio'])
        keep_fill = fill_ratios > cfg['min_fill_ratio']
        keep_combined = keep_area & keep_fill
        
        boxes = boxes[keep_combined]
        scores = scores[keep_combined]
        mask_areas = mask_areas[keep_combined]
        fill_ratios = fill_ratios[keep_combined]
        
        if boxes.shape[0] == 0:
            return self._empty_result(image_path, width, height)
        
        keep_nms = nms(boxes.float(), scores.float(), cfg['nms_threshold'])
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        mask_areas = mask_areas[keep_nms]
        fill_ratios = fill_ratios[keep_nms]
        
        # Convert to product list
        products = []
        max_products = cfg['max_products_per_image']
        border_margin = cfg['border_margin_px']
        max_touches = cfg['max_border_touches']
        min_aspect = cfg['min_aspect_ratio']
        max_aspect = cfg['max_aspect_ratio']
        
        for i in range(min(boxes.shape[0], max_products * 2)):
            box = boxes[i].cpu().numpy()
            x1, y1, x2, y2 = box
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            
            w, h = bbox[2], bbox[3]
            if h == 0 or w == 0:
                continue
            aspect = w / h
            if aspect < min_aspect or aspect > max_aspect:
                continue
            
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
        
        products.sort(key=lambda p: p["bbox"][2] * p["bbox"][3], reverse=True)
        
        # Clean up GPU tensors
        del masks, boxes, scores, mask_areas, fill_ratios
        
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
    
    def detect_products_batch(self, image_paths: list) -> list:
        """
        Process multiple images with optimized batching.
        
        Uses a producer-consumer pattern:
        - Producer: Loads images in parallel (I/O bound)
        - Consumer: Runs inference sequentially on GPU (compute bound)
        
        This keeps the GPU continuously busy while I/O happens in parallel.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of detection results (same format as detect_products)
        """
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue
        import threading
        
        batch_size = self.config.get('batch_size', 4)
        num_workers = self.config.get('num_workers', 2)
        
        results = [None] * len(image_paths)
        image_queue = Queue(maxsize=batch_size * 2)
        
        # Producer: Load images in parallel threads
        def image_loader():
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for idx, path in enumerate(image_paths):
                    future = executor.submit(self._load_image, path)
                    futures.append((idx, future))
                
                for idx, future in futures:
                    image_data = future.result()
                    image_queue.put((idx, image_data))
            
            # Signal end of images
            image_queue.put(None)
        
        # Start loader thread
        loader_thread = threading.Thread(target=image_loader, daemon=True)
        loader_thread.start()
        
        # Consumer: Process images as they become available
        processed = 0
        while True:
            item = image_queue.get()
            if item is None:
                break
            
            idx, image_data = item
            result = self._process_single_image_fast(image_data)
            results[idx] = result
            processed += 1
            
            # Periodic GPU memory cleanup
            if processed % batch_size == 0:
                torch.cuda.empty_cache()
        
        loader_thread.join()
        torch.cuda.empty_cache()
        
        return results
