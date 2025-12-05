"""
SAM 2 Product Detector - Clean HuggingFace Implementation.

This service ONLY detects products and returns bounding boxes.
Cropping is done separately by a simple Python UDF.
"""

import logging
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

logger = logging.getLogger(__name__)


class SAM2Detector:
    """
    SAM 2 Product Detector using HuggingFace Transformers.
    
    Returns bounding boxes only - no cropping.
    Clean, simple API for demo purposes.
    """
    
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small"):
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
        
        # Detection parameters
        self.points_per_side = 24
        self.pred_iou_thresh = 0.92
        self.min_area_ratio = 0.01
        self.max_area_ratio = 0.70
        
        logger.info("✓ SAM 2 Detector ready")
    
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
                    {"bbox": [x, y, w, h], "confidence": float, "area": int},
                    ...
                ]
            }
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Generate point grid (24x24 = 576 points)
        points_1d = np.linspace(0, 1, self.points_per_side)
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
            )[0]  # (576, 3, H, W)
            
            iou_scores = outputs.iou_scores[0]  # (576, 3)
        
        # Flatten multimask outputs
        masks = masks.flatten(0, 1)  # (1728, H, W)
        scores = iou_scores.flatten()  # (1728,)
        
        # Filter by IoU threshold
        keep = scores > self.pred_iou_thresh
        masks = masks[keep]
        scores = scores[keep]
        
        if masks.shape[0] == 0:
            image.close()
            return {
                "image_path": image_path,
                "width": width,
                "height": height,
                "products": []
            }
        
        # Convert masks to bounding boxes
        from torchvision.ops import masks_to_boxes, nms
        
        boxes = masks_to_boxes(masks)  # (N, 4) [x1, y1, x2, y2]
        areas = masks.sum(dim=(1, 2))
        
        # Filter by area
        image_area = width * height
        area_ratios = areas.float() / image_area
        keep_area = (area_ratios > self.min_area_ratio) & (area_ratios < self.max_area_ratio)
        
        boxes = boxes[keep_area]
        scores = scores[keep_area]
        areas = areas[keep_area]
        
        if boxes.shape[0] == 0:
            image.close()
            return {
                "image_path": image_path,
                "width": width,
                "height": height,
                "products": []
            }
        
        # NMS to remove duplicates
        keep_nms = nms(boxes.float(), scores.float(), 0.5)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        areas = areas[keep_nms]
        
        # Convert to product list
        products = []
        for i in range(min(boxes.shape[0], 10)):  # Limit to 10 products
            box = boxes[i].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Convert to [x, y, w, h] format
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            
            # Filter by aspect ratio
            w, h = bbox[2], bbox[3]
            if h == 0 or w == 0:
                continue
            aspect = w / h
            if aspect < 1/6 or aspect > 6:
                continue
            
            # Filter by border touch
            touches = sum([
                bbox[0] < 10,
                bbox[1] < 10,
                (bbox[0] + bbox[2]) > (width - 10),
                (bbox[1] + bbox[3]) > (height - 10)
            ])
            if touches >= 2:
                continue
            
            products.append({
                "bbox": bbox,
                "confidence": round(float(scores[i].cpu()), 3),
                "area": int(areas[i].cpu())
            })
        
        # Sort by bounding box area (largest first)
        products.sort(key=lambda p: p["bbox"][2] * p["bbox"][3], reverse=True)
        
        # Return top candidates - deduplication happens in orchestrator
        # This allows tuning dedup params without Docker rebuilds
        final_products = products[:10]  # Return up to 10 candidates per image
        
        image.close()
        
        # Clear GPU memory
        del masks, boxes, scores, areas
        torch.cuda.empty_cache()
        
        return {
            "image_path": image_path,
            "width": width,
            "height": height,
            "products": final_products
        }
