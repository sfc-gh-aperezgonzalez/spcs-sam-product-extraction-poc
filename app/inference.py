"""
SAM inference engine for automatic product segmentation.
"""

import os
import uuid
import gc
import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from utils import (
    read_image_from_stage,
    upload_crop_to_stage,
    filter_masks_minimal,
    create_transparent_crop
)

logger = logging.getLogger(__name__)


class SAMInference:
    """SAM-based inference engine for product extraction."""
    
    def __init__(self, model_path: str):
        """
        Initialize SAM model.
        
        Args:
            model_path: Path to SAM checkpoint file (e.g., sam_vit_h_4b8939.pth)
        """
        self.last_crop_metadata = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load SAM model (vit_h architecture)
        model_type = "vit_h"
        logger.info(f"Loading SAM model: {model_type} from {model_path}")
        
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.device)
        
        # Configure automatic mask generator for whole-product extraction
        # Tuned to avoid over-segmentation of products into parts
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=24,  # Fewer points = fewer masks (default: 32)
            pred_iou_thresh=0.92,  # Slightly relaxed to catch complete objects
            stability_score_thresh=0.93,  # Slightly relaxed for completeness
            box_nms_thresh=0.5,  # Aggressive NMS to merge nearby boxes (default: 0.7)
            min_mask_region_area=1000,  # Higher to avoid small fragments
        )
        
        logger.info("SAM mask generator initialized")
    
    def process_image(
        self,
        input_url: str,
        output_stage: str
    ) -> List[str]:
        """
        Process an ad image and extract product crops.
        
        Args:
            input_url: Stage URL to input image (e.g., '@AD_INPUT_STAGE/ads/template_1_01.jpg')
            output_stage: Output stage URL (e.g., '@AD_OUTPUT_STAGE')
        
        Returns:
            List of output stage URLs for cropped products
        
        Note:
            Uses FLAT directory structure (NO subdirectories).
            FUSE mounts in SPCS hang with mkdir() operations during intensive file writing.
            Files: template_X_YY_runid_product_NNN.png (all at root level)
        """
        logger.info(f"Processing image: {input_url}")
        
        # Extract template name from input path
        # e.g., '@AD_INPUT_STAGE/ads/template_1_01.jpg' -> 'template_1_01'
        input_filename = os.path.basename(input_url)
        template_name = os.path.splitext(input_filename)[0]
        
        # Generate unique run ID
        run_id = uuid.uuid4().hex[:8]
        
        # FLAT structure: template_X_YY_runid_product_NNN.png (NO subdirectories)
        output_prefix = f"{template_name}_{run_id}_"
        
        logger.info(f"Output prefix: {output_prefix}")
        
        # Read image from Snowflake stage
        image = read_image_from_stage(input_url)
        np_image = np.array(image)
        image.close()  # Close PIL Image to free memory
        
        logger.info(f"Image shape: {np_image.shape}")
        
        # Generate masks using SAM
        logger.info("Generating masks with SAM...")
        masks = self.mask_generator.generate(np_image)
        logger.info(f"Generated {len(masks)} initial masks")
        
        # Filter masks using product-detection heuristics
        # Returns top 3 candidates (sorted, deduplicated, limited in filter function)
        filtered_masks = filter_masks_minimal(
            masks=masks,
            image_shape=np_image.shape
        )
        logger.info(f"Returning {len(filtered_masks)} product candidates for Cortex filtering")
        
        # Clean up large mask list (each mask has numpy arrays)
        del masks
        
        # Create crops and upload to stage
        crop_urls = []
        self.last_crop_metadata = []
        image_area = np_image.shape[0] * np_image.shape[1]
        
        logger.info(f"Starting crop creation loop for {len(filtered_masks)} masks...")
        
        for idx, mask in enumerate(filtered_masks):
            logger.info(f"Processing mask {idx+1}/{len(filtered_masks)}")
            try:
                # Create bounding box crop (RGB, no transparency)
                logger.info(f"  Creating crop {idx}...")
                crop_image = create_transparent_crop(
                    image=np_image,
                    mask=mask
                )
                logger.info(f"  Crop {idx} created, size: {crop_image.size}")
                
                # Generate flat output filename (no subdirectories)
                # Structure: @STAGE/template_X_YY_runid_product_NNN.png
                filename = f"{output_prefix}product_{idx:03d}.png"
                output_url = output_stage + "/" + filename if not output_stage.endswith("/") else output_stage + filename
                logger.info(f"  Uploading {idx} to: {output_url}")
                
                # Upload to stage (flat structure, no subdirectories)
                upload_crop_to_stage(crop_image, output_url)
                logger.info(f"  Upload {idx} complete")
                crop_image.close()  # Close PIL Image to free memory
                crop_urls.append(output_url)
                
                # Store metadata for downstream Cortex filtering
                bbox = mask["bbox"]
                self.last_crop_metadata.append({
                    "crop_url": output_url,
                    "area_ratio": round(mask["area"] / image_area, 4),
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "confidence": round(mask.get("stability_score", 0.0), 3)
                })
                
                logger.info(f"Created crop {idx+1}/{len(filtered_masks)}: {filename}")
                
            except Exception as e:
                logger.warning(f"Failed to create crop {idx}: {str(e)}")
                continue
        
        logger.info(f"Successfully created {len(crop_urls)} product crops")
        
        # Clean up large objects to free memory
        del np_image
        del filtered_masks
        
        # Clear GPU memory after processing each image
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
        
        # Force garbage collection to free memory immediately
        gc.collect()
        
        # NOTE: Removed artificial delays - not needed with flat directory structure
        # Retry logic in upload_crop_to_stage handles transient failures
        logger.info("Memory cleanup complete")
        
        return crop_urls
    
    def get_crop_metadata(self):
        """Return metadata from last processing for Cortex-based filtering."""
        return self.last_crop_metadata

