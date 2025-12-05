"""
SAM 2 Detection Service - Clean HuggingFace API.

This service detects products and returns bounding boxes.
Cropping is handled separately by a Python UDF.

Configuration can be loaded from a JSON file for tuning without Docker rebuilds:
- Place detection_config.json in the input stage root
- Service reads config at startup and on /reload_config calls
"""

from fastapi import FastAPI, HTTPException, Request
import os
import json
import logging
import glob
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration file location (on mounted stage)
CONFIG_PATH = os.getenv("CONFIG_PATH", "/input/detection_config.json")

app = FastAPI(
    title="SAM 2 Product Detection Service",
    description="Detect products using HuggingFace SAM 2 - returns bounding boxes. "
                "Configure via detection_config.json in input stage.",
    version="1.1.0"
)

detector = None


def get_detector(force_reload_config: bool = False):
    """
    Get or create the SAM2 detector instance.
    
    Args:
        force_reload_config: If True, reload config from file
    """
    global detector
    if detector is None:
        model_id = os.getenv("MODEL_ID", "facebook/sam2.1-hiera-small")
        logger.info(f"Loading SAM 2 model: {model_id}")
        logger.info(f"Config path: {CONFIG_PATH}")
        
        from sam2_detector import SAM2Detector
        detector = SAM2Detector(model_id=model_id, config_path=CONFIG_PATH)
    elif force_reload_config:
        detector.reload_config(CONFIG_PATH)
    
    return detector


@app.get("/health")
async def health():
    import torch
    det = get_detector()
    return {
        "status": "healthy",
        "model": "SAM 2.1 HuggingFace",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_path": CONFIG_PATH,
        "current_config": det.config
    }


@app.post("/reload_config")
async def reload_config():
    """
    Reload detection configuration from the config file.
    
    Use this to apply new settings without restarting the service:
    1. Upload new detection_config.json to input stage
    2. Call this endpoint
    3. New detections will use updated parameters
    """
    try:
        det = get_detector(force_reload_config=True)
        return {
            "status": "config_reloaded",
            "config_path": CONFIG_PATH,
            "current_config": det.config
        }
    except Exception as e:
        logger.error(f"Config reload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect")
async def detect_single(request: Request):
    """
    Detect products in a single image.
    
    Request: {"data": [[row_num, "image_path"]]}
    Returns: {"data": [[row_num, "{products: [...]}"]]}
    """
    body = await request.json()
    
    if "data" not in body or not body["data"]:
        raise HTTPException(status_code=400, detail="Invalid request")
    
    row_data = body["data"][0]
    row_num = row_data[0]
    image_path = row_data[1]
    
    # Convert stage path to local path
    if image_path.startswith("@"):
        parts = image_path.split("/", 1)
        local_path = f"/input/{parts[1]}" if len(parts) == 2 else "/input/"
    else:
        local_path = image_path
    
    try:
        det = get_detector()
        result = det.detect_products(local_path)
        
        # Return relative path for stage reference
        result["image_path"] = image_path
        
        return {"data": [[row_num, json.dumps(result)]]}
        
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_folder")
async def detect_folder(request: Request):
    """
    Detect products in all images in a folder using batch processing.
    
    Request: {"data": [[row_num, "@INPUT_STAGE/folder"]]}
    Returns: {"data": [[row_num, "[{image_path, products}, ...]"]]}
    """
    body = await request.json()
    
    if "data" not in body or not body["data"]:
        raise HTTPException(status_code=400, detail="Invalid request")
    
    row_data = body["data"][0]
    row_num = row_data[0]
    input_folder = row_data[1]
    
    # Convert stage path to local path
    if input_folder.startswith("@"):
        parts = input_folder.split("/", 1)
        local_folder = f"/input/{parts[1]}" if len(parts) == 2 else "/input/"
    else:
        local_folder = input_folder
    
    logger.info(f"Detecting products in folder: {local_folder}")
    
    try:
        start_time = time.time()
        det = get_detector()
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(local_folder, ext)))
        image_files.sort()
        
        logger.info(f"Found {len(image_files)} images")
        batch_size = det.config.get('batch_size', 4)
        logger.info(f"Processing with batch_size={batch_size}")
        
        # Use batch processing for better throughput
        batch_start = time.time()
        results = det.detect_products_batch(image_files)
        batch_time = time.time() - batch_start
        
        # Convert local paths back to stage paths and log progress
        base_stage = input_folder.split('/')[0].replace('@', '')
        for i, result in enumerate(results):
            local_path = result["image_path"]
            relative = local_path.replace("/input/", "")
            result["image_path"] = f"@{base_stage}/{relative}"
            
            # Log individual results
            logger.info(f"[{i+1}/{len(image_files)}] {os.path.basename(local_path)}: "
                       f"{len(result['products'])} products")
        
        elapsed = time.time() - start_time
        total_products = sum(len(r["products"]) for r in results)
        
        logger.info(f"Batch inference: {len(image_files)} images in {batch_time:.2f}s "
                   f"({batch_time/len(image_files):.3f}s/image)")
        
        summary = {
            "total_images": len(results),
            "total_products": total_products,
            "elapsed_seconds": round(elapsed, 2),
            "avg_seconds_per_image": round(elapsed / len(results), 2) if results else 0,
            "throughput_per_hour": round(3600 / elapsed * len(results), 0) if elapsed > 0 else 0,
            "detections": results
        }
        
        logger.info(f"✓ Complete: {len(results)} images, {total_products} products in {elapsed:.2f}s")
        
        return {"data": [[row_num, json.dumps(summary)]]}
        
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    det = get_detector()
    return {
        "service": "SAM 2 Product Detection",
        "version": "1.1.0",
        "model": "HuggingFace SAM 2.1",
        "description": "Detects products and returns bounding boxes. Cropping done separately.",
        "config": {
            "path": CONFIG_PATH,
            "current_values": det.config,
            "how_to_tune": "Upload detection_config.json to input stage, then call /reload_config"
        },
        "endpoints": {
            "/health": "Health check with current config",
            "/detect": "POST - Detect single image",
            "/detect_folder": "POST - Detect all images in folder",
            "/reload_config": "POST - Reload config from file (no restart needed)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

