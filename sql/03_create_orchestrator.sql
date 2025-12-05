-- ============================================================================
-- Filename: 03_create_orchestrator.sql
-- Description: Create the EXTRACT_PRODUCTS stored procedure
--
-- Prerequisites: 
--   - Service deployed (02_create_service.sql)
--   - Detection functions available
-- Creates: EXTRACT_PRODUCTS stored procedure
-- ============================================================================

-- Use SYSADMIN for stored procedure creation
USE ROLE SYSADMIN;
USE DATABASE SHALION_HF_DEMO;
USE SCHEMA PRODUCT_EXTRACTION;

-- Create the orchestrator stored procedure
CREATE OR REPLACE PROCEDURE EXTRACT_PRODUCTS(
    INPUT_FOLDER VARCHAR,
    OUTPUT_STAGE VARCHAR
)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-snowpark-python', 'Pillow')
HANDLER = 'main'
AS
$$
import json
import time
import os
import tempfile
from io import BytesIO
from snowflake.snowpark.files import SnowflakeFile
from PIL import Image

def deduplicate_products(products: list, overlap_threshold: float = 0.25) -> list:
    """
    Remove duplicate/nested product detections based on bounding box overlap.
    
    Args:
        products: List of product dicts with 'bbox' [x, y, w, h]
        overlap_threshold: If overlap / smaller_bbox_area > threshold, consider duplicate
    
    Returns:
        Deduplicated list of products (largest boxes first, duplicates removed)
    """
    if not products:
        return []
    
    # Sort by bounding box area (largest first)
    sorted_products = sorted(products, key=lambda p: p['bbox'][2] * p['bbox'][3], reverse=True)
    
    final_products = []
    for product in sorted_products:
        bbox = product['bbox']
        x1, y1, w1, h1 = bbox
        bbox1_area = w1 * h1
        
        # Check overlap with already selected products
        is_duplicate = False
        for selected in final_products:
            bbox2 = selected['bbox']
            x2, y2, w2, h2 = bbox2
            bbox2_area = w2 * h2
            
            # Calculate intersection area
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            
            # Check if this is a duplicate (significant overlap with existing)
            min_bbox_area = min(bbox1_area, bbox2_area)
            if min_bbox_area > 0:
                overlap_ratio = overlap_area / min_bbox_area
                if overlap_ratio > overlap_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            final_products.append(product)
    
    return final_products

def crop_single_product(session, image_path: str, bbox: list, output_stage: str, output_filename: str) -> dict:
    """Crop a single product from an image using scoped URL."""
    try:
        # Parse stage path: @AD_INPUT_STAGE/ads/template_1_01.jpeg
        clean_path = image_path.lstrip('@')
        parts = clean_path.split('/', 1)
        stage_name = parts[0]
        file_path = parts[1] if len(parts) > 1 else ''
        
        # Get scoped URL for reading
        scoped_result = session.sql(f"""
            SELECT BUILD_SCOPED_FILE_URL('@{stage_name}', '{file_path}') AS url
        """).collect()
        scoped_url = scoped_result[0]['URL']
        
        # Read image using scoped URL
        with SnowflakeFile.open(scoped_url, 'rb') as f:
            img_bytes = f.read()
        
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        
        # Crop: bbox is [x, y, width, height]
        x, y, w, h = bbox
        cropped = img.crop((x, y, x + w, y + h))
        
        # Save to temp file and upload
        out_buf = BytesIO()
        cropped.save(out_buf, format='PNG')
        cropped_bytes = out_buf.getvalue()
        
        img.close()
        cropped.close()
        
        # Write to temp file with proper name and PUT to stage
        temp_dir = tempfile.mkdtemp()
        proper_path = os.path.join(temp_dir, output_filename)
        
        try:
            with open(proper_path, 'wb') as f:
                f.write(cropped_bytes)
            
            session.file.put(proper_path, output_stage, auto_compress=False, overwrite=True)
        finally:
            os.unlink(proper_path)
            os.rmdir(temp_dir)
        
        return {"status": "success", "output_path": f"{output_stage}/{output_filename}"}
    except Exception as e:
        return {"status": "error", "error": str(e), "output_path": output_filename}

def main(session, input_folder: str, output_stage: str):
    start_time = time.time()
    
    # Build full stage path for detection service
    detection_path = f"@AD_INPUT_STAGE/{input_folder}"
    
    # Step 1: Call GPU detection service
    print(f"Step 1: Detecting products in {detection_path}...")
    detection_start = time.time()
    
    result = session.sql(f"""
        SELECT DETECT_PRODUCTS_FOLDER('{detection_path}') AS detections
    """).collect()
    
    detection_time = time.time() - detection_start
    
    if not result:
        return json.dumps({"error": "Detection failed", "total_products": 0})
    
    detections = json.loads(result[0]['DETECTIONS'])
    
    print(f"  Detection complete: {detections['total_images']} images, "
          f"{detections['total_products']} products in {detection_time:.1f}s")
    
    # Step 2: Deduplicate and crop products
    print(f"Step 2: Deduplicating and cropping products...")
    crop_start = time.time()
    
    total_before_dedup = detections['total_products']
    total_after_dedup = 0
    
    crop_tasks = []
    for img_result in detections['detections']:
        image_path = img_result['image_path']
        filename = image_path.split('/')[-1].rsplit('.', 1)[0]
        
        # Deduplicate products for this image
        deduped_products = deduplicate_products(img_result['products'], overlap_threshold=0.25)
        total_after_dedup += len(deduped_products)
        
        for idx, product in enumerate(deduped_products):
            bbox = product['bbox']
            output_filename = f"{filename}_product_{idx:03d}.png"
            crop_tasks.append({
                "image_path": image_path,
                "bbox": bbox,
                "output_filename": output_filename
            })
    
    print(f"  Deduplication: {total_before_dedup} -> {total_after_dedup} products "
          f"({total_before_dedup - total_after_dedup} duplicates removed)")
    
    # Process crops
    successful_crops = 0
    failed_crops = 0
    
    for task in crop_tasks:
        result = crop_single_product(
            session,
            task["image_path"],
            task["bbox"],
            output_stage,
            task["output_filename"]
        )
        if result["status"] == "success":
            successful_crops += 1
        else:
            failed_crops += 1
            print(f"  Warning: {result.get('error', 'Unknown error')}")
    
    crop_time = time.time() - crop_start
    total_time = time.time() - start_time
    
    print(f"  Cropping complete: {successful_crops} products in {crop_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    
    return json.dumps({
        "total_images": detections['total_images'],
        "total_products": successful_crops,
        "failed_crops": failed_crops,
        "detection_seconds": round(detection_time, 2),
        "cropping_seconds": round(crop_time, 2),
        "total_seconds": round(total_time, 2),
        "throughput_per_hour": round(3600 / total_time * detections['total_images'], 0) if total_time > 0 else 0
    })
$$;

SELECT '✓ Orchestrator procedure created' AS progress;

SELECT 'Orchestrator setup complete!' AS status;

-- ============================================================================
-- USAGE:
-- 
-- -- Extract products from all images in ads folder
-- CALL EXTRACT_PRODUCTS('ads', '@AD_OUTPUT_STAGE');
-- 
-- -- Check output
-- LIST @AD_OUTPUT_STAGE PATTERN = '.*product.*\.png';
-- ============================================================================


