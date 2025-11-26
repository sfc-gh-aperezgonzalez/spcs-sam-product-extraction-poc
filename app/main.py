"""
FastAPI application for SAM-based product extraction from ad images.
Runs as an SPCS service with GPU acceleration.
"""

from fastapi import FastAPI, HTTPException, Request
import os
import json
import logging

from inference import SAMInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SAM Product Extraction Service",
    description="Extract individual products from ad images using Segment Anything Model",
    version="1.0.0"
)

# Global inference engine (lazy-loaded)
inference_engine = None


def get_inference_engine() -> SAMInference:
    """Lazy-load the SAM inference engine."""
    global inference_engine
    if inference_engine is None:
        model_path = os.getenv("MODEL_PATH", "/models/sam_vit_h_4b8939.pth")
        logger.info(f"Loading SAM model from {model_path}")
        inference_engine = SAMInference(model_path=model_path)
        logger.info("SAM model loaded successfully")
    return inference_engine


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None
    }


@app.post("/infer")
async def run_inference(request: Request):
    """
    Extract products from an ad image.
    Supports both direct calls and Snowflake service function batch format.
    
    Args:
        request: Either {input_url, output_stage} OR Snowflake batch {data: [[row, arg1, arg2]]}
    
    Returns:
        JSON with crop URLs and metadata: {"crops": [...], "num_products": N, ...}
    
    Note:
        Uses FLAT directory structure (no subdirectories) for improved performance.
        Files: template_X_YY_runid_product_NNN.png (all at stage root level)
    """
    # Parse JSON body
    body = await request.json()
    
    # Parse request - handle both formats
    if "data" in body:
        # Snowflake service function batch format: {"data": [[row_num, arg1, arg2, ...]]}
        # For single-row: [[0, image_path, output_stage]]
        data = body["data"]
        if len(data) > 0 and len(data[0]) >= 3:
            input_url = data[0][1]  # Second element (after row number)
            output_stage = data[0][2]  # Third element (stage URL)
            logger.info(f"Service function batch request: row={data[0][0]}")
        else:
            raise HTTPException(status_code=400, detail="Invalid batch data format")
    else:
        # Direct API call format
        input_url = body.get("input_url")
        output_stage = body.get("output_stage", os.getenv("AD_OUTPUT_STAGE_URL", "@AD_OUTPUT_STAGE"))
    
    # Ensure output_stage has proper format (with @ prefix)
    if not output_stage.startswith("@"):
        output_stage = f"@{output_stage}"
    
    logger.info(f"Inference request: input={input_url}, output_stage={output_stage}")
    
    try:
        # Get inference engine
        engine = get_inference_engine()
        logger.info("Engine loaded successfully")
        
        # Run inference and get crops
        logger.info("Starting process_image...")
        crop_urls = engine.process_image(
            input_url=input_url,
            output_stage=output_stage
        )
        logger.info(f"process_image completed")
        
        logger.info(f"Generated {len(crop_urls)} product crops")
        
        # Get crop metadata for downstream filtering
        crop_metadata = engine.get_crop_metadata()
        
        # Classify as product-likely or non-product based on heuristics
        product_likely = len(crop_urls) > 0  # Simple: if any crops extracted, likely a product
        
        # Return Snowflake service function format for scalar function
        # Format: {"data": [[row_number, return_value]]}
        # Return a JSON string with crops + metadata for Cortex filtering
        result_json = json.dumps({
            "crops": crop_urls,
            "num_products": len(crop_urls),
            "product_likely": product_likely,
            "input_image": input_url,
            "output_stage": output_stage,
            "metadata": crop_metadata  # For Cortex-based product filtering
        })
        
        # Snowflake service function response format
        response = {"data": [[0, result_json]]}
        return response
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "SAM Product Extraction",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "inference": "/infer (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

