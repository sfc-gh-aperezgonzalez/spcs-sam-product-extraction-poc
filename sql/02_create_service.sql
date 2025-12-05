-- ============================================================================
-- Filename: 02_create_service.sql
-- Description: Create SAM 2 detection service and functions
--
-- Prerequisites: 
--   - Infrastructure created (01_setup_infrastructure.sql)
--   - Docker image pushed to ML_INFERENCE_REPO
--   - service_spec.yaml uploaded to CODE_STAGE
-- Creates: Detection service and service functions
-- ============================================================================

-- Service creation requires ACCOUNTADMIN for compute pool access
USE ROLE ACCOUNTADMIN;
USE DATABASE SHALION_HF_DEMO;
USE SCHEMA PRODUCT_EXTRACTION;

-- Step 1: Drop existing service if present
DROP SERVICE IF EXISTS SAM2_DETECTOR_SERVICE;

SELECT '✓ Cleaned up existing service' AS progress;

-- Step 2: Create the detection service
CREATE SERVICE SAM2_DETECTOR_SERVICE
    IN COMPUTE POOL SAM_GPU_POOL
    FROM @CODE_STAGE
    SPEC = 'service_spec.yaml'
    MIN_INSTANCES = 1
    MAX_INSTANCES = 1
    COMMENT = 'SAM 2 Product Detection using HuggingFace';

SELECT '✓ Service created - waiting for READY status...' AS progress;

-- Step 3: Check service status
CALL SYSTEM$GET_SERVICE_STATUS('SAM2_DETECTOR_SERVICE');

-- Step 4: Create detection functions
CREATE OR REPLACE FUNCTION DETECT_PRODUCTS(IMAGE_PATH VARCHAR)
RETURNS VARCHAR
SERVICE = SAM2_DETECTOR_SERVICE
ENDPOINT = detect
MAX_BATCH_ROWS = 1
AS '/detect';

CREATE OR REPLACE FUNCTION DETECT_PRODUCTS_FOLDER(INPUT_FOLDER VARCHAR)
RETURNS VARCHAR
SERVICE = SAM2_DETECTOR_SERVICE
ENDPOINT = detect
MAX_BATCH_ROWS = 1
AS '/detect_folder';

SELECT '✓ Detection functions created' AS progress;

-- Step 5: Create cropping UDF (runs on warehouse, no GPU needed)
CREATE OR REPLACE FUNCTION CROP_PRODUCT(
    IMAGE_BYTES BINARY,
    X INT,
    Y INT,
    WIDTH INT,
    HEIGHT INT
)
RETURNS BINARY
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('pillow')
HANDLER = 'crop_image'
AS $$
from PIL import Image
from io import BytesIO

def crop_image(image_bytes: bytes, x: int, y: int, width: int, height: int) -> bytes:
    """Crop a region from an image and return as PNG bytes."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    cropped = img.crop((x, y, x + width, y + height))
    output = BytesIO()
    cropped.save(output, format="PNG")
    img.close()
    cropped.close()
    return output.getvalue()
$$;

SELECT '✓ Cropping UDF created' AS progress;

-- Step 6: Create helper function to parse detection results
CREATE OR REPLACE FUNCTION PARSE_DETECTIONS(DETECTION_JSON VARCHAR)
RETURNS TABLE (
    IMAGE_PATH VARCHAR,
    PRODUCT_INDEX INT,
    BBOX_X INT,
    BBOX_Y INT,
    BBOX_WIDTH INT,
    BBOX_HEIGHT INT,
    CONFIDENCE FLOAT,
    AREA INT
)
LANGUAGE SQL
AS $$
    SELECT 
        PARSE_JSON(DETECTION_JSON):image_path::VARCHAR AS image_path,
        f.index::INT AS product_index,
        f.value:bbox[0]::INT AS bbox_x,
        f.value:bbox[1]::INT AS bbox_y,
        f.value:bbox[2]::INT AS bbox_width,
        f.value:bbox[3]::INT AS bbox_height,
        f.value:confidence::FLOAT AS confidence,
        f.value:area::INT AS area
    FROM TABLE(FLATTEN(PARSE_JSON(DETECTION_JSON):products)) f
$$;

SELECT '✓ Helper functions created' AS progress;

-- Grant usage to SYSADMIN for service and functions
GRANT USAGE ON SERVICE SAM2_DETECTOR_SERVICE TO ROLE SYSADMIN;
GRANT USAGE ON FUNCTION DETECT_PRODUCTS(VARCHAR) TO ROLE SYSADMIN;
GRANT USAGE ON FUNCTION DETECT_PRODUCTS_FOLDER(VARCHAR) TO ROLE SYSADMIN;
GRANT USAGE ON FUNCTION CROP_PRODUCT(BINARY, INT, INT, INT, INT) TO ROLE SYSADMIN;
GRANT USAGE ON FUNCTION PARSE_DETECTIONS(VARCHAR) TO ROLE SYSADMIN;

-- Switch back to SYSADMIN for subsequent scripts
USE ROLE SYSADMIN;

SELECT 'Service deployment complete!' AS status;

-- ============================================================================
-- USAGE:
-- 
-- -- Detect products in a single image
-- SELECT DETECT_PRODUCTS('@AD_INPUT_STAGE/ads/image.jpg') AS detections;
-- 
-- -- Detect products in all images in a folder
-- SELECT DETECT_PRODUCTS_FOLDER('@AD_INPUT_STAGE/ads') AS detections;
-- 
-- -- Parse detections into rows
-- SELECT * FROM TABLE(PARSE_DETECTIONS(
--     DETECT_PRODUCTS('@AD_INPUT_STAGE/ads/image.jpg')
-- ));
-- ============================================================================
