-- ============================================================================
-- Filename: 05_configure_detection.sql
-- Description: Configure SAM2 detection parameters without Docker rebuild
--
-- Prerequisites: 
--   - Service deployed (02_create_service.sql)
--   - Config file ready in local config/ folder
-- ============================================================================

USE ROLE SYSADMIN;
USE DATABASE SHALION_HF_DEMO;
USE SCHEMA PRODUCT_EXTRACTION;

-- ============================================================================
-- STEP 1: Upload configuration file to input stage
-- ============================================================================

-- Option A: Upload default balanced config
-- PUT file://config/detection_config.json @AD_INPUT_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- Option B: Upload conservative preset (fewer false positives)
-- PUT file://config/presets/conservative.json @AD_INPUT_STAGE/detection_config.json AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- Option C: Upload aggressive preset (more detections)
-- PUT file://config/presets/aggressive.json @AD_INPUT_STAGE/detection_config.json AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

SELECT '=== Current Config Files in Stage ===' AS step;
LIST @AD_INPUT_STAGE PATTERN = '.*config.*\.json';

-- ============================================================================
-- STEP 2: Reload configuration in running service
-- ============================================================================

-- Create a function to reload config (calls service endpoint)
CREATE OR REPLACE FUNCTION RELOAD_DETECTION_CONFIG()
RETURNS VARIANT
SERVICE = SAM2_DETECTOR_SERVICE
ENDPOINT = detect
MAX_BATCH_ROWS = 1
AS '/reload_config';

-- Call it to apply new config
SELECT '=== Reloading Configuration ===' AS step;
-- Note: This requires the service to be running
-- SELECT RELOAD_DETECTION_CONFIG();

-- ============================================================================
-- STEP 3: Verify current configuration
-- ============================================================================

-- Check service health (includes current config)
SELECT '=== Current Service Configuration ===' AS step;
-- Use direct HTTP call via the service function or check logs

-- ============================================================================
-- CONFIGURATION PARAMETER GUIDE
-- ============================================================================
/*
PARAMETER               | DEFAULT | DESCRIPTION
------------------------|---------|--------------------------------------------------
pred_iou_thresh         | 0.92    | Min confidence score (0.88-0.96 typical)
min_fill_ratio          | 0.25    | Min mask/bbox ratio - FILTERS TEXT/LOGOS
min_area_ratio          | 0.02    | Min detection size (% of image)
max_area_ratio          | 0.70    | Max detection size (% of image)  
min_aspect_ratio        | 0.15    | Min width/height (tall objects)
max_aspect_ratio        | 6.0     | Max width/height (wide objects)
border_margin_px        | 10      | Pixels from edge = "touching"
max_border_touches      | 1       | Skip if touches >N borders
nms_threshold           | 0.5     | Overlap threshold for dedup
max_products_per_image  | 10      | Max products returned

KEY TUNING TIPS:
- Too many false positives? → Increase min_fill_ratio (0.30-0.40)
- Missing products? → Decrease pred_iou_thresh (0.88-0.90)
- Logos being detected? → Increase min_fill_ratio (logos have sparse masks)
- Small decorations? → Increase min_area_ratio (0.025-0.04)
*/

SELECT 'Configuration guide complete. Upload config file and call RELOAD_DETECTION_CONFIG().' AS status;

