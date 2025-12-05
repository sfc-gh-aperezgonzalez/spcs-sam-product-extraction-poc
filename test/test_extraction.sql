-- ============================================================================
-- Filename: test_extraction.sql
-- Description: End-to-end test of SAM 2 product extraction
--
-- Prerequisites: 
--   - All SQL scripts executed (01, 02, 03)
--   - Service is READY
--   - Test images uploaded to AD_INPUT_STAGE/ads
-- ============================================================================

USE ROLE SYSADMIN;
USE DATABASE SHALION_HF_DEMO;
USE SCHEMA PRODUCT_EXTRACTION;
USE WAREHOUSE SAM_DEMO_WH;

-- Step 1: Verify service is ready
SELECT '=== Step 1: Check Service Status ===' AS step;
CALL SYSTEM$GET_SERVICE_STATUS('SAM2_DETECTOR_SERVICE');

-- Step 2: Count input images
SELECT '=== Step 2: Count Input Images ===' AS step;
SELECT COUNT(*) AS input_image_count
FROM DIRECTORY(@AD_INPUT_STAGE)
WHERE RELATIVE_PATH LIKE 'ads/%.jpg' 
   OR RELATIVE_PATH LIKE 'ads/%.jpeg'
   OR RELATIVE_PATH LIKE 'ads/%.png';

-- Step 3: Clear output stage
SELECT '=== Step 3: Clear Output Stage ===' AS step;
REMOVE @AD_OUTPUT_STAGE PATTERN = '.*\.png';

-- Step 4: Run extraction
SELECT '=== Step 4: Run Product Extraction ===' AS step;
CALL EXTRACT_PRODUCTS('ads', '@AD_OUTPUT_STAGE');

-- Step 5: Verify output
SELECT '=== Step 5: Verify Output ===' AS step;
SELECT COUNT(*) AS output_product_count
FROM DIRECTORY(@AD_OUTPUT_STAGE)
WHERE RELATIVE_PATH LIKE '%_product_%.png';

-- Step 6: Sample output files
SELECT '=== Step 6: Sample Output Files ===' AS step;
SELECT RELATIVE_PATH, SIZE, LAST_MODIFIED
FROM DIRECTORY(@AD_OUTPUT_STAGE)
WHERE RELATIVE_PATH LIKE '%_product_%.png'
ORDER BY RELATIVE_PATH
LIMIT 20;

SELECT 'Test complete!' AS status;


