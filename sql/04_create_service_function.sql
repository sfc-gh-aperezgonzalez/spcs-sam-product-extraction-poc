-- ============================================================================
-- Filename: 04_create_service_function.sql
-- Description: Create service function to call SAM inference from SQL/notebooks
--
-- Prerequisites: SAM_INFERENCE_SERVICE running and ready with public endpoint
-- Creates: EXTRACT_PRODUCTS function
-- ============================================================================

USE DATABASE SHALION_HF_DEMO;
USE SCHEMA PRODUCT_EXTRACTION;

-- Step 1: Grant service role for endpoint access
GRANT SERVICE ROLE SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE!ALL_ENDPOINTS_USAGE 
    TO ROLE ACCOUNTADMIN;

SELECT '✓ Service role granted' AS progress;

-- Step 2: Create native service function
-- Uses Snowflake's native SERVICE= syntax to bind directly to SPCS endpoint
-- This is NOT Python code - it's a Snowflake SQL definition that binds to the FastAPI endpoint
-- The actual processing code is in app/main.py, app/inference.py, app/utils.py
-- Returns JSON string with all crop URLs and metadata
CREATE OR REPLACE FUNCTION SHALION_HF_DEMO.PRODUCT_EXTRACTION.EXTRACT_PRODUCTS(
    IMAGE_PATH VARCHAR,
    OUTPUT_PREFIX VARCHAR
)
RETURNS VARCHAR
SERVICE = SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE
ENDPOINT = inference
MAX_BATCH_ROWS = 1
AS '/infer';

SELECT '✓ Function EXTRACT_PRODUCTS created' AS progress;

-- Step 3: Test the function with sample image
-- SELECT SHALION_HF_DEMO.PRODUCT_EXTRACTION.EXTRACT_PRODUCTS(
--     '@SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE/demo/1.png',
--     'test/'
-- );

SELECT 'Service function ready!' AS status;
