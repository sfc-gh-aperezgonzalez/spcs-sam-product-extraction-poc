-- ============================================================================
-- Filename: 01_setup_infrastructure.sql
-- Description: Create database, schemas, stages, warehouse, image repo, and compute pool
--
-- Prerequisites: ACCOUNTADMIN role (or sufficient privileges)
-- Creates: SHALION_HF_DEMO database, stages, warehouse, GPU compute pool
-- ============================================================================

-- Step 1: Create database and schema
CREATE DATABASE IF NOT EXISTS SHALION_HF_DEMO
    COMMENT = 'Product extraction from ad images using SAM on SPCS';

CREATE SCHEMA IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION
    COMMENT = 'Schema for SAM-based product segmentation pipeline';

SELECT '✓ Database and schema created' AS progress;

-- Step 2: Create warehouse for notebook and admin operations
CREATE WAREHOUSE IF NOT EXISTS SAM_DEMO_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for SAM demo notebook operations';

SELECT '✓ Warehouse created' AS progress;

-- Step 3: Create stages for input images, output crops, and model weights
-- AD_INPUT_STAGE: Original ad images go here
CREATE STAGE IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for input ad images';

-- AD_OUTPUT_STAGE: Cropped product images written here by inference service
CREATE STAGE IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_OUTPUT_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for cropped product images output';

-- MODEL_STAGE: SAM vit-h checkpoint weights stored here
CREATE STAGE IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.MODEL_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for SAM model checkpoint files';

SELECT '✓ Stages created (AD_INPUT_STAGE, AD_OUTPUT_STAGE, MODEL_STAGE)' AS progress;

-- Step 4: Create image repository for container images
CREATE IMAGE REPOSITORY IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.ML_INFERENCE_REPO
    COMMENT = 'Container image repository for SAM inference service';

-- Show repository URL for later use
SHOW IMAGE REPOSITORIES LIKE 'ML_INFERENCE_REPO' IN SCHEMA SHALION_HF_DEMO.PRODUCT_EXTRACTION;

SELECT '✓ Image repository created' AS progress;

-- Step 5: Create GPU compute pool
-- Using GPU_NV_S: 1 NVIDIA A10G GPU with 24GB memory
-- Ideal for SAM inference workloads
-- Note: Compute pools must be created in current schema context (no fully qualified names)
USE DATABASE SHALION_HF_DEMO;
USE SCHEMA PRODUCT_EXTRACTION;

CREATE COMPUTE POOL IF NOT EXISTS SAM_GPU_POOL
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = GPU_NV_S
    AUTO_RESUME = TRUE
    AUTO_SUSPEND_SECS = 3600
    COMMENT = 'GPU compute pool for SAM inference service (1x NVIDIA A10G)';

SELECT '✓ GPU compute pool created (SAM_GPU_POOL)' AS progress;

-- Step 6: Show compute pool status
SHOW COMPUTE POOLS LIKE 'SAM_GPU_POOL';

SELECT 'Infrastructure setup complete!' AS status;

-- ============================================================================
-- NEXT STEPS:
-- 1. Download SAM vit-h checkpoint (see README.md for instructions)
-- 2. Upload checkpoint to MODEL_STAGE
-- 3. Build and push Docker container image
-- 4. Run 02_create_service.sql to deploy the inference service
-- ============================================================================

