-- ============================================================================
-- Filename: 01_setup_infrastructure.sql
-- Description: Create database, schemas, stages, warehouse, image repo, and compute pool
--
-- Prerequisites: ACCOUNTADMIN or SYSADMIN role
-- Creates: Database, stages, warehouse, GPU compute pool
-- ============================================================================

-- Use SYSADMIN for database, warehouse, and stage creation
USE ROLE SYSADMIN;

-- Step 1: Create database and schema
CREATE DATABASE IF NOT EXISTS SHALION_HF_DEMO
    COMMENT = 'Product extraction from ad images using SAM 2 on SPCS';

CREATE SCHEMA IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION
    COMMENT = 'Schema for SAM 2 product segmentation pipeline';

SELECT '✓ Database and schema created' AS progress;

-- Step 2: Create warehouse for orchestration and cropping operations
CREATE WAREHOUSE IF NOT EXISTS SAM_DEMO_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for SAM demo operations';

SELECT '✓ Warehouse created' AS progress;

-- Step 3: Create stages for input images, output crops, and code
CREATE STAGE IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for input ad images';

CREATE STAGE IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_OUTPUT_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for cropped product images output';

CREATE STAGE IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.CODE_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for service spec and code files';

SELECT '✓ Stages created' AS progress;

-- Step 4: Create image repository for container images
CREATE IMAGE REPOSITORY IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.ML_INFERENCE_REPO
    COMMENT = 'Container image repository for SAM 2 detection service';

SHOW IMAGE REPOSITORIES LIKE 'ML_INFERENCE_REPO' IN SCHEMA SHALION_HF_DEMO.PRODUCT_EXTRACTION;

SELECT '✓ Image repository created' AS progress;

-- Step 5: Create GPU compute pool (requires ACCOUNTADMIN)
USE ROLE ACCOUNTADMIN;
USE DATABASE SHALION_HF_DEMO;
USE SCHEMA PRODUCT_EXTRACTION;

CREATE COMPUTE POOL IF NOT EXISTS SAM_GPU_POOL
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = GPU_NV_S
    AUTO_RESUME = TRUE
    AUTO_SUSPEND_SECS = 3600
    COMMENT = 'GPU compute pool for SAM 2 detection (1x NVIDIA A10G)';

SELECT '✓ GPU compute pool created' AS progress;

-- Grant compute pool usage to SYSADMIN for service creation
GRANT USAGE ON COMPUTE POOL SAM_GPU_POOL TO ROLE SYSADMIN;
GRANT MONITOR ON COMPUTE POOL SAM_GPU_POOL TO ROLE SYSADMIN;

SHOW COMPUTE POOLS LIKE 'SAM_GPU_POOL';

-- Switch back to SYSADMIN for subsequent scripts
USE ROLE SYSADMIN;

SELECT 'Infrastructure setup complete!' AS status;

-- ============================================================================
-- NEXT STEPS:
-- 1. Build and push Docker container image (see DEPLOYMENT.md)
-- 2. Upload service_spec.yaml to CODE_STAGE
-- 3. Run 02_create_service.sql to deploy the detection service
-- 4. Run 03_create_orchestrator.sql to create the extraction procedure
-- ============================================================================
