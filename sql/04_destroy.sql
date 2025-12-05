-- ============================================================================
-- Filename: 04_destroy.sql
-- Description: Remove all resources created by this project
--
-- WARNING: This will permanently delete all data, images, and resources!
--
-- Removes:
--   - SAM2_DETECTOR_SERVICE (SPCS service)
--   - SAM_GPU_POOL (GPU compute pool)
--   - SHALION_HF_DEMO database (includes all stages, procedures, functions)
--   - SAM_DEMO_WH warehouse
-- ============================================================================

-- Step 1: Drop the SPCS service (requires ACCOUNTADMIN)
USE ROLE ACCOUNTADMIN;

SELECT '=== Step 1: Dropping SPCS Service ===' AS step;
DROP SERVICE IF EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM2_DETECTOR_SERVICE;
SELECT '✓ Service dropped (or did not exist)' AS progress;

-- Step 2: Suspend and drop the GPU compute pool
SELECT '=== Step 2: Dropping GPU Compute Pool ===' AS step;
ALTER COMPUTE POOL IF EXISTS SAM_GPU_POOL STOP ALL;
ALTER COMPUTE POOL IF EXISTS SAM_GPU_POOL SUSPEND;
DROP COMPUTE POOL IF EXISTS SAM_GPU_POOL;
SELECT '✓ Compute pool dropped (or did not exist)' AS progress;

-- Step 3: Drop the database (includes all schemas, stages, functions, procedures)
USE ROLE SYSADMIN;

SELECT '=== Step 3: Dropping Database ===' AS step;
DROP DATABASE IF EXISTS SHALION_HF_DEMO CASCADE;
SELECT '✓ Database dropped (or did not exist)' AS progress;

-- Step 4: Drop the warehouse
SELECT '=== Step 4: Dropping Warehouse ===' AS step;
DROP WAREHOUSE IF EXISTS SAM_DEMO_WH;
SELECT '✓ Warehouse dropped (or did not exist)' AS progress;

SELECT 'All project resources have been removed!' AS status;

-- ============================================================================
-- NOTE: This script does NOT remove:
--   - Docker images in Snowflake container registry (they are removed with the repo)
--   - Local Docker images (use: docker rmi sam2-detector:latest)
--   - Local model files (docker/hf_models/)
-- ============================================================================

