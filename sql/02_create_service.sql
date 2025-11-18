-- ============================================================================
-- Filename: 02_create_service.sql
-- Description: Create SPCS service for SAM inference
--
-- Prerequisites: 
--   - Infrastructure setup complete (01_setup_infrastructure.sql)
--   - Docker image built and pushed to ML_INFERENCE_REPO
--   - SAM checkpoint uploaded to MODEL_STAGE
-- Creates: SAM_INFERENCE_SERVICE
-- ============================================================================

USE DATABASE SHALION_HF_DEMO;
USE SCHEMA PRODUCT_EXTRACTION;

-- Get image repository URL (you'll need this to construct the full image path)
-- Run: SHOW IMAGE REPOSITORIES LIKE 'ML_INFERENCE_REPO';
-- Expected format: <org-account>.registry.snowflakecomputing.com/shalion_hf_demo/product_extraction/ml_inference_repo

-- Step 1: Create the inference service

CREATE SERVICE IF NOT EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE
    IN COMPUTE POOL SAM_GPU_POOL
    FROM SPECIFICATION $$
    spec:
      containers:
      - name: sam-inference
        image: /shalion_hf_demo/product_extraction/ml_inference_repo/sam-inference:latest
        env:
          MODEL_PATH: /models/sam_vit_h_4b8939.pth
          AD_OUTPUT_STAGE_URL: "@SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_OUTPUT_STAGE/"
        volumeMounts:
        - name: model-weights
          mountPath: /models
        - name: input-stage
          mountPath: /input
        - name: output-stage
          mountPath: /output
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
      endpoints:
      - name: inference
        port: 8080
        public: true
      volumes:
      - name: model-weights
        source: stage
        stageConfig:
          name: "@SHALION_HF_DEMO.PRODUCT_EXTRACTION.MODEL_STAGE"
      - name: input-stage
        source: stage
        stageConfig:
          name: "@SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE"
      - name: output-stage
        source: stage
        stageConfig:
          name: "@SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_OUTPUT_STAGE"
    $$
    MIN_INSTANCES = 1
    MAX_INSTANCES = 1
    COMMENT = 'SAM inference service for product extraction from ads';

SELECT '✓ Service created: SAM_INFERENCE_SERVICE' AS progress;

-- Step 2: Check service status
-- Wait for status to show READY (may take 2-5 minutes for first start)
CALL SYSTEM$GET_SERVICE_STATUS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE');

SELECT 'Service deployment initiated. Use SYSTEM$GET_SERVICE_STATUS to check readiness.' AS status;

-- ============================================================================
-- TROUBLESHOOTING:
-- - Check logs: CALL SYSTEM$GET_SERVICE_LOGS('SAM_INFERENCE_SERVICE', 0, 'sam-inference', 100);
-- - Check status: SELECT SYSTEM$GET_SERVICE_STATUS('SAM_INFERENCE_SERVICE');
-- - List running services: SHOW SERVICES IN SCHEMA PRODUCT_EXTRACTION;
-- ============================================================================

