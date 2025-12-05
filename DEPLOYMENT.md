# Deployment Guide

Step-by-step instructions for deploying SAM 2 product extraction for brand classification.

**Purpose:** Extract clean product regions from retail ads to improve Cortex brand classification  
**Architecture:** Stored Procedure → GPU Detection Service (HuggingFace SAM 2) → Warehouse Cropping → Output Stage

## Prerequisites Checklist

- [ ] Snowflake account with SPCS GPU support
- [ ] SnowCLI installed and configured as ACCOUNTADMIN
- [ ] Docker installed locally
- [ ] ~2GB free disk space for HuggingFace model
- [ ] Sample ad images for testing

## Deployment Steps

### 1. Download HuggingFace Model (~3 minutes)

```bash
# Create directory for HuggingFace model
mkdir -p docker/hf_models

# Download SAM 2.1 model using HuggingFace CLI
pip install huggingface_hub
huggingface-cli download facebook/sam2.1-hiera-small \
    --local-dir docker/hf_models/sam2.1-hiera-small \
    --local-dir-use-symlinks False

# Verify download
ls -la docker/hf_models/sam2.1-hiera-small/
```

### 2. Setup Snowflake Infrastructure (~2 minutes)

```bash
# Create database, stages, warehouse, compute pool
snow sql -f sql/01_setup_infrastructure.sql
```

**Verify GPU compute pool status:**
```bash
snow sql -q "DESCRIBE COMPUTE POOL SAM_GPU_POOL;"
```
Expected state: `STARTING` → `IDLE` (takes ~2 minutes)

### 3. Get Image Repository URL (~1 minute)

```bash
# Get your repository URL
snow sql -q "SHOW IMAGE REPOSITORIES LIKE 'ML_INFERENCE_REPO' IN SCHEMA SHALION_HF_DEMO.PRODUCT_EXTRACTION;"
```

**Save this URL** - you'll need it in steps 4 and 5.  
Format: `orgname-accountname.registry.snowflakecomputing.com/shalion_hf_demo/product_extraction/ml_inference_repo`

### 4. Build and Push Docker Image (~10 minutes)

```bash
# Login to Snowflake container registry
snow spcs image-registry login

# Build the Docker image for x86_64 (required for Snowflake GPU)
# Note: Build from project root, not docker/ directory
docker build --platform linux/amd64 -f docker/Dockerfile -t sam2-detector:latest .

# Get your registry URL
REGISTRY_URL=$(snow spcs image-registry url)
echo $REGISTRY_URL

# Tag for Snowflake
docker tag sam2-detector:latest ${REGISTRY_URL}/shalion_hf_demo/product_extraction/ml_inference_repo/sam2-detector:latest

# Push to Snowflake
docker push ${REGISTRY_URL}/shalion_hf_demo/product_extraction/ml_inference_repo/sam2-detector:latest
```

### 5. Upload Service Spec to Stage (~1 minute)

```bash
# Upload service_spec.yaml to CODE_STAGE
snow stage copy docker/service_spec.yaml @SHALION_HF_DEMO.PRODUCT_EXTRACTION.CODE_STAGE --overwrite

# Verify upload
snow stage list-files @SHALION_HF_DEMO.PRODUCT_EXTRACTION.CODE_STAGE
```

### 6. Deploy Detection Service (~3-5 minutes)

```bash
# Create the detection service and functions
snow sql -f sql/02_create_service.sql

# Wait for service to become READY (takes 2-5 minutes)
snow sql -q "CALL SYSTEM\$GET_SERVICE_STATUS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM2_DETECTOR_SERVICE');"
```

**Expected output**: `"status": "READY"`

**Check logs if issues:**
```bash
snow sql -q "CALL SYSTEM\$GET_SERVICE_LOGS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM2_DETECTOR_SERVICE', 0, 'sam2-detector', 100);"
```

### 7. Create Orchestrator Procedure (~1 minute)

```bash
# Create the EXTRACT_PRODUCTS stored procedure
snow sql -f sql/03_create_orchestrator.sql
```

### 8. Upload Test Images (~1 minute)

```bash
# Upload sample ad images to AD_INPUT_STAGE/ads folder
# Option A: Use the included sample_images folder
snow stage copy sample_images/ @SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE/ads/ --overwrite

# Option B: Use your own images
snow stage copy /path/to/your/images/*.jpg @SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE/ads/ --overwrite

# Verify upload
snow stage list-files @SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE/ads/
```

### 9. Test Extraction (~2 minutes)

```bash
# Run extraction on test images
snow sql -q "CALL SHALION_HF_DEMO.PRODUCT_EXTRACTION.EXTRACT_PRODUCTS('ads', '@AD_OUTPUT_STAGE');"

# Check output
snow stage list-files @SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_OUTPUT_STAGE
```

## Total Time: ~25-30 minutes

## Verification

After deployment complete, verify:
- ✅ Service status shows "READY"
- ✅ EXTRACT_PRODUCTS returns JSON with product counts
- ✅ Cropped products appear in AD_OUTPUT_STAGE
- ✅ Output filenames follow pattern: `{original}_product_{index}.png`

## Troubleshooting

### Service won't start
```bash
# Check detailed status
snow sql -q "CALL SYSTEM\$GET_SERVICE_STATUS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM2_DETECTOR_SERVICE');"

# Check container logs
snow sql -q "CALL SYSTEM\$GET_SERVICE_LOGS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM2_DETECTOR_SERVICE', 0, 'sam2-detector', 100);"
```

### GPU pool won't start
```bash
# Check compute pool status
snow sql -q "DESCRIBE COMPUTE POOL SAM_GPU_POOL;"

# Check available GPU instance families
snow sql -q "SHOW COMPUTE POOL INSTANCE FAMILIES;"
```

### HuggingFace model not found
```bash
# Verify model files exist in Docker build context
ls -la docker/hf_models/sam2.1-hiera-small/

# Check container logs for model loading errors
snow sql -q "CALL SYSTEM\$GET_SERVICE_LOGS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM2_DETECTOR_SERVICE', 0, 'sam2-detector', 100);"
```

### Image push fails
```bash
# Re-login to registry
snow spcs image-registry login

# Verify image repository exists
snow sql -q "SHOW IMAGE REPOSITORIES IN SCHEMA SHALION_HF_DEMO.PRODUCT_EXTRACTION;"
```

## Clean Up (Optional)

To remove everything:

```sql
-- Drop the service
DROP SERVICE IF EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM2_DETECTOR_SERVICE;

-- Suspend compute pool
ALTER COMPUTE POOL SAM_GPU_POOL SUSPEND;

-- Drop database (includes all stages, procedures, etc.)
DROP DATABASE IF EXISTS SHALION_HF_DEMO CASCADE;

-- Drop compute pool
DROP COMPUTE POOL IF EXISTS SAM_GPU_POOL;
```

## Next Steps

After successful deployment:

1. **Test with real ad images:** Upload your retail ads and verify extraction quality
2. **Build Cortex workflow:** 
   - Embed extracted crops with `AI_EMBED`
   - Classify brands with Cortex Search
3. **Optimize costs:**
   - Monitor actual credit consumption
   - Tune GPU auto-suspend settings
   - Batch process images for efficiency
4. **Scale to production:**
   - Increase `MAX_NODES` for parallel processing
   - Create Snowflake Tasks for automation
   - Set up monitoring and alerting

