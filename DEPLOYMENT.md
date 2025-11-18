# Deployment Guide

Step-by-step instructions for deploying SAM product extraction for brand classification.

**Purpose:** Extract clean product regions from retail ads to improve Cortex brand classification  
**Architecture:** SQL function → SPCS GPU service → SAM → RGB bounding box crops → Cortex embeddings

## Prerequisites Checklist

- [ ] Snowflake account with SPCS GPU support
- [ ] SnowCLI installed and configured as ACCOUNTADMIN
- [ ] Docker installed locally
- [ ] ~3GB free disk space for model checkpoint
- [ ] Sample ad images for testing

## Deployment Steps

### 1. Download SAM Model Checkpoint (~5 minutes)

```bash
# Create directory for model weights
mkdir -p model_weights
cd model_weights

# Download SAM vit-h checkpoint (~2.4GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Verify download
ls -lh sam_vit_h_4b8939.pth

cd ..
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

### 3. Upload Model Checkpoint to Snowflake (~3 minutes)

```bash
# Upload checkpoint to MODEL_STAGE (~2.5GB file)
snow stage copy \
  model_weights/sam_vit_h_4b8939.pth \
  @SHALION_HF_DEMO.PRODUCT_EXTRACTION.MODEL_STAGE \
  --overwrite

# Verify upload
snow stage list-files @SHALION_HF_DEMO.PRODUCT_EXTRACTION.MODEL_STAGE
```

### 4. Upload Sample Images (~1 minute)

```bash
# Manually upload sample ad images to the AD_INPUT_STAGE stage (JPEG or PNG)

# Verify upload
snow stage list-files @SHALION_HF_DEMO.PRODUCT_EXTRACTION.AD_INPUT_STAGE/demo/
```

### 5. Get Image Repository URL (~1 minute)

```bash
# Get your repository URL
snow sql -q "SHOW IMAGE REPOSITORIES LIKE 'ML_INFERENCE_REPO' IN SCHEMA SHALION_HF_DEMO.PRODUCT_EXTRACTION;"
```

**Save this URL** - you'll need it in steps 6 and 7.  
Format: `orgname-accountname.registry.snowflakecomputing.com/shalion_hf_demo/product_extraction/ml_inference_repo`

### 6. Build and Push Docker Image (~10 minutes)

```bash
# Login to Snowflake container registry
snow spcs image-registry login

# Build the Docker image for x86_64 (required for Snowflake GPU)
docker build --platform linux/amd64 -f docker/Dockerfile -t sam-inference:latest .

# Get your registry URL
REGISTRY_URL=$(snow spcs image-registry url)
echo $REGISTRY_URL

# Tag for Snowflake
docker tag sam-inference:latest ${REGISTRY_URL}/shalion_hf_demo/product_extraction/ml_inference_repo/sam-inference:latest

# Push to Snowflake
docker push ${REGISTRY_URL}/shalion_hf_demo/product_extraction/ml_inference_repo/sam-inference:latest
```

### 7. Deploy SPCS Service (~3-5 minutes)

```bash
# Create the inference service
snow sql -f sql/02_create_service.sql

# Wait for service to become READY (takes 2-5 minutes)
snow sql -q "CALL SYSTEM\$GET_SERVICE_STATUS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE');"
```

**Expected output**: `"status": "READY"`

**If not ready yet:** Wait 2 minutes and run the status check again.

**Check logs if issues:**
```bash
snow sql -q "CALL SYSTEM\$GET_SERVICE_LOGS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE', 0, 'sam-inference', 100);"
```

### 8. Create Service Function (~1 minute)

```bash
snow sql -f sql/04_create_service_function.sql
```

This creates `EXTRACT_PRODUCTS()` - a SQL-callable function that:
- Extracts product regions with SAM
- Returns RGB bounding box crops (no transparency)
- Includes metadata for downstream Cortex filtering

### 9. Run Demo Notebook (~5 minutes)

1. Open Snowsight (Snowflake web UI)
2. Navigate to **Projects** → **Notebooks**
3. Click **+ Notebook** → **Import .ipynb file**
4. Upload `notebooks/sam_product_extraction_demo.ipynb`
5. Select warehouse: `SAM_DEMO_WH`
6. Run all cells

## Total Time: ~30-40 minutes

## Verification

After deployment complete, verify:
- ✅ Service status shows "READY"
- ✅ Function returns JSON with crops and metadata
- ✅ Crops are RGB images (bounding boxes, no transparency)
- ✅ `product_likely` flag indicates product vs. non-product

## Troubleshooting

### Service won't start
```bash
# Check detailed status
snow sql -q "CALL SYSTEM\$GET_SERVICE_STATUS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE');"

# Check container logs
snow sql -q "CALL SYSTEM\$GET_SERVICE_LOGS('SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE', 0, 'sam-inference', 100);"
```

### GPU pool won't start
```bash
# Check compute pool status
snow sql -q "DESCRIBE COMPUTE POOL SAM_GPU_POOL;"

# Check available GPU instance families
snow sql -q "SHOW COMPUTE POOL INSTANCE FAMILIES;"
```

### Model checkpoint not found in container
```bash
# Verify checkpoint is in stage
snow stage list @SHALION_HF_DEMO.PRODUCT_EXTRACTION.MODEL_STAGE

# Check volume mount in service spec (sql/02_create_service.sql line 37)
```

### Image push fails
```bash
# Re-login to registry
snow registry login

# Verify image repository exists
snow sql -q "SHOW IMAGE REPOSITORIES IN SCHEMA SHALION_HF_DEMO.PRODUCT_EXTRACTION;"
```

## Clean Up (Optional)

To remove everything:

```sql
-- Drop the service
DROP SERVICE IF EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_INFERENCE_SERVICE;

-- Suspend compute pool
ALTER COMPUTE POOL SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_GPU_POOL SUSPEND;

-- Drop database (includes all stages, procedures, etc.)
DROP DATABASE IF EXISTS SHALION_HF_DEMO CASCADE;

-- Drop compute pool
DROP COMPUTE POOL IF EXISTS SHALION_HF_DEMO.PRODUCT_EXTRACTION.SAM_GPU_POOL;
```

## Next Steps

After successful deployment:

1. **Test with real ad images:** Upload your retail ads and verify extraction quality
2. **Build Cortex workflow:** 
   - Embed extracted crops with `AI_EMBED`
   - Filter non-products using reference embeddings
   - Classify brands with Cortex Search
3. **Optimize costs:**
   - Monitor actual credit consumption
   - Tune GPU auto-suspend settings
   - Batch process images for efficiency
4. **Scale to production:**
   - Increase `MAX_NODES` for parallel processing
   - Create Snowflake Tasks for automation
   - Set up monitoring and alerting

