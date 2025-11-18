# Snowflake SPCS + SAM Product Extraction for Brand Classification

Extract clean product regions from retail ad images to improve Cortex Search brand classification accuracy.

**Use Case:** Pre-process supermarket ad images to remove template noise before embedding/brand matching  
**Technology:** SAM vit-h on Snowflake SPCS GPU  

## Demo Video



> Watch the SAM product extraction in action - extracting clean product regions from retail ads in ~5 seconds per image on Snowflake SPCS GPU.

---

## The Problem This Solves

**Challenge:** Retail ad images have large template backgrounds that dominate embeddings:
- 60-70% template area (store logos, graphics, promotional text)
- 30-40% actual product
- Result: Cortex Search embeddings similarity → wrong brand classification ❌

**Solution:** SAM extracts product regions → embeddings focus on products → correct brands ✅

---

## Quick Start

```sql
-- Extract product regions from an ad image
SELECT SHALION_HF_DEMO.PRODUCT_EXTRACTION.EXTRACT_PRODUCTS(
    '@AD_INPUT_STAGE/my_ad.jpg',
    'output/'
);
```

**Returns:**
```json
{
  "crops": [
    "@AD_OUTPUT_STAGE/output/product_000.png",
    "@AD_OUTPUT_STAGE/output/product_001.png",
    "@AD_OUTPUT_STAGE/output/product_002.png"
  ],
  "num_products": 3, 
  "product_likely": true,
  "metadata": [
    {"crop_url": "...", "area_ratio": 0.234, "bbox": [120, 80, 400, 350], "confidence": 0.96},
    {"crop_url": "...", "area_ratio": 0.152, "bbox": [550, 100, 380, 320], "confidence": 0.94},
    {"crop_url": "...", "area_ratio": 0.089, "bbox": [200, 500, 300, 250], "confidence": 0.91}
  ]
}
```

**What you get:**
- Clean product crops (RGB bounding boxes, no transparency)
- Template/background removed
- Metadata for downstream Cortex filtering
- Top 3 candidates per ad (sorted by size)

---

## Cost Analysis

### SAM Extraction Costs (PoC Observations)

**SPCS GPU Configuration:**
- Instance: GPU_NV_S (NVIDIA A10G, 24GB VRAM)
- Credit rate: **0.57 credits/hour** (per Snowflake pricing table)
- Processing time: **~5 seconds per image** (measured across multiple test runs)
  - Test results: 4-7 seconds per image (average: 5.1 seconds)
  - Includes full inference cycle: image load, SAM processing, crop generation, stage upload

**Per-Image Cost Calculation:**
- Credits per image: (5 seconds / 3600) × 0.57 = **~0.0008 credits**
- Cost per image: 0.0008 credits × credit price

**Cost at different credit prices:**
| Credit Price | Cost per Image | Cost per 1000 | Cost per 1M |
|--------------|----------------|---------------|-------------|
| $2/credit | $0.0016 | $1.60 | $1,600 |
| $3/credit | $0.0024 | $2.40 | $2,400 |
| $4/credit | $0.0032 | $3.20 | $3,200 |

### Cost at Scale (Range Based on Credit Pricing)

| Volume | Low ($2/credit) | Mid ($3/credit) | High ($4/credit) |
|--------|-----------------|-----------------|------------------|
| **10K images** | $16 | $24 | $32 |
| **100K images** | $160 | $240 | $320 |
| **1M images** | $1,600 | $2,400 | $3,200 |
| **10M images/year** | $16,000 | $24,000 | $32,000 |

**Cost range for 1M images: $1,600 - $3,200**

### Important Disclaimers

**These are estimates based on:**
- PoC testing with GPU_NV_S instance
- Average **5 seconds processing time** per retail ad image (tested across multiple images)
- Snowflake credit pricing varies by account and contract
- Credit consumption from official [Snowflake Credit Table](https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf)

**Production costs may vary due to:**
- Image complexity (simple ads may process in 4s, complex in 7s)
- Batch processing optimizations (parallel execution)
- GPU instance selection (XS/S/M pricing differs)
- Auto-suspend configuration (minimize idle time)
- Actual credit rates in your Snowflake account

**Always monitor actual credit consumption** in production and adjust budgets accordingly.

---

## Architecture

```
SQL Query
    ↓
EXTRACT_PRODUCTS() Service Function
    ↓
SPCS Service (NVIDIA A10G GPU)
    ├─ SAM vit-h model
    ├─ Mounted volumes: /input, /output, /models
    ├─ Generate masks (40-80 per image)
    ├─ Filter with heuristics (area, aspect, quality)
    ├─ Return top 3 candidates
    └─ RGB bounding box crops (no transparency)
    ↓
JSON Response
    ├─ crops: ["@stage/product_000.png", ...]
    ├─ metadata: [{area_ratio, confidence, bbox}, ...]
    └─ product_likely: true/false
    ↓
Cortex Embeddings (on clean crops)
    ↓
Cortex Search (brand classification)
```

---

## Key Features

### 1. Template Noise Removal
- **Before:** Embed full 1200×800 ad (60% template, 40% product)
- **After:** Embed 400×350 product crop (100% product)
- **Result:** Embeddings match product features, not template

### 2. Non-Product Filtering
- Landscapes, graphics, banners → filtered out
- `product_likely` flag in response
- Saves Cortex costs on non-product images

### 3. Production-Grade Filtering
- Area: 1-70% of image
- Aspect ratio: 1:6 to 6:1 (no ribbons/edges)
- Quality: pred_iou ≥ 0.80, stability ≥ 0.90
- Border detection: drops 2+ edge touching masks
- Deduplication: 25% overlap threshold
- Returns: Top 3 candidates for Cortex selection

### 4. Metadata for Intelligent Filtering
- `area_ratio`: Size relative to original
- `confidence`: SAM quality score
- `bbox`: Position coordinates
- Enables SQL-based business logic filtering

---

## Tested Results

**5 real retail ads processed:**
- Bambi boots (holiday): 2 crops
- Gillette razors: 3 crops
- Jacobs coffee bundle: 3 crops
- SHOKZ headphones: 1 crop
- SHOKZ duplicate: 3 crops

**Performance:**
- Processing: **~5 seconds per image** (4-7 second range)
- Initial masks: 40-80 per image
- After filtering: 1-3 products per image
- Noise reduction: ~95% (80 → 2 masks)

---

## Setup

**Complete deployment guide:** See [DEPLOYMENT.md](DEPLOYMENT.md)

**Quick steps:**
1. Download SAM checkpoint (2.5GB, one-time)
2. Run SQL infrastructure setup (5 min)
3. Upload checkpoint and images (5 min)
4. Build and push Docker image (10 min)
5. Deploy SPCS service (5 min)
6. Create service function (1 min)
7. Test extraction

**Total time:** 30-40 minutes

---

## Usage

### SQL (Simplest)

```sql
-- Extract from one image
SELECT EXTRACT_PRODUCTS('@AD_INPUT_STAGE/ad1.jpg', 'results/');

-- Batch process with filtering
WITH extractions AS (
    SELECT 
        image_id,
        PARSE_JSON(EXTRACT_PRODUCTS(image_path, image_id || '/')) AS result
    FROM ad_images
)
SELECT 
    image_id,
    result:crops AS crop_urls,
    result:product_likely AS is_product,
    result:metadata AS crop_info
FROM extractions
WHERE result:product_likely = TRUE;
```

### Python (Snowflake Notebook)

```python
import json

# Extract products
result_str = session.sql("""
    SELECT EXTRACT_PRODUCTS('@AD_INPUT_STAGE/ad1.jpg', 'output/')
""").collect()[0][0]

result = json.loads(result_str)
print(f"Found {result['num_products']} products")
print(f"Product-likely: {result['product_likely']}")

# Access metadata
for crop_meta in result['metadata']:
    print(f"Crop: {crop_meta['crop_url']}")
    print(f"  Size: {crop_meta['area_ratio']:.1%} of image")
    print(f"  Quality: {crop_meta['confidence']:.2f}")
```

---

## Cost Optimization Tips

1. **Batch processing:** Process images in parallel using Snowflake Tasks
2. **Auto-suspend:** GPU pool suspends after 1 hour idle (no waste)
3. **Pre-filtering:** Skip obvious non-products before SAM (save GPU costs)
4. **Right-sizing:** Use smallest GPU that meets latency needs
5. **Monitoring:** Track actual credit consumption, adjust estimates

---

## Technical Specifications

### SPCS Infrastructure
- **Compute Pool:** GPU_NV_S (1× NVIDIA A10G, 24GB VRAM)
- **Volumes:** 3 mounted stages (MODEL, INPUT, OUTPUT)
- **Throughput:** ~240 images/hour per GPU
- **Auto-scaling:** Can increase MAX_NODES for parallel processing

### SAM Configuration
- **Model:** SAM vit-h (2.5GB checkpoint)
- **Parameters:** 
  - points_per_side: 24 (anti-fragmentation)
  - pred_iou_thresh: 0.92
  - stability_score_thresh: 0.93
  - box_nms_thresh: 0.5 (aggressive merging)
  - min_mask_region_area: 1000

### Output Format
- **Image format:** PNG (RGB, no alpha channel)
- **Crop type:** Bounding box (preserves all pixels)
- **Typical size:** 25KB - 200KB per crop
- **Candidates:** Top 3 per ad (for Cortex selection)

---

## Value Proposition

### What This Enables

**Primary value:**
- Removes template noise from ad images
- Extracts clean product regions for accurate embeddings
- Filters non-product images (saves downstream costs)
- All processing stays in Snowflake

**Cost efficiency:**
- SAM preprocessing: **$1.60-$3.20 per 1000 images** (GPU_NV_S at $2-4/credit)
- Enables accurate downstream brand classification
- Scales to millions of images
- Note: Cortex embedding/search costs are separate and depend on your usage

---

## Project Structure

```
.
├── README.md                          # This file
├── DEPLOYMENT.md                      # Step-by-step setup
├── sql/
│   ├── 01_setup_infrastructure.sql    # Database, stages, GPU pool
│   ├── 02_create_service.sql          # SPCS service with volumes
│   └── 04_create_service_function.sql # Service function definition
├── docker/
│   ├── Dockerfile                     # CUDA + PyTorch + SAM
│   └── requirements.txt               # FastAPI, OpenCV, NumPy
├── app/
│   ├── main.py                        # FastAPI endpoint (52 lines)
│   ├── inference.py                   # SAM engine (146 lines)
│   └── utils.py                       # Filtering + I/O (220 lines)
└── notebooks/
    └── sam_product_extraction_demo.ipynb  # Demo notebook
```

**Total:** 13 essential files, ~500 lines of Python

---

## Filtering Heuristics

SAM generates 40-100 masks per image. We filter to 1-3 product candidates using:

**Area bounds:**
- Min: 1% of image (removes logos, text, decorations)
- Max: 70% of image (removes full backgrounds)

**Aspect ratio:**
- Range: 1:6 to 6:1 (drops thin ribbons, wide banners)

**Quality:**
- Predicted IoU: ≥ 0.80
- Stability score: ≥ 0.90

**Border detection:**
- Drops masks touching 2+ edges (likely borders/backgrounds)

**Deduplication:**
- Removes masks with >25% overlap (keeps largest)

**Output:**
- Top 3 candidates (largest first)
- Downstream Cortex filtering selects final winner

---

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for complete step-by-step instructions.

---

## Clean Up

```sql
-- Suspend GPU pool (stops charges)
ALTER COMPUTE POOL SAM_GPU_POOL SUSPEND;

-- Drop everything
DROP SERVICE SAM_INFERENCE_SERVICE;
DROP COMPUTE POOL SAM_GPU_POOL;
DROP DATABASE SHALION_HF_DEMO CASCADE;
```

---

## References

- [SAM Model](https://github.com/facebookresearch/segment-anything) - Segment Anything
- [SPCS](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview) - Snowpark Container Services
- [Service Functions](https://docs.snowflake.com/en/sql-reference/sql/create-function-spcs) - SQL-callable services
- [Cortex Search](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview) - Vector similarity search
- [Stage Volumes](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/snowflake-stage-volume) - Mounted storage

---

**Built by:** Alejandro Perez - Solution Engineer  
