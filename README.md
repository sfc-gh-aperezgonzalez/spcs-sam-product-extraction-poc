# Snowflake SPCS + SAM 2 Product Extraction for Brand Classification

Extract clean product regions from retail ad images to improve Cortex Search brand classification accuracy.

**Use Case:** Pre-process supermarket ad images to remove template noise before embedding/brand matching  
**Technology:** SAM 2 (HuggingFace) on Snowflake SPCS GPU

### Why SAM 2?

[SAM 2 (Segment Anything Model 2)](https://huggingface.co/facebook/sam2.1-hiera-small) is Meta's state-of-the-art foundation model for image segmentation. It excels at identifying and isolating objects in images **without requiring task-specific training**. This makes it ideal for retail product extraction because:

- **Zero-shot segmentation**: Detects products of any shape, size, or category without fine-tuning
- **Robust to noise**: Handles complex promotional backgrounds, logos, and overlapping graphics
- **High precision**: Provides accurate bounding boxes for clean product crops
- **GPU-optimized**: Efficient inference on NVIDIA GPUs via HuggingFace Transformers  

## Demo Video
https://github.com/user-attachments/assets/058ab533-6715-49cd-81f9-1b5ec722492f

---

## The Problem This Solves

**Challenge:** Retail ad images have large template backgrounds that dominate embeddings:
- 60-70% template area (store logos, graphics, promotional text)
- 30-40% actual product
- Result: Cortex Search embeddings similarity → wrong brand classification ❌

**Solution:** SAM 2 extracts product regions → embeddings focus on products → correct brands ✅

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTRACT_PRODUCTS SP                          │
│                    (Orchestrator - runs on warehouse)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────┐         ┌──────────────────────────┐     │
│   │  1. GPU Detection   │         │  2. CPU Cropping         │     │
│   │  ─────────────────  │         │  ──────────────────────  │     │
│   │  SAM 2 HuggingFace  │  ───▶   │  Pillow (warehouse)      │     │
│   │  → Bounding boxes   │         │  → Cropped PNG files     │     │
│   └─────────────────────┘         └──────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Input: @AD_INPUT_STAGE/ads/*.jpg    Output: @AD_OUTPUT_STAGE/*_product_*.png
```

**Key Benefits:**
- **HuggingFace Integration**: Uses `transformers` library with SAM 2.1
- **GPU-Accelerated Detection**: Runs on NVIDIA A10G via SPCS
- **Warehouse-Based Cropping**: No GPU needed for image cropping
- **Deduplication**: Removes overlapping/nested product detections

---

## Quick Start

```sql
-- Extract product regions from all images in a folder
CALL SHALION_HF_DEMO.PRODUCT_EXTRACTION.EXTRACT_PRODUCTS(
    'ads',                    -- Input folder (relative to AD_INPUT_STAGE)
    '@AD_OUTPUT_STAGE'        -- Output stage for cropped products
);
```

**Returns:**
```json
{
  "total_images": 50,
  "total_products": 128,
  "detection_seconds": 42.21,
  "cropping_seconds": 42.65,
  "total_seconds": 84.86,
  "throughput_per_hour": 2121
}
```

**Output Files:**
```
@AD_OUTPUT_STAGE/template_1_01_product_000.png
@AD_OUTPUT_STAGE/template_1_01_product_001.png
@AD_OUTPUT_STAGE/template_1_02_product_000.png
...
```

---

## Performance

> **Note:** Results based on PoC testing with 50 sample retail ad images. Actual performance may vary based on image complexity, product density, and hardware configuration.

### PoC Benchmark Results (Single Node)

| Metric | Value |
|--------|-------|
| Images Processed | 50 |
| Products Extracted | 128 |
| Detection Time (GPU) | 42.2s |
| Cropping Time (CPU) | 42.7s |
| Total Time | 84.9s |
| **Throughput** | **~2,100 images/hour** |
| Avg Time per Image | ~1.7s |
| GPU | NVIDIA A10G (24GB) |

### Estimated Processing Times

| Volume | Single Node (1x GPU) | Multi-Node Potential* |
|--------|---------------------|----------------------|
| 1 image | ~1.7 seconds | - |
| 50 images | ~1.4 minutes | - |
| 1,000 images | ~29 minutes | ~3 min (10 nodes) |
| 10,000 images | ~4.8 hours | ~29 min (10 nodes) |

*Multi-node estimates assume linear scaling with `MAX_NODES` in compute pool and `MAX_INSTANCES` in service configuration. Actual results depend on workload distribution and overhead.

---

## Components

### 1. Detection Service (GPU)
- **Technology**: HuggingFace `transformers` with SAM 2.1-hiera-small
- **Input**: Stage path to image folder
- **Output**: JSON with bounding boxes per image
- **Runs on**: SPCS GPU compute pool

### 2. Orchestrator SP (Warehouse)
- **Technology**: Snowpark Python with Pillow
- **Functions**:
  - Calls GPU detection service
  - Deduplicates overlapping detections
  - Crops images using bounding boxes
  - Saves cropped products to output stage

### 3. Helper Functions
- `DETECT_PRODUCTS(image_path)` - Detect products in single image
- `DETECT_PRODUCTS_FOLDER(folder_path)` - Detect products in folder
- `CROP_PRODUCT(bytes, x, y, w, h)` - Crop region from image
- `PARSE_DETECTIONS(json)` - Parse detection JSON into rows

---

## Setup

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup instructions.

**Quick Overview:**
1. Create infrastructure (database, stages, compute pool)
2. Build and push Docker image
3. Deploy detection service
4. Create orchestrator procedure
5. Upload test images and run extraction

---

## File Structure

```
├── app/
│   ├── main.py              # FastAPI service endpoints
│   └── sam2_detector.py     # SAM 2 HuggingFace detector
├── docker/
│   ├── Dockerfile           # GPU container with SAM 2
│   └── service_spec.yaml    # SPCS service specification
├── sql/
│   ├── 01_setup_infrastructure.sql
│   ├── 02_create_service.sql
│   ├── 03_create_orchestrator.sql
│   └── 04_destroy.sql
├── notebooks/
│   └── sam_product_extraction_demo.ipynb
├── test/
│   └── test_extraction.sql  # End-to-end test script
├── DEPLOYMENT.md
└── README.md
```

---

## Cost Estimates

> **Disclaimer:** Cost estimates are based on PoC results and Snowflake pricing at **$2.16/credit**. Actual costs vary by region, contract terms, and workload patterns. See [Snowflake Credit Consumption Table](https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf) for latest pricing.

### Resource Costs (Per Hour)

| Resource | Configuration | Credits/Hour | Est. Cost/Hour |
|----------|---------------|--------------|----------------|
| GPU Compute Pool | GPU_NV_S (1x A10G) | 0.57 | ~$1.23 |
| Warehouse | XSMALL | 1 | ~$2.16 |
| Storage | Internal Stages | - | Minimal |

### Estimated Processing Costs

Based on PoC throughput of ~2,100 images/hour (single node). Warehouse time = total stored procedure execution time (GPU detection + CPU cropping).

| Volume | Processing Time | GPU Cost | Warehouse Cost | **Total Est. $** | **Total Credits** |
|--------|-----------------|----------|----------------|------------------|-------------------|
| 1 image | ~1.7 sec | $0.00058 | $0.00102 | **$0.0016** | 0.00074 |
| 50 images | ~1.4 min | ~$0.03 | ~$0.05 | **~$0.08** | ~0.037 |
| 1,000 images | ~29 min | ~$0.59 | ~$1.04 | **~$1.63** | ~0.76 |
| 10,000 images | ~4.8 hours | ~$5.90 | ~$10.37 | **~$16.27** | ~7.53 |

### Scaling Considerations

- **Multi-Node GPU Pool:** Increase `MAX_NODES` in compute pool for parallel processing
- **Multiple Service Instances:** Increase `MAX_INSTANCES` in service spec
- **Cost Optimization:** Tune GPU pool auto-suspends and evaluate GPU provisioned throughput in Snowflake for better pricing
- **Batch Processing:** Process images in batches during off-peak hours for potential cost savings

> **Note:** For high-volume production workloads, scaling with multiple GPU nodes can significantly reduce processing time while maintaining similar per-image costs.

---

## Owner

**Alejandro Perez**  
Solution Engineer @ Snowflake

---

## License

MIT License - See LICENSE file for details.
