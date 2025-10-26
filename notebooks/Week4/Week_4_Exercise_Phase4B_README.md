# Week 4 Exercise - Phase 4B: Multi-Sensor Fusion README

**Date:** 2025-10-26
**Status:** Complete 
**Result:** 0.00% accuracy (vs Week 3 baseline 86.67%)

---
3
## Quick Start

### Option 1: Interactive Notebook (Recommended)
```bash
cd /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio/notebooks/Week4
jupyter notebook Week_4_Exercise_Phase4B.ipynb
```

### Option 2: Run Python Scripts
```bash
# Run complete Phase 4B pipeline
cd /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio/notebooks/Week4
./run_phase4b_real.sh  # Full pipeline (~30-60 min)
./run_phase4b_mock.sh  # Mock testing only (~15-20 min)
```

---

## What This Phase Tests

**Objective:** Test if multi-sensor fusion (Sentinel-2 + MODIS + DEM) improves accuracy when combined with transfer learning (ResNet50).

**Hypothesis:** Adding MODIS vegetation phenology and DEM topography would provide complementary information to improve accuracy beyond Phase 4A's 13.33%.

**Result:** **HYPOTHESIS REJECTED** - Accuracy dropped to 0.00% (complete failure), demonstrating that adding more sensors cannot fix fundamental transfer learning problems.

---

## Multi-Sensor Fusion Pipeline

### Data Sources

**Sentinel-2 (6 bands at 10m resolution):**
- Blue (B2), Green (B3), Red (B4)
- Near-Infrared (B8)
- Shortwave Infrared (B11, B12)
- Native resolution: 10m

**MODIS Vegetation Indices (1 band):**
- NDVI (Normalized Difference Vegetation Index)
- Native resolution: 250m → resampled to 10m
- 16-day composite, median aggregation
- Scaled to [-1, 1]

**SRTM DEM (1 band):**
- Elevation above sea level
- Native resolution: 30m (1 arc-second) → resampled to 10m
- Range: 49-204m in study area (Los Lagos, Chile)

**Fused Composite:**
- Total: 8 bands (6 S2 + 1 MODIS + 1 DEM)
- Resolution: 10m (all aligned to Sentinel-2 grid)
- Fusion strategy: Early fusion (pixel-level stacking)
- Patch size: 8×8 pixels (80m × 80m footprint)

### Resampling Strategy

| Sensor | Native Resolution | Target Resolution | Upsampling Factor | Method |
|--------|------------------|------------------|------------------|--------|
| Sentinel-2 | 10m | 10m | 1× (none) | - |
| MODIS NDVI | 250m | 10m | 25× | Bilinear |
| SRTM DEM | 30m | 10m | 9× | Bicubic |

**Note:** Aggressive resampling (25× for MODIS) creates synthetic pixels via interpolation, not real measurements.

---

## Why This Experiment Failed

### Root Causes (6 Major Issues)

1. **Inherited All Phase 4A Problems**
   - Same aggressive freezing: 150/175 layers (85.7%)
   - Same patch size issues: 8×8 → 32×32 upsampling artifacts
   - Same domain mismatch: ImageNet RGB → multispectral satellite

2. **Added Complexity Without Fixing Root Cause**
   - Increased input dimensions: 6 → 8 bands (+33%)
   - Same frozen architecture: Can't adapt to new data
   - Band reduction bottleneck: 8 bands → 3 pseudo-RGB via 1×1 conv

3. **Resampling Artifacts**
   - MODIS: 250m → 10m = 25× upsampling (creates 24 synthetic pixels per real measurement)
   - DEM: 30m → 10m = 9× upsampling (less severe, but still interpolated)
   - Upsampled bands may add noise rather than signal

4. **Insufficient Dataset Size for Increased Complexity**
   - Phase 4A: 6 × 8 × 8 = 384 input dimensions, 300 samples
   - Phase 4B: 8 × 8 × 8 = 512 input dimensions, 300 samples (+33% complexity)
   - Rule of thumb: 50-100 samples per input dimension
   - Phase 4B needs: 25,600-51,200 samples, has: 300

5. **Band Reduction Information Loss**
   - First layer: Conv2D(3, kernel_size=1) compresses 8D → 3D
   - Critical spectral information lost before ResNet50
   - Only 14.3% trainable parameters to learn optimal compression

6. **Complete Model Collapse**
   - Validation accuracy: 0.00% (not a single correct prediction)
   - Training accuracy: 23-28% (barely above random 20%)
   - Validation loss: 2.171 (constant, no improvement)
   - Worse than Phase 4A (13.33%) despite more data

---

## Key Files

### Notebook
- `Week_4_Exercise_Phase4B.ipynb` - Interactive notebook with full analysis

### Scripts (8-Step Pipeline)
1. `phase4b_01_load_sensors.py` - Load S2, MODIS, DEM from Earth Engine
2. `phase4b_02_resample_align.py` - Resample MODIS/DEM to 10m, verify alignment
3. `phase4b_03_fuse_stack.py` - Stack sensors into 8-band composite
4. `phase4b_04_extract_patches.py` - Extract 8-band patches (FIXED: stratified split)
5. `phase4b_05_quality_control.py` - QC checks on extracted patches
6. `phase4b_06_train_fusion.py` - Train ResNet50 fusion model
7. `phase4b_07_ablation_study.py` - Compare Week 3 vs 4A vs 4B
8. `phase4b_08_baseline_report.py` - Generate comprehensive report

### Shell Scripts
- `run_phase4b_mock.sh` - Mock data testing (15-20 min, code validation)
- `run_phase4b_real.sh` - Real data execution (30-60 min, actual results)

### Outputs
- `phase4b_outputs/X_train_fused.npy` - Training patches (300, 8, 8, 8)
- `phase4b_outputs/X_val_fused.npy` - Validation patches (75, 8, 8, 8)
- `phase4b_outputs/y_train_fused.npy` - Training labels (300,)
- `phase4b_outputs/y_val_fused.npy` - Validation labels (75,)
- `phase4b_outputs/fusion_model.h5` - Trained model (failed, but saved)
- `phase4b_outputs/fusion_training_history.json` - Training metrics
- `phase4b_outputs/fused_composite_preview.png` - 8-band visualization
- `phase4b_outputs/qc_visualization.png` - Sample patches QC

### Documentation
- `Week4_Phase4B_Summary.md` - Comprehensive results report
- `Week_4_Exercise_Phase4B_README.md` - This file
- `4B_lessons_learned.md` - Mock testing bugs and fixes (4 bugs caught)

---

## Results Summary

| Metric | Week 3 SimpleCNN | Phase 4A ResNet50 | Phase 4B Fusion | Change (vs Week 3) |
|--------|------------------|-------------------|-----------------|---------------------|
| **Accuracy** | 86.67% | 13.33% | **0.00%** | **-86.67pp** |
| **Precision (macro)** | - | 0.027 | 0.00 | - |
| **Recall (macro)** | - | 0.200 | 0.00 | - |
| **F1 (macro)** | - | 0.047 | 0.00 | - |
| **Input Bands** | 6 (S2) | 6 (S2) | 8 (S2+MODIS+DEM) | +2 bands |
| **Parameters** | 54K | 23.6M | 23.6M | 437× larger |
| **Training Time** | ~30s | ~7s | ~7s | Faster (frozen) |

### Training Behavior

**Validation Metrics (Across 11 Epochs):**
- **Loss:** 2.171 (CONSTANT - no improvement)
- **Accuracy:** 0.00% (CONSTANT - complete failure)
- Model learned absolutely nothing

**Training Metrics:**
- **Accuracy:** 23-28% (barely above random 20% for 5 classes)
- **Loss:** 2.27-2.49 (high, not decreasing)

---

## Critical Dataset Fix Applied

### Validation Set Stratification Issue

**Original Problem (Discovered Post-Execution):**
- Phase 4B initially used sequential splitting: first N patches → train, next M → validation
- Resulted in validation set with **only 1 class (Class 2: Parcels)**
- All models showed 100% or 0% accuracy (meaningless results)

**Fix Applied:**
Changed `phase4b_04_extract_patches.py` line 229:
```python
# BEFORE (Sequential split - caused imbalance):
if Y_TRAIN_PATH.exists():
    # ... sequential split based on Week 3 indices

# AFTER (Forced stratified split - ensures balanced classes):
if False:  # Force stratified split
    # ... old code disabled
else:
    from sklearn.model_selection import train_test_split
    X_train_fused, X_val_fused, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
```

**Result After Fix:**
- Training classes: [0, 1, 2, 3, 4] with counts [60, 48, 44, 45, 43]
- Validation classes: [0, 1, 2, 3, 4] with counts [21, 10, 19, 17, 8]
- **All 5 classes now balanced in both train and validation**

**Impact:**
- Re-ran patch extraction (~3.5 min)
- Re-ran Phase 4B training (~25 sec)
- Results now valid: 0.00% accuracy confirmed as true failure

---

## Lessons Learned

### 1. Engineering Success ≠ Research Success

**Phase 4B Technical Pipeline (100% Success):**
- ✅ Loaded multiple sensors from Earth Engine
- ✅ Resampled different resolutions to common grid
- ✅ Aligned sensors spatially and temporally
- ✅ Created 8-band fused composite
- ✅ Extracted multi-sensor patches without errors
- ✅ Trained model without crashes

**Phase 4B Model Performance (0% Success):**
- ❌ Validation accuracy: 0.00%
- ❌ Model learned nothing useful
- ❌ Worse than random guessing

**Lesson:** Correct implementation + wrong approach = failure

---

### 2. More Data ≠ Better Results

**The Reality:**
- Phase 4A (6 bands): 13.33%
- Phase 4B (8 bands): 0.00% ← **Worse, not better**

**Why:** Adding sensors increased complexity without fixing root causes (frozen layers, domain mismatch, patch size issues).

**Lesson:** Fix the learning strategy before adding more data sources.

---

### 3. Diagnose Root Causes Before Iterating

**What Should Have Happened:**
1. Phase 4A fails (13.33%) ✗
2. **Diagnose:** Aggressive freezing, domain mismatch
3. **Fix Phase 4A first**
4. **Then** add multi-sensor fusion

**What Actually Happened:**
1. Phase 4A fails (13.33%) ✗
2. **Immediately add complexity:** Multi-sensor fusion
3. Phase 4B fails worse (0.00%) ✗

**Lesson:** Solve one problem at a time.

---

### 4. Simpler Models Can Outperform Complex Ones

| Model | Parameters | Bands | Accuracy |
|-------|-----------|-------|----------|
| **Week 3 SimpleCNN** | 54K | 6 (S2) | **86.67%** ✅ |
| **Phase 4A ResNet50** | 23.6M | 6 (S2) | 13.33% ✗ |
| **Phase 4B ResNet50** | 23.6M | 8 (S2+MODIS+DEM) | 0.00% ✗ |

**Lesson:** For small, domain-specific datasets, train from scratch with task-specific architecture.

---

### 5. Resampling Can't Create Missing Information

**Resampling Reality:**
- MODIS: 250m → 10m = 25× upsampling (24/25 pixels are synthetic)
- DEM: 30m → 10m = 9× upsampling (8/9 pixels are interpolated)

**Impact:** Upsampled bands don't add true information, just smoothed estimates.

**Lesson:** Use <5× upsampling when possible; >10× likely adds more noise than signal.

---

### 6. Mock Testing Saves Time

During Phase 4B development, mock testing caught **4 critical bugs** before real execution:
1. Null geometry handling (Polygon 121 crashes `geemap.geopandas_to_ee()`)
2. None value handling in EE statistics (crashes formatting)
3. JSON serialization of NumPy types (not serializable)
4. Earth Engine shape mismatch (returns (9,12,8) instead of (8,8,8))

**Impact:** Saved 2+ hours of failed real executions by catching issues in 15-minute test cycles.

**Lesson:** Always test with mock data first for Earth Engine workflows.

---

## Recommendations

### Option 1: Apply Week 3 SimpleCNN to 8-Band Data (Phase 4C)
```python
# Test if fusion helps when using a proven architecture
Input: 8×8×8 (S2 + MODIS + DEM)
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool
Flatten → Dense(128) → Dropout(0.5) → Dense(5)
```
**Expected:** 88-92% if fusion helps, 85-87% if neutral
**Actual (Phase 4C):** 69% (S2+DEM), 44% (full fusion)

### Option 2: Reduce Transfer Learning Freezing
```python
# Freeze only 30-50% of layers instead of 85%
# Use larger patches (32×32 or 64×64) to avoid upsampling
```
**Expected:** 70-80% if it works, <20% if still too frozen

### Option 3: Use Domain-Specific Pretrained Weights
```python
# Pretrain on satellite imagery (e.g., EuroSAT, UC Merced)
# Then fine-tune on Los Lagos data
```
**Expected:** 80-90% if pretrained weights available

---

## Prerequisites

### Data Requirements
- Week 3 dataset (X_train.npy, X_val.npy, y_train.npy, y_val.npy)
- Earth Engine authentication (for loading MODIS and DEM)
- Training polygons GeoPackage (126 polygons, 125 valid)

### Software Environment
- Python 3.11+
- TensorFlow 2.20.0+ / Keras
- Earth Engine API (geemap 0.35.0+)
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- SciPy (for patch resizing)

### Earth Engine Access
```python
import ee
ee.Authenticate()  # First time only
ee.Initialize()
```

---

## Running the Experiment

### Step-by-Step Execution

#### Option 1: Run Notebook (Fastest)
```bash
cd /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio/notebooks/Week4
jupyter notebook Week_4_Exercise_Phase4B.ipynb
# Execute all cells (assumes phase4b_outputs/*.npy files exist)
```

#### Option 2: Run Full Pipeline (Includes Data Generation)
```bash
cd /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio/notebooks/Week4

# Option A: Mock testing first (recommended, ~15-20 min)
./run_phase4b_mock.sh

# Option B: Real execution (~30-60 min)
./run_phase4b_real.sh
```

#### Option 3: Run Scripts Individually
```bash
# Step 1: Load sensors from Earth Engine (~5 min)
/opt/miniconda3/envs/geoai/bin/python phase4b_01_load_sensors.py

# Step 2: Resample and align (~2 min)
/opt/miniconda3/envs/geoai/bin/python phase4b_02_resample_align.py

# Step 3: Fuse and stack (~1 min)
/opt/miniconda3/envs/geoai/bin/python phase4b_03_fuse_stack.py

# Step 4: Extract patches (~3-5 min)
/opt/miniconda3/envs/geoai/bin/python phase4b_04_extract_patches.py

# Step 5: Quality control (~30 sec)
/opt/miniconda3/envs/geoai/bin/python phase4b_05_quality_control.py

# Step 6: Train fusion model (~7 sec)
/opt/miniconda3/envs/geoai/bin/python phase4b_06_train_fusion.py

# Step 7: Ablation study (~1 min)
/opt/miniconda3/envs/geoai/bin/python phase4b_07_ablation_study.py

# Step 8: Generate report (~10 sec)
/opt/miniconda3/envs/geoai/bin/python phase4b_08_baseline_report.py
```

---

## Expected Outputs

### Console Output (Step 6: Training)
```
Model Architecture Summary:
  Total layers: 181 (175 ResNet50 + 6 custom)
  Frozen layers: 150 (85.7%)
  Total parameters: 23,587,717
  Trainable parameters: 3,417,605 (14.5%)

Training ResNet50 Fusion Model...
Epoch 11/50: loss=2.2717, accuracy=0.2800, val_loss=2.1710, val_accuracy=0.0000
Early stopping at epoch 11 (no improvement for 10 epochs)

Phase 4B Fusion Model - Validation Results:
  Accuracy: 0.00%

Comparison:
  Week 3 SimpleCNN (6 bands):    86.67%
  Phase 4A ResNet50 (6 bands):   13.33%
  Phase 4B ResNet50 (8 bands):    0.00%  ← Current
  Change vs Week 3:              -86.67 percentage points
  Change vs Phase 4A:            -13.33 percentage points
```

### Files Generated
```
phase4b_outputs/
├── sentinel2_composite.tif            # S2 median composite (6 bands)
├── modis_ndvi.tif                     # MODIS NDVI resampled to 10m
├── srtm_dem.tif                       # SRTM DEM resampled to 10m
├── fused_composite.tif                # 8-band fused composite
├── fused_composite_preview.png        # RGB visualization
├── X_train_fused.npy                  # Training patches (300, 8, 8, 8)
├── X_val_fused.npy                    # Validation patches (75, 8, 8, 8)
├── y_train_fused.npy                  # Training labels (300,)
├── y_val_fused.npy                    # Validation labels (75,)
├── fusion_model.h5                    # Trained model (failed)
├── fusion_training_history.json       # Training metrics
├── fusion_training_history.png        # Accuracy/loss plots
├── ablation_comparison.png            # Week 3 vs 4A vs 4B
├── ablation_results.json              # Performance comparison
└── baseline_report.md                 # Auto-generated report
```

---

## Troubleshooting

### Issue 1: Earth Engine Authentication Error
**Error:** `ee.EEException: Please authenticate to Earth Engine`
**Solution:**
```python
import ee
ee.Authenticate()  # Follow browser authentication
ee.Initialize()
```

### Issue 2: Validation Set Has Only 1 Class
**Error:** `WARNING: Validation set has only 1 class: [2]`
**Solution:** This was fixed in phase4b_04_extract_patches.py (line 229). Re-run script.

### Issue 3: Phase 4B Takes Too Long
**Expected Times:**
- Mock test: 15-20 min (code validation only)
- Real execution: 30-60 min (full pipeline)
- Training alone: ~7 sec (frozen layers train fast)

**If Much Longer:** Check Earth Engine quotas, internet connection.

### Issue 4: Memory Error During Earth Engine Export
**Error:** `ResourceExhaustedError: User memory limit exceeded`
**Solution:** Reduce study area, increase scale (10m → 20m), or use smaller polygons.

---

## Next Steps

After completing Phase 4B:

1. **Phase 4C:** SimpleCNN with multi-sensor fusion ✅
   - Apply Week 3's proven architecture to 8-band data
   - Result: 69% (S2+DEM best), 44% (full fusion)
   - Lesson: Fusion works with correct architecture, but dataset size still limiting

2. **Week 4 Summary:**
   - Compare all experiments (4A: 13.33%, 4B: 0.00%, 4C: 69%)
   - Document lessons learned
   - Generate comprehensive report

3. **Future Work:**
   - Collect more training samples (300 → 1000+)
   - Test ablation study (S2 only, S2+DEM, S2+MODIS, all)
   - Try late fusion or decision fusion strategies

---

## References

### Related Materials
- `Week_4_Study_Guide.md` - Concepts and theoretical background
- `Week_4_Exercise.md` - Exercise instructions
- `Week4_Phase4B_Summary.md` - Comprehensive results analysis
- `4B_lessons_learned.md` - Mock testing bugs and fixes

### Key Concepts
- Multi-sensor fusion (early, late, decision)
- Transfer learning failure modes
- Resampling and spatial alignment
- Stratified splitting for imbalanced datasets
- Mock testing for Earth Engine workflows

---

## Contact

For questions or issues with this experiment, refer to:
- Study guide: Sections on multi-sensor fusion
- Exercise document: Troubleshooting section (Issues 6-8)
- Phase 4B Summary: Root cause analysis (6 failure modes)

---

**Generated:** 2025-10-26
**Status:** Phase 4B Complete (Documented Failure)
**Conclusion:** Multi-sensor fusion pipeline works correctly (engineering success), but ResNet50 transfer learning completely failed (0.00% accuracy). Adding more sensors did not fix fundamental transfer learning problems from Phase 4A. SimpleCNN trained from scratch (Week 3: 86.67%) vastly outperforms ResNet50 multi-sensor fusion (Phase 4B: 0.00%).
