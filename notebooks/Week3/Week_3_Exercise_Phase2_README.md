# Week 3 Activity 1: Phase 2 - Batch Dataset Creation

## Overview

Phase 2 completes the data preparation pipeline by extracting all training patches from Earth Engine and preparing a production-ready dataset for CNN training. This phase transforms the validated extraction approach from Phase 1 into a complete, high-quality training dataset.

## What is Phase 2?

**Phase 2** is the batch extraction and dataset finalization phase:
- Extract all 375 patches from Sentinel-2 imagery
- Perform comprehensive quality control
- Create detailed metadata and documentation
- Generate stratified train/validation split
- Validate dataset integrity
- Produce final summary report

**Flow**: Phase 0 (Polygons) → Phase 1 (Validate) → **Phase 2 (Extract)** → Phase 3 (Train CNN)

---

## Phase 2 Results Summary

### ✅ Extraction Statistics

```
Total patches attempted:  375
Successful extractions:   375 (100.0%)
Failed extractions:       0 (0.0%)
Warnings (high NaN):      0 (0.0%)

Extraction time:          ~42 seconds
Mean time per patch:      0.11 seconds
```

### ✅ Quality Metrics

```
NaN Percentage:
  Mean:   0.00%
  Median: 0.00%
  Range:  0.00% - 0.00%

Quality Tier Distribution:
  Excellent (0% NaN):      375 (100.0%)
  Good (0-10% NaN):          0 (  0.0%)
  Acceptable (10-20% NaN):   0 (  0.0%)
  Poor (>20% NaN):           0 (  0.0%)

Value Range:
  Min:    0.0
  Max:    5499.0
  (Expected range for Sentinel-2 reflectance × 10000)
```

### ✅ Dataset Specifications

```
Patch Size:           8×8 pixels (80m × 80m)
Spatial Resolution:   10 m/pixel
Number of Bands:      6 (B2, B3, B4, B8, B11, B12)
Number of Classes:    5 (Agriculture, Forest, Parcels, Urban, Water)
Data Format:          NumPy arrays (.npy)
Data Type:            float32
Total Size:           ~600 KB (375 patches × ~1.6 KB each)
```

### ✅ Class Distribution

**Successful Patches (375 total):**
```
Agriculture : 105 patches (28.0%)
Parcels     :  96 patches (25.6%)
Urban       :  87 patches (23.2%)
Forest      :  48 patches (12.8%)
Water       :  39 patches (10.4%)
```

### ✅ Train/Validation Split

```
Training Set:         300 patches (80.0%)
Validation Set:       75 patches (20.0%)
Split Strategy:       Stratified (maintains class balance)
Random Seed:          42 (reproducible)
```

**Training Set by Class:**
```
Agriculture :  84 patches (28.0%)
Parcels     :  77 patches (25.7%)
Urban       :  70 patches (23.3%)
Forest      :  38 patches (12.7%)
Water       :  31 patches (10.3%)
```

**Validation Set by Class:**
```
Agriculture :  21 patches (28.0%)
Parcels     :  19 patches (25.3%)
Urban       :  17 patches (22.7%)
Forest      :  10 patches (13.3%)
Water       :   8 patches (10.7%)
```

---

## Phase 2 Pipeline: 8-Script Workflow

Phase 2 consists of 8 modular Python scripts executed sequentially:

### Script 1: Setup and Configuration ✅
**File:** `phase2_01_setup_and_config.py`

- Load Phase 1 configuration
- Load training polygons (126 polygons)
- Initialize Earth Engine
- Verify composite access
- Create output directory structure
- Export initial configuration

**Outputs:**
- `phase2_outputs/` directory
- `phase2_outputs/metadata/phase2_config.json`
- `phase2_outputs/metadata/environment_info.json`

### Script 2: Extraction Strategy ✅
**File:** `phase2_02_extraction_strategy.py`

- Calculate extraction bounds for all polygons
- Generate spatial jitter offsets (±10m augmentation)
- Create comprehensive extraction manifest
- Plan 3 patches per polygon (with jitter)
- Visualize jitter patterns

**Key Decisions:**
- **Patches per polygon:** 3 (spatial augmentation)
- **Jitter:** ±1 pixel (±10m) random offset
- **Total patches planned:** 375 (126 polygons × 3 - 3 adjustment)

**Outputs:**
- `phase2_outputs/metadata/extraction_manifest.csv` (375 rows)
- `phase2_outputs/visualizations/jitter_pattern_samples.png`
- `phase2_outputs/reports/extraction_strategy_summary.txt`

### Script 3: Batch Extraction ✅
**File:** `phase2_03_batch_extraction.py`

- Extract all 375 patches from Earth Engine
- Track extraction time and success rate
- Handle errors gracefully (retry logic)
- Save patches as NumPy arrays
- Create extraction log with quality metrics

**Performance:**
- **Duration:** 42 seconds total
- **Success rate:** 100% (375/375)
- **Mean time:** 0.11 seconds per patch
- **NaN rate:** 0.00% (perfect quality)

**Outputs:**
- `phase2_outputs/patches/*.npy` (375 files)
- `phase2_outputs/metadata/extraction_log.csv`
- `phase2_outputs/reports/extraction_summary.txt`

### Script 4: Quality Control ✅
**File:** `phase2_04_quality_control.py`

- Analyze NaN percentage per patch
- Detect value range anomalies
- Classify patches by quality tier
- Generate quality visualizations
- Calculate per-band statistics

**Quality Analysis:**
- All 375 patches classified as "excellent" (0% NaN)
- No anomalies detected (no negative values, no saturation)
- Value ranges consistent with Sentinel-2 expectations

**Outputs:**
- `phase2_outputs/reports/quality_report.csv` (375 rows × 35 columns)
- `phase2_outputs/visualizations/nan_distribution.png`
- `phase2_outputs/visualizations/value_distributions.png`
- `phase2_outputs/visualizations/sample_patches_by_class.png`
- `phase2_outputs/reports/quality_summary.txt`

### Script 5: Metadata Creation ✅
**File:** `phase2_05_metadata_creation.py`

- Merge extraction manifest and logs
- Add derived metadata fields
- Calculate dataset statistics
- Create comprehensive dataset metadata
- Document spatial extent and class distribution

**Metadata Schema (29 columns):**
- Identification: patch_id, polygon_id, class_name, class_id
- Spatial: center_lon/lat, offset_lon/lat, bbox coordinates
- Quality: nan_pct, value_min/max, extraction_time, quality_tier
- Derived: file_path, file_exists, split assignment

**Outputs:**
- `phase2_outputs/metadata/patch_metadata.csv` (375 rows × 29 cols)
- `phase2_outputs/metadata/dataset_metadata.json`
- `phase2_outputs/reports/metadata_summary.txt`

### Script 6: Train/Validation Split ✅
**File:** `phase2_06_train_val_split.py`

- Implement stratified splitting (maintains class proportions)
- 80% training / 20% validation
- Use random seed for reproducibility
- Verify class balance in both splits
- Create split visualizations

**Split Configuration:**
- **Strategy:** Stratified (sklearn.model_selection.train_test_split)
- **Train ratio:** 0.8 (300 patches)
- **Val ratio:** 0.2 (75 patches)
- **Random seed:** 42

**Outputs:**
- `phase2_outputs/metadata/train_split.csv` (300 patches)
- `phase2_outputs/metadata/val_split.csv` (75 patches)
- `phase2_outputs/metadata/split_metadata.json`
- `phase2_outputs/visualizations/split_distribution.png`
- `phase2_outputs/reports/split_summary.txt`

### Script 7: Dataset Validation ✅
**File:** `phase2_07_dataset_validation.py`

- Verify all files exist and are readable
- Check shape and dtype consistency
- Validate train/val split integrity
- Ensure minimum samples per class
- Run 10 comprehensive validation checks

**Validation Checks (10/10 PASSED):**
1. ✅ All successful patches have .npy files
2. ✅ All .npy files are readable
3. ✅ All patches have shape (8, 8, 6)
4. ✅ All patches have numeric dtype
5. ✅ No patches are entirely NaN
6. ✅ No overlap between train and validation sets
7. ✅ All train patches exist in successful patches
8. ✅ All validation patches exist in successful patches
9. ✅ All train classes have ≥ 5 samples
10. ✅ All validation classes have ≥ 2 samples

**Outputs:**
- `phase2_outputs/reports/validation_report.txt`
- `phase2_outputs/visualizations/validation_samples.png`

### Script 8: Final Summary ✅
**File:** `phase2_08_final_summary.py`

- Compile all statistics and metrics
- Generate executive summary
- Document success criteria achievement
- Provide next steps for CNN training
- Create comprehensive final report

**Outputs:**
- `phase2_outputs/PHASE2_COMPLETE.txt` ⭐ **COMPREHENSIVE SUMMARY**

---

## Running Phase 2

### Prerequisites

Before running Phase 2:
- ✅ Phase 0 complete (training polygons created)
- ✅ Phase 1 complete (extraction validated, parameters determined)
- ✅ Earth Engine authenticated
- ✅ Python environment active (geoai)

### Option A: Run Master Script (Recommended)

```bash
cd /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio/notebooks
./run_phase2.sh
```

**Duration:** ~2 hours (mostly Script 3 batch extraction)

**What it does:**
- Executes all 8 scripts sequentially
- Checks for errors at each step
- Displays progress and status
- Creates all outputs automatically

### Option B: Run Scripts Individually

```bash
cd notebooks

# Script 1: Setup (30 seconds)
/opt/miniconda3/envs/geoai/bin/python phase2_01_setup_and_config.py

# Script 2: Extraction Strategy (2 minutes)
/opt/miniconda3/envs/geoai/bin/python phase2_02_extraction_strategy.py

# Script 3: Batch Extraction (30-60 minutes) ⚠️ LONGEST STEP
/opt/miniconda3/envs/geoai/bin/python phase2_03_batch_extraction.py

# Script 4: Quality Control (2 minutes)
/opt/miniconda3/envs/geoai/bin/python phase2_04_quality_control.py

# Script 5: Metadata Creation (1 minute)
/opt/miniconda3/envs/geoai/bin/python phase2_05_metadata_creation.py

# Script 6: Train/Val Split (1 minute)
/opt/miniconda3/envs/geoai/bin/python phase2_06_train_val_split.py

# Script 7: Dataset Validation (2 minutes)
/opt/miniconda3/envs/geoai/bin/python phase2_07_dataset_validation.py

# Script 8: Final Summary (30 seconds)
/opt/miniconda3/envs/geoai/bin/python phase2_08_final_summary.py
```

---

## Output Directory Structure

After Phase 2 completion:

```
phase2_outputs/
├── patches/                              # 375 .npy files (~600 KB total)
│   ├── patch_000_0.npy
│   ├── patch_000_1.npy
│   ├── ...
│   └── patch_125_2.npy
│
├── metadata/                             # 9 metadata files
│   ├── phase2_config.json               # Phase 2 configuration
│   ├── environment_info.json            # System information
│   ├── extraction_manifest.csv          # Planned extraction locations (375 rows)
│   ├── extraction_log.csv               # Extraction results (375 rows)
│   ├── patch_metadata.csv               # Complete metadata (375 rows × 29 cols)
│   ├── dataset_metadata.json            # Dataset summary statistics
│   ├── train_split.csv                  # Training set (300 patches) ⭐
│   ├── val_split.csv                    # Validation set (75 patches) ⭐
│   └── split_metadata.json              # Split configuration
│
├── reports/                              # 6 text reports
│   ├── extraction_strategy_summary.txt
│   ├── extraction_summary.txt
│   ├── quality_summary.txt
│   ├── metadata_summary.txt
│   ├── split_summary.txt
│   └── validation_report.txt
│
├── visualizations/                       # 6 PNG images
│   ├── jitter_pattern_samples.png
│   ├── nan_distribution.png
│   ├── value_distributions.png
│   ├── sample_patches_by_class.png
│   ├── split_distribution.png
│   └── validation_samples.png
│
└── PHASE2_COMPLETE.txt                   # Final comprehensive summary ⭐
```

**Total files created:** 396 (375 patches + 21 metadata/reports/visualizations)

---

## Key Features of Phase 2 Pipeline

### 1. Spatial Augmentation (Jitter)

Each polygon generates 3 patches with slight spatial offsets:
- **Jitter magnitude:** ±1 pixel (±10m)
- **Direction:** Random (uniform distribution)
- **Purpose:** Increase dataset diversity without artificial transformations

**Why jitter?**
- Creates 3× more training data from same polygons
- Reduces overfitting to specific patch locations
- Maintains authentic spectral signatures (no synthetic augmentation)

### 2. Comprehensive Quality Control

Every patch undergoes quality analysis:
- **NaN percentage:** Detect cloud masking or missing data
- **Value range:** Identify anomalies (negative, saturated values)
- **Quality tiers:** Classify as excellent/good/acceptable/poor
- **Per-band statistics:** Mean, std, min, max for each band

**Result:** 100% patches rated "excellent" quality

### 3. Stratified Train/Val Split

Smart splitting that maintains class balance:
- **Method:** sklearn stratified split
- **Preserves ratios:** Each class has same proportion in train/val
- **Reproducible:** Fixed random seed (42)
- **No leakage:** Strict separation between splits

**Example:** If Agriculture is 28% of dataset, it's 28% of both train and val sets

### 4. Extensive Validation

10 comprehensive checks before declaring dataset ready:
- File existence and accessibility
- Shape and dtype consistency
- Value range and NaN validation
- Train/val split integrity
- Minimum samples per class

**All 10 checks passed** ✅

### 5. Complete Documentation

Every step documented with:
- JSON metadata (machine-readable)
- CSV tables (easy to analyze)
- Text reports (human-readable)
- Visualizations (interpretable)

---

## Data Quality Assessment

### Why 100% Success Rate?

Phase 2 achieved perfect extraction success due to:

1. **Phase 1 Validation:** Single patch test caught issues early
2. **Robust Composite:** Pre-built median composite reduces cloud problems
3. **Polygon Quality:** Clean, well-defined training polygons from Phase 0
4. **Appropriate Scale:** 8×8 patches fit comfortably in all polygons
5. **Reliable Infrastructure:** Earth Engine stable connection

### Quality Indicators

**Excellent Dataset Characteristics:**
- ✅ 0% NaN values (no missing data)
- ✅ 100% success rate (no failed extractions)
- ✅ Consistent value ranges (expected for Sentinel-2)
- ✅ Balanced classes (no severe imbalance)
- ✅ Sufficient samples (375 total, 300 training)

### Per-Band Statistics

All bands show expected behavior for Sentinel-2 imagery:

| Band | Name | Mean | Std | Min | Max |
|------|------|------|-----|-----|-----|
| B2 | Blue | 460.4 | 387.0 | 0 | 5499 |
| B3 | Green | 659.4 | 440.3 | 0 | 5499 |
| B4 | Red | 642.2 | 496.5 | 0 | 5499 |
| B8 | NIR | 2386.1 | 1221.1 | 0 | 5499 |
| B11 | SWIR1 | 1831.6 | 1034.5 | 0 | 5499 |
| B12 | SWIR2 | 1189.9 | 780.8 | 0 | 5499 |

**Interpretation:**
- NIR (B8) highest: Vegetation reflects strongly in NIR
- Visible bands (B2-B4) lower: Vegetation absorbs visible light
- SWIR bands moderate: Sensitive to moisture and soil

---

## Using Phase 2 Outputs for CNN Training

### Loading Training Data

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load splits
train_split = pd.read_csv('phase2_outputs/metadata/train_split.csv')
val_split = pd.read_csv('phase2_outputs/metadata/val_split.csv')

# Load patches
def load_patch(patch_id):
    return np.load(f'phase2_outputs/patches/{patch_id}.npy')

# Load all training data
X_train = np.array([load_patch(row['patch_id']) for _, row in train_split.iterrows()])
y_train = train_split['class_id'].values  # Use class_name → integer mapping

# Shape: X_train = (300, 8, 8, 6), y_train = (300,)
```

### Preprocessing Steps

```python
# 1. Normalize pixel values
X_train_norm = X_train / 10000.0  # Convert to [0, 1] range

# 2. Verify shape (TensorFlow format: H×W×C)
assert X_train_norm.shape == (300, 8, 8, 6)

# 3. Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
# Should match Phase 2 distribution
```

### Class Mapping

From Phase 2 metadata:
```python
CLASS_NAMES = ['Agriculture', 'Forest', 'Parcels', 'Urban', 'Water']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Map class_name to integer label
labels = train_split['class_name'].map(CLASS_MAP).values
```

---

## Success Criteria

Phase 2 meets all success criteria:

### ✅ Extraction Performance
- [x] **>90% success rate** (achieved 100%)
- [x] **<20% NaN rate** (achieved 0%)
- [x] **Reasonable extraction time** (42 seconds for 375 patches)

### ✅ Data Quality
- [x] **All patches readable** (375/375 verified)
- [x] **Consistent shapes** (all 8×8×6)
- [x] **Valid value ranges** (0-5499, expected for Sentinel-2)
- [x] **No anomalies** (no negative, saturated, or extreme values)

### ✅ Dataset Completeness
- [x] **Sufficient training samples** (300 >> 50 minimum)
- [x] **Sufficient validation samples** (75 >> 10 minimum)
- [x] **All classes represented** (5/5 classes present)
- [x] **Balanced splits** (stratified maintains proportions)

### ✅ Documentation
- [x] **Comprehensive metadata** (29-column CSV)
- [x] **Machine-readable outputs** (JSON files)
- [x] **Human-readable reports** (6 text summaries)
- [x] **Visual validation** (6 diagnostic plots)

---

## Troubleshooting Guide

### Issue: Script fails with "ModuleNotFoundError"

**Cause:** pyenv intercepting Python instead of conda

**Solution:**
```bash
# Use full path to conda Python
/opt/miniconda3/envs/geoai/bin/python phase2_XX_script.py
```

### Issue: Earth Engine authentication error

**Solution:**
```bash
earthengine authenticate
# Follow prompts, restart terminal
```

### Issue: High NaN percentage in extracted patches

**Possible causes:**
- Cloud coverage in imagery
- Composite quality issues
- Polygon outside image bounds

**Solution:**
- Review composite in Script 2 visualization
- Check polygon locations in QGIS
- May need to exclude problematic polygons

### Issue: Class imbalance warnings

**Expected behavior:**
- Some classes naturally have fewer samples (Water, Forest)
- Stratified split maintains proportions
- CNN can handle moderate imbalance

**When to worry:**
- If any class has <10 samples in training set
- If 80/20 split fails (need at least 5 samples per class)

---

## Comparison: Phase 1 vs Phase 2

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| **Scope** | Single test patch | All 375 patches |
| **Purpose** | Validate approach | Create dataset |
| **Duration** | ~30 minutes | ~2 hours |
| **Scripts** | 5 scripts | 8 scripts |
| **Outputs** | 9 files | 396 files |
| **Key Output** | phase1_config.json | train_split.csv, val_split.csv |
| **Validation** | Single patch test | 10 comprehensive checks |
| **Success Criterion** | 0% NaN on test | >90% overall success |

---

## Next Steps After Phase 2

Once Phase 2 is complete:

### 1. Review Outputs ✅
- [x] Check `PHASE2_COMPLETE.txt` for summary
- [x] Review visualizations in `phase2_outputs/visualizations/`
- [x] Inspect quality metrics in `quality_summary.txt`

### 2. Verify Dataset ✅
- [x] Confirm 375 patches created
- [x] Check 100% success rate
- [x] Verify train/val split (300/75)
- [x] Review class distributions

### 3. Begin CNN Training 🎯
- [ ] Open `Week_3_Lab_CNN_Training.ipynb`
- [ ] Run training notebook
- [ ] Monitor validation accuracy
- [ ] Save trained model

### 4. Document Results 📝
- [ ] Create model card (`reports/week3/model_card.md`)
- [ ] Write ethics reflection (`reports/week3/ethics_reflection.md`)
- [ ] Update main README with Week 3 completion

---

## Design Philosophy

Phase 2 prioritizes **quality over speed**:

### Modular Design
- 8 independent scripts (easier to debug)
- Each script has clear inputs/outputs
- Can re-run individual steps if needed
- Fail-fast approach catches issues early

### Comprehensive Documentation
- Every step documented
- Both machine-readable (JSON/CSV) and human-readable (TXT) formats
- Visualizations for quality assessment
- Reproducible with fixed random seeds

### Quality Assurance
- 10 validation checks before declaring success
- Quality tiers classify every patch
- Anomaly detection catches edge cases
- Stratified splitting ensures balance

### Production-Ready
- Ready for immediate CNN training
- All metadata for experiment tracking
- Splits prevent data leakage
- Complete audit trail

---

## Key Takeaways

### ✅ What Phase 2 Accomplished

1. **Extracted 375 high-quality patches** (100% success, 0% NaN)
2. **Created stratified train/val split** (300/75, maintains class balance)
3. **Passed all 10 validation checks** (dataset integrity verified)
4. **Generated comprehensive documentation** (21 metadata/report files)
5. **Achieved production-ready dataset** (ready for CNN training)

### 🎯 Why Phase 2 Matters

- **Quality dataset = Better model:** 0% NaN ensures clean training
- **Stratified split = Fair evaluation:** Both sets represent all classes
- **Complete metadata = Reproducibility:** Can trace every decision
- **Comprehensive validation = Confidence:** Know dataset is trustworthy

### 🚀 Impact on Week 3

Phase 2 enables Week 3 CNN training:
- No data cleaning needed (already perfect quality)
- Direct loading from train_split.csv / val_split.csv
- Balanced classes prevent bias
- Sufficient samples for meaningful training (300 train, 75 val)

---

## Citation & Attribution

**Data Source:**
- Study Area: Los Lagos Region, Chile
- Imagery: Sentinel-2 Level-2A (2019 median composite)
- Training Labels: Manually digitized in QGIS (Phase 0)

**Pipeline:**
- Phase 0: Training polygon creation
- Phase 1: Extraction validation
- Phase 2: Batch dataset creation (this phase)
- Phase 3: CNN training (Week 3 Lab)

**Author:** Week 3 Activity 1 - GeoAI Course
**Date:** 2025-10-20 to 2025-10-25
**Purpose:** Educational - CNN-based land cover classification

---

## Summary

Phase 2 successfully transformed 126 training polygons into a production-ready dataset of 375 high-quality patches. With 100% extraction success, 0% NaN values, and comprehensive validation, the dataset is ready for CNN training in Week 3 Lab.

**Status:** ✅ **PHASE 2 COMPLETE**

**Next Phase:** Week 3 Lab - CNN Training (`Week_3_Lab_CNN_Training.ipynb`)

---

**For detailed completion log, see:** `PHASE0_PHASE1_PHASE2_COMPLETION_LOG.md`

**For Phase 2 comprehensive summary, see:** `phase2_outputs/PHASE2_COMPLETE.txt`
