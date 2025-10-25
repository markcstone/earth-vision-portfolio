# Week 3 Exercise: CNN Land Cover Classification

**Complete Pipeline from Training Data to Trained Model**

## Overview

Week 3 Exercise demonstrates a complete end-to-end workflow for CNN-based land cover classification using Sentinel-2 imagery. The exercise is divided into 4 phases, each building on the previous phase to create a production-ready classification system.

**Study Area:** Los Lagos Region, Chile
**Imagery:** Sentinel-2 Level-2A (2019 median composite)
**Classes:** 5 land cover types (Agriculture, Forest, Parcels, Urban, Water)
**Framework:** TensorFlow/Keras
**Duration:** ~4-6 hours total

---

## Learning Objectives

By completing this exercise, you will:

1. **Understand the full ML pipeline** - From raw imagery to trained model
2. **Master training data creation** - Manual digitization in QGIS with quality control
3. **Learn patch-based classification** - Why patch size matters for CNNs
4. **Implement data quality workflows** - Validation, splitting, and integrity checks
5. **Train and evaluate CNNs** - Complete model development and interpretation
6. **Apply ethical ML practices** - Recognize limitations and appropriate use cases

---

## Exercise Structure

The exercise is divided into **4 sequential phases**, each with its own notebook and README:

### 📐 Phase 0: Training Data Creation
**Notebook:** `Week_3_Exercise_Phase0.ipynb`
**README:** `Week_3_Exercise_Phase0_README.md`
**Duration:** ~90 minutes
**Tool:** QGIS

**What you'll do:**
- Load Sentinel-2 composite in QGIS
- Manually digitize 126 training polygons across 5 land cover classes
- Ensure spatial distribution and class balance
- Export as GeoJSON for Phase 1

**Key Skills:**
- Visual interpretation of satellite imagery
- Training data quality principles
- QGIS polygon digitization
- Avoiding common labeling errors

**Outputs:**
- `data/labels/larger_polygons.geojson` (126 polygons)
- `phase0_config.json` (metadata)

---

### ✅ Phase 1: Validation & Configuration
**Notebook:** `Week_3_Exercise_Phase1.ipynb`
**README:** `Week_3_Exercise_Phase1_README.md`
**Duration:** ~45 minutes
**Tool:** Jupyter + Earth Engine

**What you'll do:**
- Load and analyze training polygons (size distribution by class)
- Load Sentinel-2 composite from Earth Engine
- **Test single patch extraction** (fail-fast validation)
- Determine optimal patch size based on polygon constraints
- Calculate patches per polygon

**Key Skills:**
- Earth Engine Python API
- Statistical analysis of geospatial data
- Patch size trade-offs
- Pre-flight diagnostics

**Key Finding:**
- **Patch size: 8×8 pixels (80m)** - Small enough for smallest class (Urban), large enough for CNN
- **Patches per polygon: 3** - With spatial jitter for augmentation

**Outputs:**
- `phase1_outputs/` (9 files: visualizations, analysis, config)
- Validated extraction parameters

---

### 🎯 Phase 2: Batch Dataset Creation
**Notebook:** `Week_3_Exercise_Phase2.ipynb`
**README:** `Week_3_Exercise_Phase2_README.md`
**Duration:** ~60-90 minutes
**Tool:** Jupyter + Earth Engine

**What you'll do:**
- Create extraction manifest with spatial jitter (3 patches × 126 polygons = 378 planned)
- Extract 375 patches from Earth Engine (batch processing)
- Perform quality control analysis (NaN percentage, value ranges)
- Create stratified train/validation split (80/20, maintains class balance)
- Validate dataset integrity (10 automated checks)
- Generate summary statistics and visualizations

**Key Skills:**
- Spatial jitter for data augmentation
- Batch Earth Engine extraction
- Quality control workflows
- Stratified splitting
- Dataset validation

**Key Results:**
- **375 successful patches** (100% success rate, 0% NaN)
- **Training set:** 300 patches (80%)
- **Validation set:** 75 patches (20%)
- **Data quality:** Excellent (all patches usable)

**Outputs:**
- `phase2_outputs/patches/*.npy` (375 NumPy arrays, 8×8×6)
- `phase2_outputs/metadata/` (split CSVs, metadata JSON)
- `phase2_outputs/reports/` (visualizations, quality reports)

---

### 🤖 Phase 3: CNN Training & Evaluation
**Notebook:** `Week_3_Exercise_Phase3.ipynb`
**README:** `Week_3_Exercise_Phase3_README.md`
**Duration:** ~60 minutes
**Tool:** Jupyter + TensorFlow/Keras

**What you'll do:**
- Load 375 patches from Phase 2 outputs
- Normalize data (divide by 10000 → [0, 1] range)
- Build SimpleCNN architecture (2 conv blocks, 54K parameters)
- Configure training (Adam optimizer, callbacks, early stopping)
- Train model for up to 50 epochs
- Evaluate performance (accuracy, precision, recall, F1)
- Generate visualizations (confusion matrix, training curves, predictions)
- Interpret results and document limitations

**Key Skills:**
- TensorFlow/Keras model building
- Training configuration and monitoring
- Performance metrics interpretation
- Model evaluation best practices
- Error analysis

**Key Results:**
- **Overall Accuracy:** 86.67%
- **Best Classes:** Forest (100% F1), Water (100% F1)
- **Challenging Classes:** Urban-Parcels confusion (cyan roofs vs buildings)
- **Model Size:** 54,181 parameters (211 KB)

**Outputs:**
- `models/week3/best_model.h5` (trained model)
- `figures/week3/*.png` (4 visualizations)
- `reports/week3/metrics.json` (complete performance metrics)

---

## Prerequisites

### Required Software
- **QGIS 3.x** - For Phase 0 polygon digitization
- **Jupyter Notebook/Lab** - For Phases 1-3
- **Python 3.8+** with packages:
  - `earthengine-api` - Earth Engine Python API
  - `geemap` - EE visualization and data extraction
  - `geopandas` - Vector data handling
  - `tensorflow` - Deep learning framework
  - `scikit-learn` - Train/val splitting and metrics
  - `matplotlib`, `seaborn` - Visualization

### Required Accounts
- **Google Earth Engine** - Free account, sign up at https://earthengine.google.com
- Authenticate before starting: `earthengine authenticate`

### Required Data
- **AOI GeoJSON** - `data/external/larger_aoi.geojson` (study area boundary)
- **Sentinel-2 Composite** - Created using `Week_3_S2_Composites.ipynb` or use pre-existing asset

---

## Directory Structure

All Week 3 materials are organized in the `notebooks/Week3/` directory:

```
earth-vision-portfolio/
└── notebooks/
    └── Week3/
        ├── Week_3_Exercise_README.md           # This file
        ├── Week_3_Exercise_Phase0.ipynb         # Phase 0 notebook
        ├── Week_3_Exercise_Phase0_README.md     # Phase 0 docs
        ├── Week_3_Exercise_Phase1.ipynb         # Phase 1 notebook
        ├── Week_3_Exercise_Phase1_README.md     # Phase 1 docs
        ├── Week_3_Exercise_Phase2.ipynb         # Phase 2 notebook
        ├── Week_3_Exercise_Phase2_README.md     # Phase 2 docs
        ├── Week_3_Exercise_Phase3.ipynb         # Phase 3 notebook
        ├── Week_3_Exercise_Phase3_README.md     # Phase 3 docs
        ├── Week_3_Lab.ipynb                     # PyTorch alternative
        ├── Week_3_S2_Composites.ipynb           # Preprocessing utility
        ├── Week_3_Study_Guide.md                # Conceptual guide
        ├── phase0_config.json                   # Phase 0 output
        ├── phase1_outputs/                      # Phase 1 outputs
        └── phase2_outputs/                      # Phase 2 outputs
```

## Running the Exercise

### Sequential Execution (Recommended)

**Navigate to the Week3 directory and complete each phase in order:**

```bash
# Navigate to Week3 directory
cd notebooks/Week3

# Phase 0: Open QGIS and digitize polygons (see Phase0_README.md)
# Save output to: ../../data/labels/larger_polygons.geojson

# Phase 1: Validation and configuration
jupyter notebook Week_3_Exercise_Phase1.ipynb
# Expected runtime: ~10 minutes (mostly Earth Engine processing)
# Outputs: phase1_outputs/

# Phase 2: Dataset creation
jupyter notebook Week_3_Exercise_Phase2.ipynb
# Expected runtime: ~45-60 minutes (Earth Engine batch extraction)
# Outputs: phase2_outputs/

# Phase 3: CNN training
jupyter notebook Week_3_Exercise_Phase3.ipynb
# Expected runtime: ~10-15 minutes (model training)
# Outputs: ../../models/week3/, ../../figures/week3/, ../../reports/week3/
```

### Parallel Execution (Advanced)

If you already have outputs from previous phases:

```bash
cd notebooks/Week3
# Can run Phase 3 independently if phase2_outputs/ exists
jupyter notebook Week_3_Exercise_Phase3.ipynb
```

---

## Expected Outputs

### Directory Structure After Completion

```
earth-vision-portfolio/
├── data/
│   └── labels/
│       └── larger_polygons.geojson          # Phase 0 output (126 polygons)
├── notebooks/
│   └── Week3/
│       ├── Week_3_Exercise_README.md        # Master README
│       ├── Week_3_Exercise_Phase*.ipynb     # The 4 phase notebooks
│       ├── Week_3_Exercise_Phase*_README.md # Phase-specific docs
│       ├── Week_3_Lab.ipynb                 # PyTorch alternative
│       ├── Week_3_S2_Composites.ipynb       # Preprocessing
│       ├── Week_3_Study_Guide.md            # Study guide
│       ├── phase0_config.json               # Phase 0 metadata
│       ├── phase1_outputs/                  # Phase 1 outputs (9 files)
│       │   ├── composite_info.json
│       │   ├── patch_size_analysis.png
│       │   └── environment_info.json
│       └── phase2_outputs/                  # Phase 2 outputs (396 files)
│           ├── patches/*.npy                # 375 patch arrays
│           ├── metadata/                    # Split CSVs, metadata
│           └── reports/                     # Quality reports, viz
├── models/
│   └── week3/
│       └── best_model.h5                    # Phase 3 trained model (211 KB)
├── figures/
│   └── week3/
│       ├── confusion_matrix.png
│       ├── training_curves.png
│       ├── prediction_samples.png
│       └── sample_patches_by_class.png
└── reports/
    └── week3/
        ├── metrics.json
        └── training_history.json
```

### File Sizes
- Total patch data: ~600 KB (375 patches)
- Trained model: ~211 KB (54K parameters)
- Visualizations: ~2 MB (300 DPI PNG)
- Complete outputs: ~5-10 MB

---

## Key Concepts Covered

### 1. Training Data Quality
- **Manual vs automated labeling** - When to digitize manually
- **Spatial distribution** - Avoiding geographic bias
- **Class balance** - Ensuring all classes represented
- **Polygon size** - Impact on patch extraction

### 2. Patch-Based Classification
- **Why patches?** - CNNs need fixed-size inputs
- **Patch size selection** - Balancing context vs constraints
- **Spatial jitter** - Authentic data augmentation
- **Edge effects** - Handling polygon boundaries

### 3. Data Pipeline Design
- **Fail-fast validation** - Test on 1 before batch processing
- **Quality gates** - Automated checks at each stage
- **Reproducibility** - Random seeds, versioning, documentation
- **Separation of concerns** - Each phase has clear purpose

### 4. Model Development
- **Architecture design** - Shallow networks for small patches
- **Training monitoring** - Callbacks, early stopping, LR scheduling
- **Performance metrics** - Beyond accuracy (precision, recall, F1)
- **Error analysis** - Understanding failure modes

### 5. Ethical Considerations
- **Model limitations** - Small training set, single region, temporal snapshot
- **Appropriate use** - Research vs operational decisions
- **Bias and fairness** - Human labeling biases reflected in model
- **Transparency** - Documenting assumptions and constraints

---

## Common Issues and Solutions

### Issue 1: Earth Engine Authentication Fails

**Symptom:**
```
Error: Please authenticate Earth Engine
```

**Solution:**
```bash
earthengine authenticate
# Follow browser prompts to authorize
```

### Issue 2: Phase 1 Single Patch Extraction Fails

**Symptoms:**
- `geemap.ee_to_numpy()` returns wrong size
- NaN values in extracted patch
- Earth Engine timeout errors

**Solutions:**
1. **Check AOI bounds** - Ensure polygon is in study area
2. **Verify composite exists** - Asset ID correct in config
3. **Reduce patch size** - Try 6×6 if 8×8 fails
4. **Check Earth Engine quota** - May need to wait if quota exceeded

### Issue 3: Phase 2 Extraction Slow or Fails

**Symptoms:**
- Taking >2 hours for 375 patches
- Many extraction failures (>10%)
- Earth Engine "Too many requests" errors

**Solutions:**
1. **Add delays** - `time.sleep(0.5)` between extractions
2. **Reduce batch size** - Extract fewer patches per run
3. **Check network** - Stable internet required
4. **Use smaller AOI** - Reduce study area if too large

### Issue 4: Phase 3 Training Accuracy Low (<70%)

**Symptoms:**
- Validation accuracy stuck below 70%
- Training loss not decreasing
- Model predicts same class for everything

**Solutions:**
1. **Check data normalization** - Ensure divided by 10000
2. **Verify labels** - Check `y_train` is integers 0-4
3. **Increase epochs** - May need >50 epochs
4. **Review Phase 0 labels** - Mislabeled polygons?
5. **Check class balance** - Extreme imbalance (>50% one class)?

### Issue 5: Out of Memory During Training

**Symptom:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
1. **Reduce batch size** - Change from 32 to 16 or 8
2. **Use CPU** - `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`
3. **Close other programs** - Free up RAM
4. **Simplify model** - Reduce filters (32→16, 64→32)

---

## Extensions and Variations

Once you've completed the base exercise, try these extensions:

### 1. Data Augmentation
- Add rotation, flipping, brightness adjustments in Phase 2
- Compare performance with/without augmentation

### 2. Larger Patches
- Re-run Phase 1 with 16×16 patches
- Analyze impact on accuracy and training time

### 3. More Training Data
- Digitize 200+ polygons in Phase 0
- Measure accuracy improvement vs effort

### 4. Transfer Learning
- Use pre-trained ResNet/VGG on Phase 2 data
- Compare to SimpleCNN performance

### 5. Multi-Temporal Analysis
- Add 2025 composite to Phase 2
- Train model on change detection (2019→2025)

### 6. Hyperparameter Tuning
- Experiment with learning rates, architectures, optimizers
- Use grid search or random search

---

## Assessment Criteria

If using this exercise for coursework:

### Phase 0 (25 points)
- ✅ 126 polygons digitized (10 pts)
- ✅ All 5 classes represented (5 pts)
- ✅ Spatial distribution adequate (5 pts)
- ✅ Polygon sizes appropriate (5 pts)

### Phase 1 (20 points)
- ✅ Single patch extraction successful (10 pts)
- ✅ Patch size justified with analysis (5 pts)
- ✅ Configuration saved correctly (5 pts)

### Phase 2 (25 points)
- ✅ 375 patches extracted successfully (10 pts)
- ✅ Quality control performed (5 pts)
- ✅ Stratified split created (5 pts)
- ✅ Validation checks passed (5 pts)

### Phase 3 (30 points)
- ✅ Model trained successfully (10 pts)
- ✅ Performance >70% accuracy (5 pts)
- ✅ Evaluation metrics computed (5 pts)
- ✅ Visualizations created (5 pts)
- ✅ Results interpreted (5 pts)

**Total: 100 points**

---

## Time Investment

**Estimated time per phase:**
- **Phase 0:** 90 minutes (QGIS digitization)
- **Phase 1:** 45 minutes (validation + analysis)
- **Phase 2:** 90 minutes (extraction + QC)
- **Phase 3:** 60 minutes (training + evaluation)

**Total:** 4.5 hours

**With extensions/debugging:** 6-8 hours

---

## Learning Outcomes

After completing this exercise, you will be able to:

1. ✅ **Create high-quality training data** for supervised ML
2. ✅ **Design and execute geospatial ML pipelines** from scratch
3. ✅ **Implement quality control** at each pipeline stage
4. ✅ **Train and evaluate CNNs** for image classification
5. ✅ **Interpret model performance** using multiple metrics
6. ✅ **Recognize limitations** and ethical considerations
7. ✅ **Document workflows** for reproducibility

---

## Additional Resources

### Earth Engine
- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- [geemap Documentation](https://geemap.org)
- [Sentinel-2 Data](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)

### Deep Learning
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Guide](https://keras.io/guides/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

### Geospatial ML
- [Awesome Satellite Imagery Datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)
- [STAC Specification](https://stacspec.org)
- [Raster Vision](https://docs.rastervision.io)

---

## Citation

If using this exercise in research or publications:

```
Stone, M. (2025). Week 3 Exercise: CNN Land Cover Classification.
Earth Vision Portfolio, GeoAI Course.
https://github.com/[username]/earth-vision-portfolio
```

---

## Support

**For help with this exercise:**
1. Check phase-specific READMEs for detailed troubleshooting
2. Review "Common Issues and Solutions" section above
3. Consult Week 3 Study Guide for conceptual background
4. Post questions to course discussion board

---

## Summary

Week 3 Exercise provides a comprehensive introduction to CNN-based land cover classification. By completing all 4 phases, you'll gain hands-on experience with every step of the ML pipeline: from creating training data in QGIS, to extracting and validating patches, to training and evaluating a production-quality CNN model.

**Key Takeaway:** Building reliable geospatial ML systems requires careful attention to data quality, systematic validation, and honest assessment of limitations.

**Ready to begin?** Start with Phase 0 and work sequentially through each phase. Good luck! 🚀

---

**Last Updated:** 2025-10-25
**Version:** 1.0
**Phases:** 0 (Polygons) → 1 (Validate) → 2 (Extract) → 3 (Train)
