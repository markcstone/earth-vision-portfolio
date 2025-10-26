# Week 4 Exercise: Transfer Learning and Multi-Sensor Fusion

**Course:** GeoAI and Earth Vision: Foundations to Frontier Applications**Arc:** Arc 2 — Learning to Represent (Weeks 4-6)**Duration:** 8-10 hours**Prerequisite:** Week 3 complete (SimpleCNN baseline, training dataset)

---

## Overview

Week 4 builds on your Week 3 CNN baseline by introducing **transfer learning** and **multi-sensor data fusion**. Due to a known EfficientNet-B0 bug in this environment, we use ResNet-50 for all hands-on work. Outcomes may vary; the goal is to implement, evaluate, and document results (including negative findings).

### Learning Objectives

By completing this exercise, you will:

1. Implement transfer learning with pretrained CNNs (ResNet-50); EfficientNet-B0 excluded due to a known bug.
2. Prepare and fuse multi-sensor inputs (Sentinel-2, MODIS NDVI, SRTM DEM) into aligned 8-band tensors.
3. Modify and fine-tune pretrained backbones for multispectral inputs using appropriate freezing and learning rates.
4. Evaluate models with quantitative metrics and ablations (single-sensor vs fusion, frozen vs fine-tuned).
5. Document a reproducible baseline pipeline with saved configs, artifacts, and comparative results.
6. **Critically evaluate when transfer learning helps vs hurts, and understand that negative results provide valuable scientific insights.**

---

## Exercise Structure

Week 4 follows a **three-phase structure** building on Week 3's proven approach:

### Phase 4A: Transfer Learning with Single Sensor (3-4 hours)

**Objective**: Fine-tune ResNet-50 on Sentinel-2 imagery to establish a transfer learning baseline.

**Key Activities**:

- Load pretrained ResNet-50 (ImageNet weights)

- Modify architecture for 6-band Sentinel-2 input

- Freeze early layers, train classification head

- Fine-tune with low learning rate

- Compare to Week 3 SimpleCNN baseline

**Expected Result**: Stable training and an informative comparison to Week 3; improvements are not guaranteed. Transfer learning may underperform training from scratch due to domain mismatch, aggressive freezing (>80%), or small patch size (8×8→32×32 upsampling).

### Phase 4B: Multi-Sensor Fusion (4-5 hours)

**Objective**: Combine Sentinel-2, MODIS, and DEM to create 8-band fused dataset and train fusion model.

**Key Activities**:

- Create MODIS NDVI composite (austral summer 2019)

- Load SRTM DEM

- Resample and align all sources to 10m resolution

- Extract 8-band patches at training polygon locations

- Train ResNet-50 on fused data

- Conduct ablation study (single-sensor vs fusion)

**Expected Result**: Outcomes may vary; fusion may decrease performance if training data insufficient (need 500-1000 samples for 8 bands). Focus on ablation results, sensor contribution analysis, and understanding why fusion may not always help. Always use stratified splits to ensure all classes in validation set.

---

## Prerequisites

### From Week 3

You should have completed:

- ✅ Week 3 Phase 0-3 (training polygons, dataset, SimpleCNN model)

- ✅ 126 training polygons digitized in QGIS

- ✅ 375 patches extracted (300 train, 75 val)

- ✅ SimpleCNN trained (86.67% validation accuracy)

### Required Files

```
data/
├── labels/
│   └── larger_polygons.geojson          # Training polygons from Week 3
├── processed/
│   ├── X_train.npy                      # 300 training patches (8×8×6)
│   ├── y_train.npy                      # Training labels
│   ├── X_val.npy                        # 75 validation patches
│   └── y_val.npy                        # Validation labels
└── external/
    └── aoi.geojson                      # Los Lagos AOI

models/
└── week3/
    └── best_model.h5                    # Week 3 baseline model
```

### Software Requirements

**New for Week 4**:

- TensorFlow/Keras 2.20+ (tf.keras.applications.ResNet50)

- Earth Engine API (for MODIS, DEM access)

**Existing from Week 3**:

- Python 3.11+, NumPy, pandas, geopandas

- geemap, rasterio

- matplotlib, seaborn, plotly

---

## Phase Descriptions

### Phase 4A: Transfer Learning with Single Sensor

**Notebook**: `Week_4_Exercise_Phase4A.ipynb`

**Sections**:

1. **Setup and Configuration**
  - Load Week 3 dataset (X_train, y_train, X_val, y_val)
  - Verify data quality
  - Initialize experiment logging

1. **Load Pretrained Model**
  - Import ResNet-50 (ImageNet weights)
  - Inspect architecture (layers, parameters)
  - Understand layer structure and freezing strategy

1. **Modify Architecture**
  - Replace input layer for 6-band Sentinel-2
  - Remove ImageNet classification head
  - Add new 5-class classification head
  - Visualize modified architecture

1. **Configure Training**
  - Freeze early layers (first 150 layers)
  - Set low learning rate (1e-4)
  - Configure callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)

1. **Train Model**
  - Train for up to 50 epochs
  - Monitor training/validation curves
  - Save best model

1. **Evaluate and Compare**
  - Compute accuracy, precision, recall, F1
  - Generate confusion matrix
  - Compare to Week 3 baseline
  - Visualize sample predictions

1. **Document Results**
  - Save training history
  - Export metrics to JSON
  - Update experiment log

**Deliverables**:

- `models/week4/phase4a_transfer_model.h5` (trained model)

- `reports/phase4a_metrics.json` (evaluation metrics)

- `figures/phase4a_training_curves.png` (training history)

- `figures/phase4a_confusion_matrix.png` (confusion matrix)

- `reports/phase4a_comparison.md` (comparison to Week 3)

**Time Estimate**: 3-4 hours

---

### Phase 4B: Multi-Sensor Fusion

**Notebook**: `Week_4_Exercise_Phase4B.ipynb`

**Sections**:

1. **Multi-Sensor Data Preparation**
  - Load Sentinel-2 composite (from Week 3)
  - Create MODIS NDVI composite (2019 austral summer)
  - Load SRTM DEM
  - Resample MODIS and DEM to 10m resolution
  - Verify spatial alignment

1. **Fused Dataset Creation**
  - Stack 8 bands: [B2, B3, B4, B8, B11, B12, MODIS_NDVI, DEM]
  - Extract patches at training polygon locations
  - Quality control (NaN check, value ranges)
  - Save fused dataset

1. **Train Fusion Model**
  - Adapt ResNet-50 for 8-band input (e.g., 1×1 Conv band reduction or modified first conv)
  - Train with same configuration as Phase 4A (adjust freeze ratio/LR as needed)
  - Monitor convergence and stability

1. **Ablation Study**
  - Train on Sentinel-2 only (6 bands) — baseline
  - Train on Sentinel-2 + MODIS (7 bands)
  - Train on full fusion (8 bands)
  - Compare accuracies

1. **Analysis and Interpretation**
  - Which sensor contributes most?
  - Visualize feature importance
  - Analyze failure cases
  - Document findings

1. **Baseline Pipeline Documentation**
  - Create `config.yaml` with all parameters
  - Write `baseline_report.md` (methods, results, reflections)
  - Save reproducible workflow

**Deliverables**:

- `data/processed/X_train_fused.npy` (300 patches, 8×8×8)

- `data/processed/X_val_fused.npy` (75 patches, 8×8×8)

- `models/week4/phase4b_fusion_model.h5` (trained fusion model)

- `reports/phase4b_ablation_study.json` (comparison metrics)

- `figures/phase4b_sensor_comparison.png` (ablation results)

- `config.yaml` (baseline pipeline configuration)

- `reports/baseline_report.md` (comprehensive documentation)

**Time Estimate**: 4-5 hours

---

### Phase 4C: SimpleCNN + Multi-Sensor Fusion (Ablation)

**Objective**: Test whether the Week 3 SimpleCNN architecture benefits from multi-sensor fusion (S2, S2+MODIS, S2+DEM, Full Fusion) on the Phase 4 dataset.

**Key Activities**:

- Load fused dataset outputs from Phase 4B (8×8×8 patches) and derive band subsets (6, 7, 7, 8 bands).

- Train SimpleCNN variants from scratch for each configuration using the Week 3 hyperparameters (Adam, lr=1e-3, batch=32, early stopping).

- Compare accuracies, per-class metrics, and confusion matrices; analyze failure modes.

**Deliverables**:

- `phase4c_outputs/models/*.h5` (trained SimpleCNN variants)

- `phase4c_outputs/evaluation_results.json` (metrics per configuration)

- `figures/phase4c_confusion_matrices.png` and `figures/phase4c_accuracy_comparison.png`

- `reports/phase4c_report.md` (ablation narrative with conclusions and next steps)

**Time Estimate**: 30-45 minutes

---

## Expected Outcomes

### Performance Targets

Focus on process quality and analysis over specific accuracy thresholds:
- Establish an apples-to-apples baseline on the same dataset/splits
- Achieve stable training without collapse (no single-class predictions)
- Produce informative ablations (S2 vs S2+DEM vs S2+MODIS vs Full)
- Document clear hypotheses and next steps if accuracy does not improve

### Actual Results from Implementation

**Phase 4A: Transfer Learning (ResNet50, S2 only)**
- Result: **13.33% accuracy** (vs Week 3 SimpleCNN 86.67%)
- Failure mode: Model collapse (predicted only Agriculture class)
- Root causes: 85% layers frozen (too aggressive), 8×8→32×32 upsampling artifacts, ImageNet domain mismatch
- **Key lesson**: Transfer learning can fail catastrophically with aggressive freezing

**Phase 4B: Transfer Learning + Fusion (ResNet50, 8 bands)**
- Result: **0.00% accuracy** (complete failure)
- Failure mode: Validation accuracy flat at 0% across all epochs
- Root causes: Same transfer learning issues + validation set had only 1 class (fixed with stratified split)
- **Key lesson**: Adding sensors doesn't fix fundamental problems; always validate class distribution

**Phase 4C: SimpleCNN + Fusion (trained from scratch, ablation)**
- Results: S2-only (48%), S2+MODIS (53%), **S2+DEM (69% best)**, Full Fusion (44%)
- Key findings:
  - SimpleCNN trained successfully (unlike ResNet50 which collapsed)
  - DEM (30m→10m) >> MODIS (250m→10m) - elevation more useful
  - Full fusion (8 bands) worst - 300 samples insufficient for 8-dimensional input
- **Key lesson**: Training from scratch can outperform transfer learning for small, domain-specific datasets

### Revised Key Insights

**From Phase 4A:**
- Transfer learning **may not** improve over training from scratch
- Domain mismatch (ImageNet RGB vs multispectral) can be a hard limit
- Aggressive freezing (>80%) risks model collapse
- Patch size matters: 8×8→32×32 upsampling introduces artifacts

**From Phase 4B:**
- Multi-sensor fusion **does not automatically** improve performance
- MODIS (250m→10m) heavy upsampling adds noise
- DEM (30m→10m) more useful than MODIS
- **Critical**: Always use stratified splits for balanced validation

**From Phase 4C:**
- SimpleCNN (48-69%) >> ResNet50 (0-13%) for this task
- Fusion decreased performance with limited data (need 500-1000 samples for 8 bands)
- Dataset quality matters more than architecture complexity
- **Most important**: Negative results are scientifically valid and instructive

---

## Success Criteria

### Technical Criteria

✅ Stable training (no single-class collapse)✅ Clear, same-dataset baseline established✅ Proper alignment and QC (no NaN, correct shapes, stratified splits)✅ Meaningful ablation analyses completed (S2/S2+DEM/S2+MODIS/Full)✅ Reproducible configs and artifacts saved

### Documentation Criteria

✅ All notebooks run end-to-end without errors✅ `config.yaml` captures all parameters✅ `baseline_report.md` documents methods, failures, and results✅ Experiment log tracks decisions, hypotheses, and next steps✅ Code is well-commented and reproducible

### Learning Criteria

✅ Can explain when/why transfer learning can fail (domain mismatch, aggressive freezing, small patches)✅ Can describe trade-offs of early vs late/decision fusion (complexity vs performance)✅ Can compare transfer learning to training from scratch on same dataset (SimpleCNN >> ResNet50 for our case)✅ Can justify design decisions (freeze ratio <80%, LR tuning, patch size ≥32×32 native)✅ Can interpret ablation results and propose informed next steps (increase data, try late fusion)✅ **Can recognize that negative results are scientifically valid and provide valuable insights**

---

## Troubleshooting Guide

### Common Issues

**Issue 1: Input shape mismatch**

```
ValueError: expected input shape (8, 8, 3), got (8, 8, 6)
```

**Solution**: Modify input layer to accept 6 or 8 bands (see Phase 4A Section 3)

**Issue 2: Slow convergence**

```
Validation accuracy stuck at ~70% after 20 epochs
```

**Solution**: Lower learning rate (try 1e-5), unfreeze more layers

**Issue 3: MODIS alignment artifacts**

```
Blocky patterns in fused patches
```

**Solution**: Use bilinear resampling, verify CRS match

**Issue 4: Out of memory**

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution**: Reduce batch size (try 16 or 8), use mixed precision training

**Issue 5: Overfitting**

```
Training acc 95%, validation acc 85%
```

**Solution**: Increase dropout (0.3 → 0.5), add data augmentation

**Issue 6: Model collapse (predicts single class)**

```
Validation accuracy 0-20%, model predicts same class for all inputs
```

**Solution**: Check class distribution (use stratified split), reduce freezing ratio to 50-70%, increase learning rate

**Issue 7: Validation set imbalanced**

```
Validation has only 1-2 classes, training has all 5 classes
```

**Solution**: Use `train_test_split(..., stratify=y)` to ensure balanced splits; verify with `np.unique(y_val)`

**Issue 8: Transfer learning underperforms training from scratch**

```
ResNet50 gets 13%, SimpleCNN gets 86%
```

**Solution**: For small, domain-specific datasets, consider training from scratch or (a) reduce freeze ratio to 50-70%, (b) extract larger patches (32×32 native), (c) increase training data

---

## Reflection Questions

### Technical Reflection

1. **Transfer Learning Effectiveness**
  - Did transfer learning improve accuracy over Week 3, or did it underperform? Why?
  - Which layers contributed most? Or did aggressive freezing (>80%) prevent adaptation?
  - What evidence do you have for/against ImageNet features transferring to multispectral satellite data?
  - **If it failed**: What would you try next? (Less freezing? Larger patches? Train from scratch?)

1. **Multi-Sensor Fusion**
  - Which sensor contributed most? (Expected: DEM > MODIS due to resolution difference)
  - Did fusion improve accuracy, or did it decrease due to insufficient training data?
  - What are the trade-offs of fusion (complexity vs accuracy vs data requirements)?
  - When would single-sensor be preferable to fusion? (Small datasets, limited compute, interpretability)

1. **Baseline Pipeline**
  - What makes your pipeline reproducible?
  - What would you change for a different case study?
  - How will this baseline support future experiments (Weeks 5-12)?

### Ethical Reflection

1. **Data Inequality**
  - How do sensor differences (Sentinel-2 vs MODIS) reflect global data access inequities?
  - Who benefits most from high-resolution imagery?
  - How might transfer learning perpetuate biases from ImageNet?

1. **Model Transparency**
  - Can you explain why your fusion model makes specific predictions?
  - What are the limitations of your model?
  - How would you communicate model uncertainty to stakeholders?

---

## Next Steps

### Immediate (End of Week 4)

1. Complete both phases (4A and 4B)

1. Write baseline report

1. Post ethics reflection to #EthicsThread

1. Clean and commit code to GitHub

### Preparation for Week 5

Week 5 introduces **representation learning and self-supervision**:

- Visualize embeddings with UMAP

- Understand what CNNs learn (feature space analysis)

- Introduction to SimCLR (self-supervised learning)

**Connection**: Your Week 4 transfer model learns representations from ImageNet labels. Week 5 explores learning representations without labels.

### Long-Term (Capstone)

Your Week 4 baseline pipeline becomes the foundation for:

- Week 5-6: Self-supervised representation learning

- Week 7-10: Transformer fine-tuning, foundation models

- Week 11-12: Capstone synthesis and comparison

**Document everything now** — future you will thank present you!

---

## Assessment Rubric

### Phase 4A: Transfer Learning (50 points)

| Criterion | Points | Description |
| --- | --- | --- |
| **Implementation** | 20 | Model loads, modifies, trains correctly |
| **Performance/Analysis** | 10 | Provides stable training and insightful analysis (no fixed accuracy threshold) |
| **Comparison** | 10 | Thorough comparison to Week 3 baseline |
| **Documentation** | 10 | Code commented, results documented |

### Phase 4B: Multi-Sensor Fusion (50 points)

| Criterion | Points | Description |
| --- | --- | --- |
| **Data Fusion** | 15 | All sensors aligned, fused correctly |
| **Implementation** | 15 | Fusion model trains stably; analysis of outcomes is provided |
| **Ablation Study** | 10 | Systematic comparison of sensor combinations |
| **Baseline Report** | 10 | Comprehensive documentation (methods, results, failures, and next steps) |

### Total: 100 points

**Grading Scale**:

- 90-100: Excellent (all criteria met, insightful analysis)

- 80-89: Good (most criteria met, solid implementation)

- 70-79: Satisfactory (basic criteria met, some gaps)

- <70: Needs improvement (significant gaps)

---

## Additional Resources

### Transfer Learning

- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

- Pan & Yang (2010) *A Survey on Transfer Learning*

### Multi-Sensor Fusion

- [TorchGeo Multi-Sensor Tutorial](https://torchgeo.readthedocs.io/en/stable/tutorials/multisensor.html)

- Schmitt & Zhu (2016) *Data Fusion in Remote Sensing*

- Zhang et al. (2020) *Deep Learning for Remote Sensing Data Fusion*

### Earth Observation Applications

- Rußwurm & Körner (2020) *Self-attention for satellite time series*

- Tuia et al. (2023) *AI to advance Earth observation*

---

**Ready to begin? Start with Phase 4A and build on your Week 3 success!**

