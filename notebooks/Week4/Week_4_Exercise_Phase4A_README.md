# Week 4 Exercise - Phase 4A: Transfer Learning README

**Date:** 2025-10-26
**Status:** Complete (Documented Failure)
**Result:** 13.33% accuracy (vs Week 3 baseline 86.67%)

---

## Quick Start

### Option 1: Interactive Notebook (Recommended)
```bash
cd /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio/notebooks/Week4
jupyter notebook Week_4_Exercise_Phase4A.ipynb
```

### Option 2: Run Python Scripts
```bash
# Run all Phase 4A scripts sequentially
cd /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio/notebooks/Week4
./run_phase4a.sh  # If shell script exists
```

---

## What This Phase Tests

**Objective:** Test if transfer learning (ResNet50 pretrained on ImageNet) improves accuracy over Week 3's SimpleCNN (86.67% baseline).

**Hypothesis:** Pretrained ImageNet features would transfer to satellite imagery classification, achieving 90%+ accuracy.

**Result:** **HYPOTHESIS REJECTED** - Accuracy dropped to 13.33% (-73.3 percentage points), demonstrating catastrophic failure of transfer learning for this task.

---

## Why This Experiment Failed

### Root Causes (5 Major Issues)

1. **Aggressive Layer Freezing (85.7%)**
   - Froze 150/175 ResNet50 layers
   - Only 14.3% of parameters trainable (~3.4M / 23.6M)
   - Insufficient capacity to adapt to satellite imagery domain

2. **Domain Mismatch**
   - ImageNet: RGB natural photos (animals, objects, scenes)
   - Satellite: Multispectral signatures (6 bands, land cover patterns)
   - Pretrained edge/texture detectors irrelevant for spectral classification

3. **Patch Size Incompatibility**
   - ResNet50 designed for 224×224 images
   - Phase 4A uses 8×8 patches → upsampled to 32×32 (bilinear)
   - 28× smaller than expected input, upsampling creates artifacts

4. **Band Reduction Bottleneck**
   - 6 spectral bands → 3 pseudo-RGB via 1×1 convolution
   - Critical information loss at first layer
   - Model must learn optimal band combinations with frozen weights

5. **Model Collapse**
   - Validation accuracy: 13.33% (predicts only Agriculture class)
   - Training accuracy: 12-19% (barely above random 20% for 5 classes)
   - Model learned to predict single class for all inputs

---

## Key Files

### Notebook
- `Week_4_Exercise_Phase4A.ipynb` - Interactive notebook with full analysis

### Scripts (Original Implementation)
1. `phase4a_01_setup_and_load.py` - Load Week 3 data
2. `phase4a_02_load_pretrained.py` - Load ResNet50 with ImageNet weights
3. `phase4a_03_modify_architecture.py` - Adapt for 6-band input
4. `phase4a_04_train_model.py` - Train with frozen layers
5. `phase4a_05_evaluate.py` - Evaluate and compare to Week 3

### Outputs
- `phase4a_outputs/transfer_model.h5` - Trained model (failed, but saved)
- `phase4a_outputs/transfer_training_history.json` - Training metrics
- `phase4a_outputs/confusion_matrix.png` - Shows single-class predictions
- `phase4a_outputs/training_history.png` - Accuracy/loss curves

### Documentation
- `Week4_Phase4A_Summary.md` - Comprehensive results report
- `Week_4_Exercise_Phase4A_README.md` - This file

---

## Results Summary

| Metric | Week 3 SimpleCNN | Phase 4A ResNet50 | Change |
|--------|------------------|-------------------|--------|
| **Accuracy** | 86.67% | 13.33% | **-73.3pp** |
| **Precision (macro)** | - | 0.027 | - |
| **Recall (macro)** | - | 0.200 | - |
| **F1 (macro)** | - | 0.047 | - |
| **Parameters** | 54K | 23.6M | 437× larger |
| **Training Time** | ~30s | ~7s | Faster (frozen) |

### Confusion Matrix (Phase 4A)
```
Predicted:  [Agr, For, Prc, Urb, Wat]
Agriculture  [10,   0,   0,   0,   0]  ← All predicted as Agriculture
Forest       [18,   0,   0,   0,   0]
Parcels      [15,   0,   0,   0,   0]
Urban        [14,   0,   0,   0,   0]
Water        [18,   0,   0,   0,   0]
```

**Interpretation:** Model collapsed to predicting single class (Agriculture) for all samples.

---

## Lessons Learned

### 1. Transfer Learning Requires Domain Similarity
- ImageNet (RGB photos) ≠ Satellite imagery (multispectral)
- Pretrained features don't transfer across domains
- Consider domain-specific pretraining (e.g., EuroSAT, UC Merced)

### 2. Aggressive Freezing Prevents Adaptation
- 85.7% frozen is too much for new domain
- Rule of thumb: Freeze <50% for significant domain shift
- Alternative: Gradual unfreezing (start with 80%, reduce to 50% over epochs)

### 3. Input Size Matters for CNNs
- ResNet50 expects 224×224, not 32×32
- Upsampling small patches creates artifacts
- Solution: Use larger patches (32×32 or 64×64 native size)

### 4. Simpler Models Can Outperform Complex Ones
- SimpleCNN (54K params, trained from scratch): 86.67%
- ResNet50 (23.6M params, transfer learning): 13.33%
- For small datasets, train from scratch with task-specific architecture

### 5. Negative Results Are Scientifically Valuable
- Phase 4A proves what **doesn't work**
- Documents failure modes for future researchers
- Provides baseline for comparison (13.33% = transfer learning failure threshold)

---

## Recommendations

### If Retrying Transfer Learning

**Option 1: Reduce Freezing**
```python
# Freeze only 50-70% of layers instead of 85%
freeze_until = int(len(base_model.layers) * 0.5)  # 50% instead of 85%
learning_rate = 1e-3  # Increase 10× for more aggressive fine-tuning
```

**Option 2: Use Larger Patches**
```python
# Extract 32×32 or 64×64 patches (no upsampling needed)
patch_size = 32  # Closer to ResNet50's expected input size
input_shape = (32, 32, 6)  # No upsampling artifacts
```

**Option 3: Domain-Specific Pretraining**
```python
# Pretrain on satellite imagery first
# Step 1: Pretrain ResNet on EuroSAT (10 classes, 27K images)
# Step 2: Fine-tune on Los Lagos data (5 classes, 300 samples)
```

**Expected Improvement:** 70-85% accuracy (still likely below Week 3's 86.67%)

### Recommended Approach

**Train SimpleCNN from Scratch (Week 3 Method)**
- Proven to work: 86.67% accuracy
- Fast training: ~30 seconds
- Task-specific: Designed for 8×8 multispectral patches
- Interpretable: Simple architecture, easy to debug

---

## Educational Value

### This Failure is Instructive Because:

1. **Demonstrates Limits of Transfer Learning**
   - Not a universal solution
   - Domain match is critical
   - More parameters ≠ better results

2. **Shows Importance of Baselines**
   - Week 3's 86.67% clearly shows Phase 4A failed
   - Without baseline, might think 13.33% was acceptable

3. **Teaches Diagnostic Skills**
   - How to identify model collapse
   - How to trace failure to root causes
   - How to decide when to abandon an approach

4. **Validates Simple Solutions**
   - SimpleCNN outperforms ResNet50 by 73.3pp
   - Occam's Razor applies to machine learning

---

## Prerequisites

### Data Requirements
- Week 3 dataset must be generated first
- Required files:
  - `X_train.npy` (300 samples, 8×8×6)
  - `X_val.npy` (75 samples, 8×8×6)
  - `y_train.npy` (300 labels, 5 classes)
  - `y_val.npy` (75 labels, 5 classes)

### Software Environment
- Python 3.11+
- TensorFlow 2.20.0+ / Keras
- NumPy, Matplotlib, Seaborn, Scikit-learn
- Jupyter Notebook (for interactive notebook)

### Pretrained Weights
- ResNet50 ImageNet weights (downloaded automatically by Keras)
- ~100MB download on first run

---

## Running the Experiment

### Step-by-Step Execution

1. **Verify Week 3 Data Exists**
```bash
ls -lh ../Week3/phase1_outputs/X_*.npy
# Should show X_train.npy, X_val.npy, y_train.npy, y_val.npy
```

2. **Run Phase 4A Notebook**
```bash
cd /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio/notebooks/Week4
jupyter notebook Week_4_Exercise_Phase4A.ipynb
```

3. **Execute All Cells**
   - Section 1: Load data (6-band Sentinel-2)
   - Section 2: Load ResNet50 (ImageNet weights)
   - Section 3: Modify architecture (6 bands → 3 pseudo-RGB)
   - Section 4: Train model (~7 seconds)
   - Section 5: Evaluate (expect 13.33% accuracy)
   - Section 6: Root cause analysis (5 failure modes)
   - Section 7: Lessons learned

4. **Review Results**
   - Compare to Week 3 baseline (86.67%)
   - Examine confusion matrix (single-class predictions)
   - Read root cause analysis
   - Consider recommendations

---

## Expected Outputs

### Console Output
```
Model Architecture Summary:
  Total layers: 181 (175 ResNet50 + 6 custom)
  Frozen layers: 150 (85.7%)
  Trainable layers: 25 (14.3%)
  Total parameters: 23,587,717
  Trainable parameters: 3,417,605 (14.5%)

Training ResNet50 Transfer Model...
Epoch 10/50: loss=2.3123, accuracy=0.1267, val_loss=2.3079, val_accuracy=0.1333
Early stopping at epoch 10 (no improvement for 10 epochs)

Phase 4A Transfer Model - Validation Results:
  Accuracy: 13.33%

Comparison to Week 3:
  Week 3 SimpleCNN:        86.67%
  Phase 4A ResNet50:       13.33%  ← Current
  Change:                  -73.33 percentage points
```

### Visualizations
- Training history plot showing flat validation accuracy
- Confusion matrix showing single-class predictions
- Comparison bar chart (Week 3 vs Phase 4A)

---

## Troubleshooting

### Issue 1: Week 3 Data Not Found
**Error:** `FileNotFoundError: X_train.npy not found`
**Solution:** Run Week 3 pipeline first to generate training data

### Issue 2: ResNet50 Weights Download Fails
**Error:** `Exception downloading imagenet weights`
**Solution:** Check internet connection, retry, or download manually from Keras

### Issue 3: Memory Error During Training
**Error:** `ResourceExhaustedError: OOM when allocating tensor`
**Solution:** Reduce batch size from 32 to 16 or 8

### Issue 4: Training Takes Too Long
**Expected:** ~7 seconds (frozen layers train fast)
**If Longer:** Check if layers are actually frozen (`layer.trainable = False`)

---

## Next Steps

After completing Phase 4A:

1. **Phase 4B:** Multi-sensor fusion with ResNet50
   - Test if adding MODIS + DEM helps transfer learning
   - Result: 0.00% (even worse than Phase 4A)

2. **Phase 4C:** SimpleCNN with multi-sensor fusion
   - Apply Week 3's proven architecture to 8-band data
   - Result: 69% (best of fusion experiments, but below baseline)

3. **Week 4 Summary:**
   - Review all experiments (4A, 4B, 4C)
   - Compare approaches and lessons learned
   - Generate comprehensive report

---

## References

### Related Materials
- `Week_4_Study_Guide.md` - Concepts and theoretical background
- `Week_4_Exercise.md` - Exercise instructions and learning objectives
- `Week4_Phase4A_Summary.md` - Comprehensive results analysis

### Key Concepts
- Transfer learning
- Domain adaptation
- Fine-tuning vs feature extraction
- Model collapse
- Layer freezing strategies

---

## Contact

For questions or issues with this experiment, refer to:
- Study guide: Sections on transfer learning and failure modes
- Exercise document: Troubleshooting section
- Phase 4A Summary: Root cause analysis

---

**Generated:** 2025-10-26
**Status:** Phase 4A Complete (Documented Failure)
**Conclusion:** Transfer learning with aggressive freezing fails for satellite imagery classification. SimpleCNN trained from scratch (Week 3: 86.67%) vastly outperforms ResNet50 transfer learning (Phase 4A: 13.33%).
