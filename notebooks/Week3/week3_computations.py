"""
Week 3 Complete Pipeline: Generate All Outputs for Week 4

This script consolidates Week 3 Phases 0-3 into a single executable pipeline.
Run this to generate the dataset and baseline model needed for Week 4.

Outputs:
- data/processed/X_train.npy, y_train.npy (300 training patches)
- data/processed/X_val.npy, y_val.npy (75 validation patches)
- models/week3/best_model.h5 (SimpleCNN baseline model)

Runtime: 30-60 minutes (mostly Earth Engine extraction)

Usage:
    python week3_computations.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import sys

# Earth Engine
try:
    import ee
    import geemap
except ImportError:
    print("❌ Earth Engine not installed. Run: pip install earthengine-api geemap")
    sys.exit(1)

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    print("❌ TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

# Sklearn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("❌ Scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  tqdm not installed (optional). Install for progress bars: pip install tqdm")
    tqdm = lambda x, **kwargs: x

# Set seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("="*70)
print("WEEK 3 COMPLETE PIPELINE")
print("="*70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Purpose: Generate dataset and baseline for Week 4")
print(f"Estimated runtime: 30-60 minutes")
print("="*70)
print()

# ============================================================================
# SECTION 1: Setup Paths
# ============================================================================

print("[1/8] Setting up paths...")

REPO = Path.cwd().parent
DATA_DIR = REPO / 'data'
LABELS_DIR = DATA_DIR / 'labels'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = REPO / 'models' / 'week3'

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

POLYGONS_FILE = LABELS_DIR / 'larger_polygons.geojson'

if not POLYGONS_FILE.exists():
    print(f"❌ Training polygons not found: {POLYGONS_FILE}")
    print(f"   Complete Week 3 Phase 0 (QGIS digitization) first")
    sys.exit(1)

print(f"  ✓ Paths configured")
print(f"    Polygons: {POLYGONS_FILE}")
print(f"    Output: {PROCESSED_DIR}")
print()

# ============================================================================
# SECTION 2: Initialize Earth Engine
# ============================================================================

print("[2/8] Initializing Earth Engine...")

try:
    ee.Initialize()
    print(f"  ✓ Earth Engine initialized")
except Exception as e:
    print(f"  ❌ Error: {e}")
    print(f"     Run: earthengine authenticate")
    sys.exit(1)
print()

# ============================================================================
# SECTION 3: Load Training Polygons
# ============================================================================

print("[3/8] Loading training polygons...")

polygons = gpd.read_file(POLYGONS_FILE)
if polygons.crs.to_string() != 'EPSG:4326':
    polygons = polygons.to_crs('EPSG:4326')

print(f"  ✓ Loaded {len(polygons)} polygons")
print(f"    Classes: {sorted(polygons['class_name'].unique())}")

# Class distribution
class_counts = polygons['class_name'].value_counts().sort_index()
print(f"\n  Class distribution:")
for cls, count in class_counts.items():
    print(f"    {cls:12s}: {count:3d} polygons")
print()

# ============================================================================
# SECTION 4: Create Sentinel-2 Composite
# ============================================================================

print("[4/8] Creating Sentinel-2 composite...")
print(f"  Time period: 2019-01-01 to 2019-03-31 (austral summer)")
print(f"  This may take 1-2 minutes...")

bounds = polygons.total_bounds
aoi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])

START_DATE = '2019-01-01'
END_DATE = '2019-03-31'

s2_composite = (ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(aoi)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    .median()
    .clip(aoi))

bands = s2_composite.bandNames().getInfo()
print(f"  ✓ Composite created: {bands}")
print()

# ============================================================================
# SECTION 5: Extract Patches with Spatial Jitter
# ============================================================================

print("[5/8] Extracting patches from Earth Engine...")
print(f"  Total patches: {len(polygons) * 3} (3 per polygon with jitter)")
print(f"  Patch size: 8×8 pixels (80m × 80m)")
print(f"  ⚠️  This is the SLOWEST step (20-40 minutes)")
print()

PATCH_SIZE = 8
PATCHES_PER_POLYGON = 3

all_patches = []
all_labels = []
extraction_log = []

for idx, poly in tqdm(polygons.iterrows(), total=len(polygons), desc="Extracting"):
    # Skip polygons with invalid/null geometry
    if poly.geometry is None or poly.geometry.is_empty:
        print(f"\n  ⚠️  Skipping polygon {idx}: invalid geometry")
        continue

    centroid = poly.geometry.centroid
    lat, lon = centroid.y, centroid.x

    # Meters per degree at this latitude
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(lat))

    for patch_idx in range(PATCHES_PER_POLYGON):
        # Spatial jitter
        if patch_idx == 0:
            offset_m = (0, 0)  # Center
        else:
            offset_m = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))

        # Calculate patch center
        offset_lon = offset_m[0] / meters_per_deg_lon
        offset_lat = offset_m[1] / meters_per_deg_lat
        patch_lon = lon + offset_lon
        patch_lat = lat + offset_lat

        # Define patch geometry
        patch_half_m = (PATCH_SIZE * 10) / 2
        half_deg_lon = patch_half_m / meters_per_deg_lon
        half_deg_lat = patch_half_m / meters_per_deg_lat

        patch_geom = ee.Geometry.Rectangle([
            patch_lon - half_deg_lon,
            patch_lat - half_deg_lat,
            patch_lon + half_deg_lon,
            patch_lat + half_deg_lat
        ])

        # Extract patch
        try:
            patch = geemap.ee_to_numpy(
                s2_composite,
                region=patch_geom,
                scale=10,
                bands=bands
            )

            # Handle size mismatch
            if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                h, w, c = patch.shape
                resized = np.full((PATCH_SIZE, PATCH_SIZE, len(bands)), np.nan)
                h_copy = min(h, PATCH_SIZE)
                w_copy = min(w, PATCH_SIZE)
                resized[:h_copy, :w_copy, :] = patch[:h_copy, :w_copy, :]
                patch = resized

            # Quality check
            nan_pct = np.isnan(patch).sum() / patch.size * 100

            if nan_pct < 20:  # Accept if < 20% NaN
                all_patches.append(patch.astype(np.float32))
                all_labels.append(poly['class_id'])
                extraction_log.append({
                    'polygon_id': idx,
                    'class_name': poly['class_name'],
                    'patch_idx': patch_idx,
                    'nan_pct': nan_pct,
                    'success': True
                })
            else:
                extraction_log.append({
                    'polygon_id': idx,
                    'class_name': poly['class_name'],
                    'patch_idx': patch_idx,
                    'nan_pct': nan_pct,
                    'success': False
                })

        except Exception as e:
            extraction_log.append({
                'polygon_id': idx,
                'class_name': poly['class_name'],
                'patch_idx': patch_idx,
                'error': str(e),
                'success': False
            })

print(f"\n  ✓ Extraction complete!")
print(f"    Successful: {len(all_patches)} patches")
print(f"    Failed: {len(extraction_log) - len(all_patches)}")
print()

# ============================================================================
# SECTION 6: Create Train/Val Split
# ============================================================================

print("[6/8] Creating stratified train/validation split...")

X = np.array(all_patches)
y = np.array(all_labels)

# Stratified split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_SEED
)

print(f"  ✓ Split complete:")
print(f"    Training: {len(X_train)} patches")
print(f"    Validation: {len(X_val)} patches")
print()

# Verify stratification
print(f"  Class distribution:")
print(f"    {'Class':<12s} {'Train':>8s} {'Val':>8s}")
print(f"    {'-'*30}")
for cls_id in sorted(np.unique(y)):
    train_count = (y_train == cls_id).sum()
    val_count = (y_val == cls_id).sum()
    print(f"    Class {cls_id:<6d} {train_count:8d} {val_count:8d}")
print()

# ============================================================================
# SECTION 7: Save Dataset
# ============================================================================

print("[7/8] Saving dataset...")

np.save(PROCESSED_DIR / 'X_train.npy', X_train)
np.save(PROCESSED_DIR / 'y_train.npy', y_train)
np.save(PROCESSED_DIR / 'X_val.npy', X_val)
np.save(PROCESSED_DIR / 'y_val.npy', y_val)

print(f"  ✓ Dataset saved:")
print(f"    {PROCESSED_DIR / 'X_train.npy'} ({X_train.shape})")
print(f"    {PROCESSED_DIR / 'y_train.npy'} ({y_train.shape})")
print(f"    {PROCESSED_DIR / 'X_val.npy'} ({X_val.shape})")
print(f"    {PROCESSED_DIR / 'y_val.npy'} ({y_val.shape})")
print()

# ============================================================================
# SECTION 8: Train SimpleCNN Baseline
# ============================================================================

print("[8/8] Training SimpleCNN baseline model...")
print(f"  This may take 10-20 minutes...")
print()

# Normalize data
X_train_norm = X_train.astype(np.float32) / 10000.0
X_val_norm = X_val.astype(np.float32) / 10000.0

# Build SimpleCNN (from Week 3)
model = models.Sequential([
    layers.Input(shape=(8, 8, 6)),

    # Conv block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Conv block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
], name='SimpleCNN')

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Model: {model.count_params():,} parameters")
print()

# Train
checkpoint = callbacks.ModelCheckpoint(
    str(MODELS_DIR / 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop],
    verbose=2  # Less verbose output
)

best_val_acc = max(history.history['val_accuracy'])

print(f"\n  ✓ Training complete!")
print(f"    Best validation accuracy: {best_val_acc*100:.2f}%")
print(f"    Model saved: {MODELS_DIR / 'best_model.h5'}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*70)
print("✅ WEEK 3 PIPELINE COMPLETE!")
print("="*70)
print()
print("Outputs created:")
print(f"  1. Dataset ({len(X_train) + len(X_val)} patches total):")
print(f"     - {PROCESSED_DIR / 'X_train.npy'}")
print(f"     - {PROCESSED_DIR / 'y_train.npy'}")
print(f"     - {PROCESSED_DIR / 'X_val.npy'}")
print(f"     - {PROCESSED_DIR / 'y_val.npy'}")
print()
print(f"  2. Baseline Model:")
print(f"     - {MODELS_DIR / 'best_model.h5'}")
print(f"     - Validation accuracy: {best_val_acc*100:.2f}%")
print()
print("Next steps:")
print("  - Week 4 Phase 4A can now use this real data")
print("  - Run: cd notebooks/Week4 && ./run_phase4a_real.sh")
print()
print("="*70)
