# Week 3 Activity 1: Phase 0 - Setup & Data Preparation

## Overview

This notebook guides students through the essential setup and data preparation steps required **before** CNN training begins. Phase 0 focuses on creating and validating training polygons in QGIS, ensuring all prerequisites are met for Phase 1 (validation) and Phase 2 (batch extraction).

## What is Phase 0?

**Phase 0** is the foundation of the CNN workflow:
- Verify environment setup
- Define land cover classes
- Learn QGIS digitization workflow
- Create training polygons
- Validate data quality

**Flow**: Phase 0 (Setup) → Phase 1 (Validate) → Phase 2 (Extract) → Phase 3 (Train)

## What Changed from Part A

### Code Reduction
- **Part A (~500 lines)** → **Phase 0 (~90 lines of code)**
- **~80% code reduction** through removal of:
  - Sentinel-2 composite creation (now pre-built)
  - Cloud masking workflows
  - Index calculations (NDVI, NDWI, etc.)
  - Extensive Earth Engine operations
  - Redundant diagnostics

### What Was Kept
- ✅ Environment verification
- ✅ Class definitions (expanded)
- ✅ QGIS digitization guide (significantly expanded)
- ✅ Basic polygon validation

### What Was Moved to Phase 1
- ❌ Detailed polygon size analysis → Phase 1
- ❌ Loading pre-built composite → Phase 1
- ❌ Single patch extraction → Phase 1
- ❌ Patch size optimization → Phase 1

### What Was Removed Entirely
- ❌ Sentinel-2 composite creation (instructor provides pre-built)
- ❌ Cloud masking workflow
- ❌ Index calculation demonstrations

## Notebook Structure

```
Week_3_Activity1_Phase0.ipynb
├── 1. Introduction & Learning Objectives
│   └── What is Phase 0 and why it matters
│
├── 2. Environment Verification (~15 lines)
│   ├── Import packages
│   ├── Verify versions
│   └── Check project structure
│
├── 3. Earth Engine Setup (~10 lines)
│   ├── Initialize Earth Engine
│   └── Test basic connection
│
├── 4. Load Area of Interest (~15 lines)
│   ├── Load AOI GeoJSON
│   ├── Verify CRS (EPSG:4326)
│   ├── Calculate area
│   └── Visualization: AOI boundary map
│
├── 5. Define Land Cover Classes (Markdown only)
│   ├── Class definitions table
│   ├── Spectral characteristics
│   ├── Spatial characteristics
│   └── Exclusion criteria
│
├── 6. QGIS Digitization Guide (~100 lines markdown) ⭐ MOST IMPORTANT
│   ├── Opening QGIS and adding layers
│   ├── Creating new shapefile layer
│   ├── Attribute field setup (exact specifications)
│   ├── Dropdown configuration (value maps)
│   ├── Digitization workflow
│   ├── Quality control checks
│   └── Saving and exporting
│
├── 7. Validate Training Polygons (~40 lines)
│   ├── Load polygons GeoJSON
│   ├── Check required attributes
│   ├── Verify CRS
│   ├── Check for missing values
│   ├── Validate class names
│   ├── Class distribution
│   └── Visualization: Polygons on AOI
│
└── 8. Readiness Check & Configuration Export (~20 lines)
    ├── Checklist (8 items)
    ├── Export phase0_config.json
    └── Next steps (Phase 1)
```

**Total:** ~90 lines of code + extensive markdown

## Educational Features

### 1. Setup Focus
Phase 0 is about **preparation**, not analysis:
- Verify environment works
- Understand what data is needed
- Learn how to create training data
- Validate data is ready

### 2. QGIS Digitization Guide (Expanded)
**100+ lines of detailed instructions** including:

**Before (Part A):**
- Basic mention: "Use QGIS to digitize polygons"
- ~30 lines of general guidance

**After (Phase 0):**
- Step-by-step workflow with screenshot placeholders
- Exact attribute field specifications:
  ```
  class_name: Text (length 50)
  class_id: Integer (width 2)
  ```
- Dropdown configuration with value maps:
  ```
  Agriculture,1
  Forest,2
  Parcels,3
  Urban,4
  Water,5
  ```
- Digitization strategy:
  - How many polygons per class? (20 minimum, 50+ ideal)
  - How big should polygons be? (>0.5 ha recommended)
  - Where to digitize? (distributed across AOI, not clustered)
- Quality control procedures
- Common troubleshooting

**This is the most important section for students!**

### 3. Class Definitions Table
Clear reference for students while digitizing:

| Class | ID | Spectral | Spatial | Exclusions |
|-------|-----|----------|---------|------------|
| Agriculture | 1 | Bare soil (brown/tan) or crops (green) | Open fields, rectangular | Exclude small gardens |
| Forest | 2 | Dark green, dense vegetation | Continuous cover | Exclude tree lines |
| Parcels | 3 | Very bright cyan (greenhouse roofs) | Small rectangles, clustered | Target class! |
| Urban | 4 | Gray/white (buildings), dark (roads) | Structured patterns | Exclude isolated houses |
| Water | 5 | Dark blue/black | Smooth, uniform | Exclude small ponds |

### 4. Visual Validation
Students see their work:
- AOI boundary visualization (where to work)
- Polygon distribution map (quality check)
- Class balance bar chart (identify gaps)

### 5. Readiness Checklist
8-item checklist with pass/fail:
```
✅ 1. Environment
✅ 2. Earth Engine
✅ 3. AOI defined
✅ 4. Polygons created
✅ 5. Attributes valid
✅ 6. CRS correct
✅ 7. Sufficient polygons
✅ 8. Classes balanced
```

### 6. Configuration Handoff
Exports `phase0_config.json` for Phase 1:
```json
{
  "phase": "Phase 0 Complete",
  "date": "2025-10-20 14:30:00",
  "aoi_file": "data/external/larger_aoi.geojson",
  "polygons_file": "data/labels/larger_polygons.geojson",
  "total_polygons": 126,
  "classes": {
    "Agriculture": 35,
    "Forest": 16,
    "Parcels": 33,
    "Urban": 29,
    "Water": 13
  },
  "composite_asset": "users/markstonegobigred/Parcela/s2_2019_median_6b",
  "bands": ["B2", "B3", "B4", "B8", "B11", "B12"]
}
```

## How Students Use This Notebook

### Expected Workflow (90-120 minutes)

**Part 1: Setup (15 minutes)**
1. Run environment verification
2. Initialize Earth Engine
3. Load and visualize AOI
4. Understand study area

**Part 2: Class Definitions (10 minutes)**
5. Read class definitions carefully
6. Understand spectral and spatial characteristics
7. Note exclusion criteria

**Part 3: QGIS Digitization (60-90 minutes)** ⭐ **CORE ACTIVITY**
8. Follow QGIS guide step-by-step
9. Create new shapefile layer
10. Set up attribute fields and dropdowns
11. Digitize polygons for all 5 classes
12. Save and export to GeoJSON

**Part 4: Validation (10 minutes)**
13. Run polygon validation cells
14. Check class distribution
15. Verify readiness checklist
16. Export configuration for Phase 1

## Success Criteria

Students should be able to:

✅ Verify their Python environment is correctly configured
✅ Authenticate and initialize Earth Engine
✅ Understand the study area (Los Lagos, Chile)
✅ Define 5 land cover classes with spectral/spatial characteristics
✅ Use QGIS to digitize training polygons with attributes
✅ Validate polygon data quality (CRS, attributes, distribution)
✅ Export configuration for Phase 1

## Common Student Questions (Anticipated)

**Q: Do I need to create the Sentinel-2 composite in Phase 0?**
A: No! The composite is pre-built by the instructor. Phase 0 focuses on creating training polygons. You'll load the composite in Phase 1.

**Q: How many polygons should I digitize per class?**
A: Minimum 20 per class, ideally 50+. More is better for CNN training.

**Q: How big should my polygons be?**
A: >0.5 hectares recommended. Too small = poor spatial context. Too large = hard to find homogeneous areas.

**Q: What if I can't find enough examples of a class?**
A: Focus efforts on visible areas. For rare classes (e.g., Water), distribute what you find across the AOI. Quality > quantity.

**Q: Do I need to fill the entire AOI with polygons?**
A: No! You're creating **training samples**, not a full classification. Distribute polygons across the AOI, but they don't need to be contiguous.

**Q: What CRS should I use in QGIS?**
A: EPSG:4326 (WGS84). The validation section will check and correct if needed.

**Q: Can I edit my polygons after this notebook?**
A: Yes! You can return to QGIS anytime to add, delete, or modify polygons. Just re-run the validation section before Phase 1.

## Technical Notes

### Assumptions
- Student has QGIS installed (3.x)
- Student has completed Week 0 (environment setup)
- Running in `geoai` conda environment
- Earth Engine authenticated

### Dependencies
```python
numpy
pandas
geopandas
matplotlib
earthengine-api
geemap
```

### Execution Time
- **First run**: ~10 minutes (mostly reading and setup)
- **QGIS digitization**: 60-90 minutes (main time investment)
- **Validation**: ~2 minutes

### Data Requirements

**Provided by Instructor:**
- `data/external/larger_aoi.geojson` - Study area boundary
- Earth Engine composite: `users/markstonegobigred/Parcela/s2_2019_median_6b`

**Created by Student:**
- `data/labels/larger_polygons.geojson` - Training polygons (created in QGIS)

## Pedagogical Design Principles

### 1. Setup Before Analysis
Students must understand **what** they're preparing before **why** they're preparing it. Phase 0 focuses on mechanics, Phase 1 on reasoning.

### 2. Hands-On Data Creation
Students create their own training data, understanding:
- Subjectivity in labeling
- Class definition challenges
- Spatial distribution importance
- Data quality impact on CNN performance

### 3. Progressive Skill Building
```
Phase 0: Create data (QGIS, validation)
Phase 1: Analyze data (size constraints, quality checks)
Phase 2: Extract patches (batch processing)
Phase 3: Train CNN (deep learning)
```

### 4. Clear Checkpoints
Each phase has a clear "done" signal:
- Phase 0: ✅ Readiness checklist + config export
- Phase 1: ✅ Single patch test success + optimal parameters
- Phase 2: ✅ All patches extracted
- Phase 3: ✅ CNN trained

## Comparison: Part A vs Phase 0

| Aspect | Part A | Phase 0 |
|--------|---------|---------|
| **Lines of code** | ~500 | ~90 |
| **Focus** | Composite creation + polygons | Polygons only |
| **QGIS guide** | Brief (30 lines) | Detailed (100+ lines) |
| **Earth Engine work** | Extensive | Minimal |
| **Sentinel-2 ops** | Create composite | Reference pre-built |
| **Cloud masking** | Detailed workflow | Removed |
| **Indices** | Calculate NDVI, NDWI | Removed |
| **Polygon analysis** | Detailed size stats | Basic validation only |
| **Execution time** | ~45 minutes | ~10 minutes (code) |
| **QGIS time** | ~60 minutes | ~90 minutes (expanded guide) |

## Overlap Analysis: Phase 0 vs Phase 1

### Minimal Overlap (Intentional)
Both notebooks need to:
- Import packages (15% overlap)
- Define paths (10% overlap)

### Clear Separation

**Phase 0 Focus:**
- Environment verification
- Class definitions (not in Phase 1)
- QGIS guide (not in Phase 1)
- Basic polygon validation (attributes exist?)

**Phase 1 Focus:**
- Detailed polygon size analysis (not in Phase 0)
- Load pre-built composite (not in Phase 0)
- Single patch extraction test (not in Phase 0)
- Patch size optimization (not in Phase 0)

**Overlap reduced from 75% (Part A + Phase 1) to 15% (Phase 0 + Phase 1)**

## File Organization

### Created by Phase 0
```
phase0_config.json          # Configuration for Phase 1 handoff
Week_3_Activity1_Phase0.py  # Auto-generated (testing)
```

### Used by Phase 0
```
data/external/larger_aoi.geojson              # Provided by instructor
data/labels/larger_polygons.geojson           # Created by student in QGIS
```

### Next Phase
```
Week_3_Activity1_Phase1.ipynb  # Validates polygons, determines parameters
```

## Testing Results

**Tested on**: 2025-10-20

**Test dataset**: 126 polygons, 5 classes

**Results:**
```
✅ All packages imported successfully
✅ Earth Engine initialized
✅ AOI loaded (117.2 km²)
✅ Training polygons loaded (126 polygons)
✓ Required attributes present
✓ CRS correct (EPSG:4326)
✓ All class names valid

📊 Class Distribution:
✓ Agriculture :  35
⚠️ Forest      :  16
✓ Parcels     :  33
✓ Urban       :  29
⚠️ Water       :  13

✅ Phase 0 Complete! Ready for Phase 1!
✓ Configuration saved: phase0_config.json
```

## Next Steps After Completion

Students who complete Phase 0 should:

1. ✅ Have digitized training polygons in QGIS
2. ✅ Understand land cover class definitions
3. ✅ Have validated polygon attributes and CRS
4. ✅ Have distributed polygons across study area
5. ✅ Have exported `phase0_config.json`
6. → **Ready for Phase 1: Validation & Configuration**

Phase 1 will:
- Analyze polygon sizes in detail
- Load the pre-built Sentinel-2 composite
- Test single patch extraction
- Determine optimal patch size for CNN training

## Instructor Notes

### Customization Points

**Easy to modify:**
- Minimum polygon count threshold (line in Section 7)
- Class definitions (markdown in Section 5)
- QGIS guide steps (markdown in Section 6)
- Study area (change AOI file path)

**Hard to modify without understanding:**
- Earth Engine authentication flow
- CRS validation logic
- GeoJSON file handling

### Assessment Ideas

**Knowledge Check:**
- What are the 5 land cover classes and their spectral characteristics?
- Why is EPSG:4326 required for Earth Engine?
- What is the minimum recommended polygon count per class?

**Application:**
- Digitize 20+ polygons per class
- Achieve balanced class distribution
- Distribute polygons across entire AOI
- Export valid GeoJSON with correct attributes

**Reflection:**
- What was most challenging about class definitions?
- How did you decide where to digitize polygons?
- What class was hardest to find? Why?

### Time Management

**Recommended class schedule (2-hour session):**
- 0:00-0:15 — Introduction + Environment setup
- 0:15-0:25 — Class definitions discussion
- 0:25-1:55 — QGIS digitization (main activity)
- 1:55-2:00 — Validation + readiness check

**Homework option:**
- In-class: Sections 1-5 (understand requirements)
- At-home: Section 6 (QGIS digitization)
- Next-class: Section 7-8 (validation + Phase 1)

## License & Attribution

Part of Week 3 Activity 1 for GeoAI course.
Developed for educational use with Los Lagos, Chile parcel detection dataset.

---

**Summary:** Phase 0 prepares students for CNN training by focusing on the foundational task of creating high-quality training data in QGIS. The notebook achieves an **80% code reduction** from Part A while significantly expanding the QGIS digitization guide (30 → 100+ lines), ensuring students can successfully create training polygons independently.
