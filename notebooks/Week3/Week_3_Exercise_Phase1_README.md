# Week 3 Activity 1: Phase 1 - Educational Notebook

## Overview

This notebook consolidates the 5 Phase 1 scripts into a single, streamlined educational resource designed for students learning CNN-based land cover classification.

## What Changed from Scripts

### Code Reduction
- **5 scripts (1,826 lines)** → **1 notebook (~135 lines of code)**
- **~93% code reduction** through removal of:
  - Verbose error handling
  - Extensive troubleshooting sections
  - Multiple fallback strategies
  - File I/O for configuration
  - Progress bars and detailed logging

### What Was Kept
- ✅ Core working logic
- ✅ Essential validation checks
- ✅ Key visualizations
- ✅ Educational explanations

## Notebook Structure

```
Week_3_Activity1_Phase1.ipynb
├── 1. Introduction & Learning Objectives
│   └── What is Phase 1 and why it matters
│
├── 2. Setup & Configuration (~25 lines)
│   ├── Imports + Earth Engine initialization
│   └── Define paths and constants
│
├── 3. Load & Analyze Training Polygons (~25 lines)
│   ├── Load polygons + calculate areas
│   ├── Compute per-class statistics
│   └── Visualization: Box plot by class
│
├── 4. Load Sentinel-2 Composite (~10 lines)
│   ├── Load from Earth Engine asset
│   ├── Verify bands
│   └── Band information table
│
├── 5. Test Single Patch Extraction (~40 lines)
│   ├── Select test polygon (smallest class)
│   ├── Calculate patch bounds
│   ├── Extract from Earth Engine
│   ├── Quality check (NaN%, value range)
│   └── Visualization: RGB + NIR side-by-side
│
├── 6. Determine Optimal Patch Size (~35 lines)
│   ├── Calculate recommendations per class
│   ├── Apply constraints (CNN minimum)
│   ├── Determine patches per polygon
│   └── Visualization: Polygon sizes vs chosen patch
│
└── 7. Summary & Final Configuration
    ├── Key takeaways
    ├── Configuration table
    └── Next steps (Phase 2)
```

**Total:** ~135 lines of code + rich markdown explanations

## Educational Features

### 1. Learning Objectives
Each section starts with clear learning objectives explaining what the student will understand.

### 2. Conceptual Explanations
- **Why** not just **how**
- Trade-offs explained (e.g., larger vs smaller patches)
- Real-world context (e.g., why composites reduce clouds)

### 3. Inline Visualizations
- Box plots showing polygon size distribution
- RGB + NIR patch visualizations
- Decision charts (polygon sizes vs chosen patch)

### 4. Interpretation Blocks
After each major output, explains:
- What the results mean
- What to look for
- How to identify problems

### 5. Quality Checks
Built-in validation with clear feedback:
```python
if nan_pct == 0:
    print("✓ Perfect! No missing data.")
elif nan_pct < 20:
    print("✓ Acceptable NaN percentage (<20%).")
else:
    print("⚠️  High NaN percentage!")
```

### 6. Progressive Disclosure
Information revealed step-by-step:
1. Load data → See it
2. Analyze it → Understand it
3. Apply it → Use it
4. Validate it → Verify it

## Key Differences from Original Scripts

### Before (Script 4, Lines 91-126):
```python
try:
    # Load GeoDataFrame
    training_polys = gpd.read_file(TRAINING_POLYGONS_PATH)
    print(f"  ✓ Loaded {len(training_polys)} training polygons")
    print(f"  ✓ Source: {TRAINING_POLYGONS_PATH}")

except Exception as e:
    print(f"  ❌ ERROR loading file: {e}")
    print("\n  Possible causes:")
    print("     - File is corrupted")
    print("     - Not a valid GeoJSON")
    print("     - Missing geometry column")
    sys.exit(1)
```

### After (Notebook):
```python
# Load polygons and ensure WGS84 (required for Earth Engine)
polygons = gpd.read_file(POLYGONS_FILE)
if polygons.crs.to_string() != 'EPSG:4326':
    polygons = polygons.to_crs('EPSG:4326')

print(f"Loaded {len(polygons)} training polygons")
```

**Rationale:** Students working through a notebook can see errors directly. No need for extensive error messages - Python's traceback is sufficient for learning.

## How Students Use This Notebook

### Expected Workflow (30-45 minutes)

1. **Read Introduction (5 min)**
   - Understand Phase 1 objectives
   - Learn why testing first is important

2. **Setup (5 min)**
   - Run setup cells
   - Verify Earth Engine connection

3. **Polygon Analysis (10 min)**
   - Load polygons
   - Understand size distribution
   - Identify smallest class constraint

4. **Composite Loading (5 min)**
   - Load Sentinel-2 data
   - Learn about band selection

5. **Single Patch Test (10 min)** ⭐ **CRITICAL**
   - Extract one test patch
   - Verify quality
   - Visualize RGB + NIR

6. **Patch Size Decision (10 min)**
   - Understand trade-offs
   - See automated decision process
   - Validate with visualization

7. **Review Configuration (5 min)**
   - Understand final parameters
   - Prepare for Phase 2

## Success Criteria

Students should be able to:

✅ Explain why Phase 1 testing is important
✅ Understand how polygon size constrains patch size
✅ Interpret RGB and NIR visualizations
✅ Identify quality issues (NaN%, unusual values)
✅ Explain the patch size trade-off
✅ Read and understand the final configuration

## Common Student Questions (Anticipated)

**Q: Why use 8×8 patches instead of larger?**
A: See Section 5 - smallest class (Urban) constrains maximum size. 8px balances CNN needs with polygon constraints.

**Q: What if my test patch looks wrong?**
A: Quality check in Section 4 explains what to look for. NaN% and value range help diagnose issues.

**Q: Can I use different patch sizes?**
A: Yes! Section 5 shows the calculation. Can modify `MIN_CNN_SIZE` or constraints.

**Q: Why median composite instead of single image?**
A: Section 3 explains - reduces clouds, provides consistency.

## Technical Notes

### Assumptions
- Student has completed Part A (training polygons digitized)
- Earth Engine authenticated
- Running in `geoai` conda environment

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
- First run: ~5-10 minutes (Earth Engine extraction)
- Subsequent runs: ~2-3 minutes (if EE cache hit)

### Data Requirements
- `data/labels/larger_polygons.geojson` (126 polygons)
- `data/external/larger_aoi.geojson` (study area bounds)
- Earth Engine asset: `users/markstonegobigred/Parcela/s2_2019_median_6b`

## Pedagogical Design Principles

### 1. Minimal Code, Maximum Learning
Each line of code serves a clear purpose. No "dead code" or unused variables.

### 2. Visual Learning
Every major concept has a visualization:
- Polygon sizes → Box plot
- Test patch → RGB + NIR
- Patch decision → Bar chart

### 3. Immediate Feedback
Quality checks print clear success/warning messages at each step.

### 4. Progressive Complexity
Starts simple (load data) → builds to complex (determine optimal parameters).

### 5. Real-World Context
Uses actual research data (Los Lagos, Chile parcels) not toy datasets.

## Comparison to Original Scripts

| Aspect | Scripts | Notebook |
|--------|---------|----------|
| **Lines of code** | 1,826 | ~135 |
| **Files** | 5 separate | 1 integrated |
| **Error handling** | Extensive | Minimal |
| **Troubleshooting** | Detailed guides | Learn from errors |
| **Documentation** | Inline comments | Rich markdown |
| **Visualizations** | Saved to files | Inline display |
| **Configuration** | JSON files | In-memory |
| **Execution** | Sequential scripts | Interactive cells |
| **Learning curve** | Steep | Gradual |

## Output Comparison

### Scripts Create:
```
phase1_outputs/
├── environment_info.json
├── polygon_stats.json
├── polygon_size_dist.png
├── composite_info.json
├── test_patch.npy
├── test_patch_rgb.png
├── test_extraction_report.json
├── patch_size_analysis.png
└── phase1_config.json
```

### Notebook Creates:
- All visualizations inline (no files)
- Final config in memory (can export if needed)
- Cleaner, more focused learning experience

## Next Steps After Completion

Students who complete this notebook should:

1. ✅ Understand Phase 1 validation approach
2. ✅ Be able to explain their configuration choices
3. ✅ Have verified patch extraction works
4. → **Ready for Phase 2: Batch Extraction**

Phase 2 will use the configuration determined here to extract all patches for CNN training.

## Instructor Notes

### Customization Points

**Easy to modify:**
- Patch size constraints (line in Section 5)
- Number of patches per polygon (line in Section 5)
- Test polygon selection (line in Section 4)
- Visualization styles (matplotlib parameters)

**Hard to modify without understanding:**
- Earth Engine extraction logic
- CRS transformations
- Area calculations

### Assessment Ideas

**Knowledge Check:**
- What determines maximum patch size? (Polygon size)
- Why test on one patch first? (Catch errors early)
- What does high NaN% indicate? (Cloud masking/missing data)

**Application:**
- Run with different dataset
- Modify patch size and observe impacts
- Explain trade-offs to peer

## License & Attribution

Part of Week 3 Activity 1 for GeoAI course.
Developed for educational use with Los Lagos, Chile parcel detection dataset.

---

**Summary:** This notebook transforms 1,826 lines of production code into ~135 lines of educational code with rich explanations, achieving a **93% reduction** while maintaining all core functionality and adding significant pedagogical value.
