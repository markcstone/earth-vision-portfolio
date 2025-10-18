# Week 1: The Earth as an Image — Spectral Eyes and Digital Landscapes

## Custom GPT assistant
### System
You are an expert in geospatial sciences (remote sensing, GIS) with deep experience in Google Earth Engine (Python API, geemap), geospatial foundation models (PyTorch, transformers, torchgeo), and the Python geospatial stack (geopandas, rasterio, GDAL/PROJ). Provide stepwise, runnable guidance with robust paths (Pathlib), reproducibility (conda, ipykernel, python-dotenv), and concise troubleshooting. Assume the active kernel is "Python (geoai)" and the AOI is saved at `data/external/aoi.geojson`.

Context:
- Completed Week 0 environment setup:
  - Conda env: geoai (Python 3.11), kernel: "Python (geoai)"
  - Geospatial stack via conda-forge; ipykernel and python-dotenv installed
  - Earth Engine authenticated and initialized
  - AOI saved at data/external/aoi.geojson (path-robust patterns preferred)
  - Repo root: /Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio
- Updated docs: kernel registration alternatives; robust AOI save/load; AOI outline rendering; dotenv troubleshooting.

Goal:
- Focus on Week 1 (“The Earth as an Image”) using notebooks/Week_1_Study_Guide.md.
- Build practical notebooks to:
  - Load Sentinel‑2, create true/false color composites
  - Compute NDVI/NDWI/NDBI and visualize
  - Extract spectral signatures and plot
  - (Optional) NDVI time series
- Ensure reproducibility, robust paths (Pathlib), and figures saved to figures/.

How to help:
- Provide stepwise, minimal commands and code cells ready to run in "Python (geoai)".
- Use geemap + ee; assume AOI exists at data/external/aoi.geojson.
- Prefer absolute paths for Terminal commands; keep code path-robust for notebooks.
- Add quick validation/diagnostics per step and brief troubleshooting if errors occur.
- Keep responses concise; include only necessary code blocks.

First tasks:
1) Draft a Week1_Lab.ipynb outline with executable cells for:
   - EE init, AOI load, Sentinel‑2 filter, composites
   - NDVI/NDWI/NDBI layers
   - Spectral profile extraction + plot
   - Save figures to figures/
2) Provide verification prints and expected outputs for each section.

---

## Learning Objectives

By the end of Week 1, students will be able to:

1. **Explain fundamental remote sensing principles** relevant to environmental monitoring and GeoAI applications
2. **Describe how satellite sensors transform physical processes** into multi-band digital imagery
3. **Calculate and interpret spectral indices** (NDVI, NDWI, NDBI) for land-cover characterization
4. **Navigate the Google Earth Engine Python API** and perform basic image operations
5. **Visualize and communicate** satellite imagery insights through annotated composites and reflective writing

---

## Opening Discussion and Review (Self-Reflection)

### Review of Week 0 Key Concepts

Before diving into this week's content, take a moment to reflect on your Week 0 setup experience:

- Did you successfully authenticate with Google Earth Engine and run a test script?
- What challenges did you encounter during environment setup, and how did you resolve them?
- How comfortable are you with the basic structure of your GitHub repository?
- What questions do you still have about the course structure or technical environment?

### Today's Learning Journey

This week marks the true beginning of your GeoAI journey. We transition from setting up tools to understanding the fundamental question: **What does it mean for a satellite to "see" the Earth?** We will explore how electromagnetic radiation interacts with Earth's surface, how satellites capture this information as multi-band imagery, and how we can extract meaningful environmental insights from these digital representations. By the end of this week, you will have created your first spectral analysis and begun to think critically about what satellite imagery reveals—and what it conceals.

---

## Core Content: Remote Sensing Fundamentals for Environmental Monitoring

### The Physics of Remote Sensing

Remote sensing for environmental monitoring relies on a fundamental principle: different surface materials interact with electromagnetic radiation in characteristic ways. Understanding these physical principles provides the foundation for interpreting satellite observations and designing effective analysis approaches.

Electromagnetic radiation from the sun travels through space and interacts with Earth's surface materials through several physical processes. When solar radiation encounters a surface, it may be **reflected**, **absorbed**, or **transmitted**, with the proportions depending on the material properties and the wavelength of the radiation. Different land cover types exhibit distinctive **spectral signatures**—patterns of reflectance across different wavelengths—that enable their identification and classification using satellite sensors.

**Vegetation**, for example, exhibits characteristic spectral properties that reflect its physiological processes and structural characteristics. Healthy vegetation strongly absorbs red light (around 660 nanometers) for photosynthesis while strongly reflecting near-infrared radiation (around 850 nanometers) due to internal leaf structure. This creates the distinctive "red edge" spectral signature that forms the basis for vegetation indices such as the Normalized Difference Vegetation Index (NDVI). Stressed or senescent vegetation shows reduced near-infrared reflectance and altered red absorption, making spectral analysis a powerful tool for monitoring vegetation health.

**Water bodies** typically exhibit low reflectance across most wavelengths, appearing dark in satellite imagery. However, water's spectral signature varies depending on water quality, depth, and suspended sediments. Clear, deep water appears very dark, while shallow or turbid water may show higher reflectance. The presence of algae, phytoplankton, or other aquatic vegetation can significantly alter water's spectral signature, particularly in the green and near-infrared portions of the spectrum. These variations make water quality monitoring possible through remote sensing.

**Built-up areas and bare soil** present more complex spectral signatures that vary considerably depending on construction materials, soil composition, and moisture content. Concrete and asphalt typically show relatively flat spectral signatures with moderate reflectance across visible and near-infrared wavelengths. Soil reflectance varies dramatically with composition, moisture content, and organic matter, making soil-based land cover types particularly challenging to classify using spectral information alone. Urban areas often show high spatial heterogeneity, with mixed pixels containing multiple materials.

The **temporal dimension** adds another layer of complexity to remote sensing observations. Seasonal changes in vegetation phenology, agricultural practices, and weather conditions create temporal variations in spectral signatures that can either aid or complicate land-use classification efforts. Understanding these temporal patterns is crucial for developing robust classification approaches that work across different seasons and years. For parcelization monitoring in the Los Lagos region, temporal analysis helps distinguish gradual land-use conversion from seasonal agricultural cycles.

### Satellite Sensors and Data Characteristics

Modern Earth observation relies on a constellation of satellite sensors that provide complementary information about Earth's surface conditions. Understanding the characteristics and capabilities of different sensor systems is essential for selecting appropriate data sources and designing effective analysis approaches.

**Optical sensors**, such as those aboard the Landsat and Sentinel-2 satellites, measure reflected solar radiation across multiple spectral bands spanning visible, near-infrared, and shortwave infrared wavelengths. These sensors provide the foundation for most land-use mapping applications due to their ability to capture the spectral signatures that distinguish different land cover types.

The **Landsat program**, jointly operated by NASA and USGS, provides the longest continuous record of satellite observations, with data extending back to 1972. The current Landsat 8 and Landsat 9 satellites carry the Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS), providing observations in 11 spectral bands with spatial resolutions ranging from 15 to 100 meters. The 16-day repeat cycle of Landsat satellites provides regular temporal coverage suitable for monitoring land-use changes. For studying long-term parcelization trends in Chile, the Landsat archive provides invaluable historical context.

The European Space Agency's **Sentinel-2 mission** consists of two satellites (Sentinel-2A and Sentinel-2B) that together provide observations every 5 days at the equator. The MultiSpectral Instrument (MSI) aboard Sentinel-2 satellites captures data in 13 spectral bands with spatial resolutions of 10, 20, or 60 meters depending on the band. The higher spatial and temporal resolution of Sentinel-2 compared to Landsat makes it particularly valuable for monitoring small-scale land-use changes such as parcelization, where individual parcels may be only 1-5 hectares in size.

**Radar sensors**, such as the Synthetic Aperture Radar (SAR) aboard Sentinel-1 satellites, provide complementary information by measuring the backscatter of microwave radiation. SAR sensors can operate day or night and penetrate clouds, providing consistent observations regardless of weather conditions. The backscatter characteristics measured by SAR sensors are sensitive to surface roughness, moisture content, and vegetation structure, providing information that complements optical observations. For monitoring hydrological changes during Chile's megadrought, SAR data can provide consistent observations even during cloudy periods.

The **spatial resolution** of satellite sensors determines the smallest features that can be reliably detected and mapped. For parcelización monitoring, the typical size of individual parcels (1-5 hectares) requires sensors with spatial resolutions of 30 meters or better to provide adequate detail. The 10-meter resolution of Sentinel-2's visible and near-infrared bands is particularly well-suited for this application, allowing individual parcels to be distinguished from surrounding agricultural or natural areas.

**Temporal resolution**—how frequently a sensor observes the same location—affects the ability to detect and track land-use changes. Parcelización often occurs gradually over periods of months or years, requiring regular observations to capture the conversion process. The combination of Landsat and Sentinel-2 observations provides a temporal resolution of approximately 2-3 days in Chile, enabling detailed monitoring of land-use change processes. For rapid events like flooding, even higher temporal resolution may be necessary.

**Spectral resolution**—the number and width of spectral bands a sensor can detect—determines the level of detail available for distinguishing different surface materials. Sensors with more spectral bands can capture more subtle differences in spectral signatures, potentially improving classification accuracy. However, higher spectral resolution often comes at the cost of reduced spatial or temporal resolution, requiring careful consideration of which sensor characteristics are most important for a given application.

### Traditional Approaches to Land-Use Classification

Conventional remote sensing approaches to land-use classification have relied primarily on spectral analysis techniques that exploit the distinctive reflectance characteristics of different land cover types. These approaches have provided the foundation for decades of land-use mapping and monitoring applications, but also reveal important limitations when applied to complex classification problems such as parcelización detection.

**Spectral indices** represent one of the most widely used approaches for extracting information about land cover from satellite observations. These indices combine reflectance values from multiple spectral bands using mathematical formulas designed to enhance specific surface characteristics. The **Normalized Difference Vegetation Index (NDVI)**, calculated as (NIR - Red) / (NIR + Red), provides a measure of vegetation greenness and health that has been used in countless applications. NDVI values typically range from -1 to +1, with higher values indicating healthier, denser vegetation.

Additional spectral indices have been developed to highlight other surface characteristics relevant to land-use mapping. The **Normalized Difference Water Index (NDWI)** helps identify water bodies and wet areas by exploiting water's strong absorption of near-infrared radiation. The **Normalized Difference Built-up Index (NDBI)** enhances the contrast between built-up areas and other land cover types. The **Modified Normalized Difference Water Index (MNDWI)** provides improved discrimination of water features in areas with built-up land cover. These indices provide simple but powerful tools for initial land-cover characterization.

**Supervised classification approaches** use training data—examples of known land cover types—to train algorithms that can classify unknown pixels based on their spectral characteristics. Traditional supervised classification algorithms include **maximum likelihood classification**, which assumes that the spectral values for each land cover class follow a normal distribution, and **support vector machines**, which find optimal boundaries between different classes in spectral feature space. These approaches have been widely used for land-use mapping but require careful collection of representative training data.

**Unsupervised classification approaches** attempt to identify natural groupings in spectral data without prior knowledge of land cover types. Algorithms such as **k-means clustering** and **ISODATA** (Iterative Self-Organizing Data Analysis Technique) group pixels with similar spectral characteristics, which can then be interpreted and labeled by analysts. These approaches can be useful for exploratory analysis but often require significant post-processing to produce meaningful land-use maps.

**Object-based image analysis (OBIA)** represents an advancement over pixel-based approaches by first segmenting images into homogeneous objects and then classifying these objects based on spectral, spatial, and contextual characteristics. OBIA can be particularly effective for high-resolution imagery where individual land-use features are composed of multiple pixels. For parcelización monitoring, OBIA approaches can help identify and delineate individual parcels based on their spatial characteristics.

**Time series analysis approaches** exploit the temporal dimension of satellite observations to improve classification accuracy and detect land-use changes. These approaches analyze how spectral characteristics change over time, using temporal patterns to distinguish between land cover types that may have similar spectral signatures at any single point in time. For agricultural areas, temporal analysis can distinguish different crop types based on their phenological patterns.

Despite their widespread use and proven effectiveness for many applications, traditional remote sensing approaches face several **limitations** when applied to parcelización detection. The spectral similarity between parcelas de agrado (small rural residential parcels) and surrounding agricultural or natural areas can make them difficult to distinguish using spectral information alone. The gradual nature of parcelización processes and the small size of individual parcels relative to satellite pixel sizes create additional challenges for traditional approaches. These limitations motivate the exploration of more advanced approaches using machine learning and foundation models.

---

## Introduction to Google Earth Engine for GeoAI

Google Earth Engine represents a revolutionary platform for planetary-scale Earth observation analysis, providing access to petabyte-scale archives of satellite imagery along with the computational infrastructure needed to process these data at global scales. Understanding the capabilities and architecture of Google Earth Engine is essential for implementing the environmental monitoring approaches developed in this course.

### Google Earth Engine Platform Architecture

The Google Earth Engine platform consists of several key components that work together to enable large-scale Earth observation analysis. The **data catalog** contains petabytes of satellite imagery and other Earth observation datasets, including the complete Landsat archive, Sentinel missions, MODIS data, and hundreds of other datasets. This data catalog is continuously updated with new observations, providing access to the most current Earth observation data available.

The **computational infrastructure** of Google Earth Engine leverages Google's cloud computing resources to enable parallel processing of large datasets. Users can perform computations on entire satellite image collections without needing to download or manage the underlying data. This cloud-based approach eliminates many of the technical barriers that have traditionally limited access to large-scale Earth observation analysis. For parcelización monitoring across the entire Los Lagos region, Earth Engine's computational infrastructure makes it possible to process decades of satellite imagery without requiring local computational resources.

The **programming interface** for Google Earth Engine supports both JavaScript and Python, enabling users to develop analysis workflows using familiar programming languages. The JavaScript interface, accessed through the Google Earth Engine Code Editor, provides an interactive development environment with integrated visualization capabilities. The **Python interface**, available through the Earth Engine Python API, enables integration with other Python-based analysis tools and workflows. For this course, we will primarily use the Python API, which integrates seamlessly with Jupyter notebooks and other Python libraries.

### Earth Engine Data Model and Key Concepts

Google Earth Engine's data model is built around several key concepts that users must understand to develop effective analysis workflows.

**Images** represent individual satellite observations or derived products, with each image containing one or more bands of data. For example, a Sentinel-2 image contains 13 spectral bands capturing different portions of the electromagnetic spectrum. Each band is a two-dimensional array of pixel values, with associated metadata describing the acquisition time, sensor characteristics, and processing level.

**ImageCollections** group related images together, such as all Sentinel-2 observations for a particular time period and geographic area. Most Earth Engine workflows begin by filtering an ImageCollection to select the specific images needed for analysis. Filters can be applied based on date ranges, geographic bounds, cloud cover, or other metadata properties.

**Features** represent vector data such as points, lines, or polygons, while **FeatureCollections** group related features together. This vector data capability enables integration of training data, administrative boundaries, and other geographic information with satellite imagery analysis. For parcelización monitoring, FeatureCollections can store parcel boundaries, training data, and validation datasets.

The **Earth Engine API** provides a rich set of functions for manipulating and analyzing Earth observation data. These functions include image processing operations (filtering, masking, mathematical operations), machine learning algorithms (classification, regression, clustering), and reduction operations (statistical summaries, time series analysis). Understanding how to chain these operations together into efficient workflows is key to effective use of Earth Engine.

### Visualization and Export Capabilities

**Visualization capabilities** in Google Earth Engine enable interactive exploration of analysis results. The Map interface allows users to display images and analysis results with customizable visualization parameters. For Sentinel-2 imagery, visualization parameters specify which bands to display as red, green, and blue channels, along with minimum and maximum values for stretching the display. Time series charts, histograms, and other visualization tools help users understand their data and analysis results.

The **export functionality** in Google Earth Engine enables users to save analysis results for use in other applications. Results can be exported as images (GeoTIFF format), tables (CSV format), or vector data (Shapefile or GeoJSON format) to Google Drive, Google Cloud Storage, or other destinations. For this course, you will export visualization results and analysis outputs to include in your GitHub portfolio.

### Computational Efficiency Considerations

Understanding the **computational model** of Google Earth Engine is important for developing efficient analysis workflows. Earth Engine uses **lazy evaluation**, meaning that computations are not performed until results are explicitly requested through operations such as visualization or export. This approach enables Earth Engine to optimize computational workflows and handle very large datasets efficiently.

The **scale parameter** in Earth Engine operations determines the spatial resolution at which computations are performed. Understanding how scale affects analysis results and computational requirements is crucial for developing effective workflows. For parcelización monitoring, typical scale values range from 10 to 30 meters, balancing spatial detail with computational efficiency.

---

## Guided Examples and Demonstrations

### Exploring Sentinel-2 Imagery in Google Earth Engine

In this guided demonstration, we will explore Sentinel-2 satellite imagery for one of the Chilean case study regions and learn how to visualize and interpret multi-band imagery using the Earth Engine Python API.

#### Step 1: Initialize Earth Engine and Define Study Area

```python
import ee
import geemap

# Initialize Earth Engine
ee.Initialize()

# Define the Los Lagos region (approximate bounds)
los_lagos = ee.Geometry.Rectangle([-74.5, -43.5, -71.5, -40.5])

# Alternative: Define Lake Llanquihue region for ecosystem monitoring
lake_llanquihue = ee.Geometry.Rectangle([-72.95, -41.35, -72.55, -41.05])

# Alternative: Define megadrought study area (central Chile)
megadrought_region = ee.Geometry.Rectangle([-71.5, -34.0, -70.5, -33.0])

# For this demonstration, we'll use Los Lagos
study_area = los_lagos
```

Alternative: Load AOI from GeoJSON (repo `data/external/aoi.geojson`) and use it as `study_area`:

```python
from pathlib import Path
import json
import ee

geojson_path = Path('../data/external/aoi.geojson')
with open(geojson_path) as f:
    aoi_geojson = json.load(f)

# GeoJSON coordinates are [lon, lat]
AOI_EE = ee.Geometry(aoi_geojson['features'][0]['geometry'])
study_area = AOI_EE
```

#### Step 2: Load and Filter Sentinel-2 Imagery

```python
# Load Sentinel-2 Surface Reflectance collection
sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

# Filter for our study area and time period
# Using 2023 data for recent conditions
filtered_collection = (sentinel2
    .filterBounds(study_area)
    .filterDate('2023-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
)

# Get information about the collection
print('Number of images:', filtered_collection.size().getInfo())
print('Date range:', 
      filtered_collection.aggregate_min('system:time_start').getInfo(),
      'to',
      filtered_collection.aggregate_max('system:time_end').getInfo())

# Select the least cloudy image
best_image = filtered_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()

# Print image properties
print('Selected image date:', ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo())
print('Cloud cover:', best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo(), '%')
```

#### Step 3: Create True Color and False Color Composites

```python
# Define visualization parameters for true color (RGB)
true_color_vis = {
    'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue
    'min': 0,
    'max': 3000,
    'gamma': 1.4
}

# Define visualization parameters for false color (NIR, Red, Green)
# This combination highlights vegetation in red tones
false_color_vis = {
    'bands': ['B8', 'B4', 'B3'],  # NIR, Red, Green
    'min': 0,
    'max': 3000,
    'gamma': 1.4
}

# Create an interactive map
Map = geemap.Map()
Map.centerObject(study_area, 8)

# Add layers to the map
Map.addLayer(best_image, true_color_vis, 'True Color (RGB)')
Map.addLayer(best_image, false_color_vis, 'False Color (NIR-R-G)')

# Display the map
Map
```

**Discussion Points:**
- How does the false color composite differ from the true color image?
- What features are more visible in the false color composite?
- Can you identify different land cover types (forest, agriculture, urban, water)?

#### Step 4: Calculate Spectral Indices

```python
# Calculate NDVI (Normalized Difference Vegetation Index)
# NDVI = (NIR - Red) / (NIR + Red)
ndvi = best_image.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Calculate NDWI (Normalized Difference Water Index)
# NDWI = (Green - NIR) / (Green + NIR)
ndwi = best_image.normalizedDifference(['B3', 'B8']).rename('NDWI')

# Calculate NDBI (Normalized Difference Built-up Index)
# NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
ndbi = best_image.normalizedDifference(['B11', 'B8']).rename('NDBI')

# Define visualization parameters for indices
ndvi_vis = {
    'min': -0.2,
    'max': 0.8,
    'palette': ['blue', 'white', 'green']
}

ndwi_vis = {
    'min': -0.5,
    'max': 0.5,
    'palette': ['brown', 'white', 'blue']
}

ndbi_vis = {
    'min': -0.5,
    'max': 0.5,
    'palette': ['green', 'white', 'gray']
}

# Add index layers to the map
Map.addLayer(ndvi, ndvi_vis, 'NDVI')
Map.addLayer(ndwi, ndwi_vis, 'NDWI')
Map.addLayer(ndbi, ndbi_vis, 'NDBI')

Map
```

**Discussion Points:**
- What do high NDVI values indicate? What about low values?
- Where do you see high NDWI values, and what does this tell you?
- How does NDBI help identify urban or built-up areas?

---

### Understanding Spectral Signatures

#### Demonstration: Extracting and Comparing Spectral Profiles

```python
import matplotlib.pyplot as plt
import numpy as np

# Define sample points for different land cover types
forest_point = ee.Geometry.Point([-72.8, -41.2])
water_point = ee.Geometry.Point([-72.7, -41.15])  # Lake Llanquihue
urban_point = ee.Geometry.Point([-73.05, -41.47])  # Puerto Montt
agriculture_point = ee.Geometry.Point([-72.9, -41.3])

# Extract spectral values at each point
def get_spectral_profile(image, point, scale=10):
    """Extract spectral values at a point"""
    # Select visible and NIR bands
    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    values = image.select(bands).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=scale
    ).getInfo()
    return [values.get(b, 0) for b in bands]

# Get profiles for each land cover type
forest_profile = get_spectral_profile(best_image, forest_point)
water_profile = get_spectral_profile(best_image, water_point)
urban_profile = get_spectral_profile(best_image, urban_point)
ag_profile = get_spectral_profile(best_image, agriculture_point)

# Band wavelengths (approximate centers in nm)
wavelengths = [490, 560, 665, 842, 1610, 2190]
band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

# Plot spectral profiles
plt.figure(figsize=(12, 6))
plt.plot(wavelengths, forest_profile, 'o-', label='Forest', linewidth=2)
plt.plot(wavelengths, water_profile, 's-', label='Water', linewidth=2)
plt.plot(wavelengths, urban_profile, '^-', label='Urban', linewidth=2)
plt.plot(wavelengths, ag_profile, 'd-', label='Agriculture', linewidth=2)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Surface Reflectance', fontsize=12)
plt.title('Spectral Signatures of Different Land Cover Types', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(wavelengths, band_names, rotation=45)

plt.tight_layout()
plt.savefig('../figures/week1_spectral_profiles.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Discussion Points:**
- What is the most distinctive feature of the vegetation spectral signature?
- Why does water appear dark across most wavelengths?
- How do urban and agricultural areas differ spectrally?
- What challenges might arise in distinguishing similar land cover types?

---

### Temporal Analysis with Sentinel-2

#### Demonstration: Creating an NDVI Time Series

```python
# Create NDVI band and robustly extract a null-safe time series
def add_ndvi(image):
    return image.addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI'))

collection_with_ndvi = filtered_collection.map(add_ndvi)

# Region for time series: a point buffer (adjust as needed)
sample_region = ee.Geometry.Point([-72.9, -41.3]).buffer(600)

# Compute NDVI mean per image at region; drop nulls; bring arrays client-side
ts_fc = (collection_with_ndvi
    .filterBounds(sample_region)
    .map(lambda img: img.set({
        'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
        'ndvi': img.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=sample_region,
            scale=10,
            bestEffort=True,
            maxPixels=1e9,
            tileScale=2
        ).get('NDVI')
    }))
    .filter(ee.Filter.notNull(['ndvi']))
)

dates = ts_fc.aggregate_array('date').getInfo()
ndvi_values = ts_fc.aggregate_array('ndvi').getInfo()

import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(dates, ndvi_values, 'o-', linewidth=2, markersize=6)
plt.xlabel('Date', fontsize=12)
plt.ylabel('NDVI', fontsize=12)
plt.title('NDVI Time Series - Study Region', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/week1_ndvi_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Discussion Points:**
- What seasonal patterns do you observe in the NDVI time series?
- How might these patterns differ for different land cover types?
- What could cause sudden drops in NDVI values?
- How might temporal analysis help detect land-use change?

---

## Hands-on Activities

### Activity 1: Spectral Exploration and Visualization (90 minutes)

Students will work through the provided Jupyter notebook to explore Sentinel-2 imagery for their chosen case study region and create multi-panel visualizations.

**Learning Objectives:**
- Load and filter satellite imagery using Google Earth Engine Python API
- Create true color and false color composites
- Calculate and visualize spectral indices
- Export and annotate visualization results

**Activity Steps:**

1. **Select Your Case Study Region** (10 minutes)
   - Choose one of the three case study regions: Los Lagos (parcelization), Megadrought region, or Lake Llanquihue
   - Define a specific area of interest within your chosen region
   - Create a geometry object for your study area

2. **Load and Explore Sentinel-2 Data** (20 minutes)
   - Filter the Sentinel-2 collection for your study area and time period
   - Examine the number of available images and their cloud cover
   - Select the best image(s) for analysis
   - Explore the band structure and metadata

3. **Create Visualization Composites** (30 minutes)
   - Generate a true color (RGB) composite
   - Create at least two false color composites using different band combinations
   - Experiment with visualization parameters (min, max, gamma)
   - Compare how different composites highlight different features

4. **Calculate Spectral Indices** (20 minutes)
   - Calculate NDVI, NDWI, and NDBI for your study area
   - Create appropriate visualizations for each index
   - Identify areas with high/low values for each index
   - Relate index values to visible features in the imagery

5. **Create Final Composite Figure** (10 minutes)
   - Export thumbnails of your best visualizations
   - Arrange them into a 2x2 or 3x2 panel figure
   - Add clear labels and titles to each panel
   - Save the figure to your `figures/` directory

**Expected Outputs:**
- Completed Jupyter notebook (`notebooks/Week1_Lab.ipynb`)
- Multi-panel composite figure (`figures/week1_composite.png`)
- Brief annotations describing what each panel shows

---

### Activity 2: Spectral Signature Analysis (45 minutes)

Students will extract and compare spectral signatures for different land cover types in their study region.

**Exercise Tasks:**

1. **Identify Sample Locations** (10 minutes)
   - Using your true color composite, identify 4-5 distinct land cover types
   - Create point geometries for each location
   - Document the coordinates and expected land cover type

2. **Extract Spectral Profiles** (15 minutes)
   - Use the provided code template to extract spectral values at each point
   - Organize the data into a structured format (list or dictionary)
   - Verify that the extracted values make sense for each land cover type

3. **Visualize and Compare** (15 minutes)
   - Create a line plot showing spectral profiles for all land cover types
   - Use different colors and markers for each type
   - Add appropriate labels, legend, and title
   - Save the figure to your `figures/` directory

4. **Interpret Results** (5 minutes)
   - Write 2-3 sentences describing the key differences between spectral signatures
   - Identify which wavelengths are most useful for distinguishing different types
   - Note any surprising or unexpected patterns

**Expected Outputs:**
- Spectral profile plot (`figures/week1_spectral_profiles.png`)
- Brief interpretation in your notebook

---

### Activity 3: Temporal Pattern Exploration (Optional Challenge - 60 minutes)

For students who complete the core activities early, this optional challenge explores temporal patterns in satellite observations.

**Challenge Tasks:**

1. **Create an NDVI Time Series**
   - Extract NDVI values for a specific location over an entire year
   - Plot the time series showing seasonal variations
   - Identify the greenest and least green periods

2. **Compare Multiple Locations**
   - Extract time series for 2-3 different land cover types
   - Plot them on the same graph for comparison
   - Discuss how temporal patterns differ between types

3. **Detect Potential Changes**
   - Look for locations where NDVI shows unusual changes
   - Investigate whether these correspond to land-use changes
   - Document your findings with before/after imagery

**Expected Outputs:**
- NDVI time series plot(s) (`figures/week1_ndvi_timeseries.png`)
- Brief analysis of temporal patterns in your notebook

---

## Week 1 Checkpoint Assessment (Self-Assessment)

### Practical Exercise

Complete the following tasks to assess your understanding of Week 1 concepts:

1. **Load Sentinel-2 imagery** for your chosen case study region for a specific date
2. **Create a 2x2 composite figure** showing:
   - True color (RGB) composite
   - False color (NIR-R-G) composite
   - NDVI visualization
   - NDWI visualization
3. **Extract spectral profiles** for at least three different land cover types
4. **Export your composite figure** and save it to your GitHub repository

### Success Criteria

You have successfully completed Week 1 if you can:

- ✓ Successfully authenticate and initialize Google Earth Engine
- ✓ Load and filter Sentinel-2 imagery for a specific region and time period
- ✓ Create meaningful visualizations using different band combinations
- ✓ Calculate spectral indices (NDVI, NDWI, NDBI) correctly
- ✓ Interpret what different visualizations reveal about land cover
- ✓ Export and organize results in your GitHub repository
- ✓ Write clear annotations explaining your visualizations

---

## Reflection and Discussion (30 minutes)

### Key Questions for Reflection

1. **Conceptual Understanding**: How do spectral bands differ from what the human eye can see? What advantages does multi-spectral imagery provide for environmental monitoring?

2. **Practical Applications**: Based on your exploration this week, what types of environmental changes do you think would be most easily detected using spectral indices? What types might be more challenging?

3. **Limitations and Challenges**: What limitations did you encounter when working with satellite imagery? How might factors like cloud cover, shadows, or mixed pixels affect your analysis?

4. **Case Study Connection**: How might the spectral analysis techniques you learned this week apply to your chosen case study (parcelization, megadrought, or ecosystem health)?

### Self-Reflection Prompts

Write 250-500 words in your `Week1_Reflection.md` file addressing these questions:

- What was the most surprising thing you learned about how satellites "see" the Earth?
- What challenges did you encounter during the hands-on activities, and how did you overcome them?
- How has your understanding of satellite imagery changed from the beginning to the end of this week?
- What questions do you still have about remote sensing or spectral analysis?

---

## Preview of Week 2

Next week, we will transition from understanding how satellites see the Earth to exploring how computer vision evolved to interpret Earth imagery. We will cover:

- The evolution of computer vision from object recognition to Earth observation
- How machine learning approaches differ from traditional spectral analysis
- The unique challenges of applying AI to geospatial data
- Introduction to the "Ethics Thread" that will run throughout the course
- Exploring how pre-trained computer vision models perform on satellite imagery

**Preparation for Week 2:**
- Review basic machine learning concepts (classification, training data, features)
- Think about how the spectral patterns you observed this week might be learned by a machine learning model
- Consider: What biases might exist in satellite imagery and Earth observation data?

---

## Additional Resources

### Technical Documentation

- [Google Earth Engine Guides](https://developers.google.com/earth-engine/guides) - Comprehensive documentation for Earth Engine
- [Sentinel-2 User Handbook](https://sentinel.esa.int/documents/247904/685211/Sentinel-2_User_Handbook) - Detailed information about Sentinel-2 sensors and data
- [geemap Documentation](https://geemap.org/) - Python package for interactive Earth Engine mapping
- [Awesome Earth Engine](https://github.com/giswqs/Awesome-GEE) - Curated list of Earth Engine resources

### Recommended Readings

- **Jensen, J. R. (2015).** *Introductory Digital Image Processing: A Remote Sensing Perspective* (4th ed.). Pearson. Chapters 1-3.
  - Foundational text on remote sensing principles and image processing

- **Gorelick, N., et al. (2017).** Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18-27.
  - [https://doi.org/10.1016/j.rse.2017.06.031](https://doi.org/10.1016/j.rse.2017.06.031)
  - Overview of the Google Earth Engine platform and its capabilities

- **Zhu, X., et al. (2017).** Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources. *IEEE Geoscience and Remote Sensing Magazine*, 5(4), 8-36.
  - [https://doi.org/10.1109/MGRS.2017.2762307](https://doi.org/10.1109/MGRS.2017.2762307)
  - Preview of how deep learning is transforming remote sensing (we'll explore this in coming weeks)

### Practice Exercises

- Complete the [Earth Engine Beginner's Cookbook](https://developers.google.com/earth-engine/tutorials/community/beginners-cookbook) tutorials
- Explore different band combinations for Sentinel-2 imagery
- Practice calculating additional spectral indices (EVI, SAVI, MNDWI)
- Experiment with different visualization parameters to enhance specific features

### Glossary Terms Introduced This Week

- **Spectral Signature**: Pattern of reflectance across different wavelengths characteristic of specific materials
- **Spectral Index**: Mathematical combination of spectral bands designed to enhance specific surface characteristics
- **NDVI (Normalized Difference Vegetation Index)**: Measure of vegetation greenness calculated from red and near-infrared bands
- **NDWI (Normalized Difference Water Index)**: Index for identifying water bodies and wet areas
- **NDBI (Normalized Difference Built-up Index)**: Index for enhancing built-up and urban areas
- **Image Collection**: Group of related satellite images in Google Earth Engine
- **Spatial Resolution**: The size of the smallest feature that can be detected in satellite imagery
- **Temporal Resolution**: How frequently a satellite observes the same location
- **Spectral Resolution**: The number and width of spectral bands a sensor can detect
- **Surface Reflectance**: The fraction of incoming solar radiation reflected by Earth's surface
- **False Color Composite**: Visualization using non-visible bands (e.g., near-infrared) to enhance specific features

---

## Notes for Self-Paced Learners

### Time Management Suggestions

This week's content is designed to take approximately 8-10 hours to complete. Here's a suggested breakdown:

- **Day 1-2** (3 hours): Read core content sections, watch supplementary videos, review readings
- **Day 3-4** (4 hours): Complete Activity 1 (Spectral Exploration and Visualization)
- **Day 5** (2 hours): Complete Activity 2 (Spectral Signature Analysis)
- **Day 6** (1 hour): Write reflection and organize GitHub repository
- **Day 7** (Optional): Complete optional challenge activity or review concepts

### Common Challenges and Solutions

**Challenge: Earth Engine authentication issues**
- Solution: Make sure you've signed up for Earth Engine and your account is approved. Try running `ee.Authenticate()` again and carefully follow the prompts. If issues persist, check the [Earth Engine troubleshooting guide](https://developers.google.com/earth-engine/guides/troubleshooting).

**Challenge: Understanding which bands to use for different purposes**
- Solution: Refer to the Sentinel-2 band guide. Remember: B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2. For vegetation, use NIR and Red. For water, use Green and NIR. Experiment with different combinations!

**Challenge: Visualizations appear too dark or too bright**
- Solution: Adjust the `min` and `max` parameters in your visualization parameters. For Sentinel-2 surface reflectance, typical values range from 0-3000, but you may need to adjust based on your specific scene.

**Challenge: Cloud cover obscuring study area**
- Solution: Try filtering for images with lower cloud cover percentage, or expand your date range to find clearer images. You can also use the `ee.Algorithms.Sentinel2.CDI()` function for more sophisticated cloud masking.

### Extension Activities

**For Advanced Learners:**
- Experiment with Landsat imagery and compare results with Sentinel-2
- Explore radar data (Sentinel-1) and compare with optical imagery
- Create animated time-lapse visualizations showing seasonal changes
- Investigate additional spectral indices (EVI, SAVI, MSAVI)
- Explore the relationship between elevation (using SRTM data) and spectral patterns

**For Visual Learners:**
- Create additional visualization types (scatter plots, histograms, 3D plots)
- Design an infographic explaining spectral signatures
- Make annotated screenshots showing step-by-step processes
- Create a visual glossary of key terms with example images

**For Conceptual Thinkers:**
- Write an extended essay on "What does it mean for a satellite to 'see'?"
- Research the history of remote sensing and create a timeline
- Compare and contrast different satellite missions and their applications
- Explore philosophical questions about representation and measurement in Earth observation

---

## Assessment Notes

### Self-Assessment Focus Areas

As you complete this week's activities, pay attention to:

- **Conceptual understanding** rather than memorization of syntax
- **Ability to interpret** visualizations and relate them to physical processes
- **Problem-solving skills** when encountering errors or unexpected results
- **Documentation quality** in your notebooks and reflection

### Using the Checkpoint to Identify Growth Areas

If you struggled with any checkpoint criteria, consider:

- **Technical issues**: Review the setup guide and troubleshooting resources
- **Conceptual gaps**: Re-read relevant sections of the study guide and seek additional resources
- **Interpretation challenges**: Practice describing what you see in imagery before calculating indices
- **Organization**: Review the GitHub repository structure and best practices

---

**This comprehensive Week 1 study guide builds essential remote sensing foundations while maintaining accessibility for students with varying technical backgrounds. The structured progression from concepts to demonstrations to hands-on practice ensures deep learning and practical skill development.**

---

**Document Version:** 1.0  
**Last Updated:** October 10, 2025  
**Author:** Manus AI  
**Course:** GeoAI and Earth Vision: Foundations to Frontier Applications

