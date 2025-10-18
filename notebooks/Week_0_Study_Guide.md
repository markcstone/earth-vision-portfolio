# Week 0: Building Your GeoAI Workspace — From Setup to First Earth Observation

---

## Learning Objectives

By the end of Week 0, students will be able to:

1. **Set up a reproducible Python environment** using Conda for geospatial and machine learning workflows
2. **Authenticate and initialize Google Earth Engine** for cloud-based Earth observation analysis
3. **Create and organize a professional GitHub repository** with proper version control and documentation
4. **Manage secrets and credentials securely** using environment variables and .gitignore patterns
5. **Define and visualize an Area of Interest (AOI)** for their chosen case study region

---

## Opening Discussion: Why Environment Setup Matters

### The Foundation of Reproducible Science

Before we can explore how satellites see the Earth or train foundation models to interpret imagery, we need to build a solid technical foundation. Environment setup might seem tedious, but it's one of the most critical skills for modern computational science. A well-configured environment ensures that:

- **Your code runs consistently** across different machines and operating systems
- **Others can reproduce your work** by recreating your exact software environment
- **You can manage complex dependencies** without conflicts or version mismatches
- **Your credentials remain secure** and never accidentally get published to GitHub

This week is about building the workspace where all your GeoAI work will happen. Think of it as setting up a laboratory before conducting experiments—the quality of your setup directly impacts the quality of your research.

### Today's Learning Journey

Week 0 is structured as a comprehensive onboarding experience that will take you from a blank slate to a fully functional GeoAI development environment. You'll install and configure multiple tools, authenticate with cloud services, create your project repository, and run your first Earth observation script. By the end, you'll have a professional-grade workspace ready for the 12-week journey ahead.

This guide is designed to be followed step-by-step, with detailed explanations of **what** you're doing, **why** it matters, and **how** to troubleshoot common issues. Don't rush—take time to understand each component.

---

## Core Content: Building Your GeoAI Technical Stack

### Understanding the Python Environment Ecosystem

Python is the lingua franca of modern data science, machine learning, and geospatial analysis. However, Python's flexibility comes with complexity: different projects require different versions of Python and different combinations of libraries. Without proper environment management, you can quickly end up with **dependency conflicts**, where Library A requires Version X of a dependency while Library B requires Version Y.

**Conda** is an environment and package manager that solves this problem by creating isolated environments—self-contained directories that include a specific Python version and a specific set of packages. Each environment is completely independent, so you can have one environment for GeoAI work (Python 3.11 with geospatial libraries) and another for a different project (Python 3.9 with different dependencies) without any conflicts.

#### Conda vs. Pip: When to Use Each

You'll encounter two main package managers in the Python ecosystem:

- **Conda**: Manages both Python packages and non-Python dependencies (like GDAL, a critical geospatial library written in C++). Conda is particularly important for geospatial work because many libraries have complex compiled dependencies.

- **Pip**: The standard Python package installer, which only manages Python packages. Pip is faster and has access to more packages (via PyPI), but can't handle non-Python dependencies.

**Best Practice for GeoAI**: Use conda for geospatial libraries with compiled dependencies (geopandas, rasterio, fiona, gdal), and use pip for pure Python packages (torch, transformers, plotly). Install conda packages first, then pip packages.

#### Miniconda vs. Anaconda

- **Anaconda**: A full distribution that includes Conda plus 250+ pre-installed packages. Large download (~3GB), but includes everything you might need.

- **Miniconda**: A minimal installer that includes only Conda and Python. Small download (~50MB), and you install only what you need.

**For this course**: We recommend **Miniconda** because it gives you full control over your environment and teaches you to manage dependencies explicitly.

---

### Installing and Configuring Miniconda

#### Step 1: Download and Install Miniconda

**For macOS (M1/M2/M3 or Intel):**

1. Visit [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Download the **macOS Apple M1 ARM 64-bit pkg** (for M1/M2/M3) or **macOS Intel x86 64-bit pkg** (for Intel Macs)
3. Double-click the downloaded `.pkg` file and follow the installer prompts
4. Accept the license agreement and install to the default location

**Verification:**

Open a new Terminal window and run:

```bash
conda --version
```

You should see output like `conda 23.x.x`. If you get "command not found," you may need to restart your Terminal or add Conda to your PATH.

#### Step 2: Configure Conda Settings

Before creating environments, configure Conda for optimal performance:

```bash
# Set conda-forge as the default channel (more packages, faster updates)
conda config --add channels conda-forge
conda config --set channel_priority strict

# Disable automatic base environment activation (cleaner workflow)
conda config --set auto_activate_base false
```

**What this does:**
- **conda-forge**: A community-maintained repository with more up-to-date packages than the default channel
- **channel_priority strict**: Ensures packages come from the highest-priority channel, avoiding version conflicts
- **auto_activate_base false**: Prevents the base environment from activating automatically, keeping your terminal clean

---

### Creating Your GeoAI Environment

#### Step 3: Create the Environment with Core Dependencies

Now we'll create a dedicated environment called `geoai` with Python 3.11 and all the libraries you'll need for this course.

```bash
# Create the environment with Python 3.11
conda create -n geoai python=3.11 -y

# Activate the environment
conda activate geoai
```

**What's happening:**
- `-n geoai`: Names the environment "geoai"
- `python=3.11`: Specifies Python version 3.11 (stable and compatible with all our libraries)
- `-y`: Automatically answers "yes" to confirmation prompts

After activation, your terminal prompt should change to show `(geoai)` at the beginning, indicating you're now working inside this environment.

#### Step 4: Install Geospatial Libraries via Conda

Install the core geospatial stack using Conda (because these have compiled C/C++ dependencies):

```bash
conda install -c conda-forge geopandas rasterio fiona gdal earthengine-api geemap folium -y
```

**What each library does:**
- **geopandas**: Extends pandas to handle geospatial vector data (points, lines, polygons)
- **rasterio**: Reads and writes raster data (satellite imagery, DEMs)
- **fiona**: Handles vector data formats (shapefiles, GeoJSON)
- **gdal**: The foundational geospatial library that powers most other tools
- **earthengine-api**: Google Earth Engine Python API
- **geemap**: Interactive mapping with Earth Engine in Jupyter notebooks
- **folium**: Creates interactive web maps

**This step may take 5-10 minutes** as Conda resolves dependencies and downloads packages.

#### Step 5: Install Machine Learning and Visualization Libraries via Pip

Now install pure Python packages using pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets huggingface-hub
pip install matplotlib seaborn plotly ipykernel
pip install python-dotenv
```

**What each library does:**
- **torch, torchvision, torchaudio**: PyTorch deep learning framework (CPU version for now)
- **transformers**: Hugging Face library for foundation models
- **datasets**: Hugging Face library for dataset management
- **huggingface-hub**: Access to Hugging Face model and dataset repositories
- **matplotlib, seaborn, plotly**: Visualization libraries (static and interactive)
- **ipykernel**: Allows Jupyter to use this environment as a kernel
- **python-dotenv**: Loads environment variables from `.env` files

**Why CPU-only PyTorch?** We're installing the CPU version for local development. When you need GPU acceleration (Weeks 4+), you'll use university GPU resources or cloud platforms.

#### Step 6: Verify Installation

Test that all critical libraries are installed correctly:

```bash
python -c "import ee; import geemap; import torch; import geopandas; print('✓ All libraries installed successfully!')"
```

If you see the success message, you're ready to proceed. If you get an error, note which library failed and consult the troubleshooting section.

---

### Jupyter Notebooks and Kernel Management

Jupyter notebooks are the primary interface for interactive data analysis and visualization in this course. However, Jupyter needs to know about your `geoai` environment.

#### Step 7: Create a Jupyter Kernel for Your Environment

While your `geoai` environment is activated, run:

```bash
python -m ipykernel install --user --name geoai --display-name "Python (geoai)"
```

**What this does:**
- Creates a Jupyter kernel that points to your `geoai` environment
- The kernel will appear as "Python (geoai)" in Jupyter's kernel selection dropdown
- `--user`: Installs the kernel for your user account only (no admin privileges needed)

#### Step 8: Launch Jupyter and Verify the Kernel

```bash
# Make sure you're in your project directory
cd ~/geoai-course  # or wherever you want your project

# Launch Jupyter Lab (or Jupyter Notebook)
jupyter lab
```

**In Jupyter Lab:**
1. Create a new notebook
2. Click on the kernel name in the top-right corner
3. Select "Python (geoai)" from the dropdown
4. In a cell, run:

```python
import sys
print(sys.executable)
```

The output should show a path containing `.../envs/geoai/...`, confirming you're using the correct environment.

**Common Issue**: If you don't see "Python (geoai)" in the kernel list, make sure you ran the `ipykernel install` command while the `geoai` environment was activated.

---

### Understanding YAML and Environment Reproducibility

#### What is YAML?

**YAML** stands for "YAML Ain't Markup Language" (a recursive acronym). It's a human-readable data format commonly used for configuration files. YAML uses:

- **Colons** for key-value pairs: `name: geoai`
- **Dashes** for list items: `- numpy`
- **Indentation** to show hierarchy (like Python, but even more strict)

#### Step 9: Export Your Environment to environment.yml

Once your environment is fully configured, create a snapshot:

```bash
conda activate geoai
conda env export > environment.yml
```

**What this creates:**
An `environment.yml` file that lists every package in your environment with exact version numbers. This file allows anyone (including future you) to recreate the exact same environment.

**Example environment.yml structure:**

```yaml
name: geoai
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11.5
  - geopandas=0.14.0
  - rasterio=1.3.8
  - pip:
    - torch==2.1.0
    - transformers==4.35.0
```

#### Why This Matters for Reproducibility

Six months from now, when you return to this project, library versions will have changed. Without `environment.yml`, you might encounter breaking changes. With it, you can recreate the exact environment:

```bash
conda env create -f environment.yml
```

**Best Practice**: Commit `environment.yml` to your GitHub repository and update it whenever you install new packages.

---

### Git, GitHub, and Version Control Fundamentals

Version control is the practice of tracking changes to your code over time. **Git** is the version control system, and **GitHub** is a cloud platform for hosting Git repositories.

#### Why Version Control Matters

- **Track changes**: See what changed, when, and why
- **Collaborate**: Multiple people can work on the same project
- **Backup**: Your code is safely stored in the cloud
- **Portfolio**: Your GitHub repository demonstrates your skills to potential employers or collaborators

#### Step 10: Install and Configure Git

**Check if Git is installed:**

```bash
git --version
```

If not installed, download from [https://git-scm.com/downloads](https://git-scm.com/downloads).

**Configure your identity:**

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

This information will be attached to your commits.

#### Step 11: Create Your Project Directory Structure

```bash
# Create main project directory
mkdir ~/geoai-course
cd ~/geoai-course

# Create subdirectories
mkdir data notebooks scripts reports figures

# Create subdirectories within data
mkdir data/raw data/processed data/external

# Initialize Git repository
git init
```

**Recommended structure:**

```
geoai-course/
├── data/
│   ├── raw/           # Original, immutable data
│   ├── processed/     # Cleaned, transformed data
│   └── external/      # Data from external sources
├── notebooks/         # Jupyter notebooks
├── scripts/           # Python scripts (.py files)
├── reports/           # Markdown reports and reflections
├── figures/           # Visualizations and plots
├── .gitignore         # Files to exclude from Git
├── environment.yml    # Conda environment specification
└── README.md          # Project documentation
```

#### Step 12: Create a .gitignore File

Some files should **never** be committed to Git:
- Large data files
- Credentials and API keys
- Temporary files and caches

Create a `.gitignore` file:

```bash
touch .gitignore
```

Add these patterns (using a text editor or command line):

```
# Data files (too large for Git)
data/raw/*
data/processed/*
data/external/*

# Keep directory structure but ignore contents
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep

# Python
*.pyc
__pycache__/
.ipynb_checkpoints/
*.py[cod]

# Environment and secrets
.env
*.env

# macOS
.DS_Store

# Jupyter
.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
```

**Create .gitkeep files** to preserve empty directories:

```bash
touch data/raw/.gitkeep data/processed/.gitkeep data/external/.gitkeep
```

#### Step 13: Create Your GitHub Repository

1. Go to [https://github.com](https://github.com) and sign in
2. Click the "+" icon in the top-right and select "New repository"
3. Name it `geoai-earth-vision` (or your preferred name)
4. Set to **Public** (required for portfolio visibility)
5. Do **NOT** initialize with README, .gitignore, or license (we'll add these locally)
6. Click "Create repository"

#### Step 14: Create a Personal Access Token (PAT)

GitHub no longer accepts passwords for HTTPS authentication. You need a Personal Access Token.

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name: "GeoAI Course - MacBook"
4. Set expiration: 90 days (or custom)
5. Select scopes:
   - ✅ **repo** (Full control of private repositories)
   - ✅ **workflow** (if you plan to use GitHub Actions)
6. Click "Generate token"
7. **Copy the token immediately** (you won't see it again!)

**Store the token securely** in a password manager or `.env` file (covered next).

#### Step 15: Connect Local Repository to GitHub

```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/geoai-earth-vision.git

# Create initial commit
git add .
git commit -m "Initial commit: Project structure and environment setup"

# Push to GitHub (you'll be prompted for username and password)
git push -u origin main
```

**When prompted:**
- Username: Your GitHub username
- Password: **Paste your Personal Access Token** (not your GitHub password!)

#### Step 16: Cache Your Credentials (macOS)

To avoid entering your token every time:

```bash
git config --global credential.helper osxkeychain
```

The next time you push, macOS Keychain will save your credentials.

---

### Secure Secret Management with .env Files

API keys, tokens, and other secrets should **never** be committed to Git. Instead, store them in a `.env` file and load them programmatically.

#### Step 17: Create a .env File

```bash
cd ~/geoai-course
touch .env
```

**Add your GitHub PAT** (using a text editor):

```
GITHUB_PAT=ghp_your_token_here
```

**Verify .env is in .gitignore:**

```bash
echo ".env" >> .gitignore
```

#### Step 18: Load Secrets in Python

In your notebooks or scripts:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Access the token
github_token = os.getenv("GITHUB_PAT")

if github_token:
    print("✓ Token loaded successfully")
else:
    print("✗ Token not found - check your .env file")
```

**Why this matters:**
- Your `.env` file stays on your local machine
- Your notebooks can reference `os.getenv("GITHUB_PAT")` without exposing the actual token
- If you share your notebook, others can create their own `.env` file with their own tokens

---

### Google Earth Engine Authentication

Google Earth Engine (GEE) is a planetary-scale platform for Earth observation analysis. You'll use it extensively throughout this course.

#### Step 19: Sign Up for Earth Engine

1. Go to [https://earthengine.google.com](https://earthengine.google.com)
2. Click "Sign Up" and use your Google account
3. Select "Register a Noncommercial or Commercial Cloud project"
4. Create a new Google Cloud project (free tier is sufficient)
5. Wait for approval (usually instant, but can take up to 24 hours)

#### Step 20: Authenticate Earth Engine in Python

Open a Jupyter notebook and run:

```python
import ee

# Authenticate (first time only)
ee.Authenticate()
```

**What happens:**
1. A browser window opens
2. Sign in with your Google account
3. Authorize Earth Engine to access your account
4. Copy the authorization code and paste it back into the notebook

**After authentication, initialize Earth Engine:**

```python
ee.Initialize()
print("✓ Earth Engine initialized successfully")
```

#### Step 21: Test Earth Engine with a Simple Visualization

```python
import geemap

# Create an interactive map
Map = geemap.Map(center=[-41.3, -72.8], zoom=8)

# Load Sentinel-2 imagery
sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

# Filter for Los Lagos region, 2023, low cloud cover
image = (sentinel2
    .filterBounds(ee.Geometry.Point([-72.8, -41.3]))
    .filterDate('2023-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .median())

# Visualization parameters for true color
vis_params = {
    'bands': ['B4', 'B3', 'B2'],
    'min': 0,
    'max': 3000,
    'gamma': 1.4
}

# Add to map
Map.addLayer(image, vis_params, 'Sentinel-2 True Color')
Map
```

If you see an interactive map with satellite imagery, **congratulations!** Your Earth Engine setup is complete.

---

### Defining Your Area of Interest (AOI)

Each student will focus on one of three case study regions. Your Area of Interest (AOI) will be used throughout the course.

#### Step 22: Choose Your Case Study

**Option 1: Los Lagos Parcelization**
- **Region**: Los Lagos, southern Chile (~41°S - 43°S)
- **Focus**: Detecting rural parcelization (subdivision of agricultural land)
- **Challenges**: Small parcels (1-5 hectares), gradual change, spectral similarity

**Option 2: Chilean Megadrought**
- **Region**: Central Chile (~33°S - 34°S)
- **Focus**: Monitoring drought impacts on water resources and vegetation
- **Challenges**: Multi-year trends, seasonal variability, hydrological changes

**Option 3: Lake Llanquihue Ecosystem Health**
- **Region**: Lake Llanquihue and watershed (~41°S)
- **Focus**: Monitoring water quality, algal blooms, land-use impacts
- **Challenges**: Water body dynamics, watershed-scale processes, temporal variability

#### Step 23: Create Your AOI Geometry

**Method 1: Using geemap to draw a polygon**

```python
import geemap
import json

# Create a map centered on your region
Map = geemap.Map(center=[-41.3, -72.8], zoom=9)

# Use the drawing tools to create a polygon
# Click the polygon tool and draw your AOI
Map
```

After drawing, extract the geometry:

```python
# Get the last drawn feature
aoi = Map.draw_last_feature

# Convert to GeoJSON
aoi_geojson = geemap.ee_to_geojson(aoi)

# Save to file
with open('../data/external/aoi.geojson', 'w') as f:
    json.dump(aoi_geojson, f, indent=2)

print("✓ AOI saved to data/external/aoi.geojson")
```

**Method 2: Define a bounding box programmatically**

```python
import ee
import json

# Define bounding box [west, south, east, north]
# Example: Los Lagos region
bbox = [-74.5, -43.5, -71.5, -40.5]

# Create Earth Engine geometry
aoi_ee = ee.Geometry.Rectangle(bbox)

# Convert to GeoJSON
aoi_geojson = aoi_ee.getInfo()

# Save to file
with open('../data/external/aoi.geojson', 'w') as f:
    json.dump(aoi_geojson, f, indent=2)

print("✓ AOI saved to data/external/aoi.geojson")
```

#### Step 24: Visualize Your AOI

```python
import geemap

# Load your saved AOI
with open('../data/external/aoi.geojson', 'r') as f:
    aoi_geojson = json.load(f)

# Convert to Earth Engine geometry
aoi_ee = geemap.geojson_to_ee(aoi_geojson)

# Create map and add AOI
Map = geemap.Map()
Map.centerObject(aoi_ee, 8)
Map.addLayer(aoi_ee, {'color': 'red'}, 'Area of Interest')

# Add a basemap
Map.add_basemap('SATELLITE')

Map
```

---

## Guided Examples and Demonstrations

### Complete Setup Verification Script

Create a notebook called `Week0_Setup_Verification.ipynb` and run this comprehensive test:

```python
"""
Week 0: Complete Environment Verification
This notebook tests all components of your GeoAI setup.
"""

import sys
import os

print("=" * 60)
print("GeoAI Environment Verification")
print("=" * 60)

# 1. Python Version
print(f"\n1. Python Version: {sys.version}")
print(f"   Executable: {sys.executable}")
assert sys.version_info >= (3, 11), "Python 3.11+ required"
print("   ✓ Python version OK")

# 2. Core Libraries
print("\n2. Testing Core Libraries...")
try:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    import ee
    import geemap
    import torch
    import transformers
    print("   ✓ All core libraries imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")

# 3. Earth Engine Authentication
print("\n3. Testing Earth Engine...")
try:
    ee.Initialize()
    test_image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20230101T143729_20230101T143931_T19HBU')
    info = test_image.getInfo()
    print("   ✓ Earth Engine initialized and accessible")
except Exception as e:
    print(f"   ✗ Earth Engine error: {e}")

# 4. Environment Variables
print("\n4. Testing Environment Variables...")
from dotenv import load_dotenv
load_dotenv()
github_token = os.getenv("GITHUB_PAT")
if github_token:
    print(f"   ✓ GitHub PAT loaded (length: {len(github_token)})")
else:
    print("   ⚠ GitHub PAT not found in .env (optional for now)")

# 5. PyTorch
print("\n5. Testing PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
x = torch.rand(3, 3)
print(f"   ✓ PyTorch tensor operations working")

# 6. File Structure
print("\n6. Checking Project Structure...")
required_dirs = ['data', 'notebooks', 'scripts', 'reports', 'figures']
for dir_name in required_dirs:
    if os.path.exists(f'../{dir_name}'):
        print(f"   ✓ {dir_name}/ exists")
    else:
        print(f"   ✗ {dir_name}/ missing")

print("\n" + "=" * 60)
print("Verification Complete!")
print("=" * 60)
```

**Expected output**: All checks should show ✓ (checkmarks). If any show ✗, consult the troubleshooting section.

---

### Creating Your First Earth Observation Visualization

Create a notebook called `Week0_First_Visualization.ipynb`:

```python
"""
Week 0: First Earth Observation Visualization
Create a multi-panel figure showing your AOI with different visualizations.
"""

import ee
import geemap
import json
import matplotlib.pyplot as plt

# Initialize Earth Engine
ee.Initialize()

# Load your AOI
with open('../data/external/aoi.geojson', 'r') as f:
    aoi_geojson = json.load(f)
aoi = geemap.geojson_to_ee(aoi_geojson)

# Load Sentinel-2 imagery
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

# Filter for your AOI and recent date
image = (s2
    .filterBounds(aoi)
    .filterDate('2023-06-01', '2023-09-01')  # Summer imagery
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .median())

# Create visualizations
Map = geemap.Map()
Map.centerObject(aoi, 9)

# True color
true_color = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4}
Map.addLayer(image, true_color, 'True Color')

# False color (vegetation in red)
false_color = {'bands': ['B8', 'B4', 'B3'], 'min': 0, 'max': 3000, 'gamma': 1.4}
Map.addLayer(image, false_color, 'False Color (NIR-R-G)')

# NDVI
ndvi = image.normalizedDifference(['B8', 'B4'])
ndvi_vis = {'min': -0.2, 'max': 0.8, 'palette': ['blue', 'white', 'green']}
Map.addLayer(ndvi, ndvi_vis, 'NDVI')

# Add AOI outline
Map.addLayer(aoi, {'color': 'red'}, 'AOI Boundary')

Map
```

**Save a screenshot** of your map and add it to your `figures/` directory as `week0_first_visualization.png`.

---

## Hands-on Activities

### Activity 1: Complete Environment Setup (60 minutes)

**Checklist:**

- [ ] Install Miniconda
- [ ] Create `geoai` conda environment
- [ ] Install all required packages
- [ ] Create Jupyter kernel
- [ ] Export `environment.yml`
- [ ] Verify all imports work

**Deliverable:** Screenshot of successful verification script output

---

### Activity 2: GitHub Repository Setup (45 minutes)

**Checklist:**

- [ ] Create local project directory with proper structure
- [ ] Initialize Git repository
- [ ] Create `.gitignore` file
- [ ] Create GitHub repository
- [ ] Generate Personal Access Token
- [ ] Connect local to remote and push initial commit
- [ ] Create `README.md` with project description

**Deliverable:** Link to your public GitHub repository

---

### Activity 3: Earth Engine Authentication and AOI Definition (45 minutes)

**Checklist:**

- [ ] Sign up for Google Earth Engine
- [ ] Authenticate Earth Engine in Python
- [ ] Choose your case study region
- [ ] Define and save your AOI as GeoJSON
- [ ] Create a visualization of your AOI with Sentinel-2 imagery
- [ ] Save visualization to `figures/`

**Deliverable:** 
- `data/external/aoi.geojson`
- `figures/week0_first_visualization.png`
- Brief description of your chosen case study in `reports/Week0_Reflection.md`

---

### Activity 4: Create Your First Repository README (30 minutes)

Create a professional `README.md` file for your repository:

```markdown
# GeoAI and Earth Vision: Portfolio

**Student:** [Your Name]  
**Course:** GeoAI and Earth Vision: Foundations to Frontier Applications  
**Institution:** [Your Institution]  
**Semester:** [Current Semester/Year]

## Overview

This repository documents my learning journey through a 12-week course on GeoAI and Earth Vision, exploring how artificial intelligence and foundation models are transforming Earth observation and environmental monitoring.

## Case Study

I am focusing on **[Your Chosen Case Study]** in **[Region]**, Chile.

[2-3 sentences describing why this case study interests you and what questions you hope to explore]

## Repository Structure

```
├── data/               # Data files (not tracked in Git)
├── notebooks/          # Jupyter notebooks for weekly labs
├── scripts/            # Python scripts
├── reports/            # Weekly reflections and reports
├── figures/            # Visualizations and plots
├── environment.yml     # Conda environment specification
└── README.md           # This file
```

## Environment Setup

To reproduce this environment:

```bash
conda env create -f environment.yml
conda activate geoai
```

## Weekly Progress

- [x] Week 0: Environment Setup and AOI Definition
- [ ] Week 1: Spectral Eyes and Digital Landscapes
- [ ] Week 2: Computer Vision Meets Earth Observation
- [ ] ...

## Contact

[Your email or preferred contact method]

## License

This project is licensed under the MIT License - see LICENSE file for details.
```

Commit and push your README:

```bash
git add README.md
git commit -m "Add comprehensive README"
git push origin main
```

---

## Week 0 Checkpoint Assessment (Self-Assessment)

### Practical Exercise

Complete the following to verify your setup:

1. **Run the verification script** and screenshot the output
2. **Create a test notebook** that:
   - Loads your AOI
   - Filters Sentinel-2 imagery for your region
   - Creates a true color composite
   - Exports a thumbnail image
3. **Push everything to GitHub** and verify it appears in your repository

### Success Criteria

You have successfully completed Week 0 if you can:

- ✓ Activate your `geoai` conda environment
- ✓ Launch Jupyter and select the "Python (geoai)" kernel
- ✓ Import all required libraries without errors
- ✓ Initialize Google Earth Engine successfully
- ✓ Load and visualize your AOI
- ✓ Push commits to your GitHub repository
- ✓ View your public repository on GitHub
- ✓ Load environment variables from `.env` file

---

## Reflection and Discussion (20 minutes)

### Key Questions for Reflection

1. **Technical Understanding**: What is the difference between conda and pip, and why does it matter for geospatial work?

2. **Reproducibility**: How does `environment.yml` support reproducible research? Why is this important for scientific work?

3. **Security**: Why should API keys and tokens never be committed to Git? What could happen if they were?

4. **Organization**: How does a well-structured project directory help you stay organized and productive?

### Self-Reflection Prompt

Write 150-250 words in `reports/Week0_Reflection.md` addressing:

- What was the most challenging part of the setup process?
- What did you learn about environment management that you didn't know before?
- How comfortable are you with Git and GitHub? What do you still want to learn?
- What questions or concerns do you have as you begin Week 1?

---

## Preview of Week 1

Next week, we'll put your new environment to work! You'll learn:

- **How satellites "see" the Earth** through electromagnetic radiation
- **The physics of remote sensing** and spectral signatures
- **How to calculate and interpret spectral indices** (NDVI, NDWI, NDBI)
- **Google Earth Engine fundamentals** for planetary-scale analysis
- **Creating your first multi-panel visualization** of your study region

**Preparation for Week 1:**
- Make sure your environment is fully functional
- Familiarize yourself with your AOI by exploring it in Google Earth
- Review basic Python and pandas if needed
- Think about what environmental changes you might want to detect in your region

---

## Additional Resources

### Environment Management

- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### Git and GitHub

- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [GitHub Docs](https://docs.github.com/en)
- [Pro Git Book](https://git-scm.com/book/en/v2) (free online)
- [GitHub Skills](https://skills.github.com/) (interactive tutorials)

### Google Earth Engine

- [Earth Engine Get Started](https://developers.google.com/earth-engine/guides/getstarted)
- [geemap Documentation](https://geemap.org/)
- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)

### Python and Jupyter

- [Jupyter Lab Documentation](https://jupyterlab.readthedocs.io/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) (free online)

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: "conda: command not found"

**Solution:**
1. Restart your terminal
2. If still not working, add Conda to your PATH:
   ```bash
   export PATH="$HOME/miniconda3/bin:$PATH"
   ```
3. Add to your `~/.zshrc` or `~/.bash_profile` to make permanent

#### Issue: "Kernel dies when importing libraries"

**Solution:**
1. Make sure you created the Jupyter kernel while `geoai` was activated
2. Restart Jupyter completely
3. Select the "Python (geoai)" kernel explicitly
4. If still failing, reinstall ipykernel:
   ```bash
   conda activate geoai
   pip install --force-reinstall ipykernel
   python -m ipykernel install --user --name geoai --display-name "Python (geoai)"
   ```

#### Issue: "Earth Engine authentication fails"

**Solution:**
1. Make sure you've signed up for Earth Engine and been approved
2. Try authenticating in a regular Python session (not Jupyter):
   ```bash
   python
   >>> import ee
   >>> ee.Authenticate()
   ```
3. Clear previous credentials:
   ```bash
   earthengine authenticate --force
   ```

#### Issue: "Git push asks for password but token doesn't work"

**Solution:**
1. Make sure you're using HTTPS URL, not SSH:
   ```bash
   git remote -v
   ```
   Should show `https://github.com/...`, not `git@github.com:...`
2. If using SSH URL, change to HTTPS:
   ```bash
   git remote set-url origin https://github.com/USERNAME/REPO.git
   ```
3. When prompted for password, paste your PAT (Personal Access Token), not your GitHub password

#### Issue: "Cannot see hidden files (.gitignore, .env)"

**Solution (macOS):**
- Press `⌘ + Shift + .` (Command + Shift + Period) in Finder
- Or use terminal: `ls -la` to list all files including hidden ones

#### Issue: "Package conflicts when installing"

**Solution:**
1. Create a fresh environment:
   ```bash
   conda deactivate
   conda env remove -n geoai
   conda create -n geoai python=3.11 -y
   conda activate geoai
   ```
2. Install packages in this order:
   - Geospatial libraries via conda first
   - ML libraries via pip second
3. If specific conflict, try:
   ```bash
   conda install package_name --force-reinstall
   ```

---

## Glossary Terms Introduced This Week

- **Conda**: Package and environment manager for Python and other languages
- **Environment**: Isolated directory containing a specific Python version and packages
- **YAML**: Human-readable data format used for configuration files
- **Git**: Distributed version control system for tracking code changes
- **GitHub**: Cloud platform for hosting Git repositories
- **Personal Access Token (PAT)**: Secure authentication token for GitHub API access
- **.gitignore**: File specifying which files Git should ignore
- **.env**: File storing environment variables and secrets (not committed to Git)
- **Repository (Repo)**: Project directory tracked by Git
- **Commit**: Snapshot of changes in a Git repository
- **Push**: Upload local commits to a remote repository (like GitHub)
- **AOI (Area of Interest)**: Geographic region selected for analysis
- **GeoJSON**: JSON-based format for encoding geographic data structures
- **Jupyter Kernel**: Computational engine that executes code in Jupyter notebooks

---

## Notes for Self-Paced Learners

### Time Management Suggestions

Week 0 is foundational and may take longer than subsequent weeks. Budget approximately **6-8 hours** total:

- **Day 1** (2-3 hours): Install Miniconda, create environment, install packages
- **Day 2** (1-2 hours): Set up Git, GitHub, and create repository structure
- **Day 3** (1-2 hours): Authenticate Earth Engine, define AOI
- **Day 4** (1 hour): Create first visualization and README
- **Day 5** (1 hour): Write reflection and verify everything works

**Don't rush!** A solid setup now will save hours of troubleshooting later.

### When to Ask for Help

If you're stuck for more than 30 minutes on a single issue:
1. Check the troubleshooting section above
2. Search for the error message online (Stack Overflow, GitHub issues)
3. Ask in course discussion forum or reach out to instructor
4. Document what you tried—this helps others help you

### Extension Activities

**For those who complete early:**

- Explore the [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)
- Create multiple AOIs for different case studies
- Experiment with different Sentinel-2 band combinations
- Set up GitHub Actions for automated testing (advanced)
- Explore QGIS and install the Earth Engine plugin

---

**This comprehensive Week 0 study guide provides the detailed foundation you need to begin your GeoAI journey with confidence. Take your time, follow each step carefully, and don't hesitate to revisit sections as needed.**

---

**Document Version:** 2.0  
**Last Updated:** October 12, 2025  
**Author:** Manus AI  
**Course:** GeoAI and Earth Vision: Foundations to Frontier Applications

