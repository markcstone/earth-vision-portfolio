# Week 2: From Vision to GeoVision — The Evolution of Computer Vision into Earth Observation AI

## Assitant Prompt
### Week 2: Custom GPT assistant instructions

#### System
You are an expert in GeoAI and computer vision with deep experience in Google Earth Engine (Python API, geemap), PyTorch/TorchVision (pretrained models, feature extraction), and the Python geospatial stack (geopandas, rasterio, GDAL/PROJ). Provide stepwise, runnable guidance with robust paths (Pathlib), reproducibility (conda, ipykernel, python-dotenv), and concise troubleshooting. Assume the active kernel is "Python (geoai)" and the AOI is saved at `data/external/aoi.geojson`.

#### Context
- Environment ready from Week 1; Earth Engine authenticated and initialized
- AOI exists at `data/external/aoi.geojson`
- Repo root: `/Users/mstone14/QGIS/GeoAI_Class/github/earth-vision-portfolio`
- Focus this week: domain shift from ImageNet to Earth observation using Sentinel‑2 RGB patches and pre-trained ResNet‑50; activation visualization

#### Goal
- Evaluate ImageNet pre-trained models (ResNet‑50) on Sentinel‑2 imagery for the AOI
- Produce predictions, activation visualizations, and a concise domain shift analysis
- Save figures to `figures/` and results to `reports/`

#### How to help
- Provide minimal, runnable cells for "Python (geoai)" using `ee`, `geemap`, `torch`, and `torchvision`
- Use robust paths (Pathlib) and AOI loading from `data/external/aoi.geojson`
- Prefer absolute paths for Terminal commands; keep notebook code path-robust
- Add quick validations (prints/sanity checks) and brief troubleshooting

#### First tasks
1) Setup
   - EE init (with auth fallback), load AOI GeoJSON → `AOI_EE`
   - Build Sentinel‑2 median RGB composite (HARMONIZED) over `AOI_EE`
2) Model + labels
   - Load pre-trained ResNet‑50 (`eval()`), fetch ImageNet labels
3) Sampling + inference
   - Define sample points (Forest, Water, Urban, Agriculture, Bare Soil)
   - Extract RGB patches; preprocess; run top‑5 predictions
4) Visualization
   - Multi-panel: patches + predictions; save to `figures/week2_imagenet_predictions.png`
   - Activation/feature visualization for one sample; save to `figures/`
5) Results
   - Summary table → `reports/week2_prediction_summary.csv`
   - Brief notes on domain shift failure modes

Expected outputs: printed model info, AOI bounds, sample coords, per-sample top‑5 predictions, saved figures and CSV.
---

## Learning Objectives

By the end of Week 2, students will be able to:

1. **Explain the conceptual evolution** from classic computer vision to GeoAI, tracing the progression from hand-crafted features through deep learning to foundation models
2. **Identify and analyze the unique challenges** of applying AI to geospatial data, including scale mismatch, spectral complexity, temporal dynamics, and geographic context
3. **Evaluate how pre-trained vision models perform** on Earth observation imagery and diagnose the causes of domain shift
4. **Articulate the ethical and equity dimensions** of Earth observation AI, including data geography, representation bias, and responsible practice
5. **Launch the Ethics Thread** with critical reflection connecting technical choices to societal implications

---

## Opening Discussion and Review

### Review of Week 1 Key Concepts

Before embarking on this week's conceptual journey, reflect on your Week 1 experience with spectral remote sensing:

- How do different surface materials (vegetation, water, built-up areas) interact with electromagnetic radiation across different wavelengths?
- What did you learn about spectral indices (NDVI, NDWI, NDBI) and their role in characterizing land cover?
- What patterns did you observe in your study region when you created spectral composites and time series?
- What limitations did you encounter when trying to distinguish different land uses based solely on spectral signatures?

That last question is particularly important. You may have noticed that spectral indices, while powerful, have limitations. Agricultural fields and newly cleared parcels might have similar NDVI values. Water bodies and shadows can be confused. Urban areas show high spectral variability. These challenges motivate this week's exploration: **Can machine learning help us move beyond hand-crafted spectral indices to automatically learn patterns that distinguish land uses?**

### Today's Learning Journey

This week marks a critical conceptual transition in your GeoAI education. We move from understanding how satellites capture Earth's surface (Week 1) to exploring how artificial intelligence can learn to interpret these observations. This journey requires understanding three interconnected domains:

**First**, we'll trace the evolution of computer vision from its origins in hand-crafted feature engineering through the deep learning revolution to today's foundation models. This historical perspective reveals why modern AI approaches are so powerful—and why they sometimes fail.

**Second**, we'll examine what makes Earth observation fundamentally different from conventional computer vision. Satellite imagery isn't just another image dataset; it presents unique challenges of scale, spectral richness, temporal dynamics, and geographic context that require specialized approaches.

**Third**, we'll conduct hands-on experiments applying pre-trained computer vision models to satellite imagery from your study region. These experiments will reveal the **domain shift problem**—what happens when models trained on photographs of everyday objects encounter satellite views of forests, fields, and water bodies. The failures you'll observe aren't bugs; they're windows into what these models have actually learned.

**Finally**, we'll launch the **Ethics Thread**—a continuous reflection on responsible AI that will run throughout the course. Earth observation AI isn't ethically neutral. Questions of who observes, what gets classified, and who benefits from these systems have profound implications for environmental justice and equity.

By the end of this week, you'll understand why GeoAI requires specialized approaches, you'll have diagnosed domain shift in your own experiments, and you'll have begun thinking critically about the power dynamics embedded in automated Earth observation.

---

## Core Content: The Evolution and Adaptation of Computer Vision

### The History of Computer Vision: From Pixels to Planetary Intelligence

Computer vision—the field dedicated to enabling machines to interpret visual information—has undergone several paradigm shifts over the past six decades. Understanding this evolution provides essential context for appreciating both the capabilities and limitations of modern approaches when applied to Earth observation.

#### The Foundational Era: Edge Detection and Geometric Features (1960s-1990s)

The origins of computer vision lie in attempts to extract meaningful structure from digital images through mathematical operations. Early researchers developed algorithms to detect fundamental visual features such as edges, corners, and lines. The **Sobel operator** (1968) and **Canny edge detector** (1986) became foundational techniques for identifying boundaries between objects by detecting rapid changes in pixel intensity. These algorithms transformed images from arrays of pixel values into representations of geometric structure.

The challenge facing early computer vision was that edge detection alone doesn't solve recognition problems. An image of a cat contains millions of edges, but which edges matter for recognizing "catness"? This led to the development of more sophisticated **feature descriptors**—mathematical representations designed to capture distinctive visual patterns that could be used for object recognition.

**Scale-Invariant Feature Transform (SIFT)**, developed by David Lowe in 1999, represented a major advance in feature engineering. SIFT identifies distinctive keypoints in images and describes them in ways that remain consistent despite changes in scale, rotation, or illumination. A SIFT descriptor captures the distribution of gradient orientations around a keypoint, creating a 128-dimensional vector that characterizes local image structure. SIFT features enabled reliable object recognition across different viewpoints and lighting conditions, making them widely used in applications from panorama stitching to 3D reconstruction.

**Histogram of Oriented Gradients (HOG)**, introduced by Dalal and Triggs in 2005, provided another influential approach to feature representation. HOG divides an image into small cells and computes histograms of gradient orientations within each cell, creating a representation that captures object shape while remaining robust to local variations. HOG features became particularly successful for pedestrian detection and other object recognition tasks.

This era of **hand-crafted features** shared a fundamental characteristic with the spectral indices you explored in Week 1: domain experts designed mathematical formulas to highlight specific patterns they believed were important. SIFT and HOG for computer vision, NDVI and NDWI for remote sensing—both represent human-designed features based on domain knowledge. And both face the same fundamental limitation: features that work well for one task or dataset often fail on others.

#### The Machine Learning Era: Learning Classifiers from Features (1990s-2012)

As feature extraction techniques matured, researchers increasingly focused on using machine learning to learn classifiers from these features. Rather than hand-coding rules for recognition ("if the object has pointy ears and whiskers, it's a cat"), machine learning approaches learned patterns from labeled examples.

**Support Vector Machines (SVMs)**, introduced by Cortes and Vapnik in 1995, became particularly influential. SVMs find optimal decision boundaries in high-dimensional feature spaces, effectively separating different object categories. Given SIFT or HOG features extracted from images, an SVM could learn to distinguish cats from dogs, cars from bicycles, or faces from non-faces. The combination of hand-crafted features (SIFT, HOG) with learned classifiers (SVM) became the dominant paradigm in computer vision for over a decade.

This approach achieved impressive results on many tasks, but it had inherent limitations. The features were still hand-crafted, requiring domain expertise and often task-specific tuning. Different applications required different features: SIFT for object matching, HOG for pedestrian detection, Haar wavelets for face detection. Each new problem required careful feature engineering, and there was no guarantee that the features humans designed would capture the patterns most relevant for the task.

The remote sensing community followed a similar trajectory. Traditional approaches combined hand-crafted spectral indices with machine learning classifiers. You might calculate NDVI, NDWI, NDBI, and various texture measures, then train an SVM or Random Forest classifier to distinguish land cover types. This approach works reasonably well but requires significant domain expertise and often task-specific feature engineering.

#### The Deep Learning Revolution: Learning Features from Data (2012-Present)

The 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) marked a watershed moment in computer vision history. A deep convolutional neural network called **AlexNet**, developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, achieved a top-5 error rate of 15.3%—dramatically outperforming the second-place entry (26.2% error) which used traditional hand-crafted features and classifiers.

This wasn't just an incremental improvement; it represented a paradigm shift. AlexNet didn't use hand-crafted features at all. Instead, it learned hierarchical feature representations directly from raw pixel values through a process called **backpropagation**. The network consisted of multiple layers of **convolutional filters**—small matrices of learnable parameters that slide across images to detect patterns. Early layers learned to detect simple patterns like edges and color blobs. Middle layers combined these into more complex shapes and textures. Deep layers recognized high-level object parts and ultimately entire objects.

The key insight was that **features don't need to be hand-crafted; they can be learned from data**. Given enough labeled examples and sufficient computational resources, neural networks could automatically discover the features most relevant for a task. This approach proved far more effective than decades of hand-crafted feature engineering.

Following AlexNet's success, the field rapidly developed increasingly sophisticated architectures:

**VGGNet** (2014) demonstrated that deeper networks with smaller filters could achieve better performance, introducing networks with 16-19 layers.

**GoogLeNet/Inception** (2014) introduced the concept of **inception modules**—parallel convolutional operations at multiple scales combined within a single layer—enabling networks to capture patterns at different spatial scales simultaneously.

**ResNet** (2015) introduced **residual connections**—skip connections that allow information to bypass layers—enabling the training of extremely deep networks with 50, 101, or even 152 layers. ResNet won the 2015 ImageNet challenge and became one of the most influential architectures in computer vision.

**DenseNet** (2017) extended the residual connection concept by connecting every layer to every other layer in a feed-forward fashion, improving feature reuse and gradient flow.

These architectures shared a common foundation: **convolutional neural networks (CNNs)**. Understanding how CNNs work is essential for this course, as they form the basis for most GeoAI approaches.

#### Understanding Convolutional Neural Networks: Architecture and Mechanisms

A convolutional neural network processes images through a series of layers, each performing specific operations that transform the input into increasingly abstract representations.

**Convolutional layers** apply learned filters to input images. Each filter is a small matrix (typically 3×3 or 5×5) of parameters that slides across the image, computing the dot product between the filter weights and the pixel values at each location. This operation detects specific patterns: a filter might learn to respond strongly to vertical edges, another to horizontal edges, another to specific color combinations. Early convolutional layers typically learn simple, general-purpose features like edges and color blobs. Deeper layers learn more complex, task-specific features like textures, object parts, and eventually entire objects.

**Activation functions** introduce non-linearity into the network, enabling it to learn complex patterns. The most common activation function is **ReLU (Rectified Linear Unit)**, which simply outputs the maximum of zero and the input value: ReLU(x) = max(0, x). This simple function has proven remarkably effective, helping networks train faster and achieve better performance than earlier activation functions like sigmoid or tanh.

**Pooling layers** reduce the spatial dimensions of feature maps, making the network more computationally efficient and helping it learn features that are robust to small spatial variations. **Max pooling**, the most common approach, divides the feature map into small regions (typically 2×2) and outputs the maximum value from each region. This reduces the feature map size by a factor of 4 while retaining the most prominent features.

**Fully connected layers** near the end of the network combine the learned features to make final predictions. These layers connect every neuron from the previous layer to every neuron in the current layer, enabling the network to learn complex combinations of features for classification.

**Normalization layers** (particularly **batch normalization**) standardize the inputs to each layer, stabilizing training and enabling the use of higher learning rates. Batch normalization has become a standard component of modern CNN architectures.

**Dropout layers** randomly set a fraction of neuron activations to zero during training, preventing the network from relying too heavily on any particular feature and reducing overfitting.

The training process for CNNs involves showing the network many labeled examples and adjusting the filter weights to minimize the difference between the network's predictions and the true labels. This is accomplished through **backpropagation**—an algorithm that computes gradients of the loss function with respect to each parameter—and **gradient descent**—an optimization algorithm that updates parameters in the direction that reduces the loss.

#### The ImageNet Dataset and Transfer Learning

The success of deep learning in computer vision has been inextricably linked to the **ImageNet dataset**. Created by Fei-Fei Li and colleagues, ImageNet contains over 14 million labeled images spanning more than 20,000 categories. The annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) used a subset of 1.2 million training images across 1,000 categories, creating a standardized benchmark that drove rapid progress in computer vision.

ImageNet's impact extends far beyond the competition itself. Models trained on ImageNet learn general-purpose visual features that transfer remarkably well to other tasks. A network trained to distinguish between 1,000 ImageNet categories learns to detect edges, textures, object parts, and spatial relationships—features useful for many visual recognition tasks.

This observation led to the paradigm of **transfer learning**: rather than training networks from scratch for each new task, we can start with a network pre-trained on ImageNet and fine-tune it for our specific application. This approach requires far less labeled data and computational resources than training from scratch, making deep learning accessible for applications where large labeled datasets aren't available.

Transfer learning has become the dominant paradigm in computer vision. Need to classify medical images? Start with an ImageNet-pretrained network. Want to detect objects in satellite imagery? Start with ImageNet pre-training. This approach has proven remarkably effective across diverse applications.

But there's a critical question: **What happens when the target domain differs fundamentally from ImageNet?** This brings us to the central challenge of this week: understanding what happens when models trained on photographs of everyday objects encounter satellite imagery of Earth's surface.

#### The Transformer Era and Foundation Models (2020-Present)

The most recent paradigm shift in computer vision has been the adaptation of **transformer architectures**—originally developed for natural language processing—to visual tasks. The **Vision Transformer (ViT)**, introduced by Dosovitskiy et al. in 2020, treats images as sequences of patches and processes them using the same self-attention mechanisms that revolutionized NLP.

Transformers differ fundamentally from CNNs in how they process images. Rather than using convolutional filters that examine local neighborhoods, transformers use **self-attention** mechanisms that can relate any part of the image to any other part, potentially capturing long-range dependencies that CNNs might miss. This global perspective can be particularly valuable for Earth observation, where the meaning of a pixel often depends on context from distant parts of the image.

The transformer revolution has enabled a new class of models called **foundation models**—massive models trained on enormous datasets (often billions of images) that learn general-purpose representations useful across many tasks. Examples include:

**CLIP (Contrastive Language-Image Pre-training)** learns to associate images with text descriptions, enabling zero-shot classification (classifying images into categories the model was never explicitly trained on) by comparing image features to text embeddings.

**SAM (Segment Anything Model)** learns to segment objects in images with minimal prompting, demonstrating remarkable generalization across diverse visual domains.

**DINOv2** uses self-supervised learning to create powerful visual representations without requiring labeled data, learning by predicting relationships between different views of the same image.

These foundation models represent the cutting edge of computer vision, and several have been specifically adapted for Earth observation (which we'll explore in Weeks 7-10). But to understand why domain-specific foundation models are necessary, we first need to understand the **domain shift problem**—what happens when general-purpose vision models encounter Earth observation data.

---

### The Unique Challenges of Earth Observation: Why Satellite Imagery is Different

Applying computer vision to Earth observation is not simply a matter of swapping datasets. Satellite imagery presents fundamental challenges that distinguish it from the photographs used to train conventional computer vision models. Understanding these challenges is essential for developing effective GeoAI approaches.

#### Challenge 1: Scale and Spatial Resolution

**Natural images**—the photographs that comprise ImageNet and similar datasets—typically capture scenes at human scale. A photograph of a cat might show the cat occupying hundreds or thousands of pixels, with individual whiskers, eyes, and fur textures clearly visible. The spatial resolution is fine enough to see details that humans use for recognition.

**Satellite imagery** captures Earth's surface from hundreds of kilometers away. The spatial resolution—the ground area represented by each pixel—varies by sensor but is typically 10-60 meters for freely available optical satellites like Sentinel-2 and Landsat. This creates several fundamental challenges:

**The mixed pixel problem**: A single 30-meter Landsat pixel might contain parts of a forest, a clearing, and a stream. Unlike a photograph where a pixel is clearly "cat" or "background," satellite pixels often represent mixtures of multiple land cover types. This fuzzy boundary between classes complicates classification and makes precise delineation of features challenging.

**Sub-pixel features**: For parcelización monitoring, individual parcels might be only 1-5 hectares (100m × 100m to 225m × 225m). At 30-meter resolution, a small parcel might be represented by only 10-50 pixels—far less detail than the hundreds or thousands of pixels representing objects in natural images. At 10-meter Sentinel-2 resolution, the situation improves but remains challenging for the smallest parcels.

**Scale-dependent meaning**: The same spectral signature might represent different land uses depending on spatial scale and context. A small patch of bare soil might be a construction site, agricultural field preparation, or natural erosion. Context from surrounding pixels is essential for interpretation, but this context operates at much larger spatial scales than in natural images.

**Spatial autocorrelation**: In satellite imagery, nearby pixels are highly correlated—forests cluster together, urban areas form connected regions, agricultural fields show regular geometric patterns. This spatial structure is far more pronounced than in natural images, where objects can appear anywhere in the frame. This autocorrelation has important implications for how we split data into training and testing sets (a topic we'll explore in Week 3).

#### Challenge 2: Spectral Richness vs. Visual Appearance

**Natural images** use three color channels—Red, Green, and Blue—that match human visual perception. ImageNet models are trained on RGB images and learn to recognize visual appearance: what things "look like" to human eyes. The features these models learn are fundamentally tied to visual appearance in the human-visible spectrum.

**Satellite imagery** typically provides 10 or more spectral bands spanning visible, near-infrared, shortwave infrared, and sometimes thermal wavelengths. As you learned in Week 1, many of the most informative bands for environmental monitoring fall outside the visible spectrum:

**Near-infrared (NIR)** reflectance is essential for vegetation monitoring. Healthy vegetation strongly reflects NIR radiation due to internal leaf structure, creating the distinctive spectral signature that makes NDVI such a powerful index. But NIR is invisible to humans and absent from natural image datasets.

**Shortwave infrared (SWIR)** bands are sensitive to water content in vegetation and soil, making them valuable for drought monitoring and fire detection. Again, these wavelengths are invisible to humans and not represented in natural images.

**Thermal infrared** bands measure surface temperature, providing information about energy balance, evapotranspiration, and urban heat islands. This information has no analog in visual appearance.

This creates a fundamental mismatch: **ImageNet models expect 3-channel RGB input and have learned features based on visual appearance. Satellite imagery provides 10+ channels of spectral information, with the most informative bands often being invisible to humans.**

Researchers have developed several approaches to address this mismatch:

**Band selection**: Choose three spectral bands to map to RGB channels. This might be true color (Red, Green, Blue), false color (NIR, Red, Green), or other combinations. But this approach discards most of the spectral information available.

**Index calculation**: Calculate spectral indices (NDVI, NDWI, etc.) and use these as additional input channels. But this reintroduces hand-crafted features, partially defeating the purpose of deep learning.

**Multi-channel adaptation**: Modify the network architecture to accept multi-channel input. This requires retraining or fine-tuning and may not fully leverage pre-trained features.

**Separate processing**: Process different spectral bands or indices through separate network branches and combine the results. This adds architectural complexity.

None of these solutions is perfect, and the spectral richness vs. visual appearance challenge remains a fundamental issue in applying computer vision to Earth observation.

#### Challenge 3: Temporal Dynamics and Change Detection

**Natural images** are typically single snapshots capturing a moment in time. While video analysis exists, most computer vision models process individual frames independently. ImageNet contains photographs, not time series.

**Earth observation** is fundamentally temporal. Environmental processes unfold over days, seasons, years, and decades:

**Phenological cycles**: Vegetation goes through seasonal cycles of growth, senescence, and dormancy. Agricultural crops are planted, grow, and are harvested on annual cycles. These temporal patterns are essential for distinguishing crop types and monitoring agricultural practices.

**Land-use change**: Parcelización occurs gradually as agricultural land is subdivided and converted to residential use. Deforestation, urban expansion, and agricultural intensification all unfold over months to years. Detecting these changes requires comparing observations across time.

**Environmental variability**: Drought impacts accumulate over multiple seasons. Algal blooms appear and disappear within weeks. Flooding events are ephemeral. Understanding these processes requires temporal context.

**Seasonal confounds**: A deciduous forest in winter might have similar spectral characteristics to grassland or bare soil. Agricultural fields show dramatic spectral changes throughout the growing season. Without temporal context, these seasonal variations can be confused with land-use change.

This temporal dimension creates several challenges for applying computer vision models:

**Single-image models miss temporal patterns**: A model trained on individual photographs has no concept of temporal change. It might classify a forest in winter as bare soil, unable to recognize that the same location will be green in summer.

**Temporal resolution requirements**: Different processes require different temporal resolutions. Crop type mapping might require observations every 5-10 days throughout the growing season. Parcelización monitoring might require annual or seasonal observations over multiple years. Flood mapping requires observations within hours or days of an event.

**Data volume**: Analyzing time series multiplies data volume. A single Sentinel-2 scene might be 1-2 GB. A time series covering a region for a year might be hundreds of GB. Processing multi-year time series requires substantial computational resources.

We'll explore approaches to temporal analysis in Weeks 8-9, including recurrent neural networks, temporal convolutions, and attention mechanisms. But for this week, it's important to understand that the temporal dimension represents a fundamental difference between natural images and Earth observation data.

#### Challenge 4: Geographic Context and Spatial Relationships

**Natural images** show objects that can appear anywhere. A cat looks like a cat whether photographed in Tokyo, Toronto, or Timbuktu. The background might vary, but the object's appearance is largely independent of geographic location.

**Geospatial features** are inherently spatial and contextual:

**Spatial autocorrelation**: As mentioned earlier, nearby locations tend to have similar characteristics. Forests cluster together. Urban areas form connected regions. Rivers form networks. This spatial structure is far more pronounced than in natural images.

**Geographic context**: The meaning of spectral signatures depends on geographic context. The same NDVI value might represent healthy grassland in a semi-arid region or stressed vegetation in a humid tropical region. Elevation, climate, and biogeography all influence what spectral signatures mean.

**Spatial relationships**: The relationship between features matters. A small clearing in a forest might be natural or human-caused depending on its shape, size, and relationship to roads or other infrastructure. A bright spectral signature might be a beach, a salt flat, or a snow-covered field depending on location and surrounding features.

**Coordinate information**: Geographic coordinates themselves carry information. Latitude influences solar angle and day length. Elevation affects temperature and vegetation. Distance from coast influences climate. These geographic variables can improve classification but aren't present in natural images.

This geographic context creates both challenges and opportunities:

**Challenge**: Models might learn to recognize locations rather than features. If all your forest training examples come from one region and all your agricultural examples from another, the model might learn geographic patterns rather than land-cover characteristics.

**Challenge**: Train/test splits must account for spatial autocorrelation. Random splits can leak information if nearby pixels end up in both training and test sets.

**Opportunity**: Incorporating geographic context (coordinates, elevation, climate variables) as additional input features can improve classification.

**Opportunity**: Spatial relationships can be explicitly modeled using graph neural networks or attention mechanisms that consider relationships between pixels.

#### Challenge 5: Data Geography and Representation Bias

This challenge bridges technical and ethical dimensions, making it particularly important for the Ethics Thread you'll launch this week.

**ImageNet and similar datasets** are collected from the internet, reflecting the geographic distribution of internet users, photographers, and image sharing. Research has shown that ImageNet is heavily biased toward:

**Wealthy nations**: North America and Western Europe are over-represented. Africa, South America, and parts of Asia are under-represented.

**Urban areas**: Cities are over-represented compared to rural areas.

**Western cultural contexts**: Object categories, visual styles, and contexts reflect Western perspectives.

**Satellite imagery** provides global coverage—every location on Earth is observed with similar frequency (at least for freely available satellites like Landsat and Sentinel). But **ground truth labels**—the training data that tells models what they're looking at—show extreme geographic bias:

**Label availability**: High-quality land-cover labels are abundant for North America and Western Europe but sparse for much of Africa, South America, and Asia. This reflects research funding, institutional capacity, and priorities.

**Classification systems**: Land-cover classification schemes (urban, agricultural, forest, etc.) reflect Western land-use categories and may not capture indigenous land management, agroforestry systems, or other practices common in the Global South.

**Temporal bias**: Historical labels may not exist for many regions, making it difficult to study long-term land-use change outside well-studied areas.

**Resolution bias**: High-resolution commercial imagery is expensive, creating a data divide where wealthy organizations can see more clearly than others.

This data geography has profound implications:

**Model performance varies geographically**: Models trained primarily on Northern Hemisphere data may perform poorly in the Southern Hemisphere, where seasons are reversed, ecosystems differ, and land-use patterns vary.

**Representation and power**: Who gets observed, how they're classified, and who controls the classification systems are questions of power. Earth observation can be a tool for conservation and climate monitoring, but also for resource extraction, surveillance, and control.

**Equity and access**: If GeoAI tools work well for wealthy regions but poorly for the Global South, they may exacerbate rather than reduce global inequalities.

These issues motivate the Ethics Thread and the emphasis on responsible AI throughout this course. Technical choices about data, models, and evaluation have ethical implications that we must consider explicitly.

---

### The Domain Shift Problem: When ImageNet Meets Earth Observation

Now that we understand both the evolution of computer vision and the unique challenges of Earth observation, we can examine what happens when these two domains meet. This is the core of this week's hands-on experiments.

#### What is Domain Shift?

**Domain shift** occurs when a model's training distribution differs from its deployment distribution. A model trained on ImageNet has learned patterns from millions of photographs of everyday objects. When we apply this model to satellite imagery, we're asking it to make predictions on a fundamentally different type of data.

The model hasn't learned to recognize forests, agricultural fields, or water bodies. It has learned to recognize cats, cars, and buildings as they appear in photographs. When it encounters satellite imagery, it tries to match what it sees to the patterns it knows, often with nonsensical results.

#### Experimental Evidence: What ImageNet Models See in Satellite Imagery

Research and practical experience have documented numerous examples of domain shift when applying ImageNet models to Earth observation:

**Geometric pattern confusion**: Agricultural fields with regular geometric patterns are often classified as "tennis courts," "baseball diamonds," or "crossword puzzles." The model has learned that regular geometric patterns in ImageNet often correspond to human-made recreational facilities or objects.

**Texture similarity**: Forests viewed from above are sometimes classified as "broccoli" or "cauliflower" because the texture of tree canopies at certain scales resembles these vegetables in photographs. The model has learned texture patterns but not environmental meaning.

**Linear feature confusion**: Rivers and roads are sometimes confused because both appear as linear features in imagery. The model has learned to recognize linear structures but not to distinguish between water and pavement based on spectral properties.

**Color-based errors**: Urban areas with regular patterns and neutral colors might be classified as "quilts," "mosaics," or "honeycomb" based on visual appearance rather than land-use function.

These failures aren't random. They reveal what ImageNet models have actually learned: **visual patterns (geometry, texture, color) rather than environmental meaning (land cover, ecosystem function, human use)**.

#### Why Domain Shift Occurs: Mismatched Feature Hierarchies

ImageNet models learn hierarchical features optimized for recognizing everyday objects in photographs:

**Early layers** learn edges, colors, and simple textures that are general-purpose and transfer reasonably well to satellite imagery.

**Middle layers** learn object parts, complex textures, and spatial patterns specific to ImageNet categories. These features may or may not be relevant for Earth observation.

**Deep layers** learn high-level object representations (cat faces, car wheels, building facades) that are highly specific to ImageNet and largely irrelevant for satellite imagery.

When we apply an ImageNet model to satellite imagery, the early layers provide some useful features (edges, basic textures), but the deeper layers are trying to recognize objects that don't exist in satellite views. The model is fundamentally solving the wrong problem.

#### Implications for GeoAI

The domain shift problem has several important implications:

**Pre-training helps but isn't sufficient**: ImageNet pre-training provides useful low-level features, but the high-level features need to be adapted or relearned for Earth observation tasks.

**Domain-specific training is necessary**: Effective GeoAI requires training on satellite imagery, not just fine-tuning ImageNet models.

**Architecture matters**: Some architectural choices that work well for ImageNet (like certain pooling strategies or receptive field sizes) may not be optimal for satellite imagery.

**Evaluation is critical**: We need to evaluate models on satellite imagery, not assume that ImageNet performance predicts Earth observation performance.

This motivates the progression of this course:
- **Week 3**: Train CNNs specifically on satellite imagery
- **Weeks 4-6**: Explore transfer learning and domain adaptation approaches
- **Weeks 7-10**: Use foundation models trained specifically on Earth observation data

---

### The Ethics of Earth Observation AI: Power, Representation, and Responsibility

This section introduces the ethical framework that will guide the Ethics Thread throughout the course. Earth observation AI is not ethically neutral; it embodies choices about what to observe, how to classify, and who benefits.

#### The Extractive View: Earth Observation as Resource Extraction

Kate Crawford's concept of the **"extractive view"** provides a powerful framework for understanding the politics of Earth observation. Crawford argues that AI systems, including Earth observation, often treat landscapes and communities as resources to be measured, classified, and optimized without their consent or participation.

Earth observation has historically been tied to resource extraction and control:

**Colonial mapping**: Satellite imagery's predecessors—aerial photography and cartography—were tools of colonial administration, making territories "legible" for resource extraction and control.

**Military origins**: Many Earth observation satellites were originally developed for military reconnaissance. The "dual-use" nature of Earth observation technology means it serves both civilian and military purposes.

**Resource monitoring**: Earth observation is used to identify mineral deposits, assess timber resources, monitor agricultural productivity—all potentially serving extractive industries.

**Surveillance**: Earth observation can monitor borders, track population movements, and identify informal settlements—potentially enabling displacement or control.

This history doesn't mean Earth observation is inherently problematic, but it does mean we must be conscious of how these systems can be used and who they serve.

#### Data Geography: Who Sees, Who is Seen?

Earth observation is shaped by power at every stage:

**Satellite deployment**: Which satellites are launched, where they orbit, what sensors they carry, and what they observe are decisions made primarily by wealthy nations and corporations. The constellation of Earth observation satellites reflects the priorities of these actors.

**Data access**: While Landsat and Sentinel data are freely available (thanks to deliberate policy choices), high-resolution commercial imagery costs thousands to millions of dollars. This creates a **data divide** where wealthy organizations can see more clearly than others.

**Processing capacity**: Analyzing satellite imagery at scale requires computational resources and technical expertise that are unevenly distributed globally. Google Earth Engine democratizes access to some extent, but significant barriers remain.

**Ground truth collection**: Creating training data requires field visits, local knowledge, and resources. The geographic bias in available training data reflects global inequalities in research funding and capacity.

**Interpretation and classification**: The categories we use to classify land cover (urban, agricultural, forest, wetland) reflect particular worldviews and may not match how local communities understand their landscapes. Indigenous land management practices, agroforestry systems, and other approaches may be invisible or misclassified.

#### Responsible GeoAI: Principles and Practices

How do we practice responsible Earth observation AI? Several frameworks and principles can guide our work:

**1. Transparency and Documentation**

- Clearly document data sources, including their limitations and biases
- Describe model architectures, training procedures, and evaluation metrics
- Acknowledge uncertainty and limitations in results
- Make code and methods available when possible (respecting privacy and security concerns)

**2. Interpretability and Explainability**

- Ensure models provide explanations, not just predictions
- Use visualization techniques (like Grad-CAM) to understand what models are responding to
- Validate model predictions against domain knowledge and ground truth
- Be skeptical of "black box" predictions, especially for high-stakes decisions

**3. Accountability and Governance**

- Identify who is responsible when models make errors or cause harm
- Establish processes for reviewing and correcting errors
- Consider governance structures that include affected communities
- Be prepared to withdraw or modify systems that cause harm

**4. Participation and Co-Design**

- Involve affected communities in defining problems and interpreting results
- Recognize local and indigenous knowledge as valid and valuable
- Design systems that empower rather than displace human decision-making
- Share benefits of Earth observation with observed communities

**5. Equity and Justice**

- Consider how Earth observation systems might exacerbate or reduce inequalities
- Ensure systems work well across diverse geographic and social contexts
- Prioritize applications that serve public good over private profit
- Be attentive to how classification systems might stigmatize or marginalize

**6. Environmental Responsibility**

- Consider the environmental costs of computation (energy use, carbon emissions)
- Balance model complexity against environmental impact
- Prioritize efficiency and sustainability in system design

**7. Cultural Sensitivity and Humility**

- Recognize that land-use categories and environmental values vary across cultures
- Avoid imposing Western classification systems on non-Western contexts
- Be humble about the limits of remote observation
- Respect privacy and sacred sites

These principles aren't abstract ideals; they should guide concrete choices throughout your GeoAI work. In your capstone project, you'll document how you've applied these principles.

#### Ethics Thread: Continuous Reflection

The **Ethics Thread** is a continuous reflection on responsible AI that will run throughout this course. Each week, you'll consider ethical dimensions of the technical approaches you're learning:

- **Week 2**: Data geography and domain shift
- **Week 3**: Training data collection and representation
- **Week 4**: Transfer learning and model fairness
- **Week 6**: Interpretability and accountability
- **Week 10**: Foundation models and environmental justice
- **Week 12**: Synthesis and responsible practice

This isn't a separate "ethics module" but an integrated thread connecting technical choices to societal implications. The goal is to develop a habit of ethical reflection that continues beyond this course.

---

## Guided Examples and Demonstrations

### Experiment 1: Loading and Understanding Pre-Trained Models

Before we apply ImageNet models to satellite imagery, let's understand what these models are and how they work.

#### Understanding ResNet-50 Architecture

ResNet-50 is one of the most influential CNN architectures, introduced by He et al. in 2015. The "50" refers to the depth: 50 layers of computations. The key innovation of ResNet is **residual connections**—skip connections that allow information to bypass layers.

Without residual connections, very deep networks suffer from the **vanishing gradient problem**: during backpropagation, gradients become smaller as they propagate backward through layers, making it difficult to train very deep networks. Residual connections address this by allowing gradients to flow directly through the network via skip connections.

ResNet-50's architecture consists of:
- **Initial convolutional layer**: 7×7 convolution with 64 filters, stride 2
- **Max pooling layer**: 3×3 pooling, stride 2
- **Residual blocks**: Four stages with 3, 4, 6, and 3 blocks respectively
- **Global average pooling**: Reduces spatial dimensions to 1×1
- **Fully connected layer**: 1000 outputs for ImageNet classes

The model has approximately 25.6 million parameters, all learned from ImageNet training data.

#### Loading a Pre-Trained Model

```python
import torch
import torchvision.models as models

# Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode (disables dropout, batch norm training)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

When you run this code, PyTorch downloads the pre-trained weights (approximately 98 MB) and loads them into the model architecture. These weights represent the patterns the model learned from 1.2 million ImageNet training images.

#### Understanding ImageNet Categories

ImageNet's 1000 categories span a wide range of everyday objects:
- Animals: 398 categories (dogs, cats, birds, fish, insects, etc.)
- Vehicles: 47 categories (cars, trucks, bicycles, aircraft, etc.)
- Household items: 150+ categories (furniture, appliances, tools, etc.)
- Food: 89 categories (fruits, vegetables, dishes, etc.)
- Natural objects: 50+ categories (plants, geological formations, etc.)
- Structures: 40+ categories (buildings, bridges, monuments, etc.)

Notably absent are categories directly relevant to Earth observation: "agricultural field," "forest," "parcelized land," "water body," etc. The closest categories might be "seashore," "lakeside," or "valley," but these describe scenic views, not land-cover types.

This mismatch between ImageNet categories and Earth observation needs is a fundamental source of domain shift.

#### Preprocessing Requirements

ImageNet models expect specific input preprocessing:

**1. Resize to 224×224 pixels**: ImageNet training images were resized to this size. The model's architecture expects this input size.

**2. Normalize with ImageNet statistics**: Subtract the mean and divide by standard deviation computed across the ImageNet training set:
- Mean: [0.485, 0.456, 0.406] for R, G, B channels
- Std: [0.229, 0.224, 0.225] for R, G, B channels

**3. Convert to tensor**: PyTorch expects input as tensors with shape (batch_size, channels, height, width).

This preprocessing ensures that the input distribution matches what the model saw during training. When we apply these models to satellite imagery, we must apply the same preprocessing, even though satellite imagery has different statistical properties. This preprocessing mismatch contributes to domain shift.

---

### Experiment 2: Applying ImageNet Models to Satellite Imagery

Now we'll systematically test how ImageNet models perform on satellite imagery from your study region.

#### Experimental Design

To rigorously evaluate domain shift, we need:

**1. Representative samples**: Select locations representing different land cover types in your study region (forest, agriculture, water, urban, bare soil).

**2. Consistent methodology**: Use the same preprocessing and inference procedure for all samples.

**3. Systematic documentation**: Record predictions, confidence scores, and observations for each sample.

**4. Visualization**: Create activation maps to understand what the model is responding to.

#### Sample Selection Strategy

For each land cover type, select locations that are:
- **Representative**: Typical examples of that land cover
- **Unambiguous**: Clear examples without mixed pixels
- **Accessible**: Locations where you can verify land cover (using high-resolution imagery or local knowledge)
- **Distributed**: Spread across your study region to capture geographic variability

#### Running Inference

The inference process involves several steps:

**1. Download satellite imagery patch**: Use Earth Engine to download a small region (500m × 500m) around each sample point.

**2. Convert to RGB**: Select appropriate bands for RGB visualization. For Sentinel-2, this typically means bands B4 (Red), B3 (Green), B2 (Blue).

**3. Normalize to 0-255**: Satellite imagery reflectance values typically range from 0-10000. Convert to 0-255 for visualization and model input.

**4. Preprocess for model**: Resize to 224×224, normalize with ImageNet statistics, convert to tensor.

**5. Run model inference**: Pass through the network to get predictions.

**6. Interpret results**: Examine top predictions and confidence scores.

#### Interpreting Predictions

When examining predictions, consider:

**Confidence scores**: High confidence in wrong predictions indicates the model is confidently applying patterns that don't transfer to satellite imagery.

**Top-5 predictions**: Sometimes the correct or related category appears in top-5 even if not top-1.

**Patterns in errors**: Are certain land cover types consistently misclassified? Do errors show patterns (e.g., geometric features confused with human-made objects)?

**Activation patterns**: What parts of the image is the model responding to?

---

### Experiment 3: Activation Visualization with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) helps us visualize which parts of an image contribute most to a model's prediction. This technique computes gradients of the predicted class score with respect to feature maps in the final convolutional layer, creating a heatmap showing which regions influenced the prediction.

#### How Grad-CAM Works

1. **Forward pass**: Run the image through the network and get the prediction.

2. **Backward pass**: Compute gradients of the predicted class score with respect to the final convolutional layer's feature maps.

3. **Weight feature maps**: Weight each feature map by its corresponding gradient, indicating its importance for the prediction.

4. **Combine and normalize**: Sum weighted feature maps and apply ReLU (to focus on positive contributions), creating a heatmap.

5. **Upsample**: Resize the heatmap to match the input image size.

6. **Overlay**: Superimpose the heatmap on the original image to visualize which regions influenced the prediction.

#### Interpreting Activation Maps

When examining Grad-CAM visualizations on satellite imagery:

**Geometric patterns**: Does the model respond to field boundaries, road networks, or other geometric features?

**Texture**: Is the model responding to texture patterns (like forest canopy texture)?

**Edges**: Does the model focus on edges between different land cover types?

**Irrelevant features**: Is the model responding to clouds, shadows, or other artifacts?

Comparing activation maps between correct and incorrect predictions can reveal what features the model has learned and why it fails on satellite imagery.

---

## Hands-on Activities

### Activity 1: ImageNet to GeoVision Experiment (90 minutes)

**Objective**: Systematically test how ImageNet-pretrained models perform on satellite imagery from your study region.

**Learning Goals**:
- Understand the domain shift problem through direct experimentation
- Practice loading and using pre-trained models
- Develop skills in systematic evaluation and documentation

**Tasks**:

1. **Setup** (15 minutes)
   - Load ResNet-50 pre-trained model
   - Load ImageNet class labels
   - Initialize Earth Engine and load your AOI

2. **Sample Selection** (15 minutes)
   - Identify 5-7 locations representing different land cover types
   - Verify land cover using high-resolution imagery or local knowledge
   - Document coordinates and land cover types

3. **Data Collection** (20 minutes)
   - Download Sentinel-2 patches for each location
   - Create RGB composites
   - Save patches for later analysis

4. **Inference** (20 minutes)
   - Preprocess patches for ImageNet model
   - Run inference for each patch
   - Record top-5 predictions and confidence scores

5. **Visualization** (15 minutes)
   - Create multi-panel figure showing patches and predictions
   - Generate activation maps for 2-3 examples
   - Save all visualizations

6. **Documentation** (15 minutes)
   - Create summary table of results
   - Document observations and patterns
   - Note surprising or interesting findings

**Deliverables**:
- Completed Jupyter notebook: `notebooks/Week2_ImageNet_to_GeoVision.ipynb`
- Prediction visualization: `figures/week2_imagenet_predictions.png`
- Activation maps: `figures/week2_activation_*.png`
- Results table: `reports/week2_prediction_summary.csv`

**Self-Assessment Questions**:
- Did you test at least 5 different land cover types?
- Are your visualizations clear and well-labeled?
- Did you document your methodology and observations?
- Can someone else reproduce your experiment from your notebook?

---

### Activity 2: Domain Shift Analysis (60 minutes)

**Objective**: Write a structured analysis explaining the domain shift problem based on your experimental results.

**Learning Goals**:
- Develop skills in technical writing and analysis
- Connect experimental observations to theoretical concepts
- Practice explaining complex ideas clearly

**Structure**:

**Introduction** (100 words)
- Brief overview of the experiment
- Statement of the domain shift problem
- Preview of key findings

**Observations** (150-200 words)
- Summarize what you observed in your experiments
- Include specific examples of misclassifications
- Describe patterns in model failures
- Reference your visualizations

**Analysis** (250-300 words)
- Explain *why* these misclassifications occurred
- Connect to concepts from the study guide:
  - Scale mismatch
  - Spectral vs. visual information
  - Feature hierarchies learned from ImageNet
  - Geographic context
- Use specific examples to illustrate each point

**Implications** (150-200 words)
- What does this mean for applying computer vision to Earth observation?
- What approaches might address these challenges?
- How does this motivate the rest of the course?

**Conclusion** (50-100 words)
- Summarize key insights
- Connect to your case study

**Deliverable**:
- Analysis document: `reports/Week2_Domain_Shift_Analysis.md` (600-800 words)

**Self-Assessment Questions**:
- Does your analysis clearly explain *why* failures occurred, not just *what* failed?
- Have you connected observations to theoretical concepts?
- Is your writing clear and well-organized?
- Have you included specific examples to support your points?

---

### Activity 3: Ethics Thread Launch (45 minutes)

**Objective**: Launch the Ethics Thread with a critical reflection on power, representation, and responsibility in Earth observation AI.

**Learning Goals**:
- Engage critically with ethics literature
- Connect abstract ethical concepts to concrete technical choices
- Develop a practice of ethical reflection

**Required Reading**:
- Crawford, K. (2021). *Atlas of AI*, Chapter 1: "Earth" (or alternative: UNESCO AI Ethics Recommendation, Section 2)

**Reflection Prompt**:

Write 400-600 words addressing:

**1. Power and Representation** (150-200 words)
- Who decides what gets observed in Earth observation?
- How are classification systems chosen, and whose perspectives do they reflect?
- What power dynamics are embedded in automated Earth observation?
- Connect to specific examples from the reading

**2. Your Case Study** (150-200 words)
- How might the ethical issues raised in the reading apply to your chosen case study?
- For parcelización: Who benefits from monitoring? Who might be harmed?
- For megadrought: How might monitoring inform policy? Who has access to results?
- For lake ecosystem: What interests are served by water quality monitoring?

**3. Responsible Practice** (100-200 words)
- What specific practices will you adopt to ensure your GeoAI work is responsible?
- How will you ensure transparency, interpretability, and accountability?
- How will you consider equity and justice in your work?
- What challenges do you anticipate in practicing responsible GeoAI?

**Deliverable**:
- Ethics Thread post: `reports/Week2_Ethics_Thread.md`

**Self-Assessment Questions**:
- Have you engaged substantively with the reading?
- Have you made concrete connections to your case study?
- Have you identified specific, actionable responsible practices?
- Does your reflection demonstrate critical thinking about power and representation?

---

## Reflection and Discussion (30 minutes)

### Key Questions for Reflection

1. **Conceptual Integration**: How does this week's exploration of computer vision connect to Week 1's spectral analysis? What are the complementary strengths of hand-crafted indices vs. learned features?

2. **Technical Insight**: What surprised you most about how ImageNet models performed on satellite imagery? What specific failures were most revealing about what these models have learned?

3. **Methodological Understanding**: Why is it important to test models systematically across different land cover types rather than just running a few examples? What did systematic testing reveal that ad-hoc testing might have missed?

4. **Ethical Awareness**: How has this week changed your thinking about the "objectivity" of satellite imagery and AI-based Earth observation? What responsibilities arise from automating environmental monitoring?

5. **Future Directions**: Based on this week's experiments, what do you think will be most important for building effective GeoAI models in coming weeks? What challenges do you anticipate?

### Self-Reflection Prompt

Write 300-400 words in `reports/Week2_Reflection.md` addressing:

- **Most significant insight**: What was the most important thing you learned this week? This might be technical, conceptual, or ethical.

- **Conceptual evolution**: How did your understanding of AI and Earth observation change from the beginning to the end of the week?

- **Ethical awareness**: What ethical questions or concerns arose for you? How do they relate to your case study?

- **Looking forward**: What are you most curious to explore in Week 3 as you begin building your own CNN for Earth observation?

---

## Preview of Week 3: CNNs for Landscapes

Next week, you'll build on this foundation by training your first CNN specifically for Earth observation:

### What's Coming

**CNN Architecture Fundamentals**: Deep dive into how convolutional neural networks process images, including convolution operations, pooling, activation functions, and training procedures.

**Data Preparation**: Learn to create training datasets for land-cover classification, including image tiling, label encoding, data augmentation, and train/test splitting that accounts for spatial autocorrelation.

**Training Your First Model**: Build and train a CNN for land-cover classification on Sentinel-2 imagery from your study region, learning to monitor training progress, diagnose overfitting, and tune hyperparameters.

**Interpretability**: Visualize what your model learns through filter visualization, activation maps, and feature space analysis, understanding how learned features differ from ImageNet features.

**Evaluation**: Compute accuracy metrics appropriate for geospatial classification (overall accuracy, per-class accuracy, IoU, F1 scores), create confusion matrices, and analyze error patterns.

**Documentation**: Create reproducible training pipelines with clear documentation, establishing the foundation for your capstone project.

### Preparation for Week 3

To prepare for next week:

**1. Review neural network basics** (30-45 minutes)
- Forward propagation: how networks make predictions
- Backpropagation: how networks learn from errors
- Gradient descent: how parameters are updated
- Resources: [3Blue1Brown Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

**2. Familiarize with PyTorch basics** (45-60 minutes)
- Tensors and basic operations
- Building simple networks with nn.Module
- Training loops
- Resources: [PyTorch tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)

**3. Define land-cover classes** (30 minutes)
- Identify 4-6 land-cover classes you want to classify in your study region
- Consider: What classes are most relevant for your case study?
- Think about: How will you collect training data for each class?

**4. Conceptual preparation** (15 minutes)
- Reflect: What would "success" look like for a land-cover classification model?
- Consider: How will you evaluate whether your model is working well?
- Think about: What errors would be most problematic for your application?

---

## Additional Resources

### Computer Vision History and Foundations

**Textbooks and Comprehensive Resources**:

- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.
  - Free online: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
  - Chapters 5 (Machine Learning Basics), 6 (Deep Feedforward Networks), 9 (Convolutional Networks)
  - The definitive textbook on deep learning fundamentals

- **Prince, S. J. D. (2023).** *Understanding Deep Learning*. MIT Press.
  - Free online: [https://udlbook.github.io/udlbook/](https://udlbook.github.io/udlbook/)
  - Modern, accessible introduction with excellent visualizations

**Seminal Papers**:

- **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).** ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*.
  - The AlexNet paper that started the deep learning revolution
  - [https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

- **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep Residual Learning for Image Recognition. *CVPR*.
  - The ResNet paper introducing residual connections
  - [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

- **Dosovitskiy, A., et al. (2021).** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
  - Vision Transformers (ViT) paper
  - [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

### GeoAI and Remote Sensing with Deep Learning

**Review Papers**:

- **Zhu, X. X., et al. (2017).** Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources. *IEEE Geoscience and Remote Sensing Magazine*, 5(4), 8-36.
  - Comprehensive overview of deep learning applications in remote sensing
  - [https://doi.org/10.1109/MGRS.2017.2762307](https://doi.org/10.1109/MGRS.2017.2762307)

- **Reichstein, M., et al. (2019).** Deep learning and process understanding for data-driven Earth system science. *Nature*, 566, 195-204.
  - Connecting AI to Earth system science and environmental processes
  - [https://doi.org/10.1038/s41586-019-0912-1](https://doi.org/10.1038/s41586-019-0912-1)

- **Yuan, Q., et al. (2020).** Deep learning in environmental remote sensing: Achievements and challenges. *Remote Sensing of Environment*, 241, 111716.
  - Recent advances and remaining challenges
  - [https://doi.org/10.1016/j.rse.2020.111716](https://doi.org/10.1016/j.rse.2020.111716)

**Application Papers**:

- **Rolf, E., et al. (2021).** A generalizable and accessible approach to machine learning with global satellite imagery. *Nature Communications*, 12, 4392.
  - Practical approaches to global-scale machine learning with satellite imagery
  - [https://doi.org/10.1038/s41467-021-24638-z](https://doi.org/10.1038/s41467-021-24638-z)

- **Robinson, C., et al. (2019).** Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data. *CVPR*.
  - Combining multiple data sources for land-cover mapping
  - [https://arxiv.org/abs/1906.03739](https://arxiv.org/abs/1906.03739)

### Ethics and Responsible AI

**Books**:

- **Crawford, K. (2021).** *Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence*. Yale University Press.
  - Chapter 1: "Earth" (required reading)
  - Critical examination of AI's environmental and social impacts

- **Benjamin, R. (2019).** *Race After Technology: Abolitionist Tools for the New Jim Code*. Polity Press.
  - How technology can reinforce inequality
  - Relevant for thinking about representation bias

**Policy and Frameworks**:

- **UNESCO (2023).** *Recommendation on the Ethics of Artificial Intelligence*.
  - International framework for ethical AI
  - [https://unesdoc.unesco.org/ark:/48223/pf0000381137](https://unesdoc.unesco.org/ark:/48223/pf0000381137)

- **Gebru, T., et al. (2021).** Datasheets for Datasets. *Communications of the ACM*, 64(12), 86-92.
  - Framework for documenting datasets (we'll use this in later weeks)
  - [https://arxiv.org/abs/1803.09010](https://arxiv.org/abs/1803.09010)

- **Mitchell, M., et al. (2019).** Model Cards for Model Reporting. *FAT\**.
  - Framework for documenting models
  - [https://arxiv.org/abs/1810.03993](https://arxiv.org/abs/1810.03993)

**Research Papers**:

- **Hanna, A., et al. (2020).** Towards a Critical Race Methodology in Algorithmic Fairness. *FAT\**.
  - Critical perspectives on algorithmic fairness
  - [https://arxiv.org/abs/1912.03593](https://arxiv.org/abs/1912.03593)

### Interactive Tutorials and Tools

**PyTorch Resources**:

- **PyTorch Tutorials**: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
  - Especially: "Transfer Learning for Computer Vision Tutorial"
  - "Training a Classifier" tutorial

- **TorchVision Models Documentation**: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
  - Complete documentation for pre-trained models

**Visualization Tools**:

- **Grad-CAM PyTorch**: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
  - Advanced activation visualization library

- **TensorBoard**: [https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
  - Visualizing training progress

**Educational Visualizations**:

- **CNN Explainer**: [https://poloclub.github.io/cnn-explainer/](https://poloclub.github.io/cnn-explainer/)
  - Interactive visualization of how CNNs work

- **Distill.pub**: [https://distill.pub/](https://distill.pub/)
  - Visual explanations of machine learning concepts
  - Especially: "Feature Visualization" and "Building Blocks of Interpretability"

---

## Glossary Terms Introduced This Week

- **Computer Vision**: Field of artificial intelligence focused on enabling machines to interpret and understand visual information from images and videos

- **Feature Engineering**: Manual design of algorithms and mathematical transformations to extract meaningful patterns from raw data

- **Convolutional Neural Network (CNN)**: Neural network architecture specialized for processing grid-like data (images) using convolutional operations that detect spatial patterns

- **Transfer Learning**: Technique of adapting a model trained on one task/dataset to a different but related task/dataset, leveraging learned features

- **ImageNet**: Large-scale dataset of 14+ million labeled photographs spanning 20,000+ categories, used to train and evaluate computer vision models

- **Domain Shift**: Phenomenon where a model's training distribution differs from its deployment distribution, causing performance degradation

- **Mixed Pixel Problem**: Challenge in satellite imagery where a single pixel contains multiple land cover types due to coarse spatial resolution

- **Spatial Autocorrelation**: Statistical property where nearby locations tend to have similar values, pronounced in geospatial data

- **Data Geography**: Geographic distribution and biases in training datasets, reflecting uneven data collection and labeling efforts

- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Visualization technique showing which image regions most influence a model's predictions

- **Foundation Model**: Large model trained on massive datasets that learns general-purpose representations adaptable to many downstream tasks

- **Zero-shot Learning**: Capability of models to perform tasks they weren't explicitly trained for, often by leveraging learned representations

- **Self-Supervised Learning**: Training approach where models learn from unlabeled data by predicting relationships within the data itself

- **Residual Connection**: Skip connection in neural networks allowing information to bypass layers, enabling training of very deep networks

- **Backpropagation**: Algorithm for computing gradients of loss functions with respect to network parameters, enabling learning

- **Gradient Descent**: Optimization algorithm that iteratively adjusts parameters in the direction that reduces loss

- **Activation Function**: Non-linear function applied to neuron outputs, enabling networks to learn complex patterns (e.g., ReLU, sigmoid, tanh)

- **Pooling**: Operation that reduces spatial dimensions of feature maps, typically by taking maximum or average values in local regions

- **Batch Normalization**: Technique that standardizes layer inputs during training, stabilizing learning and enabling higher learning rates

- **Dropout**: Regularization technique that randomly deactivates neurons during training to prevent overfitting

- **Ethics Thread**: Continuous reflection on responsible AI practices throughout the course, connecting technical choices to societal implications

- **Extractive View**: Concept describing how AI systems can treat landscapes and communities as resources to be measured and optimized without consent

---

## Notes for Self-Paced Learners

### Time Management Suggestions

Week 2 is conceptually dense, requiring both technical work and critical reflection. Budget approximately **9-11 hours**:

**Day 1** (2-3 hours): 
- Read Core Content sections on computer vision history and evolution
- Watch supplementary videos on CNNs and deep learning
- Take notes on key concepts

**Day 2** (2 hours):
- Read Core Content on Earth observation challenges and domain shift
- Review ImageNet and pre-trained models
- Understand the domain shift problem conceptually

**Day 3** (2-3 hours):
- Complete Activity 1: ImageNet to GeoVision experiments
- Run inference on multiple land cover types
- Create visualizations

**Day 4** (1.5 hours):
- Generate activation maps
- Complete Activity 2: Domain shift analysis
- Write structured analysis document

**Day 5** (1.5-2 hours):
- Read ethics material (Crawford or UNESCO)
- Complete Activity 3: Ethics Thread launch
- Write reflection connecting reading to case study

**Day 6** (1 hour):
- Write weekly reflection
- Organize repository and check deliverables
- Prepare for Week 3

### Common Challenges and Solutions

**Challenge: Understanding CNN architectures feels overwhelming**

*Solution*: Focus on conceptual understanding rather than memorizing details. The key insights are: (1) CNNs learn hierarchical features, (2) early layers detect simple patterns, deeper layers detect complex patterns, (3) convolution operations detect spatial patterns. Don't worry about memorizing every layer in ResNet-50.

**Challenge: PyTorch model loading is slow or fails**

*Solution*: Pre-trained models are large (~100MB). First download takes time but models are cached. If download fails, check internet connection and try again. You can also manually download weights and load locally.

**Challenge: Earth Engine image download fails or times out**

*Solution*: Try smaller regions (reduce buffer from 500m to 250m) or lower resolution (increase scale parameter). Earth Engine has export limits. If persistent failures, try different times of day when server load may be lower.

**Challenge: Understanding why ImageNet models fail on satellite imagery**

*Solution*: Focus on the fundamental mismatch: ImageNet models learned to recognize everyday objects in photographs. Satellite imagery shows landscapes from above at coarse resolution with spectral information invisible to humans. The models are solving the wrong problem. Specific failures (fields → tennis courts) illustrate this mismatch.

**Challenge: Ethics reading feels abstract or disconnected from technical work**

*Solution*: For every concept in the reading, ask: "How does this apply to my case study?" Make it concrete. If reading about surveillance, think: "Could parcelización monitoring be used for surveillance? Who might be affected?" Connect abstract ideas to specific scenarios.

**Challenge: Balancing technical depth with conceptual understanding**

*Solution*: This week prioritizes conceptual understanding over technical implementation. You don't need to master PyTorch or CNN architectures this week. Focus on understanding *why* domain shift occurs and *what* it means for GeoAI. Technical mastery comes in Week 3.

### Extension Activities

**For Advanced Learners**:

- **Compare multiple architectures**: Test VGG, EfficientNet, and Vision Transformer on the same satellite imagery. Do different architectures show different failure modes?

- **Quantitative analysis**: Compute metrics (accuracy, confidence scores) across land cover types. Are some types more prone to misclassification?

- **Fine-tuning experiment**: Try fine-tuning an ImageNet model on a small satellite imagery dataset. How much does performance improve?

- **Literature review**: Read papers on domain adaptation and transfer learning for remote sensing. What approaches have researchers proposed?

**For Conceptual Thinkers**:

- **Visual timeline**: Create an infographic showing computer vision evolution from 1960s to present, highlighting key innovations.

- **Extended ethics essay**: Write a longer piece (1500-2000 words) on "The Politics of Automated Earth Observation."

- **Case study research**: Investigate how Earth observation has been used (or misused) in your study region. What power dynamics are at play?

- **Comparative analysis**: How do indigenous or local classification systems differ from Western land-cover categories? What gets lost in translation?

**For Visual Learners**:

- **Concept map**: Create a visual diagram connecting all concepts from this week (CNN, ImageNet, domain shift, data geography, ethics).

- **Annotated examples**: Create a detailed visual guide to ImageNet misclassifications on satellite imagery with explanations.

- **Architecture visualization**: Draw or diagram how information flows through a CNN, from input pixels to final predictions.

- **Infographic**: Design an infographic comparing natural images vs. satellite imagery across multiple dimensions (scale, spectral, temporal, context).

**For Hands-On Learners**:

- **Dataset exploration**: Download and explore the ImageNet dataset. What geographic biases can you identify?

- **Model comparison**: Load multiple pre-trained models and compare their predictions on the same satellite images.

- **Activation analysis**: Generate activation maps for multiple layers (not just the final layer). How do representations change through the network?

- **Custom preprocessing**: Experiment with different preprocessing approaches. What happens if you don't normalize with ImageNet statistics?

---

## Assessment Notes

### Self-Assessment Focus Areas

This week emphasizes **conceptual understanding** and **critical thinking** over technical execution:

**Conceptual Understanding (40%)**:
- Do you understand the evolution of computer vision and why it matters?
- Can you explain why Earth observation differs from natural image analysis?
- Do you grasp the domain shift problem and its causes?

**Experimental Rigor (25%)**:
- Did you systematically test multiple land cover types?
- Are your experiments well-documented and reproducible?
- Did you create clear visualizations?

**Analytical Thinking (25%)**:
- Can you explain *why* failures occurred, not just *what* failed?
- Do you connect observations to theoretical concepts?
- Are your implications thoughtful and well-reasoned?

**Ethical Reflection (10%)**:
- Have you engaged substantively with ethics readings?
- Do you make concrete connections to your case study?
- Have you identified specific responsible practices?

### Using Feedback to Improve

If you struggled with:

**Conceptual understanding**: 
- Re-read core content sections slowly
- Watch supplementary videos on CNNs and deep learning
- Create concept maps or summaries in your own words
- Discuss concepts with peers or instructor

**Technical implementation**:
- Review PyTorch basics tutorials
- Work through Earth Engine examples step-by-step
- Start with simpler examples before complex experiments
- Check code carefully for errors

**Analytical writing**:
- Practice explaining "why" not just "what"
- Use specific examples to support general claims
- Organize writing with clear structure (intro, body, conclusion)
- Have someone else read your writing and provide feedback

**Ethical reflection**:
- Connect every abstract concept to concrete examples
- Think about stakeholders: who benefits, who might be harmed?
- Consider your own positionality and assumptions
- Engage with diverse perspectives beyond Western frameworks

---

**This comprehensive Week 2 study guide provides the conceptual foundation, technical skills, and ethical framework you need to understand how computer vision evolved and why specialized approaches are necessary for Earth observation. Take your time with the material, engage deeply with the experiments, and reflect critically on the implications of automated Earth monitoring.**

---

**Document Version:** 2.0  
**Last Updated:** October 15, 2025  
**Author:** Manus AI  
**Course:** GeoAI and Earth Vision: Foundations to Frontier Applications

