# ML-Assisted Lithography Hotspot Detection System

Lithography hotspots are critical manufacturing defects in advanced ICs that reduce yield and long-term reliability. This project builds a **scalable machine learning pipeline** to automatically detect such hotspots from ICCAD-2012 layout benchmarks using engineered features and ensemble models.


---

## Overview

As technology nodes shrink, lithography-induced defects become harder to catch with traditional simulation and rule-based approaches, which are often slow and compute-heavy.
This project proposes an ML-based hotspot detection workflow that processes large PNG layout datasets in batches, extracts rich geometric and spatial features, and trains an ensemble classifier optimized for high hotspot recall under severe class imbalance.

**Key goals:**

- Efficient processing of large-scale layout images without out-of-memory issues.
- Robust hotspot detection with high recall while controlling false alarms.

---

## Dataset

The project uses the **ICCAD-2012** lithography hotspot benchmark, which provides labeled layout patterns as PNG images.
Patterns are grouped into benchmark-specific sets, each containing hotspot and non-hotspot samples suitable for training and evaluation.

Main properties:

- Binary layout images (patterns) with hotspot/non-hotspot labels.  
- Strong class imbalance: non-hotspots vastly outnumber hotspots.

---

## Methodology

### 1. Data Loading & Batch Processing

- Layout PNGs are loaded in **benchmark-wise batches** to avoid memory overflow.
- Intermediate representations and features are cached for faster experimentation.

### 2. Feature Engineering (43 Features)

For each pattern, a 43-dimensional feature vector is computed, covering:

- Density metrics: global, quadrants, center vs periphery, multiple ring densities.  
- Topological and geometric features: shape complexity, connectivity, structural patterns.  
- Spatial and gradient statistics: distribution of edges and transitions across the layout.  

These features allow traditional ML models to approximate complex lithography behavior without full physics simulation.

### 3. Handling Class Imbalance

- Class weights are used in models (e.g., hotspots weighted higher than non-hotspots).
- Additional undersampling strategies are implemented for some experiments to match hotspot and non-hotspot counts in balanced subsets.

### 4. Model Training & Threshold Tuning

The core classifier is an **ensemble** (e.g., Random Forest + other learners) trained on standardized feature vectors.
A custom threshold search loop finds a decision threshold that targets **≥ 90% recall**, prioritizing detection of hotspots over precision.

### 5. Evaluation & Visualization

Evaluation metrics include:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- False alarm rate  

Diagnostic plots such as confusion matrix, ROC curve, precision–recall curve, and feature distribution visualizations are generated to understand model behavior.

---

## Results

On the ICCAD-2012 test set, the final model achieves

| Metric            | Value   |
|-------------------|---------|
| Accuracy          | 94.08%  |
| Precision         | 20.22%  |
| Recall            | 74.40%  |
| F1-Score          | 31.79%  |
| False Alarm Rate  | 5.63%   |

These results indicate:

- High overall accuracy driven by the dominant non-hotspot class.  
- Strong recall for hotspots (most true hotspots are detected), at the cost of lower precision due to class imbalance and aggressive recall optimization.

Confusion matrix, ROC, and precision–recall curves provide deeper insight into the trade-off between missed hotspots and false alarms.

---

## Future Work

Planned and potential improvements include:

- Advanced re-sampling (SMOTE variants, hybrid over/under-sampling) to improve precision.  
- Automated feature selection or dimensionality reduction to remove redundant features.  
- Deep learning architectures (e.g., CNNs directly on layout images) for richer pattern modeling.  
- Continuous model updating with new datasets and emerging lithography defect types.

---

## References

1. **Intelectron6 GitHub – Lithography Hotspot Detection** repository, 2023.[file:46]  
2. **ICCAD-2012** Lithography Hotspot Detection Benchmark Dataset.[file:46]  
3. Yang et al., *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)*, 2017.[file:46]  
4. Prior work on fuzzy pattern matching and imbalance-aware hotspot detection methods.[file:46]
