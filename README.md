# Flood_Prediction_Analytics
Machine Learning based flood prediction and analytics system using Sentinel-2 imagery.
Here's a complete README for your project:

---

# 🌊 Chungthang GLOF Flood Analytics — Sentinel-2 & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange.svg)](https://colab.research.google.com)
[![Data](https://img.shields.io/badge/Data-Sentinel--2%20L2A-green.svg)](https://scihub.copernicus.eu)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

A remote sensing and machine learning pipeline for automated detection, spatial mapping, and quantitative measurement of land cover changes caused by the South Lhonak Lake Glacial Lake Outburst Flood (GLOF) that devastated Chungthang Valley, North Sikkim, India on October 4, 2023.

---

## 📌 Study Area & Event

| Parameter | Details |
|---|---|
| **Event** | South Lhonak Lake GLOF |
| **Date** | October 4, 2023 |
| **Location** | Chungthang Valley, North Sikkim |
| **Coordinates** | 27.58°N – 27.62°N, 88.62°E – 88.67°E |
| **Satellite** | Sentinel-2 MSI L2A (10 m resolution) |
| **Image Tile** | T45UYB (UTM Zone 45N, EPSG:32645) |
| **Image Size** | 439 × 464 pixels (~20.37 km²) |

The flood destroyed the **Teesta III hydroelectric dam**, **31 bridges**, and over **25,900 structures**, mobilising an estimated **270 million cubic metres of sediment** downstream.

---

## 🎯 What This Project Does

This project trains a computer to look at satellite pixels and automatically decide what each pixel represents — water, vegetation, buildings, or debris. Instead of manually inspecting images, the machine learning model classifies every pixel across the entire scene and produces:

- **Pre-flood vs Post-flood land cover maps** with geographic coordinates
- **Change detection maps** showing exactly where damage occurred
- **Quantitative area statistics** (km²) for all four land cover classes
- **Spectral index analysis** (NDVI, NDWI, NDBI) across both dates
- **ML accuracy assessment** with confusion matrices and feature importance

---

## 📊 Key Results

| Land Cover Class | Before (km²) | After (km²) | Change |
|---|---|---|---|
| 🌿 Vegetation | 18.62 | 18.57 | −0.3% |
| 💧 Water | 0.22 | 0.06 | **−72.0%** |
| 🟤 Barren/Debris | 0.17 | 0.73 | **+322.2%** |
| 🏠 Built-up | 1.36 | 1.01 | −25.8% |

**ML Model Performance (Random Forest):**
- Train Accuracy: **87.71%** | Test Accuracy: **80.52%** | Overfit Gap: 7.19%
- Macro Precision: 0.81 | Macro Recall: 0.80 | Macro F1: 0.81

> The +322% increase in Barren/Debris area represents the GLOF sediment deposition corridor along the Teesta River — the primary quantifiable satellite signature of the flood damage.

---

## 🗂️ Project Structure

```
├── minor.ipynb               # Main Google Colab notebook
├── Before_Flood (1).tif      # Pre-flood Sentinel-2 GeoTIFF (upload to /content/)
├── After_Flood (1).tif       # Post-flood Sentinel-2 GeoTIFF (upload to /content/)
└── outputs/
    ├── FINAL_ML_CLASSIFIED_MAP.png
    ├── classification_maps.png
    ├── change_detection_map.png
    ├── confusion_matrix.png
    └── area_statistics_chart.png
```

---

## ⚙️ Installation & Setup

**Run entirely on Google Colab — no local setup needed.**

```python
# Cell 1: Install dependencies
!pip install rasterio pyproj scikit-learn matplotlib numpy scipy -q
```

Then upload your GeoTIFF files to `/content/` via the Colab file sidebar and run all cells top to bottom (`Runtime → Run all`).

**Dependencies:**

| Library | Purpose |
|---|---|
| `rasterio` | Read/write Sentinel-2 GeoTIFF files |
| `numpy` | Numerical operations on band arrays |
| `matplotlib` | All visualisations and maps |
| `pyproj` | UTM → WGS-84 lat/lon coordinate conversion |
| `scikit-learn` | Random Forest classifier, accuracy metrics |
| `scipy.ndimage` | Local texture feature computation |

---

## 🛰️ Data Acquisition

Download your Sentinel-2 images from **Google Earth Engine** using this script:

```javascript
// Pre-flood image (clear scene before Oct 4, 2023)
var s2_before = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(ee.Geometry.Point([88.65, 27.60]))
  .filterDate('2023-08-01', '2023-09-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first()
  .select(['B2','B3','B4','B8']);

Export.image.toDrive({
  image: s2_before,
  description: 'Before_Flood',
  scale: 10,
  region: ee.Geometry.Rectangle([88.60, 27.57, 88.68, 88.63])
});
```

> **Note:** Your current GeoTIFF has **4 bands** (B2, B3, B4, B8). To enable accurate NDBI computation, re-export including **B11 (SWIR)** as a 5th band.

---

## 🔬 Methodology

### Pipeline Overview

```
Sentinel-2 GeoTIFF (Before + After)
        ↓
Band Loading & DN Normalisation (÷ 10000 → [0,1])
        ↓
Spectral Index Computation (NDVI, NDWI, NDBI)
        ↓
Change Detection (ΔNDWI, ΔNDVI pixel maps)
        ↓
Threshold-based Land Cover Classification (4 classes)
        ↓
Boundary-aware Random Forest Training
        ↓
Full-Image Classification → Land Cover Maps
        ↓
Area Statistics + Validation
```

### Spectral Indices

| Index | Formula | Physical Meaning |
|---|---|---|
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation health |
| NDWI | (Green − NIR) / (Green + NIR) | Open water detection |
| NDBI | (Red − NIR) / (Red + NIR) | Built-up / barren surfaces (proxy) |

### ML Feature Vector (7 dimensions per pixel)

Each pixel is represented by: `[Blue, Green, Red, NIR, Red/NIR ratio, Green/NIR ratio, NIR local texture]`

> NDVI and NDWI are **deliberately excluded** from the feature vector to prevent data leakage — labels are derived from raw band thresholds, not from indices, ensuring genuine model generalisation.

### Boundary-Aware Training

The training pipeline uses a **KNN-based boundary detection** step (k=7 neighbours) to identify class-boundary pixels. Training samples are drawn as **60% boundary pixels + 40% interior pixels**, ensuring the model learns to handle the hardest classification decisions at class edges.

---

## ⚠️ Known Limitations

- Cloud cover: Sentinel-2 is optical only — cloud-free post-flood imagery may be unavailable for weeks in Himalayan monsoon conditions
- No SWIR band: Current GeoTIFF lacks B11, so NDBI uses Red/NIR approximation, reducing Built-up class accuracy
- No DEM integration:Debris extent is mapped but deposit depth/volume cannot be estimated without elevation data
- 10 m resolution: Individual boulders and debris channels narrower than 10 m cannot be resolved

---

## ✅ Validation Against Ground Reports

All satellite-derived findings are consistent with published post-disaster reports:

| Finding | This Project | Ground Report |
|---|---|---|
| Debris increase | +322% | 270M m³ sediment mobilised (Sattar et al., 2025, *Science*) |
| Water decrease | −72% | South Lhonak Lake drained >100 ha (ISRO RISAT-1A) |
| Built-up damage | −25.8% | 25,900+ structures damaged (ACAPS, 2023) |
| Vegetation impact | −0.3% | Channelised damage; slopes intact (SANDRP, 2023) |

---

## 📚 References

1. Sattar, A. et al. (2025). The Sikkim flood of October 2023: Drivers, causes and impacts. *Science*, 387. DOI: 10.1126/science.adq7616
2. McFeeters, S. K. (1996). The use of NDWI in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425–1432
3. Xu, H. (2006). Modification of MNDWI to enhance open water features. *International Journal of Remote Sensing*, 27(14), 3025–3033
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32
5. SANDRP (2023). Glacial Lake Flood Destroys Teesta-3 Dam in Sikkim. [sandrp.in](https://sandrp.in)
6. ESA (2023). Sentinel-2 User Handbook. European Space Agency

---

## 👤 Authors

Anuran Sen — B.Tech Computer Science & Engineering, SRM Institute of Science and Technology, Kattankulathur
- GitHub: [@SEN-ANURAN](https://github.com/SEN-ANURAN)
- LinkedIn: [linkedin.com/in/anuran-sen](https://linkedin.com/in/anuran-sen)

---

## 🏛️ Institution

Department of Computing Technologies
SRM Institute of Science and Technology, Kattankulathur – 603 203
Minor Project | Academic Year 2024–25 | 

---

*This project is intended for academic research and publication purposes. All satellite data sourced from the Copernicus Programme (ESA), freely available under open access terms.*
