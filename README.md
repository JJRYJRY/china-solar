# PVCF: China High-Resolution Photovoltaic Comparison and Fusion Dataset

This repository contains the code used to construct the **PVCF dataset**, a
high-resolution photovoltaic (PV) vector dataset for China, as described in
the manuscript *"PVCF: A High-Resolution, Spatially Consistent Multi-Source
Fusion Dataset for Photovoltaic Infrastructure in China"*.

The PVCF dataset systematically compares, cross-validates and fuses nine
publicly available PV datasets covering China, supported by DeepLab V3+
based validation and extensive manual annotation, to produce an updated,
spatially consistent PV vector dataset.

The PVCF vector dataset itself (`PVCFv1.shp`) is distributed separately;
this repository provides only the custom processing code.

## Repository contents

| File | Description |
|------|-------------|
| `fusion.py` | Multi-source data fusion. Integrates nine publicly available PV datasets into a harmonised product. Inputs are the raw vector files of the nine source datasets; the output is a fused PV distribution map in GeoTIFF format, where each pixel value records the number of source datasets identifying it as PV (the spatial-consistency overlap count). |
| `train.py` | Training script for the DeepLab V3+ semantic segmentation model used to identify PV installations from remote sensing imagery. Includes data preprocessing, hyperparameter search, model training and validation. |
| `retrieve.py` | Validates newly retrieved candidate PV polygons with the trained DeepLab V3+ model, retaining patches whose IoU exceeds 0.5 (PASCAL VOC protocol). Supports reproduction of the validation procedure and extension of the database in future years. |
| `example_data/` | A small real example dataset (PV polygons and image tiles) for verifying the environment and testing `train.py` and `retrieve.py`. |
| `requirements.txt` | Python package dependencies. |
| `LICENSE` | MIT License. |

## Installation and environment configuration

The code is implemented in **Python 3.12**. A clean virtual environment is
recommended.

```bash
# 1. Clone the repository
git clone https://github.com/JJRYJRY/china-solar.git
cd china-solar

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt
```

The main dependencies are `torch` and `torchvision` (DeepLab V3+),
`geopandas`, `shapely`, `rasterio`, `fiona` and `pyproj` (geospatial
processing), `optuna` (hyperparameter search), and `Pillow`, `numpy`,
`pandas` and `matplotlib`. A CUDA-capable GPU is recommended for `train.py`
and `retrieve.py` but is not required; the code falls back to CPU
automatically.

## Data requirements

`fusion.py` requires the **nine source PV datasets** as input. These are
publicly available but are **not** redistributed in this repository; they
must be obtained from their original providers. The datasets, their
coverage, temporal span, spatial resolution and methodology are listed in
Table 1 of the manuscript. Place one shapefile in each of nine
sub-directories named `1` .. `9` to use them with `fusion.py`.

`train.py` and `retrieve.py` require PV polygon shapefiles and basemap
imagery. A small **real example dataset** is provided in `example_data/`
so that these two scripts can be run and the environment verified without
any additional downloads:

```
example_data/
    shp/           PV polygons for train.py
    new_polygons/  candidate PV polygons for retrieve.py
    tiles/         basemap image tiles ("<x>_<y>.png")
```

For full-scale processing rather than a quick test, replace the example
inputs with PV polygons of interest and the corresponding basemap imagery.
The manuscript used the Esri World Imagery Wayback service for model
training and validation. Imagery can either be supplied as pre-downloaded
local tiles (`--tile-source local`) or fetched from an XYZ tile service the
user is entitled to use (`--tile-source server --tile-base-url <URL>`). No
imagery service endpoint is hard-coded in the code.

### 1. Multi-source data fusion (`fusion.py`)

Fuses the nine source PV datasets into an overlap-count map. The input
directory must contain nine sub-directories (`1` .. `9`), each holding one
shapefile for one source dataset (downloaded from its original provider;
see "Data requirements" above).

```bash
python fusion.py \
    --input-dir  ./data/sources \
    --output-dir ./output/fusion \
    --pixel-size 10
```

Output: `output/fusion/merged.tif` — a GeoTIFF in which each pixel value is
the number of source datasets that identify the pixel as PV. This overlap
count is the spatial-consistency indicator used to define low-, medium- and
high-confidence regions in the manuscript.

### 2. Train the DeepLab V3+ model (`train.py`)

Reads PV polygons, builds paired image/mask tiles, runs an Optuna
hyperparameter search, trains the final model and evaluates it on a
held-out validation set (mIoU, pixel accuracy, Dice coefficient).

```bash
python train.py \
    --shp-dir      ./example_data/shp \
    --tile-source  local \
    --tile-dir     ./example_data/tiles \
    --output-dir   ./output/train \
    --optuna-trials 12
```

To download imagery from a tile service instead of reading local tiles,
use `--tile-source server --tile-base-url <URL>`, where `<URL>` is the base
URL of an XYZ tile service you are entitled to use.

Outputs (under `output/train/`): the trained model
(`models/deeplabv3plus_pv_best.pth`), training-curve and prediction plots,
and a text report.

### 3. Validate and extend the dataset (`retrieve.py`)

Screens newly retrieved candidate PV polygons with the trained model. Each
patch is assessed by Intersection over Union (IoU); a patch passes when its
IoU exceeds 0.5, following the PASCAL VOC protocol. Patches without
available imagery are reported separately as skipped.

```bash
python retrieve.py \
    --model-path  ./output/train/models/deeplabv3plus_pv_best.pth \
    --shp-dir     ./example_data/new_polygons \
    --tile-source local \
    --tile-dir    ./example_data/tiles \
    --output-dir  ./output/retrieve
```

Outputs (under `output/retrieve/`): `passed_polygons.shp` and
`failed_polygons.shp`, an IoU-distribution plot, and a validation report.

> Note: the bundled `example_data/` is a small sample for verifying the
> environment and exercising the full workflow. For dataset-scale results,
> use a larger set of PV polygons and the corresponding basemap imagery
> described in the manuscript.

## Reproducibility and versioning

All custom code in this repository is released under the MIT License (see
`LICENSE`). Updates to the code are tracked through GitHub version control,
and corresponding versioned releases are archived on Zenodo with a
permanent digital object identifier.

- GitHub: https://github.com/JJRYJRY/china-solar
- Zenodo: https://doi.org/10.5281/zenodo.20264929

## Citation

If you use this code or the PVCF dataset, please cite the associated
manuscript. The full citation will be added here once the article is
published.

## Contact

For questions about the code or the dataset, please open an issue on the
GitHub repository.
