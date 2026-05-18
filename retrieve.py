#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieve.py
===========

PV dataset retrieval, validation and extension script for the PVCF
(China High-Resolution Photovoltaic Comparison and Fusion) dataset.

This script supports reproduction and future extension of the PVCF
database. Given a set of newly retrieved candidate PV polygons (for
example, a more recent release of a source dataset, or polygons collected
for a later year), it uses the trained DeepLab V3+ model to assess each
polygon and decides whether it should be incorporated into the dataset.

The validation procedure follows Section 2.5 of the manuscript exactly:

  * Each candidate PV patch is evaluated with the Intersection over Union
    (IoU) as the sole metric of segmentation quality.
  * A patch is retained (PASSED) when its IoU exceeds 0.5, following the
    evaluation protocol of the PASCAL VOC benchmark (Everingham et al.,
    2010); otherwise it is rejected (FAILED).
  * Patches for which the required imagery cannot be obtained are reported
    separately as SKIPPED. They are neither passed nor failed, so that the
    screening result is not biased.

Workflow
--------
1. Load the trained DeepLab V3+ checkpoint produced by train.py.
2. Read the candidate PV polygons from one or more shapefiles.
3. For every polygon, obtain the overlapping image tiles, run inference,
   assemble the predicted and reference masks, and compute the IoU.
4. Split the polygons into passed / failed / skipped sets, export the
   passed and failed sets as shapefiles, and write a validation report.

Imagery input
-------------
Two imagery sources are supported and selected with --tile-source:

  * local  (default, recommended for reproduction)
        Image tiles are read from a local directory; tile files must be
        named "<x>_<y>.png". A small example set is provided under
        example_data/ in this repository.

  * server
        Image tiles are downloaded from a tiled basemap service following
        the standard {base-url}/{z}/{y}/{x} XYZ scheme. The base URL must
        be supplied by the user via --tile-base-url; no service endpoint
        is hard-coded in this script.

Usage
-----
    python retrieve.py --model-path ./output/models/deeplabv3plus_pv_best.pth \\
                       --shp-dir    ./example_data/new_polygons \\
                       --tile-source local \\
                       --tile-dir   ./example_data/tiles \\
                       --output-dir ./validation_output

Run "python retrieve.py --help" for the full list of options.

Dependencies
------------
    pip install torch torchvision geopandas shapely Pillow requests \\
                matplotlib numpy fiona pyproj pandas
"""

import argparse
import logging
import math
import os
import warnings
from io import BytesIO

import numpy as np

warnings.filterwarnings("ignore")

import requests
from PIL import Image, ImageDraw

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, box

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Constants
# ============================================================
ZOOM = 16                # XYZ tile zoom level
IMG_SIZE = 256           # tiles are resized to IMG_SIZE x IMG_SIZE
NUM_CLASSES = 2          # 0 = background, 1 = PV

# Patch-level retention rule (Section 2.5 of the manuscript):
# IoU is the sole metric, and a patch is retained when IoU > IOU_THRESHOLD.
# The 0.5 threshold follows the PASCAL VOC protocol (Everingham et al., 2010).
IOU_THRESHOLD = 0.5

# ImageNet statistics used to normalise the input imagery.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("retrieve")


# ============================================================
# Argument parsing
# ============================================================
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate and incorporate new PV polygons with the "
                    "trained DeepLab V3+ model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained DeepLab V3+ checkpoint (.pth).",
    )
    parser.add_argument(
        "--shp-dir",
        required=True,
        help="Directory containing the candidate PV shapefiles (.shp).",
    )
    parser.add_argument(
        "--output-dir",
        default="./validation_output",
        help="Directory for the passed/failed shapefiles and the report.",
    )
    parser.add_argument(
        "--tile-source",
        choices=["local", "server"],
        default="local",
        help="Where image tiles come from: a local directory or a tile "
             "server.",
    )
    parser.add_argument(
        "--tile-dir",
        default=None,
        help="Directory of local image tiles named '<x>_<y>.png'. "
             "Required when --tile-source is 'local'.",
    )
    parser.add_argument(
        "--tile-base-url",
        default=None,
        help="Base URL of an XYZ tile service ({base}/{z}/{y}/{x}). "
             "Required when --tile-source is 'server'. No endpoint is "
             "hard-coded; supply a service you are entitled to use.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=ZOOM,
        help="XYZ tile zoom level.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_THRESHOLD,
        help="IoU retention threshold; a patch passes when IoU exceeds it.",
    )
    return parser.parse_args()


# ============================================================
# 1. Coordinate <-> tile conversions (Web Mercator XYZ scheme)
# ============================================================
def lng2tile(lng, zoom):
    """Convert a longitude to an X tile index."""
    return math.floor((lng + 180) / 360 * 2 ** zoom)


def lat2tile(lat, zoom):
    """Convert a latitude to a Y tile index."""
    return math.floor(
        (1 - math.log(math.tan(math.radians(lat))
                      + 1 / math.cos(math.radians(lat))) / math.pi)
        / 2 * 2 ** zoom
    )


def num2deg(xtile, ytile, zoom):
    """Convert tile indices to the (lat, lng) of the tile's top-left corner."""
    n = 2 ** zoom
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    lng = xtile / n * 360.0 - 180.0
    return lat, lng


def tile_bounds_box(x, y, zoom):
    """Return the shapely box covering tile (x, y) at the given zoom."""
    lt_lat, lt_lng = num2deg(x, y, zoom)
    rb_lat, rb_lng = num2deg(x + 1, y + 1, zoom)
    return box(lt_lng, rb_lat, rb_lng, lt_lat)


# ============================================================
# 2. Load the trained DeepLab V3+ model
# ============================================================
def load_model(model_path):
    """Load a DeepLab V3+ checkpoint produced by train.py."""
    log.info("Loading model checkpoint: %s", model_path)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = deeplabv3_resnet50(weights=None)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE).eval()

    log.info("Model loaded. Training-stage validation metrics: "
             "mIoU=%s  PixAcc=%s  Dice=%s",
             checkpoint.get("val_miou", "N/A"),
             checkpoint.get("val_pixel_accuracy", "N/A"),
             checkpoint.get("val_dice", "N/A"))
    return model


# ============================================================
# 3. Read candidate PV polygons from shapefiles
# ============================================================
def load_candidate_polygons(shp_dir):
    """Read every shapefile in a directory into one GeoDataFrame (WGS84)."""
    shp_files = [
        os.path.join(shp_dir, f)
        for f in os.listdir(shp_dir)
        if f.lower().endswith(".shp")
    ]
    if not shp_files:
        log.error("No .shp files found in %s", shp_dir)
        return None

    log.info("Found %d shapefile(s).", len(shp_files))
    gdfs = []
    for shp_path in shp_files:
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            gdf = gdf[gdf.geometry.geom_type.isin(
                ["Polygon", "MultiPolygon"])].copy()
            gdf["_src_file"] = os.path.basename(shp_path)
            gdfs.append(gdf)
            log.info("  %s: %d polygons", os.path.basename(shp_path), len(gdf))
        except Exception as exc:
            log.error("Failed to read %s: %s", shp_path, exc)

    if not gdfs:
        return None
    merged = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), crs="EPSG:4326"
    )
    log.info("Loaded %d candidate polygons in total.", len(merged))
    return merged


# ============================================================
# 4. Obtain a tile image (local file or tile server)
# ============================================================
def load_tile_image(x, y, zoom, tile_source, tile_dir, tile_base_url):
    """Return an RGB PIL image for tile (x, y), or None if unavailable."""
    if tile_source == "local":
        path = os.path.join(tile_dir, f"{x}_{y}.png")
        if not os.path.exists(path):
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            log.warning("Failed to read local tile %s: %s", path, exc)
            return None

    # tile_source == "server"
    url = f"{tile_base_url.rstrip('/')}/{zoom}/{y}/{x}"
    try:
        response = requests.get(url, timeout=15, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        log.warning("Failed to download tile (%d,%d): %s", x, y, exc)
        return None


# ============================================================
# 5. Rasterise a polygon into a binary reference mask for one tile
# ============================================================
def make_mask_from_poly(poly, x, y, zoom):
    """Build an IMG_SIZE x IMG_SIZE reference mask (1 = PV) for tile (x, y)."""
    lt_lat, lt_lng = num2deg(x, y, zoom)
    rb_lat, rb_lng = num2deg(x + 1, y + 1, zoom)
    tile_box = box(lt_lng, rb_lat, rb_lng, lt_lat)
    lng_range = rb_lng - lt_lng
    lat_range = lt_lat - rb_lat

    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(mask)

    intersection = poly.intersection(tile_box)
    if intersection.is_empty:
        return mask
    geoms = (list(intersection.geoms)
             if isinstance(intersection, MultiPolygon)
             else [intersection])
    for geom in geoms:
        if geom.geom_type != "Polygon":
            continue
        points = [
            (int((lng - lt_lng) / lng_range * IMG_SIZE),
             int((lt_lat - lat) / lat_range * IMG_SIZE))
            for lng, lat in geom.exterior.coords
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return mask


# ============================================================
# 6. Run inference for a single polygon and compute its IoU
# ============================================================
def evaluate_polygon(model, poly, args):
    """Infer over all tiles covering a polygon and return its IoU.

    The predicted and reference PV masks of every overlapping tile are
    concatenated, and a single IoU value for the PV class is computed.
    IoU is the sole metric, in line with Section 2.5 of the manuscript.

    Returns
    -------
    dict with keys 'iou' and 'tiles', or None when no imagery is available.
    """
    minx, miny, maxx, maxy = poly.bounds
    x_min, x_max = lng2tile(minx, args.zoom), lng2tile(maxx, args.zoom)
    y_min = min(lat2tile(maxy, args.zoom), lat2tile(miny, args.zoom))
    y_max = max(lat2tile(maxy, args.zoom), lat2tile(miny, args.zoom))

    predicted_masks = []
    reference_masks = []
    tile_count = 0

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if not poly.intersects(tile_bounds_box(x, y, args.zoom)):
                continue

            img = load_tile_image(
                x, y, args.zoom, args.tile_source,
                args.tile_dir, args.tile_base_url,
            )
            if img is None:
                continue

            img_tensor = TF.to_tensor(
                img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            )
            img_tensor = TF.normalize(img_tensor, NORM_MEAN, NORM_STD)
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img_tensor)["out"]
                logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE),
                                       mode="bilinear", align_corners=False)
                prediction = logits.argmax(1).squeeze(0).cpu().numpy()

            reference = np.array(
                make_mask_from_poly(poly, x, y, args.zoom)
            )
            predicted_masks.append(prediction.flatten())
            reference_masks.append(reference.flatten())
            tile_count += 1

    if not predicted_masks:
        return None  # no imagery could be obtained for this polygon

    predicted = np.concatenate(predicted_masks)
    reference = np.concatenate(reference_masks)

    # IoU for the PV class (class 1) - the sole evaluation metric.
    pred_pv = predicted == 1
    ref_pv = reference == 1
    intersection = np.logical_and(pred_pv, ref_pv).sum()
    union = np.logical_or(pred_pv, ref_pv).sum()
    iou = float(intersection / union) if union > 0 else 1.0

    return {"iou": iou, "tiles": tile_count}


# ============================================================
# 7. Validate all candidate polygons and export the results
# ============================================================
def validate_and_export(model, gdf, args):
    """Screen every candidate polygon by IoU and export passed/failed sets.

    A polygon passes when its IoU exceeds the threshold. Polygons for which
    no imagery is available are recorded as SKIPPED and excluded from the
    passed and failed shapefiles.

    Returns
    -------
    (passed_rows, failed_rows, skipped_rows, iou_values)
    """
    passed_rows, failed_rows, skipped_rows = [], [], []
    iou_values = []
    total = len(gdf)

    for position, (_, row) in enumerate(gdf.iterrows(), start=1):
        result = evaluate_polygon(model, row.geometry, args)

        if result is None:
            log.warning("[%d/%d] SKIPPED (no imagery available)",
                        position, total)
            skipped_row = row.copy()
            skipped_row["status"] = "SKIPPED"
            skipped_rows.append(skipped_row)
            continue

        iou = result["iou"]
        iou_values.append(iou)
        status = "PASSED" if iou > args.iou_threshold else "FAILED"
        log.info("[%d/%d] IoU=%.4f  (%d tiles)  -> %s",
                 position, total, iou, result["tiles"], status)

        new_row = row.copy()
        new_row["iou"] = round(iou, 4)
        new_row["status"] = status
        if status == "PASSED":
            passed_rows.append(new_row)
        else:
            failed_rows.append(new_row)

    os.makedirs(args.output_dir, exist_ok=True)

    if passed_rows:
        passed_path = os.path.join(args.output_dir, "passed_polygons.shp")
        gpd.GeoDataFrame(passed_rows, crs="EPSG:4326").to_file(
            passed_path, encoding="utf-8"
        )
        log.info("Exported %d passed polygons to %s",
                 len(passed_rows), passed_path)
    else:
        log.info("No polygons passed validation.")

    if failed_rows:
        failed_path = os.path.join(args.output_dir, "failed_polygons.shp")
        gpd.GeoDataFrame(failed_rows, crs="EPSG:4326").to_file(
            failed_path, encoding="utf-8"
        )
        log.info("Exported %d failed polygons to %s",
                 len(failed_rows), failed_path)
    else:
        log.info("No polygons failed validation.")

    if skipped_rows:
        log.info("%d polygons were skipped (no imagery available).",
                 len(skipped_rows))

    return passed_rows, failed_rows, skipped_rows, iou_values


# ============================================================
# 8. Validation report
# ============================================================
def save_validation_report(passed_rows, failed_rows, skipped_rows,
                            iou_values, args):
    """Write an IoU-distribution plot and a plain-text validation report."""
    report_dir = os.path.join(args.output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    n_pass = len(passed_rows)
    n_fail = len(failed_rows)
    n_skip = len(skipped_rows)
    n_evaluated = n_pass + n_fail
    total = n_evaluated + n_skip
    pass_rate = n_pass / n_evaluated if n_evaluated > 0 else 0.0

    # IoU distribution histogram.
    hist_path = os.path.join(report_dir, "iou_distribution.png")
    if iou_values:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(iou_values, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(args.iou_threshold, color="red", linestyle="--",
                   label=f"threshold = {args.iou_threshold}")
        ax.set_title("IoU distribution of candidate PV patches")
        ax.set_xlabel("IoU")
        ax.set_ylabel("Number of patches")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)

    # Plain-text report.
    if iou_values:
        iou_summary = (
            f"  IoU  mean={np.mean(iou_values):.4f}  "
            f"min={np.min(iou_values):.4f}  "
            f"max={np.max(iou_values):.4f}"
        )
    else:
        iou_summary = "  IoU  (no polygons were evaluated)"

    report_lines = [
        "=" * 62,
        "DeepLab V3+ PV segmentation - new-data validation report",
        "=" * 62,
        "",
        "Validation configuration:",
        f"  candidate shapefile directory : {args.shp_dir}",
        f"  model checkpoint              : {args.model_path}",
        f"  tile source                   : {args.tile_source}",
        f"  tile zoom level               : Z={args.zoom}",
        f"  IoU retention threshold       : {args.iou_threshold}",
        "",
        "  Metric: IoU is the sole metric (Section 2.5 of the manuscript).",
        "  A patch is retained when its IoU exceeds the threshold,",
        "  following the PASCAL VOC protocol (Everingham et al., 2010).",
        "  Patches without available imagery are reported as SKIPPED and",
        "  are excluded from the passed and failed sets.",
        "",
        "Validation results:",
        f"  total candidate polygons : {total}",
        f"  evaluated                : {n_evaluated}",
        f"  passed (IoU > threshold) : {n_pass}  ({pass_rate * 100:.1f}%)",
        f"  failed                   : {n_fail}  "
        f"({(1 - pass_rate) * 100:.1f}%)",
        f"  skipped (no imagery)     : {n_skip}",
        "",
        "Metric summary (evaluated polygons):",
        iou_summary,
        "",
        "Output files:",
        f"  passed shapefile : "
        f"{os.path.join(args.output_dir, 'passed_polygons.shp')}",
        f"  failed shapefile : "
        f"{os.path.join(args.output_dir, 'failed_polygons.shp')}",
        f"  IoU distribution : {hist_path}",
        "=" * 62,
    ]
    report_text = "\n".join(report_lines)
    report_path = os.path.join(report_dir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    print("\n" + report_text)
    log.info("Validation report written to %s", report_path)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_arguments()

    # Validate imagery-source arguments.
    if args.tile_source == "local" and not args.tile_dir:
        raise SystemExit(
            "Error: --tile-dir is required when --tile-source is 'local'."
        )
    if args.tile_source == "server" and not args.tile_base_url:
        raise SystemExit(
            "Error: --tile-base-url is required when --tile-source is "
            "'server'. No tile service endpoint is hard-coded in this "
            "script; please supply one you are entitled to use."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "retrieve.log"),
                encoding="utf-8",
            ),
        ],
    )
    log.info("Device: %s", DEVICE)

    # Step 1: load the trained model.
    model = load_model(args.model_path)

    # Step 2: read the candidate PV polygons.
    gdf = load_candidate_polygons(args.shp_dir)
    if gdf is None or len(gdf) == 0:
        log.error("No valid candidate polygons were read; aborting.")
        return

    # Step 3: validate every polygon and export the passed/failed sets.
    passed_rows, failed_rows, skipped_rows, iou_values = validate_and_export(
        model, gdf, args
    )

    # Step 4: write the validation report.
    save_validation_report(
        passed_rows, failed_rows, skipped_rows, iou_values, args
    )
    log.info("Validation complete.")


if __name__ == "__main__":
    main()
