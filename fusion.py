#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion.py
=========

Multi-source data fusion script for the PVCF (China High-Resolution
Photovoltaic Comparison and Fusion) dataset.

This script integrates nine publicly available photovoltaic (PV) datasets
into a harmonised product. The inputs are the raw vector files of the nine
source datasets; the output is the fused PV distribution map in GeoTIFF
format. Each source dataset is reprojected to a common metric coordinate
reference system, rasterised onto a shared grid, and the resulting binary
PV masks are summed. The value of every output pixel is therefore the
"overlap count": the number of source datasets that identify that pixel as
PV. This overlap count is the spatial-consistency indicator used in the
manuscript to define low-, medium- and high-confidence regions.

Workflow
--------
1. Read all source shapefiles and determine a unified CRS and bounding box.
2. If the unified CRS is geographic, reproject to a metric (UTM) CRS so that
   a fixed pixel size in metres can be used.
3. Rasterise each source dataset onto the shared grid (1 = PV, 0 = non-PV).
4. Sum all rasters to obtain the fused overlap-count map.

Inputs
------
A root directory containing nine sub-directories (named "1" .. "9" by
default), each holding one raw vector file (shapefile) for a source PV
dataset. The mapping between sub-directory number and source dataset follows
Table 1 of the manuscript.

Outputs
-------
- merged.tif : the fused PV distribution map in GeoTIFF format (uint16),
  where each pixel value is the overlap count across the source datasets.
- One intermediate single-source raster per dataset (temp_<n>.tif), removed
  after merging unless --keep-temp is given.

Usage
-----
    python fusion.py --input-dir  ./data/sources \\
                     --output-dir ./data/output \\
                     --pixel-size 10

Run "python fusion.py --help" for the full list of options.

Notes
-----
The nine source datasets are NOT redistributed with this repository. They
must be downloaded from their original providers (see Table 1 of the
manuscript and the project README). A small synthetic example is provided
under example_data/ so that the workflow can be tested end to end.
"""

import argparse
import os
import sys

import numpy as np
import rasterio
from rasterio.features import rasterize

import geopandas as gpd
from shapely.validation import make_valid


# Number of source datasets expected by the fusion workflow.
N_SOURCES = 9


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Integrate nine source PV vector datasets into a harmonised "
            "PV distribution map in GeoTIFF format."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing the numbered source sub-directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where intermediate and fused rasters are written.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=10.0,
        help="Output pixel size in metres.",
    )
    parser.add_argument(
        "--subdir-names",
        nargs="+",
        default=[str(i) for i in range(1, N_SOURCES + 1)],
        help="Names of the source sub-directories, in fusion order.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the intermediate single-source rasters after merging.",
    )
    return parser.parse_args()


def safe_make_valid(geom):
    """Return a valid version of a geometry, or None if it cannot be repaired."""
    if geom is None:
        return None
    try:
        if hasattr(geom, "is_valid") and not geom.is_valid:
            return make_valid(geom)
        return geom
    except Exception:
        return None


def load_source_geodataframes(input_dir, subdir_names):
    """Read one shapefile from each numbered source sub-directory.

    Returns a list of (name, GeoDataFrame) tuples for every sub-directory in
    which a shapefile is found.
    """
    sources = []
    for name in subdir_names:
        folder = os.path.join(input_dir, name)
        if not os.path.isdir(folder):
            print(f"  Warning: source directory not found, skipping: {folder}")
            continue

        shp_files = sorted(f for f in os.listdir(folder) if f.endswith(".shp"))
        if not shp_files:
            print(f"  Warning: no shapefile in {folder}, skipping")
            continue

        shp_path = os.path.join(folder, shp_files[0])
        print(f"  Reading source '{name}': {shp_files[0]}")
        gdf = gpd.read_file(shp_path)
        sources.append((name, gdf))

    return sources


def unify_crs(sources):
    """Reproject every GeoDataFrame to a common CRS.

    The CRS of the first source is used as the target. If that CRS is
    geographic, a metric UTM CRS centred on the combined extent is derived
    instead, so that a pixel size expressed in metres is meaningful.

    Returns
    -------
    (target_crs, unified_sources)
    """
    target_crs = sources[0][1].crs
    print(f"  Initial target CRS: {target_crs}")

    unified = []
    for name, gdf in sources:
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        unified.append((name, gdf))

    if target_crs is not None and target_crs.is_geographic:
        bounds = np.array([gdf.total_bounds for _, gdf in unified])
        center_lon = (bounds[:, 0].min() + bounds[:, 2].max()) / 2.0
        center_lat = (bounds[:, 1].min() + bounds[:, 3].max()) / 2.0

        utm_zone = int((center_lon + 180) / 6) + 1
        hemisphere = "north" if center_lat >= 0 else "south"
        target_crs = (
            f"+proj=utm +zone={utm_zone} +{hemisphere} "
            f"+datum=WGS84 +units=m +no_defs"
        )
        print(
            f"  Geographic CRS detected; reprojecting to "
            f"UTM zone {utm_zone} ({hemisphere})."
        )
        unified = [(name, gdf.to_crs(target_crs)) for name, gdf in unified]

    return target_crs, unified


def compute_global_grid(unified_sources, pixel_size):
    """Compute the shared raster grid covering all source datasets.

    Returns
    -------
    (width, height, transform, bounds)
    """
    bounds = np.array([gdf.total_bounds for _, gdf in unified_sources])
    minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
    maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()

    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))
    transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)

    print(f"  Global bounds: ({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")
    print(f"  Raster size:   {width} x {height} pixels @ {pixel_size} m")
    return width, height, transform, (minx, miny, maxx, maxy)


def clean_geometries(gdf):
    """Drop empty/None geometries, keep only polygons, and repair invalid ones."""
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if len(gdf) == 0:
        return gdf

    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(safe_make_valid)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def rasterize_source(gdf, width, height, transform, target_crs, raster_path):
    """Rasterise one source dataset to a binary uint8 GeoTIFF (1 = PV)."""
    shapes = ((geom, 1) for geom in gdf.geometry)
    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=target_crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(raster, 1)
    return raster_path


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: read sources and determine a unified CRS and grid.
    # ------------------------------------------------------------------
    print("Step 1: Reading source datasets and computing the shared grid...")
    sources = load_source_geodataframes(args.input_dir, args.subdir_names)
    if not sources:
        sys.exit("Error: no source shapefiles were found. Check --input-dir.")
    if len(sources) < N_SOURCES:
        print(
            f"  Note: {len(sources)} of {N_SOURCES} source datasets were found. "
            f"Fusion will proceed with the available datasets."
        )

    target_crs, unified_sources = unify_crs(sources)
    width, height, transform, _ = compute_global_grid(
        unified_sources, args.pixel_size
    )

    # ------------------------------------------------------------------
    # Step 2: rasterise each source dataset onto the shared grid.
    # ------------------------------------------------------------------
    print("\nStep 2: Rasterising individual source datasets...")
    temp_rasters = []
    for idx, (name, gdf) in enumerate(unified_sources, start=1):
        print(f"  Processing source '{name}' ({idx}/{len(unified_sources)})")
        gdf = clean_geometries(gdf)
        if len(gdf) == 0:
            print(f"    Warning: no valid polygons for source '{name}', skipping")
            continue

        raster_path = os.path.join(args.output_dir, f"temp_{name}.tif")
        rasterize_source(gdf, width, height, transform, target_crs, raster_path)
        temp_rasters.append(raster_path)
        print(f"    Written: {raster_path}")

    if not temp_rasters:
        sys.exit("Error: no rasters were produced; nothing to merge.")

    # ------------------------------------------------------------------
    # Step 3: sum all single-source rasters into the overlap-count map.
    # ------------------------------------------------------------------
    print(f"\nStep 3: Merging {len(temp_rasters)} rasters into the fused map...")
    overlap_count = np.zeros((height, width), dtype=np.uint16)
    for raster_path in temp_rasters:
        with rasterio.open(raster_path) as src:
            overlap_count += src.read(1).astype(np.uint16)
        print(f"  Added: {os.path.basename(raster_path)}")

    print("\nFused overlap-count statistics:")
    print(f"  Min value:        {overlap_count.min()}")
    print(f"  Max value:        {overlap_count.max()}")
    print(f"  Mean value:       {overlap_count.mean():.4f}")
    print(f"  Non-zero pixels:  {int(np.count_nonzero(overlap_count))}")

    final_raster = os.path.join(args.output_dir, "merged.tif")
    with rasterio.open(
        final_raster,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=target_crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(overlap_count, 1)
    print(f"\nDone. Fused overlap-count map written to: {final_raster}")

    # ------------------------------------------------------------------
    # Optional cleanup of intermediate rasters.
    # ------------------------------------------------------------------
    if not args.keep_temp:
        for raster_path in temp_rasters:
            try:
                os.remove(raster_path)
            except OSError:
                pass
        print("Intermediate single-source rasters removed "
              "(use --keep-temp to retain them).")


if __name__ == "__main__":
    main()
