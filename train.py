#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
========

Training script for the DeepLab V3+ semantic segmentation model used to
identify photovoltaic (PV) installations from remote sensing imagery, as
described in the Methods section of the PVCF manuscript.

The script provides a complete and reproducible pipeline:

    1. Data preprocessing - PV vector polygons are read from shapefiles, the
       intersecting image tiles are gathered, and a binary segmentation mask
       (0 = background, 1 = PV) is generated for every tile.
    2. Model training      - a DeepLab V3+ network with a ResNet-50 backbone
       and an Atrous Spatial Pyramid Pooling (ASPP) module is trained for
       binary PV / non-PV segmentation.
    3. Hyperparameter search - learning rate, weight decay, batch size,
       number of epochs and momentum are optimised with the Optuna
       framework, using validation mIoU as the objective.
    4. Validation          - under the optimal hyperparameter settings, the
       trained model is evaluated on the held-out validation set, reporting
       the mean Intersection over Union (mIoU), pixel accuracy and the Dice
       coefficient, and a small set of predictions is visualised.

Note on metrics
----------------
This training script and the manuscript use metrics at two different
stages, and the two should not be conflated:

  * Training and hyperparameter optimisation (the Methods section of the
    manuscript). Under the optimal hyperparameter settings the trained
    model is assessed on the held-out validation set with three metrics:
    mIoU, pixel accuracy and the Dice coefficient. These three metrics are
    what this script computes (see `compute_metrics`) and reports.

  * Screening of the medium agreement regions. In that later step the
    Intersection over Union (IoU) is used as the sole metric to assess the
    segmentation quality of each PV patch, and a patch is retained only
    when its IoU exceeds 0.5 (the PASCAL VOC protocol). That patch-level
    screening belongs to the dataset-construction workflow and is not
    performed by this training script.

Imagery input
-------------
Two imagery sources are supported and selected with --tile-source:

  * local  (default, recommended for reproduction)
        Image tiles are read from a local directory. Tile files must be
        named "<x>_<y>.png". A small example set is provided under
        example_data/ in this repository.

  * server
        Image tiles are downloaded from a tiled basemap service following
        the standard {base-url}/{z}/{y}/{x} XYZ scheme. The base URL must be
        supplied by the user via --tile-base-url; no service endpoint is
        hard-coded in this script. The manuscript used the Esri World
        Imagery Wayback service (see the Methods section of the manuscript); reproduction users should
        provide a basemap service for which they hold valid access rights.

Outputs
-------
- models/deeplabv3plus_pv_best.pth : trained model weights, best
  hyperparameters and validation metrics.
- report/training_curves.png       : training-loss and validation-metric
  curves.
- report/val_predictions.png       : qualitative validation predictions.
- report/training_report.txt       : a plain-text run summary.

Usage
-----
Reproduce with the bundled example data (no network required)::

    python train.py --shp-dir      ./example_data/shp \\
                    --tile-source  local \\
                    --tile-dir     ./example_data/tiles \\
                    --output-dir   ./output

Run "python train.py --help" for the full list of options.

Dependencies
------------
    pip install torch torchvision geopandas shapely Pillow requests \\
                optuna matplotlib numpy pyproj fiona
"""

import argparse
import logging
import math
import os
import random
import warnings
from io import BytesIO

import numpy as np

warnings.filterwarnings("ignore")

import requests
from PIL import Image, ImageDraw

import geopandas as gpd
from shapely.geometry import MultiPolygon, box

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50

import optuna

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Constants
# ============================================================
ZOOM = 14                # XYZ tile zoom level
IMG_SIZE = 256           # tiles are resized to IMG_SIZE x IMG_SIZE
NUM_CLASSES = 2          # 0 = background, 1 = PV
VAL_FRACTION = 0.2       # validation split ratio (8:2 train/val)
SEED = 42

# ImageNet statistics used to normalise the input imagery.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("train")


# ============================================================
# Argument parsing
# ============================================================
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DeepLab V3+ model for PV segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shp-dir",
        required=True,
        help="Directory containing the PV vector shapefiles (.shp).",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for data, model checkpoints and reports.",
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
        "--optuna-trials",
        type=int,
        default=20,
        help="Number of Optuna hyperparameter-search trials.",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=30,
        help="Minimum number of epochs for the final training run.",
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
# 2. Read PV polygons from shapefiles (unify to WGS84)
# ============================================================
def load_shp_features(shp_dir):
    """Read every shapefile in a directory and return a list of polygons.

    All geometries are reprojected to WGS84 (EPSG:4326). Polygon,
    MultiPolygon and GeometryCollection inputs are handled.
    """
    shp_files = [
        os.path.join(shp_dir, f)
        for f in os.listdir(shp_dir)
        if f.lower().endswith(".shp")
    ]
    if not shp_files:
        log.error("No .shp files found in %s", shp_dir)
        return []

    log.info("Found %d shapefile(s): %s", len(shp_files),
             [os.path.basename(f) for f in shp_files])

    all_polygons = []
    for shp_path in shp_files:
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is None:
                log.warning("%s has no CRS; assuming WGS84.",
                            os.path.basename(shp_path))
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "Polygon":
                    all_polygons.append(geom)
                elif geom.geom_type == "MultiPolygon":
                    all_polygons.extend(geom.geoms)
                elif geom.geom_type == "GeometryCollection":
                    for sub in geom.geoms:
                        if sub.geom_type == "Polygon":
                            all_polygons.append(sub)
                        elif sub.geom_type == "MultiPolygon":
                            all_polygons.extend(sub.geoms)

            log.info("  %s: %d records", os.path.basename(shp_path), len(gdf))
        except Exception as exc:
            log.error("Failed to read %s: %s", shp_path, exc)

    log.info("Loaded %d valid polygons in total.", len(all_polygons))
    return all_polygons


# ============================================================
# 3. Find image tiles that intersect PV polygons
# ============================================================
def get_intersecting_tiles(features, zoom):
    """Return the set of (x, y) tile indices that intersect any polygon."""
    tiles = set()
    for poly in features:
        minx, miny, maxx, maxy = poly.bounds
        x_min, x_max = lng2tile(minx, zoom), lng2tile(maxx, zoom)
        y_min = min(lat2tile(maxy, zoom), lat2tile(miny, zoom))
        y_max = max(lat2tile(maxy, zoom), lat2tile(miny, zoom))
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if poly.intersects(tile_bounds_box(x, y, zoom)):
                    tiles.add((x, y))
    log.info("%d tiles intersect PV polygons.", len(tiles))
    return tiles


# ============================================================
# 4. Obtain a tile image (local file or tile server)
# ============================================================
def load_tile_image(x, y, zoom, tile_source, tile_dir, tile_base_url):
    """Return an RGB PIL image for tile (x, y), or None if unavailable.

    With tile_source='local' the tile is read from
    <tile_dir>/<x>_<y>.png. With tile_source='server' it is downloaded
    from <tile_base_url>/<zoom>/<y>/<x>.
    """
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
# 5. Rasterise PV polygons into a binary segmentation mask
# ============================================================
def make_mask(x, y, zoom, features):
    """Build an IMG_SIZE x IMG_SIZE mask (1 = PV) for tile (x, y)."""
    lt_lat, lt_lng = num2deg(x, y, zoom)
    rb_lat, rb_lng = num2deg(x + 1, y + 1, zoom)
    tile_box = box(lt_lng, rb_lat, rb_lng, lt_lat)
    lng_range = rb_lng - lt_lng
    lat_range = lt_lat - rb_lat

    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(mask)

    for poly in features:
        intersection = poly.intersection(tile_box)
        if intersection.is_empty:
            continue
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
# 6. Build the paired image/mask dataset
# ============================================================
def prepare_dataset(features, args, data_dir):
    """Generate paired image/mask tiles and return their directories."""
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    tiles = get_intersecting_tiles(features, args.zoom)
    total = len(tiles)
    saved = 0

    for i, (x, y) in enumerate(sorted(tiles), 1):
        img_path = os.path.join(img_dir, f"{x}_{y}.png")
        mask_path = os.path.join(mask_dir, f"{x}_{y}.png")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            saved += 1
            continue

        img = load_tile_image(
            x, y, args.zoom, args.tile_source,
            args.tile_dir, args.tile_base_url,
        )
        if img is None:
            continue

        img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).save(img_path)
        make_mask(x, y, args.zoom, features).save(mask_path)
        saved += 1
        if i % 25 == 0 or i == total:
            log.info("  [%d/%d] tiles processed (saved %d)", i, total, saved)

    log.info("Dataset ready: %d image/mask pairs.", saved)
    return img_dir, mask_dir


# ============================================================
# 7. Dataset with augmentation
# ============================================================
class PVSegmentationDataset(Dataset):
    """Paired image/mask dataset for binary PV segmentation."""

    def __init__(self, img_dir, mask_dir, augment=False):
        paired = sorted(
            {f for f in os.listdir(img_dir) if f.endswith(".png")}
            & {f for f in os.listdir(mask_dir) if f.endswith(".png")}
        )
        self.images = [os.path.join(img_dir, n) for n in paired]
        self.masks = [os.path.join(mask_dir, n) for n in paired]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)
            if random.random() > 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))

        img_tensor = TF.normalize(TF.to_tensor(img), NORM_MEAN, NORM_STD)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        mask_tensor = (mask_tensor > 0).long()  # ensure strictly binary
        return img_tensor, mask_tensor


# ============================================================
# 8. Evaluation metrics
# ============================================================
def compute_metrics(preds, targets, num_classes=NUM_CLASSES):
    """Compute the three validation metrics reported in the Methods section of the manuscript.

    Under the optimal hyperparameter settings the manuscript reports mIoU,
    pixel accuracy and the Dice coefficient on the held-out validation set;
    this function computes exactly those three metrics. The per-class IoU
    values are also returned through `miou`.

    Note: this is the training-stage evaluation. The separate medium
    agreement screening described in the Methods section of the manuscript
    uses IoU alone, with a 0.5 retention threshold, and is not part of
    this function.

    Returns
    -------
    (miou, pixel_accuracy, dice)
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    device = preds.device

    ious = []
    for c in range(num_classes):
        intersection = ((preds == c) & (targets == c)).sum().float()
        union = ((preds == c) | (targets == c)).sum().float()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(torch.tensor(1.0, device=device))
    miou = torch.stack(ious).mean().item()

    accuracy = (preds == targets).float().mean().item()

    pred_fg = (preds == 1).float()
    target_fg = (targets == 1).float()
    dice = (
        2 * (pred_fg * target_fg).sum()
        / (pred_fg.sum() + target_fg.sum() + 1e-6)
    ).item()
    return miou, accuracy, dice


# ============================================================
# 9. Build the DeepLab V3+ model
# ============================================================
def build_model(num_classes=NUM_CLASSES):
    """Build a DeepLab V3+ model with a ResNet-50 backbone.

    The torchvision implementation already contains the Atrous Spatial
    Pyramid Pooling (ASPP) module and the enhanced decoder described in the
    manuscript. The classifier and auxiliary head are re-initialised for the
    binary PV / non-PV task.
    """
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model.to(DEVICE)


# ============================================================
# 10. Training loop
# ============================================================
def run_training(params, img_dir, mask_dir, trial=None):
    """Train a model with a given hyperparameter configuration.

    Parameters
    ----------
    params : dict
        Hyperparameters: 'lr', 'wd', 'bs', 'epochs', 'momentum'.
    trial : optuna.Trial, optional
        If provided, intermediate validation mIoU values are reported so
        that unpromising trials can be pruned.

    Returns
    -------
    (model, history, val_loader, best_miou)
    """
    dataset = PVSegmentationDataset(img_dir, mask_dir, augment=True)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_set, batch_size=params["bs"],
                              shuffle=True, num_workers=0,
                              pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=params["bs"],
                            shuffle=False, num_workers=0,
                            pin_memory=(DEVICE.type == "cuda"))

    model = build_model()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params["lr"],
        momentum=params["momentum"],
        weight_decay=params["wd"],
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=params["epochs"], power=0.9
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_miou": [], "val_acc": [], "val_dice": []}
    best_miou = 0.0
    best_state = None

    for epoch in range(1, params["epochs"] + 1):
        # ---- training ----
        model.train()
        epoch_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(imgs)["out"]
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # ---- validation ----
        model.eval()
        mious, accuracies, dices = [], [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                logits = model(imgs)["out"]
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
                miou, accuracy, dice = compute_metrics(
                    logits.argmax(1), masks
                )
                mious.append(miou)
                accuracies.append(accuracy)
                dices.append(dice)

        val_miou = float(np.mean(mious))
        val_acc = float(np.mean(accuracies))
        val_dice = float(np.mean(dices))
        history["train_loss"].append(avg_loss)
        history["val_miou"].append(val_miou)
        history["val_acc"].append(val_acc)
        history["val_dice"].append(val_dice)
        log.info(
            "  Epoch %3d/%d  loss=%.4f  mIoU=%.4f  PixAcc=%.4f  Dice=%.4f",
            epoch, params["epochs"], avg_loss, val_miou, val_acc, val_dice,
        )

        if val_miou > best_miou:
            best_miou = val_miou
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if trial is not None:
            trial.report(val_miou, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, val_loader, best_miou


# ============================================================
# 11. Optuna hyperparameter search
# ============================================================
def optuna_search(img_dir, mask_dir, n_trials):
    """Search lr, weight decay, batch size, epochs and momentum.

    Returns the best hyperparameter dictionary.
    """
    log.info("Starting Optuna hyperparameter search (%d trials)...", n_trials)

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            "bs": trial.suggest_categorical("bs", [4, 8, 16]),
            "epochs": trial.suggest_int("epochs", 10, 30),
            "momentum": trial.suggest_float("momentum", 0.85, 0.99),
        }
        _, _, _, best_miou = run_training(
            params, img_dir, mask_dir, trial=trial
        )
        return best_miou

    study = optuna.create_study(
        direction="maximize",
        study_name="deeplabv3plus_pv",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5
        ),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log.info("Best trial #%d  mIoU=%.4f",
             study.best_trial.number, study.best_value)
    log.info("Best hyperparameters: %s", study.best_params)
    return study.best_params


# ============================================================
# 12. Reporting and model export
# ============================================================
def save_report(model, history, val_loader, best_params, output_dir):
    """Save training curves, prediction visualisations, model and report."""
    model_dir = os.path.join(output_dir, "models")
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # ---- training-curve plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], color="steelblue")
    axes[0].set_title("Training loss")
    axes[1].plot(history["val_miou"], color="darkorange")
    axes[1].set_title("Validation mIoU")
    axes[2].plot(history["val_dice"], color="green")
    axes[2].set_title("Validation Dice")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    curve_path = os.path.join(report_dir, "training_curves.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    # ---- final validation pass and prediction visualisation ----
    model.eval()
    mious, accuracies, dices = [], [], []
    samples = []
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs_d = imgs.to(DEVICE)
            masks_d = masks.to(DEVICE)
            logits = model(imgs_d)["out"]
            logits = F.interpolate(logits, size=masks_d.shape[-2:],
                                   mode="bilinear", align_corners=False)
            preds = logits.argmax(1)
            miou, accuracy, dice = compute_metrics(preds, masks_d)
            mious.append(miou)
            accuracies.append(accuracy)
            dices.append(dice)

            if len(samples) < 6:
                for i in range(min(imgs.shape[0], 6 - len(samples))):
                    rgb = (imgs[i] * std + mean).clamp(0, 1)
                    rgb = rgb.permute(1, 2, 0).numpy()
                    samples.append((rgb, masks[i].numpy(),
                                    preds[i].cpu().numpy()))

    final_miou = float(np.mean(mious))
    final_acc = float(np.mean(accuracies))
    final_dice = float(np.mean(dices))

    if samples:
        n = len(samples)
        fig2, axes2 = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes2 = [axes2]
        for i, (rgb, gt, pred) in enumerate(samples):
            axes2[i][0].imshow(rgb)
            axes2[i][0].set_title("Image")
            axes2[i][1].imshow(gt, cmap="gray")
            axes2[i][1].set_title("Ground truth")
            axes2[i][2].imshow(pred, cmap="gray")
            axes2[i][2].set_title("Prediction")
            for ax in axes2[i]:
                ax.axis("off")
        fig2.tight_layout()
        vis_path = os.path.join(report_dir, "val_predictions.png")
        fig2.savefig(vis_path, dpi=150)
        plt.close(fig2)

    # ---- save the model ----
    total_params = sum(p.numel() for p in model.parameters())
    model_path = os.path.join(model_dir, "deeplabv3plus_pv_best.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_params": best_params,
            "val_miou": final_miou,
            "val_pixel_accuracy": final_acc,
            "val_dice": final_dice,
            "total_params": total_params,
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
        },
        model_path,
    )

    # ---- text report ----
    report_lines = [
        "=" * 60,
        "DeepLab V3+ PV segmentation - training report",
        "=" * 60,
        "",
        "Best hyperparameters (Optuna search):",
        f"  learning rate  : {best_params['lr']:.6f}",
        f"  weight decay   : {best_params['wd']:.6f}",
        f"  batch size     : {best_params['bs']}",
        f"  epochs         : {best_params['epochs']}",
        f"  momentum       : {best_params['momentum']:.4f}",
        "",
        "Final validation metrics:",
        f"  mIoU             : {final_miou:.4f}",
        f"  pixel accuracy   : {final_acc:.4f}",
        f"  Dice coefficient : {final_dice:.4f}",
        "",
        f"Total parameters : {total_params:,}",
        f"Model checkpoint : {model_path}",
        f"Training curves  : {curve_path}",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    with open(os.path.join(report_dir, "training_report.txt"),
              "w", encoding="utf-8") as handle:
        handle.write(report_text)
    print("\n" + report_text)
    log.info("Report written to %s", report_dir)


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

    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "train.log"), encoding="utf-8"
            ),
        ],
    )
    log.info("Device: %s", DEVICE)

    # Step 1: read PV vector polygons.
    features = load_shp_features(args.shp_dir)
    if not features:
        log.error("No valid PV polygons were read; aborting.")
        return

    # Step 2: build paired image/mask tiles.
    img_dir, mask_dir = prepare_dataset(features, args, data_dir)

    # Step 3: check the dataset size.
    n_tiles = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
    if n_tiles < 2:
        log.error("Too few samples (%d); aborting.", n_tiles)
        return
    log.info("Dataset size: %d tiles.", n_tiles)

    # Step 4: Optuna hyperparameter search.
    best_params = optuna_search(img_dir, mask_dir, args.optuna_trials)

    # Step 5: final training run (at least --final-epochs epochs).
    best_params["epochs"] = max(best_params.get("epochs", args.final_epochs),
                                args.final_epochs)
    log.info("Final training run with hyperparameters: %s", best_params)
    model, history, val_loader, _ = run_training(
        best_params, img_dir, mask_dir
    )

    # Step 6: export the model, report and visualisations.
    save_report(model, history, val_loader, best_params, args.output_dir)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
