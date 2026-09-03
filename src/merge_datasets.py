"""Merge multiple waste-classification datasets into one TRAIN/TEST structure.

Supports 2-class (O/R), 4-class (H/N/O/R), or 5-class (E/H/N/O/R) output.
Download datasets from Kaggle, extract into extra_datasets/<name>/, and run.

Usage:
    python -m src.merge_datasets                  # merge all found datasets (5-class)
    python -m src.merge_datasets --classes 4      # merge as 4-class (E→H)
    python -m src.merge_datasets --dry-run        # preview what would happen
    python -m src.merge_datasets --list           # show dataset registry
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import zipfile
from dataclasses import dataclass, field

# ── class mapping: folder-name -> O (Organic) or R (Recyclable) ──────────

# Organic (wet / biodegradable): food scraps, biological matter, vegetation
# Recyclable (dry / recyclable): paper, cardboard, glass, metal, plastic

ORGANIC = "O"
RECYCLABLE = "R"
HAZARDOUS = "H"
E_WASTE = "E"
NON_RECYCLABLE = "N"
SKIP = None  # ambiguous items excluded in 2-class mode

# 5-class universal map — the most granular
UNIVERSAL_MAP_5: dict[str, str] = {
    # organic / biodegradable
    "o": ORGANIC, "O": ORGANIC, "organic": ORGANIC, "Organic": ORGANIC,
    "biological": ORGANIC, "food_waste": ORGANIC, "food": ORGANIC,
    "Food_Organics": ORGANIC, "food_organics": ORGANIC,
    "Vegetation": ORGANIC, "vegetation": ORGANIC,
    # recyclable
    "r": RECYCLABLE, "R": RECYCLABLE, "recyclable": RECYCLABLE, "Recyclable": RECYCLABLE,
    "paper": RECYCLABLE, "Paper": RECYCLABLE,
    "cardboard": RECYCLABLE, "Cardboard": RECYCLABLE,
    "glass": RECYCLABLE, "Glass": RECYCLABLE,
    "metal": RECYCLABLE, "Metal": RECYCLABLE,
    "plastic": RECYCLABLE, "Plastic": RECYCLABLE,
    "brown-glass": RECYCLABLE, "green-glass": RECYCLABLE, "white-glass": RECYCLABLE,
    "aluminum": RECYCLABLE, "tin": RECYCLABLE,
    # hazardous
    "battery": HAZARDOUS, "Battery": HAZARDOUS, "batteries": HAZARDOUS,
    "Batteries": HAZARDOUS, "chemicals": HAZARDOUS,
    # e-waste (electronics)
    "computer": E_WASTE, "Computer": E_WASTE, "computers": E_WASTE,
    "keyboard": E_WASTE, "Keyboard": E_WASTE, "keyboards": E_WASTE,
    "mouse": E_WASTE, "Mouse": E_WASTE, "mice": E_WASTE, "Mice": E_WASTE,
    "mobile": E_WASTE, "Mobile": E_WASTE, "mobiles": E_WASTE, "Mobiles": E_WASTE,
    "phone": E_WASTE, "Phone": E_WASTE, "phones": E_WASTE,
    "printer": E_WASTE, "Printer": E_WASTE, "printers": E_WASTE,
    "television": E_WASTE, "Television": E_WASTE, "TV": E_WASTE,
    "PCB": E_WASTE, "pcb": E_WASTE, "PCBs": E_WASTE,
    "microwave": E_WASTE, "Microwave": E_WASTE, "microwaves": E_WASTE,
    "washing_machine": E_WASTE, "WashingMachine": E_WASTE,
    "speaker": E_WASTE, "Speaker": E_WASTE, "speakers": E_WASTE,
    "player": E_WASTE, "Player": E_WASTE,
    "electronics": E_WASTE, "Electronics": E_WASTE,
    "laptop": E_WASTE, "Laptop": E_WASTE,
    "monitor": E_WASTE, "Monitor": E_WASTE,
    "cable": E_WASTE, "Cable": E_WASTE, "wires": E_WASTE,
    # non-recyclable
    "clothes": NON_RECYCLABLE, "Clothes": NON_RECYCLABLE,
    "shoes": NON_RECYCLABLE, "Shoes": NON_RECYCLABLE,
    "Textile": NON_RECYCLABLE, "textile": NON_RECYCLABLE,
    "trash": NON_RECYCLABLE, "Trash": NON_RECYCLABLE,
    "Miscellaneous": NON_RECYCLABLE, "miscellaneous": NON_RECYCLABLE,
}

# 4-class: merge E-waste into Hazardous
UNIVERSAL_MAP_4: dict[str, str] = {
    k: (HAZARDOUS if v == E_WASTE else v) for k, v in UNIVERSAL_MAP_5.items()
}

# 2-class: keep only O and R
UNIVERSAL_MAP_2: dict[str, str | None] = {
    k: (v if v in (ORGANIC, RECYCLABLE) else SKIP) for k, v in UNIVERSAL_MAP_5.items()
}

UNIVERSAL_MAP = UNIVERSAL_MAP_5  # default


@dataclass
class DatasetInfo:
    name: str
    slug: str  # kaggle dataset slug
    desc: str
    has_split: bool = False  # True if dataset has its own TRAIN/TEST
    train_dir: str = ""  # relative path inside extracted folder
    test_dir: str = ""
    images_dir: str = ""  # for datasets without train/test split
    class_map: dict[str, str | None] = field(default_factory=dict)
    class_map_4: dict[str, str | None] = field(default_factory=dict)
    class_map_5: dict[str, str | None] = field(default_factory=dict)


# ── dataset registry ─────────────────────────────────────────────────────

DATASETS: dict[str, DatasetInfo] = {
    "trashnet": DatasetInfo(
        name="TrashNet (6-class)",
        slug="mostafaabla/garbage-classification",
        desc="~2,500 images: cardboard, glass, metal, paper, plastic, trash",
        images_dir="Garbage classification/Garbage classification",
        class_map={
            "cardboard": RECYCLABLE, "glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE, "trash": ORGANIC,
        },
        class_map_4={
            "cardboard": RECYCLABLE, "glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE, "trash": NON_RECYCLABLE,
        },
        class_map_5={
            "cardboard": RECYCLABLE, "glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE, "trash": NON_RECYCLABLE,
        },
    ),
    "garbage12": DatasetInfo(
        name="Garbage Classification (12-class)",
        slug="mostafaabla/garbage-classification-12",
        desc="~15,000 images: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass",
        images_dir="",
        class_map={
            "biological": ORGANIC, "trash": ORGANIC,
            "cardboard": RECYCLABLE, "brown-glass": RECYCLABLE,
            "green-glass": RECYCLABLE, "white-glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE,
            "battery": SKIP, "clothes": SKIP, "shoes": SKIP,
        },
        class_map_4={
            "biological": ORGANIC,
            "cardboard": RECYCLABLE, "brown-glass": RECYCLABLE,
            "green-glass": RECYCLABLE, "white-glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE,
            "battery": HAZARDOUS,
            "clothes": NON_RECYCLABLE, "shoes": NON_RECYCLABLE,
            "trash": NON_RECYCLABLE,
        },
        class_map_5={
            "biological": ORGANIC,
            "cardboard": RECYCLABLE, "brown-glass": RECYCLABLE,
            "green-glass": RECYCLABLE, "white-glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE,
            "battery": HAZARDOUS,
            "clothes": NON_RECYCLABLE, "shoes": NON_RECYCLABLE,
            "trash": NON_RECYCLABLE,
        },
    ),
    "realwaste": DatasetInfo(
        name="RealWaste (9-class)",
        slug="joebeachcapital/realwaste",
        desc="~4,700 images: Cardboard, Food_Organics, Glass, Metal, Miscellaneous, Paper, Plastic, Textile, Vegetation",
        images_dir="RealWaste",
        class_map={
            "Food_Organics": ORGANIC, "Vegetation": ORGANIC,
            "Cardboard": RECYCLABLE, "Glass": RECYCLABLE,
            "Metal": RECYCLABLE, "Paper": RECYCLABLE,
            "Plastic": RECYCLABLE,
            "Miscellaneous": SKIP, "Textile": SKIP,
        },
        class_map_4={
            "Food_Organics": ORGANIC, "Vegetation": ORGANIC,
            "Cardboard": RECYCLABLE, "Glass": RECYCLABLE,
            "Metal": RECYCLABLE, "Paper": RECYCLABLE,
            "Plastic": RECYCLABLE,
            "Miscellaneous": NON_RECYCLABLE, "Textile": NON_RECYCLABLE,
        },
        class_map_5={
            "Food_Organics": ORGANIC, "Vegetation": ORGANIC,
            "Cardboard": RECYCLABLE, "Glass": RECYCLABLE,
            "Metal": RECYCLABLE, "Paper": RECYCLABLE,
            "Plastic": RECYCLABLE,
            "Miscellaneous": NON_RECYCLABLE, "Textile": NON_RECYCLABLE,
        },
    ),
    "waste_v2": DatasetInfo(
        name="Waste Classification v2",
        slug="techsash/waste-classification-data",
        desc="~25,000 images: Organic / Recyclable (same format as primary dataset)",
        has_split=True,
        train_dir="TRAIN",
        test_dir="TEST",
        class_map={"O": ORGANIC, "R": RECYCLABLE},
        class_map_4={"O": ORGANIC, "R": RECYCLABLE},
        class_map_5={"O": ORGANIC, "R": RECYCLABLE},
    ),
    "household": DatasetInfo(
        name="Household Waste (6-class)",
        slug="sumn2u/garbage-classification-v2",
        desc="~2,500 images: cardboard, glass, metal, paper, plastic, trash",
        images_dir="",
        class_map={
            "cardboard": RECYCLABLE, "glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE, "trash": ORGANIC,
        },
        class_map_4={
            "cardboard": RECYCLABLE, "glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE, "trash": NON_RECYCLABLE,
        },
        class_map_5={
            "cardboard": RECYCLABLE, "glass": RECYCLABLE,
            "metal": RECYCLABLE, "paper": RECYCLABLE,
            "plastic": RECYCLABLE, "trash": NON_RECYCLABLE,
        },
    ),
    "ewaste1": DatasetInfo(
        name="E-Waste Image Dataset",
        slug="akshat103/e-waste-image-dataset",
        desc="~3,600 images: batteries, computers, keyboards, mice, mobiles, printers, TVs, PCBs, microwaves, washing machines, speakers, player devices",
        images_dir="",
        class_map={
            "batteries": SKIP, "computer": SKIP, "keyboard": SKIP,
            "mouse": SKIP, "mobile": SKIP, "printer": SKIP,
            "television": SKIP, "PCB": SKIP, "microwave": SKIP,
            "washing_machine": SKIP, "speaker": SKIP, "player": SKIP,
        },
        class_map_4={
            "batteries": HAZARDOUS, "computer": HAZARDOUS, "keyboard": HAZARDOUS,
            "mouse": HAZARDOUS, "mobile": HAZARDOUS, "printer": HAZARDOUS,
            "television": HAZARDOUS, "PCB": HAZARDOUS, "microwave": HAZARDOUS,
            "washing_machine": HAZARDOUS, "speaker": HAZARDOUS, "player": HAZARDOUS,
        },
        class_map_5={
            "batteries": HAZARDOUS, "computer": E_WASTE, "keyboard": E_WASTE,
            "mouse": E_WASTE, "mobile": E_WASTE, "printer": E_WASTE,
            "television": E_WASTE, "PCB": E_WASTE, "microwave": E_WASTE,
            "washing_machine": E_WASTE, "speaker": E_WASTE, "player": E_WASTE,
        },
    ),
    "ewaste2": DatasetInfo(
        name="E-Waste Dataset",
        slug="kaustubh2402/ewaste-dataset",
        desc="E-waste images across multiple electronic device categories",
        images_dir="",
        class_map={},
        class_map_4={},
        class_map_5={},
    ),
    "wastecls": DatasetInfo(
        name="Waste Classification (30K+)",
        slug="phenomsg/waste-classification",
        desc="~30,000 images across multiple waste categories",
        images_dir="",
        class_map={
            "organic": ORGANIC, "recyclable": RECYCLABLE,
        },
        class_map_4={
            "organic": ORGANIC, "recyclable": RECYCLABLE,
        },
        class_map_5={
            "organic": ORGANIC, "recyclable": RECYCLABLE,
        },
    ),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def _list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]


ALL_VALID = {ORGANIC, RECYCLABLE, HAZARDOUS, E_WASTE, NON_RECYCLABLE}


def _find_class_dirs(
    root: str,
    class_map: dict[str, str | None],
    valid: set[str] = ALL_VALID,
) -> list[tuple[str, str]]:
    found = []
    if not os.path.isdir(root):
        return found
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        mapped = class_map.get(entry, UNIVERSAL_MAP.get(entry))
        if mapped is None or mapped not in valid:
            continue
        found.append((full, mapped))
    return found


def _extract_zips(base: str) -> None:
    if not os.path.isdir(base):
        return
    for f in os.listdir(base):
        if not f.lower().endswith(".zip"):
            continue
        zp = os.path.join(base, f)
        try:
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(base)
            print(f"  Extracted {f}")
        except Exception as e:
            print(f"  Could not extract {f}: {e}")


def _auto_find_root(base: str, ds: DatasetInfo) -> str | None:
    _extract_zips(base)
    if ds.images_dir:
        candidate = os.path.join(base, ds.images_dir)
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(base):
        entries = os.listdir(base)
        known_classes = set(ds.class_map.keys()) | set(ds.class_map_4.keys()) | set(ds.class_map_5.keys())
        if known_classes & set(entries):
            return base
        for e in entries:
            sub = os.path.join(base, e)
            if os.path.isdir(sub):
                sub_entries = set(os.listdir(sub))
                if known_classes & sub_entries:
                    return sub
    return base


def merge(
    extra_dir: str,
    target_dir: str,
    test_ratio: float = 0.2,
    dry_run: bool = False,
    seed: int = 42,
    num_classes: int = 2,
) -> dict[str, int]:
    rng = random.Random(seed)
    if num_classes == 2:
        all_classes = ["O", "R"]
    elif num_classes == 4:
        all_classes = ["H", "N", "O", "R"]
    else:
        all_classes = ["E", "H", "N", "O", "R"]
    train_dirs = {c: os.path.join(target_dir, "TRAIN", c) for c in all_classes}
    test_dirs = {c: os.path.join(target_dir, "TEST", c) for c in all_classes}

    if not dry_run:
        for d in list(train_dirs.values()) + list(test_dirs.values()):
            os.makedirs(d, exist_ok=True)

    stats: dict[str, int] = {}
    total_added = 0

    valid = set(all_classes)

    for ds_key, ds_info in DATASETS.items():
        ds_dir = os.path.join(extra_dir, ds_key)
        if not os.path.isdir(ds_dir):
            continue

        if num_classes == 5:
            cmap = ds_info.class_map_5
        elif num_classes == 4:
            cmap = ds_info.class_map_4
        else:
            cmap = ds_info.class_map
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {ds_info.name} ({ds_key}/)")
        added = 0

        if ds_info.has_split:
            for split, tgt in [("TRAIN", train_dirs), ("TEST", test_dirs)]:
                split_path = os.path.join(ds_dir, ds_info.train_dir if split == "TRAIN" else ds_info.test_dir)
                if not os.path.isdir(split_path):
                    sub_entries = os.listdir(ds_dir) if os.path.isdir(ds_dir) else []
                    for entry in sub_entries:
                        candidate = os.path.join(ds_dir, entry, ds_info.train_dir if split == "TRAIN" else ds_info.test_dir)
                        if os.path.isdir(candidate):
                            split_path = candidate
                            break
                pairs = _find_class_dirs(split_path, cmap, valid)
                for src_folder, mapped_class in pairs:
                    images = _list_images(src_folder)
                    for img in images:
                        dst = os.path.join(tgt[mapped_class], f"{ds_key}_{img}")
                        if not dry_run and not os.path.exists(dst):
                            shutil.copy2(os.path.join(src_folder, img), dst)
                        added += 1
        else:
            root = _auto_find_root(ds_dir, ds_info)
            if root is None:
                print(f"  Could not find class folders in {ds_dir}")
                continue
            pairs = _find_class_dirs(root, cmap, valid)
            if not pairs:
                print(f"  No recognized class folders in {root}")
                print(f"  Found: {os.listdir(root) if os.path.isdir(root) else 'nothing'}")
                continue
            for src_folder, mapped_class in pairs:
                images = _list_images(src_folder)
                rng.shuffle(images)
                split_idx = max(1, int(len(images) * (1 - test_ratio)))
                train_imgs = images[:split_idx]
                test_imgs = images[split_idx:]

                for img in train_imgs:
                    dst = os.path.join(train_dirs[mapped_class], f"{ds_key}_{img}")
                    if not dry_run and not os.path.exists(dst):
                        shutil.copy2(os.path.join(src_folder, img), dst)
                for img in test_imgs:
                    dst = os.path.join(test_dirs[mapped_class], f"{ds_key}_{img}")
                    if not dry_run and not os.path.exists(dst):
                        shutil.copy2(os.path.join(src_folder, img), dst)
                added += len(images)

        stats[ds_key] = added
        total_added += added
        print(f"  -> {added:,} images {'would be' if dry_run else ''} added")

    return stats


DIR_LABELS = {"E": "E-waste", "H": "Hazardous", "N": "Non-Recyclable", "O": "Organic", "R": "Recyclable"}


def show_counts(dataset_dir: str, num_classes: int = 5) -> None:
    if num_classes == 2:
        classes = ["O", "R"]
    elif num_classes == 4:
        classes = ["H", "N", "O", "R"]
    else:
        classes = ["E", "H", "N", "O", "R"]
    print("\nCurrent dataset counts:")
    for split in ("TRAIN", "TEST"):
        for cls in classes:
            p = os.path.join(dataset_dir, split, cls)
            n = len(_list_images(p)) if os.path.isdir(p) else 0
            print(f"  {split}/{cls} ({DIR_LABELS[cls]}): {n:,} images")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge extra waste datasets")
    ap.add_argument("--extra-dir", default="./extra_datasets",
                    help="Folder containing downloaded dataset subfolders")
    ap.add_argument("--target-dir", default="./dataset/DATASET",
                    help="Target dataset directory with TRAIN/TEST structure")
    ap.add_argument("--classes", type=int, choices=[2, 4, 5], default=5,
                    help="Number of output classes (2=O/R, 4=H/N/O/R, 5=E/H/N/O/R)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview what would happen without copying")
    ap.add_argument("--list", action="store_true",
                    help="Show dataset registry and exit")
    ap.add_argument("--counts", action="store_true",
                    help="Show current dataset image counts")
    args = ap.parse_args()

    nc = args.classes

    if args.list:
        label = {2: "2-class (O/R)", 4: "4-class (H/N/O/R)", 5: "5-class (E/H/N/O/R)"}[nc]
        print("=" * 65)
        print(f"Dataset Registry [{label}] — download from Kaggle, extract into extra_datasets/<key>/")
        print("=" * 65)
        for key, ds in DATASETS.items():
            cmap = ds.class_map_5 if nc == 5 else (ds.class_map_4 if nc == 4 else ds.class_map)
            print(f"\n  {key}/")
            print(f"    {ds.name}")
            print(f"    Kaggle: kaggle datasets download -d {ds.slug}")
            print(f"    {ds.desc}")
            mapping = {k: v for k, v in cmap.items() if v is not None}
            skipped = [k for k, v in cmap.items() if v is None]
            for code, lbl in DIR_LABELS.items():
                cls_list = [k for k, v in mapping.items() if v == code]
                if cls_list:
                    print(f"    {lbl:15s} <- {', '.join(cls_list)}")
            if skipped:
                print(f"    {'Skipped':15s} <- {', '.join(skipped)}")
        return

    if args.counts:
        show_counts(args.target_dir, nc)
        return

    class_label = {2: "Organic / Recyclable", 4: "H / N / O / R", 5: "E / H / N / O / R"}[nc]
    print("=" * 60)
    print(f"Multi-Dataset Merger — {class_label}")
    print("=" * 60)
    print(f"Extra datasets dir : {os.path.abspath(args.extra_dir)}")
    print(f"Target dataset dir : {os.path.abspath(args.target_dir)}")
    if args.dry_run:
        print("MODE: DRY RUN (no files will be copied)")

    if not os.path.isdir(args.extra_dir):
        print(f"\nFolder '{args.extra_dir}' not found.")
        print("Create it and place extracted datasets inside:")
        print("  extra_datasets/")
        for key in DATASETS:
            print(f"    {key}/")
        print("\nRun --list to see download commands.")
        return

    found = [k for k in DATASETS if os.path.isdir(os.path.join(args.extra_dir, k))]
    if not found:
        print(f"\nNo recognized datasets in {args.extra_dir}/")
        print(f"Found: {os.listdir(args.extra_dir)}")
        print(f"Expected folder names: {', '.join(DATASETS.keys())}")
        return

    print(f"\nFound {len(found)} dataset(s): {', '.join(found)}")
    stats = merge(args.extra_dir, args.target_dir, dry_run=args.dry_run, num_classes=nc)

    print("\n" + "=" * 60)
    print("Summary:")
    total = 0
    for key, count in stats.items():
        print(f"  {key}: {count:,} images")
        total += count
    print(f"  TOTAL added: {total:,} images")

    if not args.dry_run:
        show_counts(args.target_dir, nc)
        print("\nDone! Now retrain: python -m src.train --model both")


if __name__ == "__main__":
    main()
