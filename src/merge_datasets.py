"""Merge multiple waste-classification datasets into one TRAIN/TEST structure.

Five additional Kaggle datasets are pre-configured with class mappings
to Organic (O) / Recyclable (R).  Download them, extract into
extra_datasets/<name>/, and run this script.

Usage:
    python -m src.merge_datasets                  # merge all found datasets
    python -m src.merge_datasets --dry-run        # preview what would happen
    python -m src.merge_datasets --list           # show dataset registry
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass, field

# ── class mapping: folder-name -> O (Organic) or R (Recyclable) ──────────

# Organic (wet / biodegradable): food scraps, biological matter, vegetation
# Recyclable (dry / recyclable): paper, cardboard, glass, metal, plastic

ORGANIC = "O"
RECYCLABLE = "R"
SKIP = None  # ambiguous items excluded

UNIVERSAL_MAP: dict[str, str | None] = {
    # organic / wet
    "o": ORGANIC, "O": ORGANIC, "organic": ORGANIC, "Organic": ORGANIC,
    "biological": ORGANIC, "food_waste": ORGANIC, "food": ORGANIC,
    "Food_Organics": ORGANIC, "food_organics": ORGANIC,
    "Vegetation": ORGANIC, "vegetation": ORGANIC,
    "trash": ORGANIC, "Trash": ORGANIC,
    # recyclable / dry
    "r": RECYCLABLE, "R": RECYCLABLE, "recyclable": RECYCLABLE, "Recyclable": RECYCLABLE,
    "paper": RECYCLABLE, "Paper": RECYCLABLE,
    "cardboard": RECYCLABLE, "Cardboard": RECYCLABLE,
    "glass": RECYCLABLE, "Glass": RECYCLABLE,
    "metal": RECYCLABLE, "Metal": RECYCLABLE,
    "plastic": RECYCLABLE, "Plastic": RECYCLABLE,
    "brown-glass": RECYCLABLE, "green-glass": RECYCLABLE, "white-glass": RECYCLABLE,
    "aluminum": RECYCLABLE, "tin": RECYCLABLE,
    # skip (ambiguous / hazardous)
    "battery": SKIP, "Battery": SKIP,
    "clothes": SKIP, "Clothes": SKIP,
    "shoes": SKIP, "Shoes": SKIP,
    "Textile": SKIP, "textile": SKIP,
    "Miscellaneous": SKIP, "miscellaneous": SKIP,
    "electronics": SKIP,
}


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
    ),
    "waste_v2": DatasetInfo(
        name="Waste Classification v2",
        slug="techsash/waste-classification-data",
        desc="~25,000 images: Organic / Recyclable (same format as primary dataset)",
        has_split=True,
        train_dir="TRAIN",
        test_dir="TEST",
        class_map={"O": ORGANIC, "R": RECYCLABLE},
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
    ),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def _list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]


def _find_class_dirs(root: str, class_map: dict[str, str | None]) -> list[tuple[str, str]]:
    found = []
    if not os.path.isdir(root):
        return found
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        mapped = class_map.get(entry, UNIVERSAL_MAP.get(entry))
        if mapped is None:
            continue
        if mapped in (ORGANIC, RECYCLABLE):
            found.append((full, mapped))
    return found


def _auto_find_root(base: str, ds: DatasetInfo) -> str | None:
    if ds.images_dir:
        candidate = os.path.join(base, ds.images_dir)
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(base):
        entries = os.listdir(base)
        known_classes = set(ds.class_map.keys())
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
) -> dict[str, int]:
    rng = random.Random(seed)
    train_o = os.path.join(target_dir, "TRAIN", "O")
    train_r = os.path.join(target_dir, "TRAIN", "R")
    test_o = os.path.join(target_dir, "TEST", "O")
    test_r = os.path.join(target_dir, "TEST", "R")

    if not dry_run:
        for d in (train_o, train_r, test_o, test_r):
            os.makedirs(d, exist_ok=True)

    stats: dict[str, int] = {}
    total_added = 0

    for ds_key, ds_info in DATASETS.items():
        ds_dir = os.path.join(extra_dir, ds_key)
        if not os.path.isdir(ds_dir):
            continue

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {ds_info.name} ({ds_key}/)")
        added = 0

        if ds_info.has_split:
            for split, target_map in [("TRAIN", {ORGANIC: train_o, RECYCLABLE: train_r}),
                                       ("TEST", {ORGANIC: test_o, RECYCLABLE: test_r})]:
                split_path = os.path.join(ds_dir, ds_info.train_dir if split == "TRAIN" else ds_info.test_dir)
                if not os.path.isdir(split_path):
                    sub_entries = os.listdir(ds_dir) if os.path.isdir(ds_dir) else []
                    for entry in sub_entries:
                        candidate = os.path.join(ds_dir, entry, ds_info.train_dir if split == "TRAIN" else ds_info.test_dir)
                        if os.path.isdir(candidate):
                            split_path = candidate
                            break
                pairs = _find_class_dirs(split_path, ds_info.class_map)
                for src_folder, mapped_class in pairs:
                    images = _list_images(src_folder)
                    target = target_map[mapped_class]
                    for img in images:
                        dst = os.path.join(target, f"{ds_key}_{img}")
                        if not dry_run and not os.path.exists(dst):
                            shutil.copy2(os.path.join(src_folder, img), dst)
                        added += 1
        else:
            root = _auto_find_root(ds_dir, ds_info)
            if root is None:
                print(f"  Could not find class folders in {ds_dir}")
                continue
            pairs = _find_class_dirs(root, ds_info.class_map)
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

                train_target = train_o if mapped_class == ORGANIC else train_r
                test_target = test_o if mapped_class == ORGANIC else test_r

                for img in train_imgs:
                    dst = os.path.join(train_target, f"{ds_key}_{img}")
                    if not dry_run and not os.path.exists(dst):
                        shutil.copy2(os.path.join(src_folder, img), dst)
                for img in test_imgs:
                    dst = os.path.join(test_target, f"{ds_key}_{img}")
                    if not dry_run and not os.path.exists(dst):
                        shutil.copy2(os.path.join(src_folder, img), dst)
                added += len(images)

        stats[ds_key] = added
        total_added += added
        print(f"  -> {added:,} images {'would be' if dry_run else ''} added")

    return stats


def show_counts(dataset_dir: str) -> None:
    print("\nCurrent dataset counts:")
    for split in ("TRAIN", "TEST"):
        for cls in ("O", "R"):
            p = os.path.join(dataset_dir, split, cls)
            n = len(_list_images(p)) if os.path.isdir(p) else 0
            label = "Organic" if cls == "O" else "Recyclable"
            print(f"  {split}/{cls} ({label}): {n:,} images")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge extra waste datasets")
    ap.add_argument("--extra-dir", default="./extra_datasets",
                    help="Folder containing downloaded dataset subfolders")
    ap.add_argument("--target-dir", default="./dataset/DATASET",
                    help="Target dataset directory with TRAIN/TEST structure")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview what would happen without copying")
    ap.add_argument("--list", action="store_true",
                    help="Show dataset registry and exit")
    ap.add_argument("--counts", action="store_true",
                    help="Show current dataset image counts")
    args = ap.parse_args()

    if args.list:
        print("=" * 65)
        print("Dataset Registry — download from Kaggle, extract into extra_datasets/<key>/")
        print("=" * 65)
        for key, ds in DATASETS.items():
            print(f"\n  {key}/")
            print(f"    {ds.name}")
            print(f"    Kaggle: kaggle datasets download -d {ds.slug}")
            print(f"    {ds.desc}")
            mapping = {k: v for k, v in ds.class_map.items() if v is not None}
            skipped = [k for k, v in ds.class_map.items() if v is None]
            o_classes = [k for k, v in mapping.items() if v == ORGANIC]
            r_classes = [k for k, v in mapping.items() if v == RECYCLABLE]
            print(f"    Organic  <- {', '.join(o_classes)}")
            print(f"    Recycle  <- {', '.join(r_classes)}")
            if skipped:
                print(f"    Skipped  <- {', '.join(skipped)}")
        return

    if args.counts:
        show_counts(args.target_dir)
        return

    print("=" * 60)
    print("Multi-Dataset Merger — Organic / Recyclable")
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
    stats = merge(args.extra_dir, args.target_dir, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("Summary:")
    total = 0
    for key, count in stats.items():
        print(f"  {key}: {count:,} images")
        total += count
    print(f"  TOTAL added: {total:,} images")

    if not args.dry_run:
        show_counts(args.target_dir)
        print("\nDone! Now retrain: python -m src.train --model both")


if __name__ == "__main__":
    main()
