#!/usr/bin/env python3
"""Build TAPIS region-feature .pth files directly from Mask2Former predictions,
with no ground-truth matching.

Unlike match_annots_n_preds.py, this script does not require ground-truth boxes
to IoU-match against predictions and assign labels -- it is for datasets with no
instance-level annotations at all (e.g. MultiBypass). Predicted `category_id` /
`score_dist` are dropped: for MultiBypass they are GraSP's 7-class robotic
instrument ontology, which is semantically wrong for a laparoscopic instrument
set. Only box + mask_embd (the region feature) are carried forward.

Output schema matches tapis/datasets/surgical_dataset_helper.py:load_features_boxes,
one dict per image:
    {'image_id', 'file_name', 'width', 'height',
     'obj_features': {(x1, y1, x2, y2): [float, ...]}}
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


def xywh_to_x1y1x2y2(bbox):
    """[x, y, w, h] -> [x1, y1, x2, y2], rounded to the nearest integer."""
    bbox = list(map(round, bbox))
    return [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]


def build_feature_entries(preds_by_id, images, feature_keys, min_score):
    """Mask2Former's query-based decoder frequently emits several near/exact-duplicate
    box predictions for the same object at low score thresholds (no NMS is applied).
    obj_features is keyed by (rounded) box coordinates, so any such duplicates would
    silently collide; resolve collisions by keeping the highest-scoring instance,
    mirroring match_annots_n_preds.py:remove_duplicates_n_features.
    """
    entries = []
    n_raw = 0
    for im in tqdm(images, desc="Building features"):
        insts = [i for i in preds_by_id.get(im["id"], []) if i["score"] >= min_score]
        n_raw += len(insts)
        insts.sort(key=lambda i: i["score"])  # ascending, so later (higher-score) wins ties
        obj_features = {}
        for inst in insts:
            box_key = tuple(xywh_to_x1y1x2y2(inst["bbox"]))
            feat = []
            for key in feature_keys:
                feat.extend(inst[key])
            obj_features[box_key] = feat
        entries.append({
            "image_id": im["id"],
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
            "obj_features": obj_features,
        })
    n_kept = sum(len(e["obj_features"]) for e in entries)
    print(f"  raw score-filtered instances: {n_raw}, after box-collision dedup: {n_kept} "
          f"({n_raw - n_kept} collisions, {100 * (n_raw - n_kept) / max(n_raw, 1):.1f}%)")
    return entries


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pth", type=Path, required=True,
                     help="Path to instances_predictions.pth (raw Mask2Former output, "
                          "e.g. .../tapis_region_proposals/output/swinl_cutmargin512/inference/instances_predictions.pth).")
    ap.add_argument("--coco-json", type=Path, required=True,
                     help="Images-only COCO json used at inference time "
                          "(id -> file_name/width/height), e.g. data_shim/multibypass_infer.json.")
    ap.add_argument("--folds-json", type=Path,
                     default=here / "multibypasst40_challenge_trainval/folds.json")
    ap.add_argument("--out-dir", type=Path,
                     default=here / "multibypasst40_challenge_trainval/features")
    ap.add_argument("--min-score", type=float, default=0.05,
                     help="Drop detections below this score (default 0.05, matches "
                          "the threshold already used for coco_per_video/*.json).")
    ap.add_argument("--feature-keys", nargs="+", default=["mask_embd"],
                     help="Instance dict keys to concatenate into the region feature "
                          "vector (default: mask_embd alone, 256-d, matching TAPIS's "
                          "cfg.FEATURES.DIM_FEATURES=256 convention).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO shim json: {args.coco_json}")
    with args.coco_json.open() as f:
        shim = json.load(f)
    images_by_case = defaultdict(list)
    for im in shim["images"]:
        case = im["file_name"].split("/")[0]
        images_by_case[case].append(im)

    print(f"Loading predictions .pth: {args.pth} (large file, this may take a few minutes)")
    preds = torch.load(args.pth, map_location="cpu")
    preds_by_id = {e["image_id"]: e["instances"] for e in preds}
    del preds

    with args.folds_json.open() as f:
        folds = json.load(f)
    fold1_videos = set(folds["fold_1"]["videos"])
    fold2_videos = set(folds["fold_2"]["videos"])
    all_videos = fold1_videos | fold2_videos
    missing = all_videos - set(images_by_case)
    if missing:
        raise ValueError(f"Videos in {args.folds_json} not found in {args.coco_json}: {missing}")

    def images_for(videos):
        out = []
        for v in sorted(videos):
            out.extend(images_by_case[v])
        return out

    # Naming mirrors data/GraSP/features/*_region_features.pth: for a given fold,
    # both _train and _val hold that fold's own videos -- run_files/*.sh picks which
    # whole fold trains and which validates (e.g. TRAIN_FOLD=fold2, TEST_FOLD=fold1),
    # so the held-out split happens across folds, not within one fold's file. GraSP's
    # _train/_val differ by GT-matching mode (Hungarian-matched vs filtered-only);
    # with no GT here, both modes degenerate to the same score-filter+dedup content.
    splits = {
        "fold1_train": images_for(fold1_videos),
        "fold1_val": images_for(fold1_videos),
        "fold2_train": images_for(fold2_videos),
        "fold2_val": images_for(fold2_videos),
        "train_train": images_for(all_videos),
    }

    expected_dim = None
    for name, images in splits.items():
        print(f"\nBuilding {name}: {len(images)} images")
        entries = build_feature_entries(preds_by_id, images, args.feature_keys, args.min_score)

        n_boxes = sum(len(e["obj_features"]) for e in entries)
        dims = {len(v) for e in entries for v in e["obj_features"].values()}
        if len(dims) > 1:
            raise ValueError(f"Inconsistent feature dims in {name}: {dims}")
        if dims:
            expected_dim = expected_dim or next(iter(dims))
            assert next(iter(dims)) == expected_dim, \
                f"{name} feature dim {dims} != previous splits' {expected_dim}"

        out_path = args.out_dir / f"{name}_region_features.pth"
        torch.save(entries, out_path)
        print(f"  -> {out_path} ({n_boxes} boxes, dim={dims})")


if __name__ == "__main__":
    main()
