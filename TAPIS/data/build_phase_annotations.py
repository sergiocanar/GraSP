#!/usr/bin/env python3
"""Build TAPIS-format GT annotation JSONs for MultiBypass's phases task.

label_files_challenge/<video>.json already carries a dense phase_label (0-11)
per frame, with each image's 0-indexed `id` matching the frame's filename in
videos_cutmargin512/<video>/ (verified: id range == frame count per video, and
frame_lists/*.csv was built with this same 0-indexed numbering). No boxes are
needed here -- REGIONS.ENABLE is False for the phases task, so this only needs
one phase label per frame.

A handful of frames (5 total, across C2V4/C2V5) have phase_label == None --
presumably past the end of the recorded procedure -- and are skipped.
"""
import argparse
import json
from pathlib import Path

PHASE_CATEGORIES = None  # filled in from the first label file read


def load_video_rows(label_path):
    with label_path.open() as f:
        d = json.load(f)
    global PHASE_CATEGORIES
    if PHASE_CATEGORIES is None:
        PHASE_CATEGORIES = d["categories"]["phase"]
    else:
        assert d["categories"]["phase"] == PHASE_CATEGORIES, \
            f"{label_path} has a different phase category list"
    return [(im["id"], im["phase_label"]) for im in d["images"] if im["phase_label"] is not None]


def build_split(videos, label_dir, frame_size):
    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    n_skipped = 0
    for video in sorted(videos):
        rows = load_video_rows(label_dir / f"{video}.json")
        for frame_num, phase_label in rows:
            file_name = f"{video}/{frame_num:06d}.jpg"
            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": frame_size,
                "height": frame_size,
                "video_name": video,
                "frame_num": frame_num,
            })
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "image_name": file_name,
                "phases": phase_label,
            })
            img_id += 1
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "phases_categories": PHASE_CATEGORIES,
    }


def main():
    here = Path(__file__).resolve().parent
    dataset_dir = here / "multibypasst40_challenge_trainval"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-dir", type=Path, default=dataset_dir / "label_files_challenge")
    ap.add_argument("--folds-json", type=Path, default=dataset_dir / "folds.json")
    ap.add_argument("--out-dir", type=Path, default=dataset_dir / "annotations")
    ap.add_argument("--frame-size", type=int, default=512,
                     help="videos_cutmargin512 frames are 512x512 (the label "
                          "files' own width/height=224 field describes a "
                          "different, unrelated resize).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.folds_json.open() as f:
        folds = json.load(f)
    fold1_videos = set(folds["fold_1"]["videos"])
    fold2_videos = set(folds["fold_2"]["videos"])
    all_videos = fold1_videos | fold2_videos

    for name, videos in [("fold1", fold1_videos), ("fold2", fold2_videos), ("train", all_videos)]:
        split = build_split(videos, args.label_dir, args.frame_size)
        out_path = args.out_dir / f"multibypass_phases_{name}.json"
        with out_path.open("w") as f:
            json.dump(split, f)
        print(f"  -> {out_path} ({len(split['images'])} images, {len(split['annotations'])} annotations)")


if __name__ == "__main__":
    main()
