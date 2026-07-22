#!/usr/bin/env python3
"""Build TAPIS-format frame_lists CSVs for MultiBypass from the frames already on disk.

Each output file has one row per frame, space-separated, no header, matching the
format read by tapis/datasets/surgical_dataset_helper.py:load_image_lists:

    CASE_NAME CASE_NUMBER FRAME_NUMBER FRAME_NAME

Only CASE_NAME (video name) and FRAME_NAME (path relative to FRAME_DIR) are
actually consumed by load_image_lists; CASE_NUMBER/FRAME_NUMBER just need to be
present to satisfy its `len(row) == 4` check.
"""
import argparse
import json
import re
from pathlib import Path


def case_number(video_name):
    """'C1V1' -> 101, 'C2V14' -> 214 -- unique, sortable, mirrors GraSP's CASE_NUMBER
    (an integer parsed from the video name) without needing a case registry."""
    m = re.match(r"C(\d+)V(\d+)$", video_name)
    if not m:
        raise ValueError(f"Unrecognized video name format: {video_name}")
    center, video = m.groups()
    return int(center) * 100 + int(video)


def write_list(out_path, videos, frames_root):
    rows = []
    for video in sorted(videos):
        frame_paths = sorted((frames_root / video).glob("*.jpg"))
        if not frame_paths:
            raise ValueError(f"No frames found for {video} under {frames_root}")
        num = case_number(video)
        for frame_idx, frame_path in enumerate(frame_paths):
            rows.append(f"{video} {num} {frame_idx} {video}/{frame_path.name}")
    out_path.write_text("\n".join(rows) + "\n")
    print(f"  -> {out_path} ({len(rows)} frames, {len(videos)} videos)")


def main():
    here = Path(__file__).resolve().parent
    dataset_dir = here / "multibypasst40_challenge_trainval"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-root", type=Path,
                     default=dataset_dir / "videos_cutmargin512",
                     help="Directory containing one subdir of frames per video "
                          "(default: videos_cutmargin512, the resolution the "
                          "region features were extracted at).")
    ap.add_argument("--folds-json", type=Path, default=dataset_dir / "folds.json")
    ap.add_argument("--out-dir", type=Path, default=dataset_dir / "frame_lists")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.folds_json.open() as f:
        folds = json.load(f)
    fold1_videos = set(folds["fold_1"]["videos"])
    fold2_videos = set(folds["fold_2"]["videos"])
    all_videos = fold1_videos | fold2_videos

    print(f"Building frame lists from {args.frames_root}")
    write_list(args.out_dir / "fold1.csv", fold1_videos, args.frames_root)
    write_list(args.out_dir / "fold2.csv", fold2_videos, args.frames_root)
    write_list(args.out_dir / "train.csv", all_videos, args.frames_root)


if __name__ == "__main__":
    main()
