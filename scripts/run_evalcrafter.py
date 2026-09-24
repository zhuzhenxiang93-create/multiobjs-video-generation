#!/usr/bin/env python3
"""Run EvalCrafter's original video_detection for one video-category unit."""
import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--index", required=True, type=int)
    ap.add_argument("--evalcrafter-repo", required=True, type=Path)
    ap.add_argument("--save-masks", action="store_true")
    args = ap.parse_args()
    units = json.loads((args.batch / "units.json").read_text())
    unit = units[args.index]
    out = args.batch / "evalcrafter" / unit["unit_id"]
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / "result.json"
    if result_file.exists():
        old = json.loads(result_file.read_text())
        assert old["video"] == unit["video"] and old["category"] == unit["category"]
        return
    video = Path(unit["video"])
    assert video.is_file()
    if unit.get("video_sha256"):
        assert hashlib.sha256(video.read_bytes()).hexdigest() == unit["video_sha256"]
    module_dir = args.evalcrafter_repo.resolve() / "metrics/Segment-and-Track-Anything"
    source = module_dir / "object_attributes_eval.py"
    assert source.is_file()
    os.chdir(module_dir)
    sys.path.insert(0, str(module_dir))
    import cv2
    import numpy as np
    import torch
    module = importlib.import_module("object_attributes_eval")
    torch.manual_seed(42)
    sam = dict(module.sam_args)
    sam["generator_args"] = dict(points_per_side=30, pred_iou_thresh=.8,
        stability_score_thresh=.9, crop_n_layers=1,
        crop_n_points_downscale_factor=2, min_mask_region_area=200)
    seg = dict(sam_gap=49, min_area=200, max_obj_num=255, min_new_obj_iou=.8)
    frames, masks, counts = module.video_detection(
        {"input_video": str(video)}, seg, sam, module.aot_args, unit["category"],
        .6, .5, .5, True)
    assert counts and len(frames) == len(masks) == len(counts)
    if args.save_masks:
        for j, mask in enumerate(masks):
            np.savez_compressed(out / f"mask_{j:03d}.npz", mask=mask)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    target = unit["requested_count"]
    score = 1 - min(1, sum(abs(c - target) / target for c in counts) / len(counts))
    result = {**unit, "status": "complete", "count_scope": "sampled_frame_visible",
        "score_reference": "prompt_requested_count_not_observed_video_ground_truth",
        "sample_stride_source_frames": 5, "fps": fps, "source_frame_indices": list(range(0, len(counts) * 5, 5)),
        "counts": list(map(int, counts)), "count_score_against_request": score,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameters": {"seg": seg, "sam": sam, "box_threshold": .6,
                       "text_threshold": .5, "box_size_threshold": .5},
        "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, default=str) + "\n")
    tmp.replace(result_file)
    print(json.dumps({"unit_id": unit["unit_id"], "score": score}))

if __name__ == "__main__":
    main()
