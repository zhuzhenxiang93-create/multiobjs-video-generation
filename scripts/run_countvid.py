#!/usr/bin/env python3
"""Run one CountVid video-category query with server1's 3 Hz sampling protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--countvid-repo", type=Path, required=True)
    args = ap.parse_args()
    units = json.loads((args.batch / "units.json").read_text())
    unit = units[args.index]
    repo = args.countvid_repo.resolve()
    out = args.batch / "countvid" / unit["unit_id"]
    out.mkdir(parents=True, exist_ok=True)
    result = out / "result.json"
    if result.exists():
        old = json.loads(result.read_text())
        assert old["video"] == unit["video"] and old["category"] == unit["category"]
        return
    video = Path(unit["video"])
    assert video.is_file()
    if unit.get("video_sha256"):
        assert hashlib.sha256(video.read_bytes()).hexdigest() == unit["video_sha256"]
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps > 0 and n > 0
    indices = sorted({round(k * fps / 3) for k in range(int(n / fps * 3))})
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    for k, idx in enumerate(indices):
        path = frames / f"{k:05d}.jpg"
        if path.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        assert ok, (video, idx)
        assert cv2.imwrite(str(path), frame)
    cap.release()
    predicted = out / "predicted.json"
    if not predicted.exists():
        predicted.write_text("{}\n")
    cmd = [sys.executable, str(repo / "count_in_videos.py"),
        "--video_dir", str(frames), "--input_text", unit["category"],
        "--sam_checkpoint", str(repo / "checkpoints/sam2.1_hiera_large.pt"),
        "--sam_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml",
        "--obj_batch_size", "30", "--img_batch_size", "10", "--downsample_factor", "1",
        "--pretrain_model_path", str(repo / "checkpoints/countgd_box.pth"),
        "--temp_dir", str(out / "inference_frames"), "--output_file", str(predicted),
        "--temporal_filter", "--w", "3", "--convert_to_rgb"]
    with (out / "execution.log").open("w") as log:
        subprocess.run(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(predicted.read_text())
    assert len(data) == 1
    estimate = next(iter(data.values()))[unit["category"]]
    source = repo / "count_in_videos.py"
    output = {**unit, "status": "complete", "count_scope": "whole_video_unique_instances",
        "predicted_count": estimate, "difference_from_request": estimate - unit["requested_count"],
        "score_reference": "prompt_requested_count_not_observed_video_ground_truth",
        "sampling": {"policy": "nearest original frame at 3 Hz", "source_fps": fps, "indices": indices},
        "countvid_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(output, indent=2) + "\n")
    tmp.replace(result)
    print(json.dumps({"unit_id": unit["unit_id"], "predicted_count": estimate}))

if __name__ == "__main__":
    main()
