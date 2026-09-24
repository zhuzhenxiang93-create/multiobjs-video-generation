#!/usr/bin/env python3
"""Run one video through the unmodified NUMINA CountBench evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED_NUMINA_SHA = "15c647a53c75c7f8b5246d66af33d5c236e2b5793108cafe30b3980496038b69"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--numina-script", type=Path, required=True)
    ap.add_argument("--gdino-config", type=Path, required=True)
    ap.add_argument("--gdino-weights", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads((args.batch / "selection.json").read_text())
    row = rows[args.index]
    assert hashlib.sha256(args.numina_script.read_bytes()).hexdigest() == EXPECTED_NUMINA_SHA
    out = args.batch / "numina" / row["prompt_id"]
    out.mkdir(parents=True, exist_ok=True)
    result = out / "result.json"
    if result.exists():
        old = json.loads(result.read_text())
        assert old["prompt_id"] == row["prompt_id"]
        return
    video = Path(row["video"])
    assert video.is_file()
    if row.get("video_sha256"):
        assert hashlib.sha256(video.read_bytes()).hexdigest() == row["video_sha256"]
    videos = out / "videos"
    videos.mkdir(exist_ok=True)
    link = videos / ("001_" + row["prompt_id"] + ".mp4")
    if not link.exists():
        link.symlink_to(video)
    targets = {x["category"]: x["target_count"] for x in row["targets"]}
    noun_file = out / "noun_counts.jsonl"
    noun_file.write_text(json.dumps(targets) + "\n")
    prompt_file = out / "prompts.txt"
    prompt_file.write_text(row["prompt"].replace("\n", " ") + "\n")
    raw = out / "official_result.json"
    command = [sys.executable, str(args.numina_script), "--video_dir", str(videos),
        "--noun_counts_file", str(noun_file), "--prompt_file", str(prompt_file),
        "--gdino_config", str(args.gdino_config), "--gdino_weights", str(args.gdino_weights),
        "--start_idx", "1", "--end_idx", "1", "--save_results", str(raw)]
    with (out / "execution.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(raw.read_text())
    assert data["num_prompts"] == 1 and len(data["per_prompt"]) == 1
    record = data["per_prompt"][0]
    assert record["targets"] == targets and record["num_frames"] > 1
    temporal_by_noun = {noun: sum(a == b for a, b in zip(counts, counts[1:])) / (len(counts) - 1)
                        for noun, counts in record["per_noun"].items()}
    temporal = sum(temporal_by_noun.values()) / len(temporal_by_noun)
    output = {"prompt_id": row["prompt_id"], "video": str(video), "targets": targets,
        "count_accuracy": record["prompt_accuracy"], "temporal_count_consistency": temporal,
        "temporal_by_noun": temporal_by_noun, "num_frames": record["num_frames"],
        "official_source_sha256": EXPECTED_NUMINA_SHA,
        "score_reference": "prompt_requested_count_not_observed_video_ground_truth",
        "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(output, indent=2) + "\n")
    tmp.replace(result)
    print(json.dumps({"prompt_id": row["prompt_id"], "count_accuracy": output["count_accuracy"]}))

if __name__ == "__main__":
    main()
