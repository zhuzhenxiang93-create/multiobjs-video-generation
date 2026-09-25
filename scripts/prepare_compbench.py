#!/usr/bin/env python3
"""Prepare the official T2V-CompBench Numeracy subset without dropping categories."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True, type=Path)
    a = p.parse_args()
    rows = json.loads((a.batch / "selection.json").read_text())
    out = a.batch / "compbench_numeracy"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    videos = out / "videos"
    videos.mkdir(parents=True)
    metadata, included, excluded = [], [], []
    for row in rows:
        targets = row["targets"]
        if not 1 <= len(targets) <= 2:
            excluded.append({"prompt_id": row["prompt_id"],
                "reason": "original scorer supports only first one/two categories; all targets retained by exclusion"})
            continue
        name = f"{len(metadata)+1:04d}.mp4"
        (videos / name).symlink_to(Path(row["video"]))
        metadata.append({"prompt": row["prompt"],
            "objects": ",".join(t["category"] for t in targets),
            "numbers": ",".join(str(t["target_count"]) for t in targets)})
        included.append({"prompt_id": row["prompt_id"], "input_name": name,
                         "video_sha256": row.get("video_sha256")})
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "eligibility.json").write_text(json.dumps({"included": included, "excluded": excluded}, indent=2) + "\n")
    print(json.dumps({"included": len(included), "excluded": len(excluded)}))

if __name__ == "__main__":
    main()
