#!/usr/bin/env python3
"""Freeze a video batch for server-independent count evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, type=Path, help="JSON rows: prompt_id, prompt, targets[{category,target_count}]")
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--verify-sha256", action="store_true")
    args = ap.parse_args()
    suite = json.loads(args.suite.read_text())
    assert isinstance(suite, list) and suite
    ids = [r["prompt_id"] for r in suite]
    assert len(ids) == len(set(ids)), "duplicate prompt_id"
    args.output.mkdir(parents=True, exist_ok=True)
    selection, units = [], []
    for row in suite:
        pid = row["prompt_id"]
        video = (args.video_dir / (pid + ".mp4")).resolve(strict=True)
        targets = row["targets"]
        assert isinstance(targets, list) and targets
        assert len({t["category"] for t in targets}) == len(targets)
        assert all(isinstance(t["target_count"], int) and t["target_count"] > 0 for t in targets)
        expected = row.get("video_sha256")
        if args.verify_sha256 and expected:
            actual = hashlib.sha256(video.read_bytes()).hexdigest()
            assert actual == expected, (pid, actual, expected)
        selection.append({"prompt_id": pid, "prompt": row["prompt"], "video": str(video),
                          "targets": targets, "video_sha256": expected})
        for j, target in enumerate(targets):
            units.append({"unit_id": f"{pid}_c{j:02d}", "prompt_id": pid, "video": str(video),
                          "video_sha256": expected, "category": target["category"],
                          "requested_count": target["target_count"]})
    for name, data in [("selection.json", selection), ("units.json", units)]:
        p = args.output / name
        if p.exists():
            assert json.loads(p.read_text()) == data, f"refusing to change frozen {p}"
        else:
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"videos": len(selection), "category_queries": len(units), "output": str(args.output)}))

if __name__ == "__main__":
    main()
