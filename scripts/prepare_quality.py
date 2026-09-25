#!/usr/bin/env python3
"""Stage exact video paths and prompts for VBench custom input."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True, type=Path)
    a = p.parse_args()
    rows = json.loads((a.batch / "selection.json").read_text())
    videos = a.batch / "videos"
    videos.mkdir(exist_ok=True)
    mapping = {}
    for row in rows:
        link = videos / (row["prompt_id"] + ".mp4")
        source = Path(row["video"])
        if link.exists():
            assert link.resolve() == source.resolve()
        else:
            link.symlink_to(source)
        mapping[str(link.resolve())] = row["prompt"]
    output = a.batch / "vbench_prompt_map.json"
    output.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"videos": len(mapping), "prompt_map": str(output)}))

if __name__ == "__main__":
    main()
