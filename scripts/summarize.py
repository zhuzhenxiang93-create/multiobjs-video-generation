#!/usr/bin/env python3
"""Aggregate available results without treating missing jobs as failures or zeros."""
import argparse
import json
from pathlib import Path
from statistics import mean

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True, type=Path)
    a = p.parse_args()
    selection = json.loads((a.batch / "selection.json").read_text())
    units = json.loads((a.batch / "units.json").read_text())
    result = {"planned_videos": len(selection), "planned_category_queries": len(units)}
    for method, rows, label in [
        ("countvid", units, "predicted_count"),
        ("evalcrafter", units, "count_score_against_request"),
        ("numina", selection, "count_accuracy")]:
        found, missing, invalid = [], [], []
        for row in rows:
            key = row["unit_id"] if "unit_id" in row else row["prompt_id"]
            path = a.batch / method / key / "result.json"
            if not path.exists():
                missing.append(key)
                continue
            try:
                data = json.loads(path.read_text())
                assert data.get("status", "complete") == "complete"
                assert label in data
                found.append(data)
            except Exception as exc:
                invalid.append({"id": key, "error": str(exc)})
        item = {"completed": len(found), "planned": len(rows), "missing": missing, "invalid": invalid}
        if method == "countvid" and found:
            item["prompt_match_rate_query_macro"] = mean(x["predicted_count"] == x["requested_count"] for x in found)
            item["absolute_error_vs_request"] = mean(abs(x["predicted_count"] - x["requested_count"]) for x in found)
            item["signed_bias_vs_request"] = mean(x["predicted_count"] - x["requested_count"] for x in found)
        elif method == "evalcrafter" and found:
            item["count_score_query_macro"] = mean(x[label] for x in found)
        elif method == "numina" and found:
            item["count_accuracy_video_macro"] = mean(x[label] for x in found)
            item["temporal_count_consistency_video_macro"] = mean(x["temporal_count_consistency"] for x in found)
        result[method] = item
    result["reference_note"] = "All current scores compare automatic counts to prompt targets; no observed-video ground truth is implied."
    out = a.batch / "summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: {x:v for x,v in val.items() if x not in ("missing","invalid")} if isinstance(val,dict) else val for k,val in result.items()}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
