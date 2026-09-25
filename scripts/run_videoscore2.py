import argparse
import contextlib
import io
import json
from pathlib import Path
import random
import re
import runpy
import sys
import traceback
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--official-script",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    rows=json.loads(a.selection.read_text())
    row=rows[a.index]
    if not a.official_script.is_file():raise FileNotFoundError(a.official_script)
    if not (a.model_dir/"config.json").is_file():raise FileNotFoundError(a.model_dir/"config.json")
    a.output_root.mkdir(parents=True,exist_ok=True)
    path=a.output_root/(row["prompt_id"]+".json")
    if path.exists():raise FileExistsError(path)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    oldargv=sys.argv
    sys.argv=[str(a.official_script),"--video_path",row["video"],"--t2v_prompt",row["prompt"],"--model_name",str(a.model_dir)]
    buffer=io.StringIO()
    record={"prompt_id":row["prompt_id"],"video":row["video"],"model_dir":str(a.model_dir),"official_script":str(a.official_script),"seed":a.seed}
    try:
        with contextlib.redirect_stdout(buffer):
            runpy.run_path(str(a.official_script),run_name="__main__")
        output=buffer.getvalue()
        record["raw_output"]=output
        record["scores"]={}
        for label,key in [("Visual Quality","visual_quality"),("Text-to-Video Alignment","text_video_alignment"),("Physical Consistency","physical_consistency")]:
            match=re.search(r"^"+re.escape(label)+r":\s*([0-9.]+|None)\s*$",output,re.M)
            record["scores"][key]=None if not match or match.group(1)=="None" else float(match.group(1))
        record["status"]="complete" if all(v is not None for v in record["scores"].values()) else "invalid_score"
    except Exception:
        record["status"]="failed"
        record["error"]=traceback.format_exc()
        record["raw_output"]=buffer.getvalue()
        raise
    finally:
        sys.argv=oldargv
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps({"prompt_id":row["prompt_id"],"status":record["status"],"scores":record.get("scores")},ensure_ascii=False))

if __name__=="__main__":
    main()
