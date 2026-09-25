# Multi-object video generation evaluation

Portable Slurm adapters for a frozen prompt suite. This repository contains evaluation code only. Videos, model checkpoints, server names, and site-specific paths stay outside GitHub.

## Metrics

| Method | Count policy | Output |
|---|---|---|
| CountVid | Distinct instances across the whole video | Estimated unique count per video and category |
| NUMINA CountBench | Detected objects in every frame | Count accuracy and adjacent-frame count stability |
| EvalCrafter Count-Score | Visible objects every fifth source frame, with tracking between detection anchors | Relative count score against the prompt target |
| T2V-CompBench Numeracy | Sampled-frame exact count for one or two categories | Official Numeracy score on eligible prompts |
| VBench six quality dimensions | Appearance and motion quality | Quality scores, not count accuracy |
| VideoScore2 | Visual quality, alignment, physical consistency | Quality and alignment scores, not count accuracy |

VBench Multiple Objects measures pairwise class co-presence and is not an exact number metric. It is not launched by these scripts.

Scores compared to prompt targets are automatic prompt-match estimates. CountVid accuracy against the objects actually present in generated videos requires independent observed-video labels.

## Inputs

Supply a JSON list with `prompt_id`, `prompt`, `targets: [{"category": "...", "target_count": 2}]`, and optionally `video_sha256`. Store each video as `<VIDEO_DIR>/<prompt_id>.mp4`.

```bash
python3 scripts/prepare_batch.py --suite "$SUITE" --video-dir "$VIDEO_DIR" --output "$BATCH_DIR" --verify-sha256
python3 scripts/prepare_quality.py --batch "$BATCH_DIR"
python3 scripts/prepare_compbench.py --batch "$BATCH_DIR"
```

The preparer freezes the selected videos and video-category queries. The CompBench preparer excludes prompts with more than two categories rather than silently ignoring targets.

## Dependencies

The following upstream repositories and model weights must be installed separately; no third-party checkpoints are published here.

- CountVid: https://github.com/niki-amini-naieni/CountVid
- NUMINA: https://github.com/H-EmbodVis/NUMINA (tested commit `18381ad6b05ba445b20e4c25ff09729d8cb6cdb2`)
- EvalCrafter: https://github.com/evalcrafter/EvalCrafter
- T2V-CompBench V2: https://github.com/KaiyueSun98/T2V-CompBench
- VBench: https://github.com/Vchitect/VBench
- VideoScore2: https://github.com/TIGER-AI-Lab/VideoScore2

Set `SOURCE_ROOT` to a directory containing those repositories and set the Python executable variables in the Slurm launchers to matching environments. Check official source versions and checkpoint integrity before execution.

## Slurm

`slurm/run_metric.sbatch` accepts `METRIC=countvid|numina|evalcrafter`, `BATCH_DIR`, `SOURCE_ROOT`, and metric-specific Python executable variables. Use an array of one task per unit for CountVid/EvalCrafter and one task per video for NUMINA. The other launchers run T2V-CompBench, VBench quality dimensions, and VideoScore2. Set Slurm output paths to a writable batch log directory when submitting.

```bash
python3 scripts/summarize.py --batch "$BATCH_DIR"
```

The summarizer reports missing outputs explicitly and never substitutes zero. Keep category-query and video-level averages separate.

## Provenance

- CountVid uses nearest original frames at 3 Hz and temporal filter `w=3`, matching the frozen pilot protocol.
- EvalCrafter calls upstream `video_detection` using the pilot thresholds and original relative-count formula.
- T2V-CompBench V2 Numeracy source SHA-256: `b1b77eee3f82af0320f2c1c2abddace1c0b5acc461a9fa35dd3ec05bbe7b6412`.
- NUMINA official evaluator source SHA-256: `15c647a53c75c7f8b5246d66af33d5c236e2b5793108cafe30b3980496038b69`.
