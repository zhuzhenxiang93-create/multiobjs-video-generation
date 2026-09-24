# Multi-object video generation evaluation

Reproducible evaluation scripts for prompt-conditioned object number in generated videos.

## Scope

- **CountVid** estimates distinct object instances over a whole video.
- **NUMINA / CountBench** measures frame-level count accuracy and adjacent-frame count stability.
- **EvalCrafter Count-Score** measures relative error of visible object counts in sampled frames.
- **T2V-CompBench Numeracy** measures frame-level exact count on its supported one/two-category prompts.
- **VBench Multiple Objects** measures pairwise class co-presence; it is not a count metric.

Results against prompt-requested counts are automatic prompt-match estimates. CountVid accuracy against actual generated-video counts requires independent human annotation.

The repository contains adapters and Slurm launchers, not videos, pretrained weights, or vendored benchmark repositories. Those inputs stay on the research servers.

## Status

Initial migration from server1 to server2 is in progress. The runner and setup instructions will be added on this branch before it is used for evaluation.
