# TrainingModel

Basketball object-detection project built around Ultralytics YOLO, Grounding DINO-assisted labelling, temporal ball tracking and a curated-data workflow for controlled model iterations.

## Current model

The promoted model remains **v1.0.0**. Training a candidate does not automatically promote it.

Default promoted checkpoint:

```text
runs/detect/yolo26b/weights/model_v1.0.0.pt
```

The evaluated v1.1.0 candidate used for BGD hard-case mining is:

```text
runs/detect/model_v1.1.0_promoted/weights/best.pt
```

Weights, datasets, local videos, review batches and generated runs are intentionally excluded from Git.

## Canonical classes and priorities

| ID | Class | Priority |
| --- | --- | --- |
| 0 | basketball | highest; prefer fewer false positives |
| 1 | player | high; prefer higher recall |
| 2 | referee | secondary |
| 3 | backboard | secondary |
| 4 | rim | high; prefer higher recall |

`rim` is the canonical project name. External classes named `hoop` are mapped to `rim`.

# INSTRUCTIONS

This is the operational runbook for the current **v1.2.0** workflow. Follow it top-to-bottom. Optional/diagnostic steps are explicitly marked as such.

## 0. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

On later terminal sessions:

```bash
source .venv/bin/activate
```

After pulling pipeline changes, run:

```bash
python3 -m unittest discover -s tests -v
```

## 1. Non-negotiable data rules

- Keep independent games/videos as independent sources whenever source identity is known.
- Never randomly split neighbouring frames from one source across train/val/test.
- Grounding DINO is only a pre-annotation tool, never ground truth.
- Exact and near-duplicate images are removed before the final build.
- `basketball` false positives are especially costly, so hard negatives matter.
- `player` and `rim` recall are important and must not materially regress.
- Protected evaluation sources must never enter training or training-oriented hard-case mining.
- The v1.2.0 project hard ceiling is **6,000 images after deduplication**. It is a ceiling, not a target.

The active limit is defined in:

```text
configs/data_sources.toml
training/data.py
```

Training validates the same policy before starting.

## 2. Existing curated source: E-BARD

E-BARD support is already built into the repository.

```bash
mkdir -p local_data/external/e_bard

hf download \
  GabrieleGiudici/E-BARD-detection \
  all.zip \
  --repo-type dataset \
  --local-dir local_data/external/e_bard

cd local_data/external/e_bard
unzip all.zip
cd ../../..

python3 scripts/import_e_bard.py
python3 scripts/enrich_ebard_backboard.py
python3 scripts/prepare_ebard_review.py
python3 scripts/review_labels.py ebard_backboard_qa
python3 scripts/finalize_ebard.py
```

A healthy import is approximately 1,800 frames from 60 games/sources. E-BARD has no native `backboard` class, so backboard enrichment is a separate TrainingModel modification.

See [`EBARD_GUIDE.md`](EBARD_GUIDE.md) for the detailed E-BARD workflow.

## 3. v1.2.0 source: Basketball Game Detections

TrainingModel uses **Basketball Game Detections Dataset** by **FirstTime** from Roboflow Universe.

- Source: https://universe.roboflow.com/firsttime/basketball-game-detections
- Author/attribution: FirstTime
- Host: Roboflow Universe
- Declared license: **CC BY 4.0**
- License text: https://creativecommons.org/licenses/by/4.0/

When sharing or redistributing adapted data, retain appropriate attribution, identify/link the license and indicate modifications. The dataset license does not automatically resolve every possible third-party right in underlying broadcast imagery, logos or likenesses.

### Class mapping

| Basketball Game Detections | TrainingModel | Treatment |
| --- | --- | --- |
| `Ball` | `basketball` (0) | remapped |
| `Player` | `player` (1) | remapped |
| `Ref` | `referee` (2) | remapped |
| `Hoop` | `rim` (4) | remapped |
| `FG Attempt` | — | ignored |
| `FG Made` | — | ignored |

BGD has **no native backboard label**. Do not infer `backboard` from `Hoop`, and do not add backboards only to isolated BGD images. Any future backboard enrichment must be complete/consistent across the source and documented separately.

### Expected local path

Current Roboflow YOLO26 export:

```text
local_data/Basketball Game Detections.v9i.yolo26/
```

The repository resolver supports common Roboflow YAML paths such as `../train/images`.

## 4. Analyze BGD once

Run the analyzer over the external dataset:

```bash
python3 scripts/analyze_basketball_game_detections.py \
  --input "local_data/Basketball Game Detections.v9i.yolo26"
```

By default it uses:

```text
runs/detect/model_v1.1.0_promoted/weights/best.pt
```

It records duplicates, ball-size buckets, current-model ball false negatives/false positives, no-ball hard negatives, `ball + rim` context and source grouping when provenance can actually be proven.

Outputs:

```text
runs/dataset_analysis/basketball_game_detections/
├── analysis_manifest.csv
├── selected_manifest.csv
└── analysis_summary.json
```

Do **not** automatically import `selected_manifest.csv`. It is only a first-pass selection.

For the current Roboflow v9 export, original game/source identity was not proven, so BGD is intentionally registered as **training-only** (`bgd_training_pool`). It must not contribute to validation/test.

## 5. Diagnostic QA — use when analyzer results look suspicious

This is a **qualification/debugging step**, not a dataset-export step.

For the current BGD v9 run, the very large `model_fn` population justified this check:

```bash
python3 scripts/prepare_bgd_analysis_qa.py \
  --input "local_data/Basketball Game Detections.v9i.yolo26"

python3 scripts/review_labels.py bgd_model_fn_diagnostic
python3 scripts/summarize_review.py bgd_model_fn_diagnostic

python3 scripts/analyze_review_changes.py \
  bgd_model_fn_diagnostic \
  --ignore-class backboard
```

The diagnostic reviewer status alone can be misleading because tiny geometry edits and adding a non-native `backboard` can mark a row as `corrected`. Use `analyze_review_changes.py` to measure meaningful native-class edits.

Observed diagnostic result for the current run:

```text
160 frames reviewed
32 frames with relevant non-backboard changes
model_fn: 26 / 100 with relevant changes
model_fp: 1 / 20 with relevant changes
model_hit_control: 5 / 20 with relevant changes
clean_no_ball_control: 0 / 20 with relevant changes
```

This was accepted as sufficient to continue.

**Never run `export_reviewed.py` on this diagnostic batch.**

## 6. Create the balanced BGD selection

Reuse the existing analysis manifest; no YOLO inference is rerun:

```bash
python3 scripts/refine_bgd_selection.py --force
```

Current target: **3,200 images**.

Target quotas:

```text
tiny_ball              180
small_ball             420
model_fn              1050
model_fp               300
no_ball_hard_negative  350
ball_and_rim           550
other                  350
```

Quotas are targets/ceilings. If one bucket contains fewer usable unique images, remaining capacity is filled from other diverse images.

Current accepted selection:

```text
selected               3200
tiny_ball               180
small_ball              420
model_fn               1050
model_fp                300
no_ball_hard_negative   247
ball_and_rim            550
other                   453
```

Ball-size distribution in the accepted selection:

```text
tiny    186
small   824
medium  499
large  1122
none    569
```

Outputs:

```text
runs/dataset_analysis/basketball_game_detections/
├── refined_selection_manifest.csv
└── refined_selection_summary.json
```

## 7. Stage the refined selection

```bash
python3 scripts/import_basketball_game_detections.py \
  --input "local_data/Basketball Game Detections.v9i.yolo26"
```

Default staging output:

```text
local_data/bgd_import/sources/bgd_training_pool/
├── images/
├── labels/
├── manifest.csv
└── source.json
```

## 8. Final BGD QA: optional gate

There are two supported finalization paths.

### Path A — with final QA

Use this when you want a final stratified manual sample before promoting staged BGD:

```bash
python3 scripts/prepare_bgd_review.py
python3 scripts/review_labels.py bgd_qa
python3 scripts/summarize_review.py bgd_qa
python3 scripts/finalize_bgd.py
```

Default sample size is 320 frames. Every row must finish as `already_correct` or `corrected` before normal finalization.

Do not add `backboard` labels to BGD QA. The source's native/mapped classes are:

```text
basketball
player
referee
rim
```

If you need a larger QA sample:

```bash
python3 scripts/prepare_bgd_review.py --sample-size 600 --force
python3 scripts/review_labels.py bgd_qa
```

### Path B — deliberately skip final QA

Use this only when the source has already been qualified and you intentionally accept the staged labels as-is:

```bash
python3 scripts/finalize_bgd.py --skip-qa
```

This mode:

- does not require a completed `bgd_qa` batch;
- does not apply unreviewed QA edits;
- promotes staged BGD labels directly;
- marks the source manifest with `review_status = qa_skipped_source`;
- keeps BGD restricted to its native/mapped classes.

The current v1.2.0 run used **Path B** after the diagnostic qualification step.

Expected successful output is approximately:

```text
Frames exported: 3200
Final QA skipped: yes
QA-reviewed labels applied: 0
```

Final curated source:

```text
local_data/sources/bgd_training_pool/
```

## 9. Current v1.2.0 dataset budget

The active project hard cap is now:

```text
6000 images after deduplication
```

Expected pre-global-dedup volume for the current run is approximately:

```text
existing curated/E-BARD   ~1796
BGD refined selection     +3200
                          -----
before global dedup       ~4996
```

This fits deliberately below the 6,000-image safety ceiling.

## 10. Build and audit

After curated sources are ready:

```bash
python3 scripts/build_dataset.py --force
python3 scripts/audit_dataset.py
```

The builder must:

- preserve explicit/protected source roles;
- keep `bgd_training_pool` in train only;
- remove exact/near duplicates globally;
- keep complete proven sources together;
- target roughly 70/15/15 where source constraints allow;
- refuse a final dataset above 6,000 images.

Do not train if the audit fails.

## 11. Train v1.2.0

Once tests and dataset audit pass:

```bash
python3 training/train.py --version 1.2.0 --init promoted
```

Expected candidate checkpoint:

```text
runs/detect/model_v1.2.0_promoted/weights/best.pt
```

The training command independently re-checks the dataset image budget before starting.

Training a candidate does not promote it automatically.

## 12. Evaluate v1.2.0

Validation:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_v1.2.0_promoted/weights/best.pt \
  --split val
```

Held-out test:

```bash
python3 evaluation/evaluate.py \
  --weights runs/detect/model_v1.2.0_promoted/weights/best.pt \
  --split test
```

Compare against the previous model on the **same test split**:

```bash
python3 evaluation/compare_models.py \
  runs/evaluation/<previous>/metrics_summary.json \
  runs/evaluation/<candidate>/metrics_summary.json
```

Promotion priorities:

- `basketball`: precision is critical; avoid a major false-positive regression;
- `basketball`: also measure operational recall, because static mAP alone was misleading for v1.1.0;
- `player`: recall must not materially regress;
- `rim`: recall must not materially regress;
- compare temporal/tracking behaviour on representative real videos as well as static metrics.

The held-out test must not be used to tune training parameters or thresholds.

## 13. Current run status / resume point

For the current v1.2.0 preparation, BGD has already been finalized into curated sources with:

```bash
python3 scripts/finalize_bgd.py --skip-qa
```

and the active dataset cap has been raised to 6,000.

Therefore, after pulling the latest `main`, the current resume sequence is:

```bash
git pull origin main
python3 -m unittest discover -s tests -v
python3 scripts/build_dataset.py --force
python3 scripts/audit_dataset.py
python3 training/train.py --version 1.2.0 --init promoted
```

Only run the training command if tests and audit pass.

## 14. Generic workflow for adding your own source

Collect sparse frames:

```bash
python3 scripts/collect_video_frames.py \
  videos/game_01.mp4 \
  --source-id game_01 \
  --kind broadcast
```

Create Grounding DINO pre-annotations:

```bash
python3 scripts/autolabel.py \
  --root local_data/collected/game_01 \
  --skip-copy
```

Prepare/review that source:

```bash
python3 scripts/prepare_review.py \
  --batch game_01_review \
  --source-id game_01 \
  --images local_data/collected/game_01/images \
  --labels local_data/collected/game_01/labels

python3 scripts/review_labels.py game_01_review
```

Export approved rows:

```bash
python3 scripts/export_reviewed.py game_01_review
```

Only `already_correct` and `corrected` rows are eligible. `skip` and `uncertain` are excluded.

Normal collection guidance remains sparse by design:

```text
sampling: ~1 frame/second
normal default: 120 frames/source
normal maximum: 200 frames/source
```

Prefer adding a new independent source over oversampling one video.

## 15. Protected gold test

Protected source:

```text
source id: OwiQgAMqyUs
role: gold_test
segment: approximately first 60 seconds
training: forbidden
```

When collecting an authorized local copy for evaluation:

```bash
python3 scripts/collect_video_frames.py \
  videos/gold_test_01.mp4 \
  --source-id OwiQgAMqyUs \
  --kind broadcast \
  --purpose evaluation
```

The gold test must never be used for training, training-oriented hard-case mining or threshold tuning.

# Lessons carried forward from v1.1.0

The v1.1.0 data policy used a 4,000-image ceiling. v1.2.0 deliberately increases that ceiling to **6,000** to accommodate a larger, still-curated dataset.

The v1.1.0 candidate clearly improved the supervised static benchmark versus v1.0.0, but did not automatically improve the P1 temporal pipeline at the previously tuned threshold. Therefore a candidate is not considered better based on global mAP alone.

For future external datasets:

- do not trust external train/val/test splits when original source grouping is unknown;
- do not assume filename patterns prove game identity;
- qualify surprising model-FN/FP populations before using them as hard cases;
- distinguish reviewer status changes from meaningful geometry/class changes;
- do not partially enrich a class that is absent from the rest of the source;
- keep high-value hard cases balanced so one bucket does not dominate the training set;
- keep a hard dataset-size ceiling even when deliberately increasing data volume.

# Tests and CI

```bash
python3 -m unittest discover -s tests -v
```

GitHub Actions runs syntax and lightweight unit tests on pushes to `main` and pull requests. Training itself is never run in CI.

# Apple Silicon

Automatic device selection prefers CUDA, then Apple MPS, then CPU.

Check MPS with:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```
