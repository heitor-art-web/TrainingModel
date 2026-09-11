# TrainingModel

Basketball object-detection project built around Ultralytics YOLO, Grounding DINO-assisted labelling, temporal ball tracking and a curated-data workflow for controlled model iterations.

## Current model

The promoted model remains **v1.0.0**. Training a candidate does not automatically promote it.

Default promoted checkpoint:

```text
runs/detect/yolo26b/weights/model_v1.0.0.pt
```

Current evaluated candidate used for hard-case mining:

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

This section is the operational runbook. Follow it top-to-bottom for the current **v1.2.0 data preparation and training flow**.

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

Run lightweight tests whenever pipeline code changes:

```bash
python3 -m unittest discover -s tests -v
```

## 1. Data rules that must always hold

- Keep independent games/videos as independent sources whenever source identity is known.
- Never split random neighbouring frames from the same source across train/val/test.
- Grounding DINO is only a pre-annotation tool, never ground truth.
- Exact and near-duplicate images must be removed before the final build.
- `basketball` false positives are especially costly, so hard negatives matter.
- `player` and `rim` recall are important and must not regress.
- The protected gold-test source `OwiQgAMqyUs` must never be used for training or training-oriented hard-case mining.
- The current v1.1.0 build/training code still enforces a **4,000-image hard cap**. Do not silently raise it. v1.2.0 requires an explicit cap decision before the final build.

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

A healthy import is approximately 1,800 frames from 60 games/sources. E-BARD has no native `backboard` class, so the backboard step is a separate TrainingModel enrichment.

See [`EBARD_GUIDE.md`](EBARD_GUIDE.md) for the detailed E-BARD workflow.

## 3. v1.2.0 source: Basketball Game Detections

TrainingModel uses **Basketball Game Detections Dataset** by **FirstTime** from Roboflow Universe.

- Source: https://universe.roboflow.com/firsttime/basketball-game-detections
- Author/attribution: FirstTime
- Host: Roboflow Universe
- Declared license: **CC BY 4.0**
- License text: https://creativecommons.org/licenses/by/4.0/

When shared or redistributed, retain appropriate attribution, identify/link the license, and indicate modifications. The dataset license does not automatically resolve every possible third-party right in underlying broadcast imagery, logos or likenesses.

### 3.1 Class mapping

| Basketball Game Detections | TrainingModel | Treatment |
| --- | --- | --- |
| `Ball` | `basketball` (0) | remapped |
| `Player` | `player` (1) | remapped |
| `Ref` | `referee` (2) | remapped |
| `Hoop` | `rim` (4) | remapped |
| `FG Attempt` | — | ignored |
| `FG Made` | — | ignored |

This source has **no native backboard label**. Do not infer `backboard` from `Hoop`, and do not add backboards only to isolated QA images. A future backboard enrichment must cover the source consistently and be documented separately.

### 3.2 Expected local path

The current Roboflow YOLO26 export can remain at:

```text
local_data/Basketball Game Detections.v9i.yolo26/
```

A normal export contains `data.yaml` plus YOLO image/label folders. The repository resolver supports Roboflow YAML paths such as `../train/images`.

## 4. Analyze Basketball Game Detections

Run the analyzer once over the external dataset:

```bash
python3 scripts/analyze_basketball_game_detections.py \
  --input "local_data/Basketball Game Detections.v9i.yolo26"
```

By default it uses:

```text
runs/detect/model_v1.1.0_promoted/weights/best.pt
```

It records:

- exact/near duplicates;
- ball-size buckets (`tiny`, `small`, `medium`, `large`);
- current-model ball false negatives;
- current-model ball false positives;
- no-ball hard negatives;
- `ball + rim` context;
- source grouping when it can be proven.

Outputs:

```text
runs/dataset_analysis/basketball_game_detections/
├── analysis_manifest.csv
├── selected_manifest.csv
└── analysis_summary.json
```

For the current Roboflow v9 export, source/game identity is not proven. Therefore this dataset must remain **training-only** and must not contribute frames to validation/test.

## 5. Diagnostic QA before selection

If the analyzer reports an unusually large `model_fn` population, validate it before importing anything.

Prepare a diagnostic review batch without rerunning inference:

```bash
python3 scripts/prepare_bgd_analysis_qa.py \
  --input "local_data/Basketball Game Detections.v9i.yolo26"

python3 scripts/review_labels.py bgd_model_fn_diagnostic
```

Summarize reviewer statuses:

```bash
python3 scripts/summarize_review.py bgd_model_fn_diagnostic
```

Then analyze actual box changes while ignoring `backboard`, because BGD does not provide that class:

```bash
python3 scripts/analyze_review_changes.py \
  bgd_model_fn_diagnostic \
  --ignore-class backboard
```

For the current reviewed diagnostic sample:

```text
160 frames reviewed
32 frames with relevant non-backboard changes
model_fn: 26 / 100 with relevant changes
model_fp: 1 / 20 with relevant changes
model_hit_control: 5 / 20 with relevant changes
clean_no_ball_control: 0 / 20 with relevant changes
```

This was considered acceptable to proceed with a balanced selection. The diagnostic batch itself must **not** be exported into `local_data/sources/`.

## 6. Create the balanced v1.2.0 BGD selection

Reuse the existing analysis manifest; this does not rerun YOLO inference:

```bash
python3 scripts/refine_bgd_selection.py --force
```

Current target: **3,200 images**.

Target mix:

```text
tiny_ball              180
small_ball             420
model_fn              1050
model_fp               300
no_ball_hard_negative  350
ball_and_rim           550
other                  350
```

Quotas are targets/ceilings. If a bucket has fewer usable images, the selector fills remaining capacity from other diverse non-duplicate images.

Current approved refined selection:

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

Outputs:

```text
runs/dataset_analysis/basketball_game_detections/
├── refined_selection_manifest.csv
└── refined_selection_summary.json
```

## 7. Stage the refined selection

Import only the refined selection into a staging area:

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

`bgd_training_pool` is intentionally training-only while original source/game grouping remains unknown.

## 8. Final BGD QA

Prepare the final stratified QA sample from staged data:

```bash
python3 scripts/prepare_bgd_review.py
python3 scripts/review_labels.py bgd_qa
```

Default sample size: **320 frames**.

Do **not** add backboard boxes during this BGD QA. Review only the classes natively supplied/mapped by this source:

```text
basketball
player
referee
rim
```

After review, summarize it:

```bash
python3 scripts/summarize_review.py bgd_qa
```

If the QA exposes a systematic problem, stop and inspect it before finalizing. You can create a larger sample if needed:

```bash
python3 scripts/prepare_bgd_review.py --sample-size 600 --force
python3 scripts/review_labels.py bgd_qa
python3 scripts/summarize_review.py bgd_qa
```

Every final QA row must end as `already_correct` or `corrected` before finalization.

## 9. Finalize BGD into curated sources

When final QA is acceptable:

```bash
python3 scripts/finalize_bgd.py
```

This copies the staged BGD source into:

```text
local_data/sources/bgd_training_pool/
```

Reviewed corrections are applied only for the native BGD classes. Backboard labels are stripped from BGD during finalization so the source remains internally consistent.

## 10. STOP: choose the v1.2.0 dataset cap

At this point the expected curated volume is approximately:

```text
existing curated/E-BARD   ~1796
BGD refined selection     +3200
                          -----
before global dedup       ~4996
```

The repository still enforces the v1.1.0 **4,000-image hard cap** in the builder/training path.

**Do not run the final build for v1.2.0 until the cap has been explicitly changed and documented.**

The intended v1.2.0 policy is to allow deliberate growth while retaining a hard safety ceiling; the exact new ceiling must be chosen explicitly before editing the configuration/code.

## 11. Build and audit the final dataset

Only after the v1.2.0 cap has been approved and updated:

```bash
python3 scripts/build_dataset.py --force
python3 scripts/audit_dataset.py
```

The build must:

- keep proven sources together;
- force `bgd_training_pool` to train only;
- preserve protected evaluation sources;
- remove exact/near duplicates globally;
- target roughly 70/15/15 where source constraints allow;
- pass the dataset audit before training.

Do not train if the audit fails.

## 12. Train v1.2.0

Initialize from the currently promoted model unless the experiment explicitly says otherwise:

```bash
python3 training/train.py --version 1.2.0 --init promoted
```

Expected output:

```text
runs/detect/model_v1.2.0_promoted/weights/best.pt
```

Training a candidate does not promote it automatically.

## 13. Evaluate v1.2.0

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

- `basketball`: precision is critical; do not accept a major false-positive regression;
- `basketball`: operational recall must also be checked because static mAP alone was misleading for v1.1.0;
- `player`: recall must not materially regress;
- `rim`: recall must not materially regress;
- compare temporal/tracking behaviour on representative real videos as well as static metrics.

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

Prepare and review that source only:

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

Only `already_correct` and `corrected` rows are eligible for curated data. `skip` and `uncertain` are excluded.

## 15. Protected gold test

The first protected source is:

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

The gold test must never be used for training, threshold tuning or training-oriented hard-case mining.

# Historical v1.1.0 policy

The v1.1.0 dataset policy used a maximum of **4,000 images after deduplication** and source-aware 70/15/15 splitting. That ceiling remains enforced by current builder/training code until the v1.2.0 policy is explicitly adopted.

The v1.1.0 candidate clearly improved the supervised static benchmark versus v1.0.0, but did not automatically improve the P1 temporal pipeline at the previously tuned threshold. Therefore v1.2.0 evaluation must include both static detection metrics and operational/temporal behaviour.

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
