# E-BARD integration guide

The project uses `GabrieleGiudici/E-BARD-detection` as an external source for v1.1.0.

Agreed class mapping:

| E-BARD | TrainingModel |
| --- | --- |
| basketball | basketball (0) |
| player | player (1) |
| referee | referee (2) |
| hoop | rim (4) |

E-BARD does not annotate `backboard`, so TrainingModel adds class 3 with Grounding DINO and validates those generated boxes with a distributed manual QA sample before the E-BARD source is admitted to the curated dataset.

The E-BARD dataset is CC BY 4.0. Preserve attribution/provenance in derived dataset records and model documentation.

## 1. Download and unpack

From the repository root:

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
```

`local_data/` is ignored by Git.

## 2. Import E-BARD into staging

```bash
python3 scripts/import_e_bard.py
```

The importer finds the E-BARD YOLO YAML and reads its class order rather than assuming source class IDs. It converts `hoop` to TrainingModel `rim` and preserves every converted E-BARD label before enrichment.

Expected staging layout:

```text
local_data/ebard_import/sources/
├── ebard_<game_1>/
│   ├── images/
│   ├── labels_ebard/   # immutable converted E-BARD labels
│   ├── labels/         # working copy, later enriched with backboard
│   ├── manifest.csv
│   └── source.json
├── ebard_<game_2>/
└── ...
```

The importer attempts to infer the original game ID from each image filename. E-BARD is derived from 60 games, so inspect the reported source count. If filename conventions prevent reliable automatic inference, provide an explicit regex:

```bash
python3 scripts/import_e_bard.py \
  --source-regex '^(?P<game>YOUR_GAME_PATTERN)_'
```

Do not continue to dataset construction if game/source grouping is wrong, because frames from the same game must never leak across train/validation/test.

## 3. Add backboard pre-annotations

```bash
python3 scripts/enrich_ebard_backboard.py
```

This runs Grounding DINO using the project's configured thresholds and adds only class `3` (`backboard`) to the working `labels/` files. `labels_ebard/` remains unchanged.

To intentionally regenerate class-3 predictions:

```bash
python3 scripts/enrich_ebard_backboard.py --overwrite-backboard
```

Grounding DINO predictions are pre-annotations, not ground truth.

## 4. Prepare the backboard QA sample

The default QA sample takes four images from each original game/source, preferring three images with a predicted backboard and one image without a prediction when available. This checks both false positives and likely misses.

```bash
python3 scripts/prepare_ebard_review.py
```

For approximately 60 games this produces about 240 review frames.

Open the reviewer:

```bash
python3 scripts/review_labels.py ebard_backboard_qa
```

Review the entire QA sample. Correct any bad backboard box, add missed backboards, and verify that the inherited E-BARD boxes still look consistent. Every QA row must end as either:

```text
corrected
already_correct
```

If the sample reveals systematic Grounding DINO errors, do not finalize E-BARD yet. Increase the sample size and review again, for example:

```bash
python3 scripts/prepare_ebard_review.py --per-source 8 --force
python3 scripts/review_labels.py ebard_backboard_qa
```

## 5. Finalize E-BARD into curated sources

Only after the QA batch is fully approved:

```bash
python3 scripts/finalize_ebard.py
```

This copies all staged E-BARD frames into `local_data/sources/` while applying the manually corrected labels from the QA sample. The command refuses to run if any QA sample is unreviewed, `skip`, or `uncertain`.

Result:

```text
local_data/sources/
├── ebard_<game_1>/
├── ebard_<game_2>/
└── ...
```

These source directories then participate in the normal source-aware 70/15/15 dataset build.

## 6. Continue with the normal pipeline

```bash
python3 scripts/build_dataset.py
python3 scripts/audit_dataset.py
```

The global v1.1.0 hard cap remains 4,000 images after deduplication. E-BARD contributes about 1,800 frames before deduplication, leaving room for own-source footage, hard cases, hard negatives and other selected data.

Only after the final dataset passes audit should the controlled A/B training runs be started.
