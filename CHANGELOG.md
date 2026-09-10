# Changelog

All notable project and model-pipeline changes are documented here.

## Unreleased

### Added
- Source registry in `configs/data_sources.toml` with explicit train/validation/test policy and protected sources.
- Initial protected gold-test source `OwiQgAMqyUs` (first ~60 seconds), prohibited from training data.
- Source-aware dataset builder with 70/15/15 target split, source-level grouping, exact/near-duplicate filtering, manifests and split reports.
- Local Gradio bounding-box reviewer with five classes, non-destructive original/reviewed labels, autosave on navigation, progress resume and filters.
- Review-batch preparation and approved-label export into curated local sources.
- Controlled `promoted` vs `pretrained` training initialization for v1.1.0 A/B comparison.
- Per-class evaluation metrics and detection-priority comparison helper.
- Unit tests for source protection, split policy, dataset-size policy and promotion criteria.
- Compact local-video frame collector that preserves `source_id`, camera type and source metadata.
- `DATASET_GUIDE.md` describing the v1.1.0 acquisition and stopping strategy.

### Changed
- Retired random frame-level 80/20 splitting; `scripts/split_dataset.py` now routes to the source-aware 70/15/15 builder.
- `dataset.yaml` now defines train, validation and test splits.
- Gold-test images must be marked `corrected` or `already_correct` before dataset construction.
- Reviewer dependencies added through `gradio_image_annotation`.
- v1.1.0 dataset policy now has a hard ceiling of 4,000 images after deduplication; the builder and training entry point both enforce it.
- Normal video collection defaults to 1 frame/second and 120 frames/source, with a 200-frame normal-sampling cap per source to favor diversity over repetition.

### Existing
- Hybrid basketball detection + temporal ball tracking.
- Per-run `tracking_report.json` with continuity, detection, prediction and loss metrics.
- Hard-case extraction for predicted, lost, low-confidence and large-jump frames.
- Dataset audit, inference/evaluation helpers and lightweight CI tests.

## Model v1.0.0

- Current promoted basketball detector.
- Classes: basketball, player, referee, backboard, rim.
- Used as the baseline detector for inference, evaluation and hybrid ball tracking.
