# Changelog

All notable project and model-pipeline changes are documented here.

## Unreleased

### Added
- Hybrid basketball detection + temporal ball tracking.
- Per-run `tracking_report.json` with continuity, detection, prediction and loss metrics.
- Hard-case extraction for predicted, lost, low-confidence and large-jump frames.
- Unit tests for tracking metrics, hard-case classification and portable dataset paths.
- Lightweight GitHub Actions workflow for syntax checks and unit tests.

### Changed
- Tracking configuration now includes hard-case extraction thresholds.
- `training.common` imports PyTorch only when automatic device selection is used, keeping utility tests lightweight.

## Model v1.0.0

- Current promoted basketball detector.
- Classes: basketball, player, referee, backboard, rim.
- Used as the baseline detector for inference, evaluation and hybrid ball tracking.
