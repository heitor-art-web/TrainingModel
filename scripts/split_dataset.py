"""Compatibility wrapper for the source-aware dataset builder.

The former implementation performed a random frame-level 80/20 split. That
method is intentionally retired because adjacent frames from one video can leak
between train and validation.
"""

from build_dataset import main


if __name__ == "__main__":
    print(
        "split_dataset.py now uses source-level 70/15/15 splitting. "
        "See scripts/build_dataset.py and configs/data_sources.toml."
    )
    main()
