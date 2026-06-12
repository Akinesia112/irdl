"""Generate dataset docs by introspecting dataset classes."""

import sys
from pathlib import Path

# Add src to path so we can import irdl
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import enum

import irdl
from irdl.base import DatasetCategory, _get_dataset_classes

DOCS_DIR = Path(__file__).parent
FRAGMENTS_DIR = DOCS_DIR / "_generated"
DATASETS_DIR = DOCS_DIR / "datasets"


def _class_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[1]


def _class_slug(class_name: str) -> str:
    """Convert class name to slug (e.g., MiracleDataset -> miracle)."""
    return "".join((f"_{ch.lower()}" if ch.isupper() else ch) for ch in class_name).lstrip("_")


def _find_description_rst(dataset_name: str) -> str | None:
    """Auto-detect description RST file for a dataset.

    Looks for files like datasets/miracle.rst or datasets/MiracleDataset.rst
    """
    # Try lowercase name first
    candidates = [
        DATASETS_DIR / f"{dataset_name}.rst",
        DATASETS_DIR / f"{dataset_name.capitalize()}.rst",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.relative_to(DOCS_DIR))
    return None


def _autosummary_block(dataset_classes: list[type]) -> list[str]:
    lines = [
        ".. autosummary::",
        "",
    ]
    lines.extend(f"   ~{cls.__module__}.{cls.__name__}" for cls in dataset_classes)
    lines.append("")
    return lines


def _hidden_toctree_block(dataset_classes: list[type]) -> list[str]:
    lines = [
        ".. toctree::",
        "   :hidden:",
        "",
    ]
    lines.extend(f"   {_class_slug(cls.__name__)}" for cls in dataset_classes)
    lines.append("")
    return lines


def _write_dataset_page(dataset_class: type) -> None:
    class_name = dataset_class.__name__
    slug = _class_slug(class_name)
    title = f"``{class_name}``"
    lines = [
        title,
        "=" * len(title),
        "",
        ".. currentmodule:: irdl",
        "",
    ]

    # Auto-detect description RST
    intro_rst = _find_description_rst(dataset_class.name)
    if intro_rst is not None:
        lines.extend(
            [
                f".. include:: {intro_rst}",
                "",
            ]
        )

    lines.extend(
        [
            f".. autoclass:: {class_name}",
            "   :members:",
            "   :undoc-members: false",
            "   :show-inheritance:",
            "",
        ]
    )
    (DATASETS_DIR / f"{slug}.rst").write_text("\n".join(lines))


def _write_category_page(category: DatasetCategory, dataset_classes: list[type]) -> None:
    title = category.value.replace("_", " ").title()
    slug = category.value
    lines = [
        title,
        "=" * len(title),
        "",
        *_autosummary_block(dataset_classes),
        *_hidden_toctree_block(dataset_classes),
    ]
    (DATASETS_DIR / f"{slug}.rst").write_text("\n".join(lines))


def main() -> None:
    """Generate all dataset tables and doc pages."""
    FRAGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Get all dataset classes
    dataset_classes = _get_dataset_classes(irdl)

    # Group by category
    datasets_by_category: dict[DatasetCategory, list[type]] = {}
    uncategorized: list[type] = []

    for cls in dataset_classes:
        category = getattr(cls, "_category", None)
        if category is None:
            uncategorized.append(cls)
        else:
            if category not in datasets_by_category:
                datasets_by_category[category] = []
            datasets_by_category[category].append(cls)

    body_lines = [
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]

    seen_datasets: set[str] = set()

    # Process categorized datasets (in DatasetCategory definition order)
    for category in DatasetCategory:
        if category in datasets_by_category:
            datasets = datasets_by_category[category]
            title = category.value.replace("_", " ").title()
            slug = category.value

            body_lines.append(f"   {title} <{slug}>")

            _write_category_page(category, datasets)
            for cls in datasets:
                if cls.name not in seen_datasets:
                    _write_dataset_page(cls)
                    seen_datasets.add(cls.name)

    # Process uncategorized datasets
    if uncategorized:
        # Create a special category for uncategorized

        class UncategorizedCategory(enum.StrEnum):
            UNCATEGORIZED = "uncategorized"

        uncategorized_category = UncategorizedCategory.UNCATEGORIZED
        body_lines.append(f"   Uncategorized <{uncategorized_category.value}>")

        # Write category page
        lines = [
            "Uncategorized",
            "=" * len("Uncategorized"),
            "",
            *_autosummary_block(uncategorized),
            *_hidden_toctree_block(uncategorized),
        ]
        (DATASETS_DIR / f"{uncategorized_category.value}.rst").write_text("\n".join(lines))

        for cls in uncategorized:
            if cls.name not in seen_datasets:
                _write_dataset_page(cls)
                seen_datasets.add(cls.name)

    body_lines.extend(["", ""])

    # Generate summary section (in DatasetCategory definition order)
    for category in DatasetCategory:
        if category in datasets_by_category:
            datasets = datasets_by_category[category]
            title = category.value.replace("_", " ").title()
            body_lines.extend(
                [
                    title,
                    "-" * len(title),
                    "",
                    *_autosummary_block(datasets),
                ]
            )

    if uncategorized:
        body_lines.extend(
            [
                "Uncategorized",
                "-" * len("Uncategorized"),
                "",
                *_autosummary_block(uncategorized),
            ]
        )

    (FRAGMENTS_DIR / "datasets_body.inc").write_text("\n".join(body_lines))


if __name__ == "__main__":
    main()
