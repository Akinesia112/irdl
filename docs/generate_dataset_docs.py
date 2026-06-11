"""Generate dataset docs from a single category mapping."""

from pathlib import Path

from dataset_categories import DATASET_CATEGORIES

DOCS_DIR = Path(__file__).parent
FRAGMENTS_DIR = DOCS_DIR / "_generated"
DATASETS_DIR = DOCS_DIR / "datasets"


def _class_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[1]


def _class_slug(qualified_name: str) -> str:
    class_name = _class_name(qualified_name)
    return "".join((f"_{ch.lower()}" if ch.isupper() else ch) for ch in class_name).lstrip("_")


def _dataset_name(dataset: dict[str, object]) -> str:
    return str(dataset["name"])


def _dataset_description_rst(dataset: dict[str, object]) -> str | None:
    value = dataset.get("description_rst")
    return None if value is None else str(value)


def _autosummary_block(datasets: list[dict[str, object]]) -> list[str]:
    lines = [
        ".. autosummary::",
        "",
    ]
    lines.extend(f"   ~{_dataset_name(dataset)}" for dataset in datasets)
    lines.append("")
    return lines


def _hidden_toctree_block(datasets: list[dict[str, object]]) -> list[str]:
    lines = [
        ".. toctree::",
        "   :hidden:",
        "",
    ]
    lines.extend(f"   {_class_slug(_dataset_name(dataset))}" for dataset in datasets)
    lines.append("")
    return lines


def _write_dataset_page(dataset: dict[str, object]) -> None:
    qualified_name = _dataset_name(dataset)
    class_name = _class_name(qualified_name)
    slug = _class_slug(qualified_name)
    title = f"``{class_name}``"
    lines = [
        title,
        "=" * len(title),
        "",
        ".. currentmodule:: irdl",
        "",
    ]
    intro_rst = _dataset_description_rst(dataset)
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


def _write_category_page(category: dict[str, object]) -> None:
    title = str(category["title"])
    slug = str(category["slug"])
    datasets = list(category["datasets"])
    lines = [
        title,
        "=" * len(title),
        "",
        *_autosummary_block(datasets),
        *_hidden_toctree_block(datasets),
    ]
    (DATASETS_DIR / f"{slug}.rst").write_text("\n".join(lines))


def main() -> None:
    """Main entry point that generates all dataset tables and doc pages."""
    FRAGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    body_lines = [
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]

    seen_datasets: set[str] = set()
    for category in DATASET_CATEGORIES:
        title = str(category["title"])
        slug = str(category["slug"])
        datasets = list(category["datasets"])

        body_lines.append(f"   {title} <{slug}>")

        _write_category_page(category)
        for dataset in datasets:
            name = _dataset_name(dataset)
            if name not in seen_datasets:
                _write_dataset_page(dataset)
                seen_datasets.add(name)

    body_lines.extend(["", ""])

    for category in DATASET_CATEGORIES:
        title = str(category["title"])
        datasets = list(category["datasets"])
        body_lines.extend(
            [
                title,
                "-" * len(title),
                "",
                *_autosummary_block(datasets),
            ]
        )

    (FRAGMENTS_DIR / "datasets_body.inc").write_text("\n".join(body_lines))


if __name__ == "__main__":
    main()
