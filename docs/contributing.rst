Contributing
============

Contributions are very welcome. The sections below describe the development setup and the conventions that make
contributions easier to review, but they are not hard gates. If you would like to contribute a useful dataset
implementation and can verify that it works, maintainers can help with formatting, documentation rebuilds, and release
polish.

Developer setup
---------------

``irdl`` uses `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ for development. ``uv`` handles virtual environment creation and dependency installation automatically. Just prepend dev commands with ``uv run``! To run tests, do

.. code-block:: console

   $ uv run python -m pytest

For linting and formatting with ``ruff``:

.. code-block:: console

   $ uv run ruff check --fix
   $ uv run ruff format

Build the documentation with:

.. code-block:: console

   $ uv run make -C docs html

The documentation Makefile regenerates the CLI help snippets that are included in the Sphinx documentation. Only the
``README.md`` might need manual updating.

Coding style
------------

The source of truth for formatting and linting is ``pyproject.toml`` and are enforced/autofixed by ``ruff``.
Important rules are:

- Line length: 120 characters.
- Double quotes.
- NumPy-style docstrings.
- Prefer explicit ``ValueError`` exceptions for invalid user input rather than ``assert``.
- After parameter validation, access required parameters directly, for example
  ``dataset_kwargs["scenario"]`` rather than ``dataset_kwargs.get("scenario")``.

Run ``ruff`` when convenient or before a push, but do not let formatting stop you from opening a contribution.
Maintainers are happy to apply final formatting and polishing.

Architecture and processing flow
--------------------------------

For the contributor-facing overview of the package architecture and the ``get`` processing
flow, see :doc:`processing_flow`.

Adding a Dataset
----------------

For a step-by-step guide, see :doc:`adding_dataset`.

.. include:: _includes/minimum_useful_contribution.inc
