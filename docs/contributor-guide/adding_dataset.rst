.. _adding_new_dataset_heading:

Adding a new Dataset
====================

A new Dataset should fit into the shared :class:`~irdl.base.BaseDataset` flow rather than
implementing its own retrieval pipeline. The Dataset-specific code should focus on acquiring
provider data, preparing one ingest-ready file, and reading that file into SOFA.

Choose a base class
-------------------

Use :class:`~irdl.base.BaseDataset` for most Datasets.

If the provider data is already SOFA-native, consider inheriting from
:class:`~irdl.sofa.SofaBaseDataset`. :class:`~irdl.sofa.SofaBaseDataset` preserves the same shared flow but
avoids unnecessary SOFA output rewrites when ``output_format="sofa"`` is requested.

Start with a concrete Dataset class. Extract an intermediate shared base class only when at
least two Datasets share provider behavior or source layout.

Implement the Dataset class
---------------------------

A skeletal Dataset looks like this:

.. code-block:: python

   from pathlib import Path

   import sofar as sf

   from irdl.base import BaseDataset


   class NewDataset(BaseDataset):
       """Retrieve and process the NEW Dataset."""

       name = "new"
       doi = "10.xxxx/example"

       @classmethod
       def get(
           cls,
           cache_dir: str | None = None,
           export_dir: str | None = None,
           output_format: str = "pyfar",
           *,
           scenario: str = "default",
       ):
           """

           Parameters
           ----------
           scenario : str
               Dataset-specific scenario to retrieve.
           """
           return cls()._get(
               cache_dir=cache_dir,
               export_dir=export_dir,
               output_format=output_format,
               scenario=scenario,
           )

       def _validate_params(self, **dataset_kwargs) -> None:
           scenario = dataset_kwargs["scenario"]
           if scenario not in {"default"}:
               raise ValueError("scenario must be 'default'")

       def _source_filename(self, **dataset_kwargs) -> str:
           scenario = dataset_kwargs["scenario"]
           return f"new-{scenario}.sofa"

       def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
           # Retrieve provider file(s) into provider_dir.
           # Return the primary provider artifact.
           raise NotImplementedError

       def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
           # Optional: extract, merge, rename, or convert provider data into ingest_path.
           # If the provider artifact is already a single ingest-ready file, the
           # BaseDataset implementation may be enough and this override can be removed.
           raise NotImplementedError

       def _ingest(self, ingest_path: Path) -> sf.Sofa:
           # Read ingest_path and return a sofar.Sofa object.
           raise NotImplementedError

Keep this template intentionally small. Do not copy processing logic from another Dataset
unless the new Dataset has the same provider format and needs the same transformation.

Public API and CLI
------------------

After implementing the class, add it to ``src/irdl/__init__.py``. This exposes the Dataset
as part of the public API:

.. code-block:: python

   from .new_module import NewDataset as NewDataset

The CLI is generated automatically from concrete :class:`~irdl.base.BaseDataset`
subclasses imported by ``irdl``. Do not add hand-written CLI code for a new Dataset. The CLI
command name comes from ``Dataset.name``; parameters and help text come from the typed
``get()`` signature and NumPy-style docstring. See :doc:`/reference/python_api` for API
links.

Document the Dataset
--------------------

Dataset docs use one mapping as single source of truth: ``docs/dataset_categories.py``.
Add the new Dataset to the right category there. The generator uses that mapping to build:

- Datasets page tables
- category pages in the left sidebar
- per-Dataset pages

If the Dataset needs handwritten prose, add an ``.rst`` fragment and point
``description_rst`` at it in ``docs/dataset_categories.py``. Set ``description_rst`` to
``None`` for the current behavior. See :doc:`/reference/python_api` for the generated
Dataset class pages.

The referenced ``.rst`` file is injected before the generated ``.. autoclass::`` block on
the Dataset page. Use a path relative to ``docs/datasets/``. Example:

.. code-block:: python

   {
       "name": "irdl.MiracleDataset",
       "description_rst": "descriptions/miracle.rst",
   }

If no handwritten prose is needed, leave ``description_rst`` unset or ``None``.

The documentation Makefile regenerates the dataset docs and CLI help while building the
docs. Maintainers can refresh generated docs during review.

Manual verification
-------------------

Final verification should use the public ``get()`` path, not a direct private method call.
For example:

.. code-block:: python

   from irdl import NewDataset

   path = NewDataset.get(..., output_format="sofa")
   print(path)

This exercises validation, provider acquisition, processing, ingest, SOFA verification, and
output conversion. :class:`~irdl.base.BaseDataset` verifies the SOFA convention and prints diagnostics that
are useful while implementing a Dataset, including during agent-assisted coding.

In your contribution notes, include the command or Python snippet you ran and the smallest
useful evidence that the retrieved data is correct, such as expected filenames, dimensions,
sampling rate, coordinates, or Dataset metadata.
