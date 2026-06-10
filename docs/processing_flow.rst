Processing flow
===============

This page describes the contributor-facing architecture of IRDL and the conceptual flow of
a ``Dataset.get(...)`` call. It intentionally focuses on stable concepts and extension
points rather than every private implementation detail.

Core architecture
-----------------

``BaseDataset``
   The core abstraction for all Datasets. It owns the shared ``get`` pipeline: common
   parameter validation, cache path handling, download/process orchestration, SOFA
   verification, and output conversion.

Optional shared bases
   A Dataset family can introduce an intermediate base class when multiple Datasets share
   provider behavior or source layout. Do this only when the behavior is genuinely shared.

Concrete Dataset classes
   Each Dataset has one concrete class. Its public entry point is a typed ``get()``
   classmethod that delegates to the shared pipeline. The typed signature and NumPy-style
   docstring are also used to generate CLI parameters and help text.

Support modules
   Download/repository helpers, CLI generation, logging/progress helpers, and small utility
   functions live outside the Dataset classes.

Generic shape:

.. code-block:: text

   BaseDataset
   └── Optional shared base
       └── Concrete Dataset

Cache stages
------------

IRDL uses three canonical Cache Stages:

``provider``
   Files exactly as the Dataset Provider delivers them.

``ingest``
   The single ingest-ready file that IRDL can read into the internal SOFA representation.

``output``
   Cached files produced by converting the internal SOFA representation to disk-based
   Output Formats.

Use these names in code comments and documentation. "Ingest-ready" is an adjective for a
file in the ``ingest`` stage, not a separate stage name.

``get`` processing flow
-----------------------

A public ``Dataset.get(...)`` call delegates to the shared ``BaseDataset`` flow:

.. code-block:: text

   Dataset.get(...)
     └─ BaseDataset._get(...)
         ├─ validate common and Dataset-specific parameters
         ├─ resolve provider / ingest / output paths
         ├─ raw output: download provider artifact and return/copy it
         ├─ reuse cached output if available
         ├─ reuse ingest file if available
         ├─ download provider artifact if needed
         ├─ process provider → ingest if needed
         ├─ ingest to internal SOFA representation
         ├─ verify and upgrade SOFA convention
         └─ convert SOFA → requested Output Format

The important extension points for a new Dataset are:

``_validate_params()``
   Validate Dataset-specific parameters and invalid parameter combinations.

``_source_filename()``
   Return the canonical basename for the ingest-ready file.

``_download()``
   Acquire provider-stage file(s) and return the primary provider artifact.

``_process()``
   Optional. Transform provider-stage files into the single ingest-ready file. The default
   implementation handles simple single-file promotion from ``provider`` to ``ingest``.

``_ingest()``
   Read the ingest-ready file and return the internal SOFA representation.

``_to_output()`` and related conversion methods normally stay in ``BaseDataset``. New
Datasets should not implement output-specific conversion unless the shared conversion layer
itself needs to change.

Output behavior
---------------

``output_format="raw"`` returns the provider-stage artifact before IRDL processing. For all
other Output Formats, IRDL ingests the data to SOFA first and then converts from SOFA to the
requested representation.

When ``export_dir`` is provided, IRDL copies the requested artifact to the export directory.
The cache remains intact so later calls can reuse provider, ingest, or output artifacts.
