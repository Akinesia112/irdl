Architecture and Processing Flow
===============

This page describes the contributor-facing architecture of ``irdl`` and the conceptual flow of
a ``Dataset.get(...)`` call. It intentionally focuses on stable concepts and extension
points rather than every private implementation detail.

Core architecture
-----------------

``BaseDataset``
   The core abstraction for all Datasets. See :class:`~irdl.base.BaseDataset`. It owns the
   shared ``get`` pipeline: common parameter validation, cache path handling,
   retrieval/process orchestration, SOFA verification, and output conversion.

Optional Dataset Family classes
   When several Datasets share dataset-specific steps in the get pipeline
   (see :ref:get-processing-flow), a Dataset Family class can be introduced to lift that
   shared logic into an intermediate base class. Do this only when the behavior is genuinely
   shared; otherwise keep it in the individual Dataset class.

Individual Dataset classes
   Each Dataset is implemented as an individual class. Its typed get() classmethod delegates to 
   the shared pipeline, while the class itself implements that pipeline's dataset-specific steps 
   (see :ref:get-processing-flow). The typed signature and NumPy-style docstring are also used to 
   generate CLI parameters and help text.

Support modules
   Retrieval/repository helpers, CLI generation, logging/progress helpers, and small
   utility functions live outside the Dataset classes.

Class hierarchy:

.. code-block:: text

   BaseDataset
   └── Optional Dataset Family class
       └── Individual Dataset class

Cache stages
------------

``irdl`` uses three canonical Cache Stages:

``provider``
   Files exactly as the Dataset Provider delivers them.

``ingest``
   The single ingest-ready file that ``irdl`` can read into the internal SOFA representation.

``output``
   Files produced by converting the internal SOFA representation to disk-based
   Output Formats. 

Use these names in code comments and documentation. "Ingest-ready" is an adjective for a
file in the ``ingest`` stage, not a separate stage name.

``get`` processing flow
-----------------------

A public ``Dataset.get(...)`` call delegates to the shared :class:`~irdl.base.BaseDataset` flow:

.. code-block:: text

   Dataset.get(...)
     └─ BaseDataset._get(...)
         ├─ validate common and Dataset-specific parameters
         ├─ resolve provider / ingest / output paths
         ├─ [optional] raw output: retrieve provider artifact and return it
         ├─ reuse cached output if available
         ├─ reuse ingest file if available
         ├─ retrieve provider artifact if needed
         ├─ [optional] process provider file(s) into ingest file
         ├─ ingest to internal SOFA representation
         ├─ verify and upgrade SOFA convention
         └─ convert SOFA → requested Output Format

Each Dataset must implement the following extension points:

``_validate_params()``
   Validate Dataset-specific parameters and invalid parameter combinations.

``_source_filename()``
   Return the canonical basename for the ingest-ready file.

``_download()``
   Acquire provider-stage file(s) and return the primary provider artifact.

``_process()``
   Optional. Transform provider-stage files into the single ingest-ready file. The default
   implementation skips processing and handles simple single-file promotion 
   from ``provider`` to ``ingest``.

``_ingest()``
   Read the ingest-ready file and return the internal SOFA representation.

``_to_output()`` and related conversion methods normally stay in
:class:`~irdl.base.BaseDataset`. New Datasets should not implement output-specific conversion
unless the shared conversion layer itself needs to change.

Output behavior
---------------

``output_format="raw"`` returns the provider-stage artifact before ``irdl`` processing. For
all other Output Formats, ``irdl`` ingests the data to SOFA first and then converts from SOFA
to the requested representation. See the public class docs in :doc:`/reference/python_api`.

When ``export_dir`` is provided, ``irdl`` copies the requested artifact to the Export Directory.
The cache remains intact so later calls can reuse provider, ingest, or output artifacts.
