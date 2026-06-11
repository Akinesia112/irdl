Python API
==========

.. currentmodule:: irdl

The public Python API is centered on concrete Dataset classes and a shared retrieval flow.

Key classes
-----------

- :class:`irdl.base.BaseDataset`
- :class:`irdl.ista.IstaBaseDataset`
- :class:`irdl.sofa.SofaBaseDataset`
- :class:`irdl.FabianDataset`
- :class:`irdl.MiracleDataset`
- :class:`irdl.SrirachaDataset`

Dataset modules
---------------

.. autosummary::
   :caption: Core Python modules
   :toctree: ../_autosummary

   base
   ista
   sofa

Internal modules
----------------

.. autosummary::
   :caption: Internal modules
   :toctree: ../_autosummary

   cli
   downloader
   logging
   repositories
   utils
