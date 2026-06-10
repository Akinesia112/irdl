``irdl``: Impulse Response Downloader
=====================================

``irdl`` retrieves, caches, and converts impulse response Datasets in a unified way.

.. code-block:: python

   from irdl import MiracleDataset

   data = MiracleDataset.get(scenario="A1")
   print(data["impulse_response"])

.. code-block:: bash

   $ irdl miracle --scenario A1

``irdl`` follows a simple user-facing flow:

1. Choose a Dataset and parameters.
2. ``irdl`` retrieves and reuses cached source artifacts.
3. ``irdl`` prepares the ingest-ready representation when needed.
4. ``irdl`` returns the requested Output Format.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Install ``irdl`` and run the first MIRACLE retrieval from Python or the CLI.

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Compare ``uv`` and ``pip`` installation paths and global tool setup.

   .. grid-item-card:: Datasets
      :link: datasets/index
      :link-type: doc

      Browse MIRACLE, SRIRACHA, and FABIAN with grouped navigation in the sidebar.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      Jump to the Python API, CLI reference, and internal modules.

.. toctree::
   :hidden:

   Introduction <self>
   Getting started <getting_started>
   Installation <installation>
   Datasets <datasets/index>
   Reference <reference/index>
   Contributor Guide <contributor-guide/index>
