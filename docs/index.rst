``irdl``: Impulse Response Downloader
=====================================

Python package to download, unpack and process impulse response datasets in a unified way.

Usage (Python API)
------------------

The package can be included in a Python script as simple as:

.. code-block:: python

  from irdl import FabianDataset

  data = FabianDataset.get(kind='measured', hato=10)
  print(data)

Will output:

.. code-block:: bash

  {'impulse_response': time domain energy Signal:
  (11950, 2) channels with 256 samples @ 44100.0 Hz sampling rate and none FFT normalization,
   'receiver_coordinates': 2D Coordinates object with 2 points of cshape (2, 1)
  Does not contain sampling weights,
   'source_coordinates': 1D Coordinates object with 11950 points of cshape (11950,)
  Does not contain sampling weights}

For more details, see the :ref:`python-api-reference` section.


Usage (CLI)
-----------

Once installed, the package provides a convenient command line script which can be invoked with ``irdl``.

.. code-block:: bash

  $ irdl --help

.. literalinclude:: cli-help.txt
  :caption: Output:
  :language: bash
  :encoding: utf-8

The supported datasets are available as subcommands, i.e.

.. code-block:: bash

  $ irdl miracle --help

.. literalinclude:: cli-miracle-help.txt
  :caption: Output:
  :language: bash
  :encoding: utf-8

For more details, see the :ref:`cli-reference` section.

.. toctree::
   :maxdepth: 2

   Installation <installation>
   Available Datasets <datasets>
   Reference <reference>
   Contributing <contributing>
   Processing flow <processing_flow>
   Adding a new Dataset <adding_dataset>
