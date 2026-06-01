Reference
=========

Python API
----------

.. currentmodule:: irdl

Dataset modules
~~~~~~~~~~~~~~~
.. autosummary::
   :caption: Dataset modules
   :toctree: _autosummary

   base
   ista
   sofa

Internal modules
~~~~~~~~~~~~~~~~
.. autosummary::
  :caption: Internal modules
  :toctree: _autosummary

  cli
  downloader
  logger
  repositories
  utils


Command Line Interface
------------------------

The package offers a command line interface (CLI). Installing the
package (into an environment) will make the command ``irdl`` available (in that
environment). All supported datasets are offered as subcommands. Please refer to
below documentation for reference or use ``--help`` to display the documentation
in the terminal.

.. click:: irdl.cli:typer_click_object
    :prog: irdl
    :nested: full
