CLI
===

.. _cli-reference:

The package offers a command line interface (CLI). Installing the package into an
environment makes the ``irdl`` command available in that environment. All supported
Datasets are exposed as subcommands.

.. click:: irdl.cli:typer_click_object
   :prog: irdl
   :nested: full
