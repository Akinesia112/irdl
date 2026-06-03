Installation
============

We recommend either ``uv`` or ``pip`` with ``venv`` for installation. First, we create a virtual environment:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv venv

        .. note::
           ``uv`` will handle environment activation implicitly (the environment is created at ``.venv``).

    .. tab-item:: ``venv``
        :sync: pip

        .. code-block:: console

            $ python3 -m venv my-env

        and activate the environment with:

        .. code-block:: console

            $ source my-env/activate/bin

Then, we install ``irdl`` in that environment with:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv pip install acoular

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ pip install -U acoular

We can check the installation by checking for CLI-tool availability with:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv run irdl --help

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ irdl --help

If you would like the ``irdl`` command to be available globally you can either add the binary to your path or install irdl with ``uv tool``!

.. tab-set::
    :sync-group: tool

    .. tab-item:: ```uv``
        :sync: uv

        Skip all of the above and simply run

        .. code-block:: console

            $ uv tool install irdl

        You can check for new available datasets with ``uv tool upgrade irdl`` 🤗

    .. tab-item:: ``pip``
        :sync: pip

        On Linux and MacOs:

        .. code-block:: console

            $ export PATH=$PATH:path/to/my-env/bin/irdl
