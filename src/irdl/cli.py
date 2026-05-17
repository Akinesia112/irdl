"""Automatic generation of a Typer script for all datasets that can be used for download."""

from inspect import signature
from typing import Annotated

import typer
from numpydoc.docscrape import FunctionDoc

import irdl

# Typer app that can be invoked by calling ``irdl`` from the CLI.
app = typer.Typer(no_args_is_help=True)

# Automatically register all supported datasets as subcommands to the app.
for name in irdl.__all__:
    # Skip non-Dataset items (like CACHE_DIR)
    if name == "CACHE_DIR":
        continue

    dataset_class = getattr(irdl, name)
    get_method = dataset_class.get

    # Get docstring and signature from the classmethod
    doc = FunctionDoc(get_method)
    sig = signature(get_method)

    # Build Typer parameters with help text from docstring
    # Match parameters by name to handle docstring composition ordering
    param_docs = {p.name: " ".join(p.desc).replace("`", "") for p in doc["Parameters"]}
    typer_parameters = [
        p.replace(annotation=Annotated[p.annotation, typer.Option(help=param_docs.get(p.name, ""))])
        for p in sig.parameters.values()
    ]

    # Set the modified signature on the underlying function
    # For classmethods, we need to set it on __func__
    get_method.__func__.__signature__ = sig.replace(parameters=typer_parameters)

    # Build help text (DOI already included in docstring via __init_subclass__)
    help_text = doc["Summary"][0] + "\n\n" + " ".join(doc["Extended Summary"])

    # Register subcommand using dataset_class.name for the command name
    app.command(
        name=dataset_class.name,
        help=help_text,
    )(get_method)

# expose click object for sphinx_click autodoc feature.
typer_click_object = typer.main.get_command(app)
