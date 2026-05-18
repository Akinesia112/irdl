"""Automatic generation of a Typer script for all datasets that can be used for download."""

import pathlib
import types
from inspect import signature
from typing import Annotated, Optional, Union, get_args, get_origin

import typer
from numpydoc.docscrape import FunctionDoc

import irdl


def _resolve_union_type(annotation):
    """Resolve Union types to Typer-compatible types."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle types.UnionType (Python 3.10+) and typing.Union
    if origin is types.UnionType or origin is Union:
        # Filter out NoneType and Path types, keep str
        non_none_args = [arg for arg in args if arg is not type(None)]
        # If we have str and/or Path, use str (paths can be passed as strings in CLI)
        if any(arg is str or arg is pathlib.Path or arg == pathlib.Path for arg in non_none_args):
            # Check if None was in the original args
            if type(None) in args:
                return Optional[str]
            return str
        # For other unions, just take the first type
        if type(None) in args:
            return Optional[non_none_args[0]]
        return non_none_args[0]

    return annotation


# Typer app that can be invoked by calling ``irdl`` from the CLI.
app = typer.Typer(no_args_is_help=True)

# Automatically register all supported datasets as subcommands to the app.
for name in irdl.__all__:
    dataset_class = getattr(irdl, name)
    get_method = dataset_class.get

    # Get docstring and signature from the classmethod
    doc = FunctionDoc(get_method)
    # For classmethods, we need to get the signature from __func__ to include the cls parameter
    # This ensures that when we set the modified signature back, it maintains the correct structure
    sig = signature(get_method.__func__)

    # Build Typer parameters with help text from docstring
    # Match parameters by name to handle docstring composition ordering
    param_docs = {p.name: " ".join(p.desc).replace("`", "") for p in doc["Parameters"]}

    # Build new parameters list, preserving cls for classmethods
    new_params = []
    for name, p in sig.parameters.items():
        if name == "cls":
            # Keep cls parameter as-is (no typer.Option annotation)
            new_params.append(p)
        else:
            # Add typer.Option annotation to dataset-specific parameters
            new_params.append(
                p.replace(
                    annotation=Annotated[_resolve_union_type(p.annotation), typer.Option(help=param_docs.get(name, ""))]
                )
            )

    # Set the modified signature on the underlying function
    get_method.__func__.__signature__ = sig.replace(parameters=new_params)

    # Build help text from docstring
    help_text = doc["Summary"][0] + "\n\n" + " ".join(doc["Extended Summary"])

    # Register subcommand using dataset_class.name for the command name
    app.command(
        name=dataset_class.name,
        help=help_text,
    )(get_method)

# expose click object for sphinx_click autodoc feature.
typer_click_object = typer.main.get_command(app)
