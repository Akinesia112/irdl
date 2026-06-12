"""Automagic generation of a Typer script for all dataset downloads."""

import pathlib
import types
from inspect import signature
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin

import numpy as np
import pyfar as pf
import typer
from numpydoc.docscrape import FunctionDoc

import irdl
from irdl.base import _get_dataset_classes
from irdl.logging import configure_cli_logging

# Configure CLI logging
configure_cli_logging()


def _resolve_union_type(annotation: type) -> type:
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
                return str | None
            return str
        # For other unions, just take the first type
        if type(None) in args:
            return non_none_args[0] | None
        return non_none_args[0]

    return annotation


def _format_cli_value(value: Any) -> str:
    """Format CLI return values for readable terminal output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        sections = []
        for key, item in value.items():
            formatted = _format_cli_item(item)
            indented = "\n".join(f"  {line}" for line in formatted.splitlines())
            sections.append(f"{key}:\n{indented}")
        return "\n\n".join(sections)
    return str(value)


def _format_cli_item(value: Any) -> str:
    """Format one item inside a CLI result mapping."""
    if isinstance(value, np.ndarray):
        summary = f"ndarray shape={value.shape} dtype={value.dtype}"
        if value.ndim == 0:
            return f"{summary}\n{value.item()}"
        return f"{summary}\n{np.array2string(value, threshold=12, edgeitems=2)}"
    if isinstance(value, pf.Signal | pf.Coordinates):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _make_wrapper(cls, method, params, help_text, dataset_name, param_docs):
    def wrapper(**kwargs) -> Any:
        result = method.__func__(cls, **kwargs)
        if result is not None:
            typer.echo(_format_cli_value(result))
        return result

    # Build the signature for the wrapper
    new_params = []
    for name, p in params.items():
        if name == "cls":
            continue
        resolved_type = _resolve_union_type(p.annotation)
        # Don't pass default to typer.Option - it's already in the parameter
        new_params.append(p.replace(annotation=Annotated[resolved_type, typer.Option(help=param_docs.get(name, ""))]))

    wrapper.__signature__ = signature(wrapper).replace(parameters=new_params)
    wrapper.__doc__ = help_text
    wrapper.__name__ = f"{dataset_name}_wrapper"
    return wrapper


#: Typer app that can be invoked by calling ``irdl`` from the CLI.
app = typer.Typer(no_args_is_help=True)

# Automatically register all supported datasets as subcommands to the app.
for dataset_class in _get_dataset_classes(irdl):
    get_method = dataset_class.get

    # Get docstring and signature from the classmethod
    doc = FunctionDoc(get_method)
    sig = signature(get_method.__func__)

    # Build Typer parameters with help text from docstring
    param_docs = {p.name: " ".join(p.desc).replace("`", "") for p in doc["Parameters"]}

    # Build help text from docstring
    help_text = doc["Summary"][0] + "\n\n" + " ".join(doc["Extended Summary"])

    wrapper = _make_wrapper(dataset_class, get_method, sig.parameters, help_text, dataset_class.name, param_docs)

    # Register subcommand using dataset_class.name for the command name
    app.command(
        name=dataset_class.name,
        help=help_text,
    )(wrapper)
