import importlib.metadata

metadata = importlib.metadata.metadata("irdl")

project = "irdl"
author = "Art J. R. Pelling"
copyright = f"2025-%Y, {author}"
version = metadata["Version"]

extensions = [
    "numpydoc",
    "sphinx_click",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]

html_static_path = ["_static"]
html_theme = "pydata_sphinx_theme"
html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
}
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["search-button-field"],
    "navbar_persistent": ["theme-switcher"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_align": "content",
    "logo": {
        "alt_text": "irdl",
        "text": "irdl",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/artpelling/irdl",
            "icon": "fa-brands fa-square-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/irdl",
            "icon": "_static/pypi.svg",
            "type": "local",
        },
    ],
    "collapse_navigation": False,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}

# sphinx_copybutton config
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "  # strips prompts
copybutton_prompt_is_regexp = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "exclude-members": "make_wrapper,wrapper,_abc_impl",
    "private-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "Path": "pathlib.Path",
}


intersphinx_mapping = {
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pooch": ("https://www.fatiando.org/pooch/latest", None),
    "pyfar": ("https://pyfar.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3/", None),
    "rich": ("https://rich.readthedocs.io/en/stable/", None),
    "sofar": ("https://sofar.readthedocs.io/en/latest/", None),
    "typer": ("https://typer.tiangolo.com/", None),
}
