# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = --fail-on-warning -W
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = irdl
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile generated-docs

generated-docs:
	mkdir -p _generated
	uv run python generate_dataset_docs.py
	uv run python -m typer ../src/irdl/cli.py utils docs --name irdl --output _generated/cli.md

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile generated-docs
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf build _autosummary _generated
	rm -f datasets/room_impulse_responses.rst datasets/head_related_impulse_responses.rst
	rm -f datasets/miracle_dataset.rst datasets/sriracha_dataset.rst datasets/fabian_dataset.rst
