# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import re
import pytest
import html
import os

_nbdir = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks')
_maindir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

_nbsubdirs = ['.']  # TODO: add AutoML notebooks

# filter directories by regex if the NOTEBOOK_DIR_PATTERN environment variable is set
_nbsubdirs = [d for d in _nbsubdirs if re.match(os.getenv('NOTEBOOK_DIR_PATTERN', '.*'), d)]

_notebooks = [
    os.path.join(subdir, path)
    for subdir in _nbsubdirs
    for path in os.listdir(os.path.join(_nbdir, subdir))
    if path.endswith('.ipynb')]
# omit the lalonde notebook
_notebooks = [nb for nb in _notebooks if "Lalonde" not in nb]


@pytest.mark.parametrize("file", _notebooks)
@pytest.mark.notebook
def test_notebook(file):
    import nbformat
    import nbconvert

    nb = nbformat.read(os.path.join(_nbdir, file), as_version=4)

    # make sure that coverage outputs reflect notebook contents
    nb.cells.insert(0, nbformat.v4.new_code_cell(f"""
    import os, coverage
    cwd = os.getcwd()
    os.chdir({_maindir!r}) # change to the root directory, so that setup.cfg is found
    coverage.process_startup()
    os.chdir(cwd) # change back to the original directory"""))

    for i in range(len(nb.cells), 9, -1):
        if nb.cells[i - 1].cell_type == 'code':
            nb.cells.insert(i, nbformat.v4.new_code_cell("""assert(matplotlib.get_backend() != 'agg')"""))

    # require all cells to complete within 15 minutes, which will help prevent us from
    # creating notebooks that are annoying for our users to actually run themselves
    ep = nbconvert.preprocessors.ExecutePreprocessor(
        timeout=1800, ignore_errors=True, extra_arguments=["--HistoryManager.enabled=False"])

    ep.preprocess(nb, {'metadata': {'path': _nbdir}})

    # remove added coverage cell, then decrement execution_count for other cells to account for it
    nb.cells.pop(0)
    for cell in nb.cells:
        if "execution_count" in cell:
            if cell["execution_count"] is not None:  # could be None if the cell errored
                cell["execution_count"] -= 1
        if "outputs" in cell:
            for output in cell["outputs"]:
                if "execution_count" in output:
                    output["execution_count"] -= 1

    output_file = os.path.join(_nbdir, 'output', file)
    # create directory if necessary
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nbformat.write(nb, output_file, version=4)

    errors = [nbconvert.preprocessors.CellExecutionError.from_cell_and_msg(cell, output)
              for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]
    if errors:
        err_str = "\n".join(html.unescape(str(err)) for err in errors)
        raise AssertionError("Encountered {0} exception(s):\n{1}".format(len(errors), err_str))
