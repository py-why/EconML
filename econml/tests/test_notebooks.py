# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import re
import pytest
import html
import os

_nbdir = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks')
_maindir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

_nbsubdirs = ['.', 'CustomerScenarios', 'Solutions']  # TODO: add AutoML notebooks

# filter directories by regex if the NOTEBOOK_DIR_PATTERN environment variable is set
_nbsubdirs = [d for d in _nbsubdirs if re.match(os.getenv('NOTEBOOK_DIR_PATTERN', '.*'), d)]

_notebooks = [
    os.path.join(subdir, path)
    for subdir in _nbsubdirs
    for path in os.listdir(os.path.join(_nbdir, subdir))
    if path.endswith('.ipynb')]
# omit the lalonde notebook
_notebooks = [nb for nb in _notebooks if "Lalonde" not in nb]


def _openml_is_down():
    """Quick liveness probe for ``sklearn.datasets.fetch_openml``.

    The Ames Housing notebook (and only that one) calls ``fetch_openml``,
    which depends on OpenML's API endpoints. Those endpoints go down
    periodically, which is not something this repo can fix; when they're down
    we don't want the notebook test to register as a real failure.

    We drive the probe through ``fetch_openml`` itself rather than poking a
    hardcoded URL, so this tracks whatever endpoints the installed sklearn
    actually uses (e.g. if sklearn migrates to an OpenML v2 API in the
    future). We point at a small dataset (``iris``) so the metadata round-trip
    is cheap; on success we ignore the returned bunch entirely.
    """
    import urllib.error
    from sklearn.datasets import fetch_openml

    try:
        fetch_openml(data_id=61, as_frame=False, parser="liac-arff")  # iris
        return False
    except urllib.error.HTTPError as e:
        # 5xx = OpenML's servers are unhappy; treat as "down".
        # 4xx would indicate a real bug (bad request, removed dataset, ...).
        return e.code >= 500
    except (urllib.error.URLError, TimeoutError):
        return True


def _notebook_params():
    openml_down = None  # probe lazily, only if needed
    for nb in _notebooks:
        if "Ames Housing" in nb:
            if openml_down is None:
                openml_down = _openml_is_down()
            marks = [pytest.mark.xfail(openml_down,
                                       reason="OpenML appears to be unavailable",
                                       strict=False)] if openml_down else []
            yield pytest.param(nb, marks=marks)
        else:
            yield nb


@pytest.mark.parametrize("file", list(_notebook_params()))
@pytest.mark.notebook
def test_notebook(file):
    import nbformat
    import nbconvert

    nb = nbformat.read(os.path.join(_nbdir, file), as_version=4)

    # require all cells to complete within 15 minutes, which will help prevent us from
    # creating notebooks that are annoying for our users to actually run themselves
    ep = nbconvert.preprocessors.ExecutePreprocessor(
        timeout=1800, allow_errors=True)

    ep.preprocess(nb, {'metadata': {'path': '.'}})

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
