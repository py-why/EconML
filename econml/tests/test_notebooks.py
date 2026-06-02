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
        fetch_openml(data_id=61, as_frame=False)  # iris
        return False
    except urllib.error.HTTPError as e:
        # 5xx = server-side problem, 408 = request timeout, 429 = rate-limited:
        # all are transient "OpenML can't serve this right now" conditions.
        # Other 4xx (e.g. 404) indicate a real bug we shouldn't paper over.
        return e.code >= 500 or e.code in (408, 429)
    except OSError:
        # Any other network/transport-level failure: URLError, socket.timeout /
        # TimeoutError, http.client.RemoteDisconnected, ConnectionResetError,
        # ConnectionRefusedError, BrokenPipeError, ... All of these inherit
        # from OSError and all mean "we couldn't talk to OpenML right now."
        return True


@pytest.mark.parametrize("file", _notebooks)
@pytest.mark.notebook
def test_notebook(file):
    import nbformat
    import nbconvert

    # The Ames Housing notebook is the only one that depends on OpenML. When
    # OpenML's API is down the notebook is doomed to fail, so probe first and
    # xfail before paying the cost of running every cell. There's a small
    # window where OpenML could recover between this probe and the test
    # actually running, but the time savings from skipping the full notebook
    # execution are worth it.
    if "Ames Housing" in file and _openml_is_down():
        pytest.xfail("OpenML appears to be unavailable")

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
