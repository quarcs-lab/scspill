"""The containers moved in v0.3; their historical import paths must still work."""

import subprocess
import sys

import pytest

from scspill.utils.scspill_helpers import containers
from scspill.utils.scspill_helpers.sar import structures as sar_structures


def test_old_structures_import_path():
    from scspill.utils.scspill_helpers.structures import (
        AlphaPosterior,
        SARPosterior,
        SCSPILLEffects,
        SCSPILLFit,
        SCSPILLInputs,
    )

    assert AlphaPosterior is sar_structures.AlphaPosterior
    assert SARPosterior is sar_structures.SARPosterior
    assert SCSPILLEffects is containers.SCSPILLEffects
    assert SCSPILLInputs is sar_structures.SARInputs
    assert SCSPILLFit is sar_structures.SARFit


def test_old_setup_import_path():
    from scspill.utils.scspill_helpers.setup import (
        normalize_w,
        prepare_scspill_inputs,
        row_normalize,
    )

    assert callable(prepare_scspill_inputs)
    assert callable(normalize_w) and callable(row_normalize)


def test_sar_inputs_extend_the_shared_panel():
    assert issubclass(sar_structures.SARInputs, containers.PanelInputs)


@pytest.mark.parametrize(
    "module",
    [
        "scspill.utils.scspill_helpers.containers",
        "scspill.utils.scspill_helpers.sar.structures",
        "scspill.utils.scspill_helpers.setup",
        "scspill.utils.scspill_helpers.structures",
    ],
)
def test_fresh_import_has_no_cycle(module):
    """Each module imports cleanly in a fresh interpreter (no import cycle)."""
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True)
