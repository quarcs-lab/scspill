"""Generate the Colab notebooks from the executed docs tutorials.

Converts ``docs/get-started.qmd`` -> ``notebooks/california.ipynb`` and
``docs/sudan.qmd`` -> ``notebooks/sudan.ipynb`` with ``quarto convert``,
strips the YAML front matter and raw-HTML blocks, and prepends a title cell
(with a build stamp) plus a pip-install cell. CI drift-checks the outputs
ignoring the build stamp line.
"""

from __future__ import annotations

import datetime
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

CONVERSIONS = [
    ("docs/get-started.qmd", "notebooks/california.ipynb", "scspill — California Proposition 99"),
    ("docs/sudan.qmd", "notebooks/sudan.ipynb", "scspill — the 2011 Sudan secession"),
]

INSTALL_CELL = '%pip install -q "scspill[numba] @ git+https://github.com/quarcs-lab/scspill.git"'


def convert(qmd_rel: str, ipynb_rel: str, title: str) -> None:
    """Convert one qmd tutorial into a self-contained Colab notebook."""
    qmd = ROOT / qmd_rel
    out = ROOT / ipynb_rel
    with tempfile.TemporaryDirectory() as tmp:
        tmp_ipynb = Path(tmp) / "converted.ipynb"
        subprocess.run(
            ["quarto", "convert", str(qmd), "--output", str(tmp_ipynb)],
            check=True,
            capture_output=True,
        )
        nb = nbformat.read(tmp_ipynb, as_version=4)

    cells = []
    for cell in nb.cells:
        src = cell.source
        if cell.cell_type == "markdown":
            # Drop the front matter block and raw HTML (banners/buttons).
            src = re.sub(r"\A---\n.*?\n---\n?", "", src, flags=re.S)
            src = re.sub(r"```\{=html\}.*?```", "", src, flags=re.S)
            src = src.strip()
            if not src:
                continue
            cell.source = src
        else:
            # Strip Quarto cell options (#| ...) which Colab renders literally.
            cell.source = "\n".join(
                line for line in src.splitlines() if not line.startswith("#|")
            ).strip()
        cells.append(cell)

    stamp = datetime.date.today().isoformat()
    header = nbformat.v4.new_markdown_cell(
        f"# {title}\n\n"
        f"Generated from the scspill documentation — "
        f"see <https://quarcs-lab.github.io/scspill/> for the rendered version.\n\n"
        f"_Notebook version: built {stamp}_"
    )
    install = nbformat.v4.new_code_cell(INSTALL_CELL)
    nb.cells = [header, install, *cells]
    for cell in nb.cells:  # Colab chokes on ids from other tools
        cell.pop("id", None)
    nb.metadata["colab"] = {"provenance": []}
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    print(f"wrote {out}")


def main() -> int:
    for qmd, ipynb, title in CONVERSIONS:
        convert(qmd, ipynb, title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
