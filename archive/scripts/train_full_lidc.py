"""Train the notebook ResNet50 on D:\\datasetLIDC_completo\\LIDC-IDRI.

Executes notebook cells through the test-metrics cell and stops before Grad-CAM
and Integrated Gradients.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03-resnet50-sprint3.ipynb"
STOP_MARKER = "test_results_resnet50.csv"


def main() -> None:
    os.chdir(ROOT / "notebooks")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {"__name__": "__main__"}
    plt.show = lambda *args, **kwargs: None
    ns["display"] = lambda *args, **kwargs: None

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        print(f"Executing cell {index}", flush=True)
        try:
            exec(source, ns, ns)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        if STOP_MARKER in source:
            print("Training and test metrics finished.", flush=True)
            return
    print("Stop marker not found.", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
