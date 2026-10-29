"""Matplotlib backend that saves demo figures as PNG files."""

import os
import sys
from pathlib import Path

from matplotlib import cycler, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureManagerBase, _Backend
from matplotlib.backends.backend_agg import FigureCanvasAgg


rcParams["axes.prop_cycle"] = cycler(
    "color",
    [
        "#1f78b4",
        "#d73027",
        "#1b9e77",
        "#d95f02",
        "#e6ab02",
        "#a6761d",
        "#66a61e",
        "#666666",
        "#b2df8a",
        "#7570b3",
    ],
)


class _SaveBackend(_Backend):
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase

    @staticmethod
    def show(*, block=None):
        """Save all open figures to the configured assets directory."""
        output_dir = Path(os.environ.get("MORPHO_DEMO_OUTPUT_DIR", "docs/assets"))
        output_dir.mkdir(parents=True, exist_ok=True)

        demo_name = Path(sys.argv[0]).stem if sys.argv else "demo"
        managers = Gcf.get_all_fig_managers()
        for index, manager in enumerate(managers, start=1):
            suffix = f"-{index}" if len(managers) > 1 else ""
            destination = output_dir / f"{demo_name}{suffix}.png"
            manager.canvas.figure.savefig(
                destination,
                format="png",
                dpi=180,
                bbox_inches="tight",
            )
            print(f"Saved {destination}")


FigureCanvas = _SaveBackend.FigureCanvas
FigureManager = _SaveBackend.FigureManager
show = _SaveBackend.show
