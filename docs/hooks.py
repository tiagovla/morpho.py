"""Generate demo figures before MkDocs collects documentation assets."""

import os
import subprocess
import sys
from pathlib import Path


def on_pre_build(config) -> None:
    """Run every demo with the PNG-saving Matplotlib backend."""
    root = Path(config.config_file_path).resolve().parent
    docs = root / "docs"
    env = os.environ.copy()
    output_dir = Path(env.get("MORPHO_DEMO_OUTPUT_DIR") or docs / "assets")
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    env["MORPHO_DEMO_OUTPUT_DIR"] = str(output_dir)
    env["MPLBACKEND"] = "module://demo_backend"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(docs), str(root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    force_regenerate = env.get("MORPHO_REBUILD_DEMOS", "").lower() in {
        "1",
        "true",
        "yes",
    }

    demos = sorted((root / "demo").rglob("*.py"))
    for demo in demos:
        if demo.name.startswith("_"):
            continue
        existing_figure = output_dir / f"{demo.stem}.png"
        if existing_figure.is_file() and not force_regenerate:
            print(f"Using existing figure {existing_figure}", flush=True)
            continue
        print(f"Generating figure for {demo.relative_to(root)}", flush=True)
        subprocess.run([sys.executable, str(demo)], cwd=root, env=env, check=True)
