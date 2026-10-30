"""Generate demo figures and coverage for the documentation site."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_coverage_markdown = ""
_coverage_badge = {}


def on_pre_build(config) -> None:
    """Run demos and tests before MkDocs renders the site."""
    global _coverage_markdown, _coverage_badge
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

    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "coverage.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=morpho",
                "--cov-report=term-missing",
                f"--cov-report=json:{report}",
            ],
            cwd=root,
            check=True,
        )
        coverage = json.loads(report.read_text())

    totals = coverage["totals"]
    _coverage_badge = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{totals['percent_covered_display']}%",
        "color": "blue",
    }
    rows = [
        "# Coverage",
        "",
        f"**Total: {totals['percent_covered_display']}%** "
        f"({totals['num_statements']} statements, {totals['missing_lines']} missed).",
        "",
        "| File | Statements | Missed | Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in sorted(coverage["files"].items()):
        summary = result["summary"]
        rows.append(
            f"| `{name}` | {summary['num_statements']} | "
            f"{summary['missing_lines']} | {summary['percent_covered_display']}% |"
        )
    rows.extend(["", "## Missing lines", ""])
    for name, result in sorted(coverage["files"].items()):
        if result["missing_lines"]:
            rows.append(f"- `{name}`: {', '.join(map(str, result['missing_lines']))}")
    _coverage_markdown = "\n".join(rows)


def on_page_markdown(markdown, page, config, files):
    """Render the coverage page with the MkDocs theme."""
    if page.file.src_uri == "coverage.md":
        return _coverage_markdown


def on_post_build(config) -> None:
    """Publish the coverage percentage for the README badge."""
    (Path(config["site_dir"]) / "coverage-badge.json").write_text(
        json.dumps(_coverage_badge), encoding="utf-8"
    )
