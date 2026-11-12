"""Set up the review environment and reproduce the three analysis stages.

Usage: python3.11 reviewer_run.py --assets-zip MSH-private-reviewer-assets.zip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import venv
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRIVATE_PREFIXES = ("02_frozen_literature_audit/assets/", "03_frozen_m2_m5/assets/")


def extract_private_assets(archive: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = member.filename.replace("\\", "/")
            target = (ROOT / name).resolve()
            if name == "PRIVATE_ASSETS_README.md":
                continue
            if not name.startswith(PRIVATE_PREFIXES) or not target.is_relative_to(ROOT):
                raise ValueError(f"Unexpected private archive member: {name}")
            if member.is_dir():
                continue
            if target.exists():
                raise FileExistsError(f"Refusing to replace existing asset: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)


def run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-zip", type=Path, help="Private ZIP provided through the journal submission system")
    parser.add_argument("--skip-install", action="store_true", help="Use the current Python environment when dependencies are already installed")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Python 3.11 is required; run this script with a Python 3.11 interpreter")
    if args.assets_zip:
        extract_private_assets(args.assets_zip.resolve())
    required = [
        ROOT / "02_frozen_literature_audit/assets/frozen_inputs/literature_hard_cleaned_all_ages.csv",
        ROOT / "02_frozen_literature_audit/assets/frozen_inputs/experiment_cleaned_all_ages.csv",
        ROOT / "03_frozen_m2_m5/assets/data/experiment_cleaned_all_ages.csv",
        ROOT / "03_frozen_m2_m5/assets/models/M2_local_XGBoost_reconstructed.joblib",
        ROOT / "03_frozen_m2_m5/assets/models/M5_weighted_source_XGBoost.joblib",
        ROOT / "03_frozen_m2_m5/assets/models/M5_target_calibrator_SVR.joblib",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Private assets are missing: " + ", ".join(missing))

    if args.skip_install:
        python = Path(sys.executable)
    else:
        environment = ROOT / ".venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements-publication.txt")], check=True)

    result = ROOT / "results/reviewer_run"
    if result.exists():
        raise FileExistsError(f"Results already exist: {result}; move them before running again")
    result.mkdir(parents=True)
    commands = {
        "stage1": [
            str(python), "01_literature_retraining/scripts/run_pipeline.py",
            "--literature", "02_frozen_literature_audit/assets/frozen_inputs/literature_hard_cleaned_all_ages.csv",
            "--experiment", "02_frozen_literature_audit/assets/frozen_inputs/experiment_cleaned_all_ages.csv",
            "--output", str(result / "stage1"), "--mode", "all",
        ],
        "stage2": [
            str(python), "02_frozen_literature_audit/scripts/run.py",
            "--output", str(result / "stage2"),
        ],
        "stage3": [str(python), "03_frozen_m2_m5/scripts/verify_archive.py"],
        "stage3_m5_rebuild": [str(python), "03_frozen_m2_m5/scripts/rebuild_m5.py",
                              "--output", str(result / "stage3_m5_rebuild")],
    }
    for name, command in commands.items():
        print(f"Running {name}...", flush=True)
        run(command, result / f"{name}.log")
    manifest = json.loads((result / "stage1/RUN_MANIFEST.json").read_text(encoding="utf-8"))
    checks = {name: value["status"] for name, value in manifest["benchmark_checks"].items()}
    (result / "reviewer_summary.json").write_text(
        json.dumps({"python": str(python), "stage1_checks": checks, "stage2": "completed",
                    "stage3": "completed", "stage3_m5_rebuild": "completed"}, indent=2),
        encoding="utf-8",
    )
    print(f"All three stages completed. Logs and summary: {result}")


if __name__ == "__main__":
    main()
