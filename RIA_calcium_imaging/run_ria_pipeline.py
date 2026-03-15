"""
Unified RIA Calcium Imaging pipeline runner.

This script provides one user-friendly entrypoint for the current RIA workflow.
It runs the existing step scripts in sequence, writes per-step logs, and produces a
run manifest so outputs are easy to inspect.

Default pipeline order:
1) validate prompts
2) crop RIA region
3) RIA compartment segmentation
4) brightness + orientation extraction
5) head segmentation
6) head-angle extraction

Example:
  python run_ria_pipeline.py
  python run_ria_pipeline.py --cycles 5 --continue-on-error
  python run_ria_pipeline.py --steps validate,crop,segment
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class StepResult:
    name: str
    script: str
    status: str  # success | noop | failed | skipped
    exit_code: int
    log_file: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RIA Calcium Imaging pipeline from a single command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated step list. Options: "
            "prepare,validate,crop,segment,brightness,head_segment,head_angle or 'all'."
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="How many pipeline cycles to run (useful when processing many videos incrementally).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps/cycles when a step fails.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip prompt asset validation before segmentation.",
    )
    return parser.parse_args()


def ensure_directories(project_root: Path) -> Dict[str, Path]:
    ria_script_dir = project_root / "RIA_calcium_imaging"
    base = project_root / "data/processed_files/RIA_calcium_imaging"
    dirs = {
        "project_root": project_root,
        "ria_script_dir": ria_script_dir,
        "base": base,
        "crop_outputs": base / "crop_outputs",
        "segmentation_outputs": base / "segmentation_outputs",
        "head_segmentation_outputs": base / "head_segmentation_outputs",
        "final_data": base / "final_data",
        "pipeline_runs": base / "pipeline_runs",
    }
    for key in ["crop_outputs", "segmentation_outputs", "head_segmentation_outputs", "final_data", "pipeline_runs"]:
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def discover_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def selected_step_order(step_arg: str, skip_validate: bool) -> List[str]:
    canonical = ["prepare", "validate", "crop", "segment", "brightness", "head_segment", "head_angle"]
    if step_arg.strip().lower() == "all":
        selected = canonical.copy()
    else:
        selected = [s.strip().lower() for s in step_arg.split(",") if s.strip()]
        unknown = [s for s in selected if s not in canonical]
        if unknown:
            raise ValueError(f"Unknown step(s): {unknown}. Valid: {canonical} or 'all'.")

    if skip_validate:
        selected = [s for s in selected if s != "validate"]

    return selected


def classify_status(step_name: str, exit_code: int, output: str) -> Tuple[str, str]:
    if exit_code == 0:
        return "success", "Step completed successfully."

    lowered = output.lower()
    noop_markers = [
        "all videos have been processed",
        "no videos found that need head angle processing",
        "all segmentation files already have brightness csv outputs",
        "no ria segmentation files found",
        "no head segmentation files found",
    ]
    if any(marker in lowered for marker in noop_markers):
        return "noop", "No pending work for this step."

    if "validate" in step_name and "no numbered .jpg/.jpeg files found" in lowered:
        return "failed", (
            "Prompt assets missing numbered images. Add files in "
            "RIA_calcium_imaging/prompt_frames and rerun."
        )

    return "failed", "Step failed. Check log for details."


def run_step(
    python_exe: str,
    script_dir: Path,
    run_log_dir: Path,
    step_name: str,
    script_file: str,
) -> StepResult:
    script_path = script_dir / script_file
    log_file = run_log_dir / f"{step_name}.log"

    if not script_path.exists():
        message = f"Missing script: {script_path}"
        log_file.write_text(message + "\n", encoding="utf-8")
        return StepResult(step_name, script_file, "failed", 2, str(log_file), message)

    cmd = [python_exe, str(script_path)]
    proc = subprocess.run(
        cmd,
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    combined_output = proc.stdout or ""
    log_file.write_text(combined_output, encoding="utf-8")
    status, message = classify_status(step_name, proc.returncode, combined_output)

    return StepResult(
        name=step_name,
        script=script_file,
        status=status,
        exit_code=proc.returncode,
        log_file=str(log_file),
        message=message,
    )


def summarize_outputs(dirs: Dict[str, Path]) -> Dict[str, int]:
    def count_matches(path: Path, suffix: str) -> int:
        if not path.exists():
            return 0
        return len([p for p in path.iterdir() if p.is_file() and p.name.endswith(suffix)])

    def count_subdirs(path: Path) -> int:
        if not path.exists():
            return 0
        return len([p for p in path.iterdir() if p.is_dir()])

    return {
        "crop_folders": count_subdirs(dirs["crop_outputs"]),
        "ria_segmentation_h5": count_matches(dirs["segmentation_outputs"], "_riasegmentation.h5"),
        "head_segmentation_h5": count_matches(dirs["head_segmentation_outputs"], "_headsegmentation.h5"),
        "brightness_csv": count_matches(dirs["final_data"], ".csv"),
        "head_angles_csv": count_matches(dirs["final_data"], "_headangles.csv"),
    }


def print_user_summary(
    run_dir: Path,
    selected_steps: List[str],
    all_results: List[StepResult],
    output_counts: Dict[str, int],
) -> None:
    print("\n" + "=" * 72)
    print("RIA Pipeline Run Summary")
    print("=" * 72)
    print(f"Run folder: {run_dir}")
    print(f"Steps requested: {', '.join(selected_steps)}")
    print("\nStep status:")
    for result in all_results:
        print(f"- {result.name:12s} -> {result.status:7s} (exit={result.exit_code})")
        print(f"  log: {result.log_file}")
        print(f"  note: {result.message}")

    print("\nCurrent output counts:")
    print(f"- crop folders:          {output_counts['crop_folders']}")
    print(f"- RIA segmentation h5:   {output_counts['ria_segmentation_h5']}")
    print(f"- head segmentation h5:  {output_counts['head_segmentation_h5']}")
    print(f"- brightness CSV files:  {output_counts['brightness_csv']}")
    print(f"- head-angle CSV files:  {output_counts['head_angles_csv']}")

    print("\nOutput layout:")
    print("- data/processed_files/RIA_calcium_imaging/")
    print("  - crop_outputs/")
    print("  - segmentation_outputs/")
    print("  - head_segmentation_outputs/")
    print("  - final_data/")
    print("  - pipeline_runs/<timestamp>/")
    print("    - logs/")
    print("    - run_manifest.json")


def main() -> int:
    args = parse_args()

    if args.cycles < 1:
        print("ERROR: --cycles must be >= 1")
        return 2

    project_root = discover_project_root()
    dirs = ensure_directories(project_root)

    python_exe = sys.executable
    if not python_exe:
        print("ERROR: Unable to determine Python executable.")
        return 2

    step_map = {
        "prepare": "1_tiftojpg.py",
        "validate": "validate_prompt_assets.py",
        "crop": "2_crop_RIAregion.py",
        "segment": "3_autoprompted_RIAsegmentation.py",
        "brightness": "4_extract_RIAbrightness_and_orientation.py",
        "head_segment": "5_head_segmentation.py",
        "head_angle": "6_extract_head_angle.py",
    }

    try:
        selected = selected_step_order(args.steps, args.skip_validate)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dirs["pipeline_runs"] / run_stamp
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[StepResult] = []

    print("Starting unified RIA pipeline...")
    print(f"Project root: {project_root}")
    print(f"Python: {python_exe}")
    print(f"Run folder: {run_dir}")

    for cycle in range(1, args.cycles + 1):
        print(f"\n--- Cycle {cycle}/{args.cycles} ---")

        for step_name in selected:
            script_name = step_map[step_name]
            print(f"Running step: {step_name} ({script_name})")
            result = run_step(
                python_exe=python_exe,
                script_dir=dirs["ria_script_dir"],
                run_log_dir=log_dir,
                step_name=f"cycle{cycle}_{step_name}",
                script_file=script_name,
            )
            all_results.append(result)
            print(f"  -> {result.status} (exit={result.exit_code})")

            if result.status == "failed" and not args.continue_on_error:
                print("Stopping early due to step failure. Use --continue-on-error to keep going.")
                output_counts = summarize_outputs(dirs)
                write_manifest(run_dir, project_root, selected, args.cycles, all_results, output_counts)
                print_user_summary(run_dir, selected, all_results, output_counts)
                return 1

    output_counts = summarize_outputs(dirs)
    write_manifest(run_dir, project_root, selected, args.cycles, all_results, output_counts)
    print_user_summary(run_dir, selected, all_results, output_counts)

    failed = [r for r in all_results if r.status == "failed"]
    return 1 if failed else 0


def write_manifest(
    run_dir: Path,
    project_root: Path,
    selected_steps: List[str],
    cycles: int,
    results: List[StepResult],
    output_counts: Dict[str, int],
) -> None:
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "selected_steps": selected_steps,
        "cycles": cycles,
        "results": [asdict(r) for r in results],
        "output_counts": output_counts,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
