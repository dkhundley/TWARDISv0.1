"""
Unified Multi-Worm Feature Extraction pipeline runner.

This script provides one user-friendly entrypoint for the current multi-worm workflow.
It runs the existing step scripts in sequence, writes per-step logs, and produces a
run manifest so outputs are easy to inspect.

Default pipeline order:
1) TIFF -> JPEG conversion
2) SAM cutouts + classifier + metrics extraction

Example:
    python run_multiworm_pipeline.py
  python run_multiworm_pipeline.py --cycles 3 --continue-on-error --skip-existing
    python run_multiworm_pipeline.py --steps extract --classifier-source torchvision --classifier-weights /path/to/weights.pth
    python run_multiworm_pipeline.py --steps extract --classifier-source huggingface
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
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
        description="Run the Multi-Worm Feature Extraction pipeline from a single command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated step list. Options: convert,extract or 'all'.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="How many pipeline cycles to run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps/cycles when a step fails.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip TIFF->JPEG conversion step.",
    )

    parser.add_argument(
        "--classifier-weights",
        default=os.environ.get("WORM_CLASSIFIER_WEIGHTS", ""),
        help="Path to worm-vs-not-worm classifier .pth weights (required for torchvision classifier source).",
    )
    parser.add_argument(
        "--classifier-source",
        choices=["huggingface", "torchvision"],
        default="huggingface",
        help="Classifier backend used by extract step.",
    )
    parser.add_argument(
        "--auto-download-classifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically download classifier weights from Hugging Face when missing.",
    )
    parser.add_argument(
        "--classifier-hf-repo",
        default="lillyguisnet/celegans-classifier-vit-h-14-finetuned",
        help="Hugging Face repo ID used for classifier auto-download.",
    )
    parser.add_argument(
        "--classifier-hf-filename",
        default="",
        help="Specific checkpoint filename in the Hugging Face repo. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--classifier-cache-dir",
        default="models",
        help="Directory to store auto-downloaded classifier checkpoint.",
    )
    parser.add_argument(
        "--input-folder",
        default="images/processed-files/multiworm_feature_extraction/converted_images",
        help="Folder containing converted images for extraction step.",
    )
    parser.add_argument(
        "--output-base",
        default="images/processed-files/multiworm_feature_extraction",
        help="Base output folder for extraction step outputs and run artifacts.",
    )
    parser.add_argument(
        "--sam2-repo",
        default="sam2",
        help="Path to local SAM2 repository root.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="models/sam2_hiera_large.pt",
        help="Path to SAM2 checkpoint file.",
    )
    parser.add_argument(
        "--sam2-config",
        default="sam2_hiera_l.yaml",
        help="SAM2 config name (preferred) or path to SAM2 config yaml.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have a metrics pickle in extraction step.",
    )
    parser.add_argument(
        "--valid-margin",
        type=int,
        default=5,
        help="Erosion margin for valid imaging area detection.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.95,
        help="Overlap threshold for merging worm masks.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=25,
        help="Minimum mask area to keep after cleaning.",
    )
    parser.add_argument(
        "--area-filter-threshold",
        type=float,
        default=0.75,
        help="Relative area threshold used to keep detected worms.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Execution device for extract step.",
    )
    return parser.parse_args()


def discover_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def ensure_directories(project_root: Path, output_base: Path) -> Dict[str, Path]:
    script_dir = project_root / "multiworm_feature_extraction"
    dirs = {
        "project_root": project_root,
        "script_dir": script_dir,
        "base": output_base,
        "pipeline_runs": output_base / "pipeline_runs",
    }
    dirs["pipeline_runs"].mkdir(parents=True, exist_ok=True)
    return dirs


def selected_step_order(step_arg: str, skip_convert: bool) -> List[str]:
    canonical = ["convert", "extract"]
    if step_arg.strip().lower() == "all":
        selected = canonical.copy()
    else:
        selected = [s.strip().lower() for s in step_arg.split(",") if s.strip()]
        unknown = [s for s in selected if s not in canonical]
        if unknown:
            raise ValueError(f"Unknown step(s): {unknown}. Valid: {canonical} or 'all'.")

    if skip_convert:
        selected = [s for s in selected if s != "convert"]

    return selected


def choose_remote_checkpoint_filename(repo_id: str, explicit_filename: str) -> str:
    if explicit_filename:
        return explicit_filename

    api_url = f"https://huggingface.co/api/models/{repo_id}"
    with urllib.request.urlopen(api_url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    sibling_names = [entry.get("rfilename", "") for entry in payload.get("siblings", [])]

    pth_candidates = [name for name in sibling_names if name.lower().endswith(".pth")]
    if pth_candidates:
        return pth_candidates[0]

    pt_candidates = [name for name in sibling_names if name.lower().endswith(".pt")]
    if pt_candidates:
        return pt_candidates[0]

    safetensors_candidates = [name for name in sibling_names if name.lower().endswith(".safetensors")]
    if safetensors_candidates:
        raise RuntimeError(
            "Hugging Face repo contains only .safetensors checkpoints, but this pipeline currently expects "
            "a .pth/.pt classifier state dict for torchvision ViT-H/14. "
            "Provide --classifier-weights with a compatible local .pth file."
        )

    raise RuntimeError(
        "Could not auto-detect a .pth/.pt checkpoint in Hugging Face repo. "
        "Pass --classifier-hf-filename explicitly."
    )


def ensure_classifier_weights(args: argparse.Namespace, project_root: Path, selected_steps: List[str]) -> str:
    if "extract" not in selected_steps:
        return args.classifier_weights

    if args.classifier_source == "huggingface":
        return args.classifier_weights

    if args.classifier_weights:
        candidate = resolve_path(project_root, args.classifier_weights)
        if candidate.exists():
            return str(candidate)
        if not args.auto_download_classifier:
            raise FileNotFoundError(f"Classifier weights not found: {candidate}")
    elif not args.auto_download_classifier:
        raise ValueError("--classifier-weights is required when using --classifier-source=torchvision.")

    cache_dir = resolve_path(project_root, args.classifier_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    remote_filename = choose_remote_checkpoint_filename(args.classifier_hf_repo, args.classifier_hf_filename)
    local_path = cache_dir / Path(remote_filename).name
    if local_path.exists():
        print(f"Using existing classifier weights: {local_path}")
        return str(local_path)

    download_url = f"https://huggingface.co/{args.classifier_hf_repo}/resolve/main/{remote_filename}?download=true"
    print(f"Downloading classifier weights from Hugging Face: {args.classifier_hf_repo}/{remote_filename}")
    urllib.request.urlretrieve(download_url, str(local_path))
    print(f"Saved classifier weights: {local_path}")
    return str(local_path)


def classify_status(step_name: str, exit_code: int, output: str) -> Tuple[str, str]:
    if exit_code == 0:
        lowered = output.lower()
        noop_markers = [
            "no images found for processing in this cycle",
            "images queued: 0",
        ]
        if any(marker in lowered for marker in noop_markers):
            return "noop", "No pending work for this step."
        return "success", "Step completed successfully."

    return "failed", "Step failed. Check log for details."


def build_extract_command(args: argparse.Namespace, project_root: Path, script_path: Path) -> List[str]:
    sam2_config_text = str(args.sam2_config).strip()
    if Path(sam2_config_text).is_absolute() or "/" in sam2_config_text or "\\" in sam2_config_text:
        sam2_config_arg = str(resolve_path(project_root, sam2_config_text))
    else:
        sam2_config_arg = sam2_config_text

    cmd = [
        sys.executable,
        str(script_path),
        "--input-folder",
        str(resolve_path(project_root, args.input_folder)),
        "--output-base",
        str(resolve_path(project_root, args.output_base)),
        "--sam2-repo",
        str(resolve_path(project_root, args.sam2_repo)),
        "--sam2-checkpoint",
        str(resolve_path(project_root, args.sam2_checkpoint)),
        "--sam2-config",
        sam2_config_arg,
        "--classifier-source",
        args.classifier_source,
        "--classifier-hf-repo",
        args.classifier_hf_repo,
        "--valid-margin",
        str(args.valid_margin),
        "--overlap-threshold",
        str(args.overlap_threshold),
        "--min-area",
        str(args.min_area),
        "--area-filter-threshold",
        str(args.area_filter_threshold),
        "--device",
        args.device,
    ]

    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.classifier_weights:
        cmd.extend(["--classifier-weights", str(resolve_path(project_root, args.classifier_weights))])

    return cmd


def run_step(
    script_dir: Path,
    run_log_dir: Path,
    step_name: str,
    script_file: str,
    command: List[str],
) -> StepResult:
    script_path = script_dir / script_file
    log_file = run_log_dir / f"{step_name}.log"

    if not script_path.exists():
        message = f"Missing script: {script_path}"
        log_file.write_text(message + "\n", encoding="utf-8")
        return StepResult(step_name, script_file, "failed", 2, str(log_file), message)

    proc = subprocess.run(
        command,
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


def summarize_outputs(project_root: Path, output_base: Path) -> Dict[str, int]:
    converted_base = resolve_path(project_root, "images/processed-files/multiworm_feature_extraction/converted_images")
    metrics_dir = output_base / "metrics"
    cutouts_dir = output_base / "final_cutouts"
    noworms_file = output_base / "noworms.csv"

    converted_jpg = len(list(converted_base.rglob("*.jpg"))) if converted_base.exists() else 0
    metrics_pkl = len(list(metrics_dir.glob("*.pkl"))) if metrics_dir.exists() else 0
    cutout_png = len(list(cutouts_dir.glob("*.png"))) if cutouts_dir.exists() else 0

    if noworms_file.exists():
        lines = [line for line in noworms_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        noworm_rows = max(len(lines) - 1, 0)
    else:
        noworm_rows = 0

    return {
        "converted_jpg": converted_jpg,
        "metrics_pickle": metrics_pkl,
        "cutout_png": cutout_png,
        "noworm_rows": noworm_rows,
    }


def print_user_summary(
    run_dir: Path,
    selected_steps: List[str],
    all_results: List[StepResult],
    output_counts: Dict[str, int],
) -> None:
    print("\n" + "=" * 72)
    print("Multi-Worm Pipeline Run Summary")
    print("=" * 72)
    print(f"Run folder: {run_dir}")
    print(f"Steps requested: {', '.join(selected_steps)}")
    print("\nStep status:")
    for result in all_results:
        print(f"- {result.name:18s} -> {result.status:7s} (exit={result.exit_code})")
        print(f"  log: {result.log_file}")
        print(f"  note: {result.message}")

    print("\nCurrent output counts:")
    print(f"- converted JPG files:   {output_counts['converted_jpg']}")
    print(f"- metrics PKL files:     {output_counts['metrics_pickle']}")
    print(f"- cutout PNG files:      {output_counts['cutout_png']}")
    print(f"- noworm CSV rows:       {output_counts['noworm_rows']}")

    print("\nOutput layout:")
    print("- images/processed-files/multiworm_feature_extraction/")
    print("  - temp_cutouts/")
    print("  - final_cutouts/")
    print("  - metrics/")
    print("  - noworms.csv")
    print("  - pipeline_runs/<timestamp>/")
    print("    - logs/")
    print("    - run_manifest.json")


def write_manifest(
    run_dir: Path,
    project_root: Path,
    selected_steps: List[str],
    cycles: int,
    results: List[StepResult],
    output_counts: Dict[str, int],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "selected_steps": selected_steps,
        "cycles": cycles,
        "args": vars(args),
        "results": [asdict(r) for r in results],
        "output_counts": output_counts,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.cycles < 1:
        print("ERROR: --cycles must be >= 1")
        return 2

    project_root = discover_project_root()
    output_base = resolve_path(project_root, args.output_base)
    dirs = ensure_directories(project_root, output_base)

    try:
        selected = selected_step_order(args.steps, args.skip_convert)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        args.classifier_weights = ensure_classifier_weights(args, project_root, selected)
    except (ValueError, FileNotFoundError, RuntimeError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}")
        print(
            "Hint: use --classifier-source huggingface for the HF safetensors model, or "
            "--classifier-source torchvision --classifier-weights <path_to_.pth> for local state dicts."
        )
        return 2

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dirs["pipeline_runs"] / run_stamp
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    step_map = {
        "convert": "1_convert_images.py",
        "extract": "2_extract_wormcutouts.py",
    }

    all_results: List[StepResult] = []

    print("Starting unified multi-worm pipeline...")
    print(f"Project root: {project_root}")
    print(f"Python: {sys.executable}")
    print(f"Run folder: {run_dir}")

    for cycle in range(1, args.cycles + 1):
        print(f"\n--- Cycle {cycle}/{args.cycles} ---")

        for step_name in selected:
            script_name = step_map[step_name]
            script_path = dirs["script_dir"] / script_name

            if step_name == "convert":
                command = [sys.executable, str(script_path)]
            else:
                command = build_extract_command(args, project_root, script_path)

            print(f"Running step: {step_name} ({script_name})")
            result = run_step(
                script_dir=dirs["script_dir"],
                run_log_dir=log_dir,
                step_name=f"cycle{cycle}_{step_name}",
                script_file=script_name,
                command=command,
            )
            all_results.append(result)
            print(f"  -> {result.status} (exit={result.exit_code})")

            if result.status == "failed" and not args.continue_on_error:
                print("Stopping early due to step failure. Use --continue-on-error to keep going.")
                output_counts = summarize_outputs(project_root, output_base)
                write_manifest(run_dir, project_root, selected, args.cycles, all_results, output_counts, args)
                print_user_summary(run_dir, selected, all_results, output_counts)
                return 1

    output_counts = summarize_outputs(project_root, output_base)
    write_manifest(run_dir, project_root, selected, args.cycles, all_results, output_counts, args)
    print_user_summary(run_dir, selected, all_results, output_counts)

    failed = [result for result in all_results if result.status == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
