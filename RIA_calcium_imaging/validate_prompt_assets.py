"""
Validate prompt assets used by 3_autoprompted_RIAsegmentation.py.

Usage:
  python validate_prompt_assets.py
  python validate_prompt_assets.py /path/to/prompt_frames /path/to/prompt_data.json
"""

import json
import os
import sys
from pathlib import Path


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_prompt_indices(prompt_dir):
    indices = set()
    for name in os.listdir(prompt_dir):
        lower = name.lower()
        if not lower.endswith(VALID_IMAGE_EXTENSIONS):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            indices.add(int(stem))
    return indices


def validate_prompt_assets(prompt_dir: Path, prompt_data_file: Path) -> int:
    errors = []

    if not prompt_dir.exists():
        errors.append(f"Missing prompt directory: {prompt_dir}")
    elif not prompt_dir.is_dir():
        errors.append(f"Prompt path is not a directory: {prompt_dir}")

    if not prompt_data_file.exists():
        errors.append(f"Missing prompt data file: {prompt_data_file}")
    elif not prompt_data_file.is_file():
        errors.append(f"Prompt data path is not a file: {prompt_data_file}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    prompt_indices = _load_prompt_indices(str(prompt_dir))
    if not prompt_indices:
        print(f"ERROR: No numbered .jpg/.jpeg files found in {prompt_dir}")
        return 1

    try:
        with open(prompt_data_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {prompt_data_file}: {exc}")
        return 1

    if not isinstance(data, dict) or not data:
        print("ERROR: prompt_data.json must be a non-empty object")
        return 1

    for frame_key, frame_prompts in data.items():
        if not str(frame_key).isdigit():
            errors.append(f"Frame key must be numeric string, got: {frame_key}")
            continue

        frame_idx = int(frame_key)
        if frame_idx not in prompt_indices:
            errors.append(
                f"Frame key {frame_key} has no matching prompt image in {prompt_dir}"
            )

        if not isinstance(frame_prompts, dict) or not frame_prompts:
            errors.append(f"Frame {frame_key} must map to a non-empty object-id dictionary")
            continue

        for obj_id, payload in frame_prompts.items():
            if not str(obj_id).isdigit():
                errors.append(f"Frame {frame_key}: object id must be numeric string, got: {obj_id}")
                continue

            if not isinstance(payload, dict):
                errors.append(f"Frame {frame_key}, object {obj_id}: payload must be an object")
                continue

            if "points" not in payload or "labels" not in payload:
                errors.append(
                    f"Frame {frame_key}, object {obj_id}: missing 'points' or 'labels'"
                )
                continue

            points = payload["points"]
            labels = payload["labels"]

            if not isinstance(points, list) or not isinstance(labels, list):
                errors.append(
                    f"Frame {frame_key}, object {obj_id}: 'points' and 'labels' must be lists"
                )
                continue

            if len(points) == 0:
                errors.append(f"Frame {frame_key}, object {obj_id}: 'points' cannot be empty")

            if len(points) != len(labels):
                errors.append(
                    f"Frame {frame_key}, object {obj_id}: points/labels length mismatch ({len(points)} vs {len(labels)})"
                )

            for i, point in enumerate(points):
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or not all(_is_number(v) for v in point)
                ):
                    errors.append(
                        f"Frame {frame_key}, object {obj_id}, point {i}: must be [x, y] numeric"
                    )

            for i, label in enumerate(labels):
                if label not in (0, 1):
                    errors.append(
                        f"Frame {frame_key}, object {obj_id}, label {i}: must be 0 or 1"
                    )

    if errors:
        print("Prompt asset validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Prompt asset validation passed.")
    print(f"- Prompt images: {len(prompt_indices)}")
    print(f"- Prompt frames in JSON: {len(data)}")
    print(f"- Prompt dir: {prompt_dir}")
    print(f"- Prompt JSON: {prompt_data_file}")
    return 0


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_prompt_dir = project_root / "RIA_calcium_imaging/prompt_frames"
    default_prompt_data = project_root / "RIA_calcium_imaging/prompt_data.json"

    if len(sys.argv) == 1:
        prompt_dir = default_prompt_dir
        prompt_data_file = default_prompt_data
    elif len(sys.argv) == 3:
        prompt_dir = Path(sys.argv[1]).expanduser().resolve()
        prompt_data_file = Path(sys.argv[2]).expanduser().resolve()
    else:
        print("Usage: python validate_prompt_assets.py [prompt_dir prompt_data_json]")
        sys.exit(2)

    sys.exit(validate_prompt_assets(prompt_dir, prompt_data_file))


if __name__ == "__main__":
    main()
