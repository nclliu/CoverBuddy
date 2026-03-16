import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .beat_track import (
        AUDIO_EXTENSIONS,
        BeatTrackingMetrics,
        ANNOTATION_EXTENSIONS,
        evaluate_audio_beats,
        find_annotation_file,
    )
except ImportError:
    from beat_track import (  # type: ignore
        AUDIO_EXTENSIONS,
        BeatTrackingMetrics,
        ANNOTATION_EXTENSIONS,
        evaluate_audio_beats,
        find_annotation_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate beat tracking on a folder of audio files plus same-stem annotations."
    )
    parser.add_argument("dataset_dir", help="Folder containing audio files and annotation files.")
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Audio sample rate used for beat tracking.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.07,
        help="Beat matching tolerance in seconds for F-measure.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of audio files to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file path for saving evaluation results.",
    )
    parser.add_argument(
        "--backend",
        choices=("librosa", "beat_this"),
        default="librosa",
        help="Beat tracking backend to evaluate.",
    )
    return parser.parse_args()


def iter_audio_files(dataset_dir: Path) -> list[Path]:
    audio_files = [
        path
        for path in sorted(dataset_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return audio_files


def build_annotation_index(dataset_dir: Path) -> dict[str, Path]:
    annotation_roots = [
        dataset_dir,
        dataset_dir / "Annotations" / "beats",
        dataset_dir / "annotations" / "beats",
    ]
    index: dict[str, Path] = {}

    for root in annotation_roots:
        if not root.exists():
            continue
        for extension in ANNOTATION_EXTENSIONS:
            for path in root.rglob(f"*{extension}"):
                index.setdefault(path.stem, path)

    return index


def resolve_annotation_path(audio_path: Path, annotation_index: dict[str, Path]) -> Path | None:
    direct_match = find_annotation_file(audio_path)
    if direct_match is not None:
        return direct_match
    return annotation_index.get(audio_path.stem)


def summarize_metrics(metrics_list: list[BeatTrackingMetrics]) -> dict:
    if not metrics_list:
        return {
            "songs_evaluated": 0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f_measure": 0.0,
            "mean_absolute_error_ms": 0.0,
            "tempo_accuracy_8pct": 0.0,
        }

    return {
        "songs_evaluated": len(metrics_list),
        "mean_precision": float(np.mean([m.precision for m in metrics_list])),
        "mean_recall": float(np.mean([m.recall for m in metrics_list])),
        "mean_f_measure": float(np.mean([m.f_measure for m in metrics_list])),
        "mean_absolute_error_ms": float(
            np.mean([m.mean_absolute_error_ms for m in metrics_list])
        ),
        "tempo_accuracy_8pct": float(np.mean([m.tempo_accuracy_8pct for m in metrics_list])),
    }

def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    audio_files = iter_audio_files(dataset_dir)
    annotation_index = build_annotation_index(dataset_dir)
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    results: list[dict] = []
    metrics_list: list[BeatTrackingMetrics] = []

    for audio_path in audio_files:
        annotation_path = resolve_annotation_path(audio_path, annotation_index)
        if annotation_path is None:
            print(f"SKIP {audio_path}: no same-stem annotation file found")
            continue

        grid, metrics = evaluate_audio_beats(
            audio_path=audio_path,
            annotation_path=annotation_path,
            sr=args.sr,
            tolerance_seconds=args.tolerance,
            backend=args.backend,
        )
        metrics_list.append(metrics)

        print(
            f"{audio_path.name}: "
            f"F={metrics.f_measure:.3f} "
            f"P={metrics.precision:.3f} "
            f"R={metrics.recall:.3f} "
            f"tempo={grid.tempo_bpm:.1f}"
        )

        results.append(
            {
                "audio_path": str(audio_path),
                "annotation_path": str(annotation_path),
                "grid": grid.to_dict(),
                "metrics": metrics.to_dict(),
            }
        )

    summary = summarize_metrics(metrics_list)
    print("\nSummary")
    print(json.dumps(summary, indent=2))

    default_name = f"beat_eval_results_{args.backend}.json"
    output_path = Path(args.output) if args.output else dataset_dir / default_name
    payload = {"backend": args.backend, "summary": summary, "results": results}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
