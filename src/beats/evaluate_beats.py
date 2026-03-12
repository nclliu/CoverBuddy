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
        plot_beats,
    )
except ImportError:
    from beat_track import (  # type: ignore
        AUDIO_EXTENSIONS,
        BeatTrackingMetrics,
        ANNOTATION_EXTENSIONS,
        evaluate_audio_beats,
        find_annotation_file,
        plot_beats,
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
        "--plot",
        action="store_true",
        help="Show beat/downbeat plots for each evaluated song.",
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
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save aggregate evaluation plots next to the JSON output.",
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


def save_visualizations(results: list[dict], summary: dict, output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    if not results:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    names = [Path(result["audio_path"]).name for result in results]
    f_measures = np.asarray([result["metrics"]["f_measure"] for result in results], dtype=float)
    tempo_ref = np.asarray(
        [result["metrics"]["tempo_reference_bpm"] for result in results], dtype=float
    )
    tempo_est = np.asarray(
        [result["metrics"]["tempo_estimated_bpm"] for result in results], dtype=float
    )
    mean_abs_error = np.asarray(
        [result["metrics"]["mean_absolute_error_ms"] for result in results], dtype=float
    )
    meter_labels = [
        f"{result['grid']['meter_numerator']}/{result['grid']['meter_denominator']}"
        for result in results
    ]

    histogram_path = output_dir / "beat_f_measure_histogram.png"
    plt.figure(figsize=(8, 4.5))
    plt.hist(f_measures, bins=15, range=(0.0, 1.0), edgecolor="black")
    plt.axvline(summary["mean_f_measure"], color="red", linestyle="--", linewidth=1.5)
    plt.title("Beat Tracking F-measure Distribution")
    plt.xlabel("F-measure")
    plt.ylabel("Song count")
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=150)
    plt.close()
    saved_paths.append(str(histogram_path))

    scatter_path = output_dir / "tempo_reference_vs_estimated.png"
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(tempo_ref, tempo_est, alpha=0.75)
    max_tempo = max(np.max(tempo_ref), np.max(tempo_est), 1.0)
    plt.plot([0, max_tempo], [0, max_tempo], linestyle="--", linewidth=1.0)
    plt.title("Reference vs Estimated Tempo")
    plt.xlabel("Reference tempo (BPM)")
    plt.ylabel("Estimated tempo (BPM)")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    saved_paths.append(str(scatter_path))

    unique_meters = sorted(set(meter_labels))
    meter_counts = [meter_labels.count(label) for label in unique_meters]
    meter_path = output_dir / "predicted_meter_counts.png"
    plt.figure(figsize=(6.5, 4.0))
    plt.bar(unique_meters, meter_counts)
    plt.title("Predicted Meter Counts")
    plt.xlabel("Predicted meter")
    plt.ylabel("Song count")
    plt.tight_layout()
    plt.savefig(meter_path, dpi=150)
    plt.close()
    saved_paths.append(str(meter_path))

    worst_count = min(10, len(results))
    worst_idx = np.argsort(f_measures)[:worst_count]
    worst_names = [names[i] for i in worst_idx]
    worst_scores = [float(f_measures[i]) for i in worst_idx]
    worst_path = output_dir / "worst_f_measure_tracks.png"
    plt.figure(figsize=(9, 5))
    plt.barh(worst_names, worst_scores)
    plt.gca().invert_yaxis()
    plt.xlim(0.0, 1.0)
    plt.title("Lowest F-measure Tracks")
    plt.xlabel("F-measure")
    plt.tight_layout()
    plt.savefig(worst_path, dpi=150)
    plt.close()
    saved_paths.append(str(worst_path))

    error_path = output_dir / "mean_absolute_error_histogram.png"
    plt.figure(figsize=(8, 4.5))
    plt.hist(mean_abs_error, bins=15, edgecolor="black")
    plt.title("Mean Absolute Beat Error")
    plt.xlabel("Error (ms)")
    plt.ylabel("Song count")
    plt.tight_layout()
    plt.savefig(error_path, dpi=150)
    plt.close()
    saved_paths.append(str(error_path))

    return saved_paths


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
            f"tempo={grid.tempo_bpm:.1f} "
            f"meter={grid.meter_numerator}/{grid.meter_denominator}"
        )

        if args.plot:
            plot_beats(audio_path, grid)

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

    if args.save_viz:
        viz_dir = output_path.parent / "beat_eval_viz"
        saved_paths = save_visualizations(results, summary, viz_dir)
        print(f"Saved visualizations to {viz_dir}")
        for path in saved_paths:
            print(path)


if __name__ == "__main__":
    main()
