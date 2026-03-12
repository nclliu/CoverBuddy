import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import librosa
import numpy as np


SUPPORTED_METERS: tuple[tuple[int, int], ...] = ((3, 4), (4, 4), (6, 8))
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
ANNOTATION_EXTENSIONS = (".beats", ".txt", ".lab", ".csv")


@dataclass
class BeatStageConfidence:
    beat_strength_mean: float
    meter_confidence: float
    downbeat_strength_mean: float


@dataclass
class BeatGrid:
    tempo_bpm: float
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    beat_confidence: BeatStageConfidence
    sr: int
    duration: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["beat_times"] = self.beat_times.tolist()
        payload["downbeat_times"] = self.downbeat_times.tolist()
        return payload


@dataclass
class BeatTrackingMetrics:
    precision: float
    recall: float
    f_measure: float
    matched_beats: int
    reference_beats: int
    estimated_beats: int
    mean_absolute_error_ms: float
    median_absolute_error_ms: float
    tempo_reference_bpm: float
    tempo_estimated_bpm: float
    tempo_accuracy_8pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Load audio as mono and normalize amplitude.
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs
    return y, sr


def _estimate_downbeat_confidence(
    beat_times: np.ndarray,
    downbeat_times: np.ndarray,
    supported_meters: Sequence[tuple[int, int]],
) -> float:
    """
    Estimate meter from beat/downbeat spacing by counting beats per bar.
    """
    if len(beat_times) < 2 or len(downbeat_times) < 2:
        return 0.0

    beats_per_bar: list[int] = []
    for start, end in zip(downbeat_times[:-1], downbeat_times[1:]):
        count = int(np.sum((beat_times >= start - 1e-6) & (beat_times < end - 1e-6)))
        if count > 0:
            beats_per_bar.append(count)

    if not beats_per_bar:
        return 0.0

    counts = np.asarray(beats_per_bar, dtype=float)
    best_error = np.inf
    for meter in supported_meters:
        error = float(np.mean(np.abs(counts - meter[0])))
        if error < best_error:
            best_error = error

    reference = max(min(meter[0] for meter in supported_meters), 1)
    confidence = max(0.0, 1.0 - best_error / reference)
    return confidence


def detect_beats_beat_this(
    path: str | Path,
    supported_meters: Sequence[tuple[int, int]] = SUPPORTED_METERS,
    device: str = "cpu",
    checkpoint: str = "final0",
) -> BeatGrid:
    """
    Beat tracking using the Beat This! model.
    """
    try:
        from beat_this.inference import File2Beats
    except ImportError as exc:
        raise ImportError(
            "beat_this is not installed. Install it before using backend='beat_this'."
        ) from exc

    cache_dir = os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    predictor = _get_beat_this_predictor(checkpoint=checkpoint, device=device)
    beat_times, downbeat_times = predictor(path)
    beat_times = np.asarray(beat_times, dtype=float)
    downbeat_times = np.asarray(downbeat_times, dtype=float)

    tempo_bpm = estimate_tempo_from_beats(beat_times)
    downbeat_confidence = _estimate_downbeat_confidence(
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        supported_meters=supported_meters,
    )

    return BeatGrid(
        tempo_bpm=tempo_bpm,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        beat_confidence=BeatStageConfidence(
            beat_strength_mean=0.0,
            meter_confidence=downbeat_confidence,
            downbeat_strength_mean=0.0,
        ),
        sr=sr,
        duration=duration,
    )


@lru_cache(maxsize=4)
def _get_beat_this_predictor(checkpoint: str, device: str):
    from beat_this.inference import File2Beats

    return File2Beats(checkpoint_path=checkpoint, device=device, float16=False, dbn=False)


def detect_beats_librosa(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100.0,
) -> tuple[float, np.ndarray, np.ndarray, int, float]:
    """
    Baseline beat tracking with librosa.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo_bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=tightness,
        units="frames",
    )
    tempo_bpm = float(np.asarray(tempo_bpm).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    beat_strength_mean = 0.0
    if len(beat_frames) > 0:
        beat_frames = np.clip(np.asarray(beat_frames, dtype=int), 0, len(onset_env) - 1)
        beat_strength_mean = float(np.mean(onset_env[beat_frames]))

    return tempo_bpm, beat_times, onset_env, hop_length, beat_strength_mean


def fix_half_double_tempo(tempo_bpm: float) -> float:
    """
    Very simple heuristic for pop-ish music.
    """
    if tempo_bpm < 70:
        return tempo_bpm * 2
    if tempo_bpm > 180:
        return tempo_bpm / 2
    return tempo_bpm


def build_bar_grid(
    beat_times: np.ndarray,
    meter_numerator: int = 4,
    first_downbeat_index: int = 0,
) -> list[tuple[int, int]]:
    """
    Assign each beat a (bar_number, beat_in_bar).
    """
    mapping: list[tuple[int, int]] = []
    for i in range(len(beat_times)):
        rel = i - first_downbeat_index
        if rel < 0:
            bar_number = 0
            beat_in_bar = (rel % meter_numerator) + 1
        else:
            bar_number = (rel // meter_numerator) + 1
            beat_in_bar = (rel % meter_numerator) + 1
        mapping.append((bar_number, beat_in_bar))
    return mapping


def infer_downbeats_from_beats(
    beat_times: np.ndarray,
    meter_numerator: int = 4,
    first_downbeat_index: int = 0,
) -> np.ndarray:
    """
    Fallback downbeat estimate: assume every N-th beat is a bar start.
    """
    if len(beat_times) == 0:
        return np.array([], dtype=float)

    first_downbeat_index = max(0, min(first_downbeat_index, len(beat_times) - 1))
    return beat_times[first_downbeat_index::meter_numerator]


def _candidate_meter_score(
    beat_times: np.ndarray,
    sr: int,
    onset_envelope: np.ndarray,
    hop_length: int,
    meter_numerator: int,
) -> tuple[float, int, float]:
    """
    Score a candidate meter by how consistently strong its downbeat positions look.
    """
    if len(beat_times) == 0:
        return 0.0, 0, 0.0

    onset_mean = float(np.mean(onset_envelope)) if len(onset_envelope) > 0 else 0.0
    best_score = -np.inf
    best_offset = 0
    best_strength = 0.0

    for offset in range(min(meter_numerator, len(beat_times))):
        candidate_times = beat_times[offset::meter_numerator]
        if len(candidate_times) == 0:
            continue

        frames = librosa.time_to_frames(candidate_times, sr=sr, hop_length=hop_length)
        frames = np.clip(frames, 0, len(onset_envelope) - 1)
        downbeat_strength = float(np.mean(onset_envelope[frames]))

        next_times = beat_times[offset + 1 :: meter_numerator]
        if len(next_times) > 0:
            next_frames = librosa.time_to_frames(next_times, sr=sr, hop_length=hop_length)
            next_frames = np.clip(next_frames, 0, len(onset_envelope) - 1)
            non_downbeat_strength = float(np.mean(onset_envelope[next_frames]))
        else:
            non_downbeat_strength = onset_mean

        score = downbeat_strength - non_downbeat_strength
        if score > best_score:
            best_score = score
            best_offset = offset
            best_strength = downbeat_strength

    return float(best_score), best_offset, best_strength


def infer_meter_and_downbeats(
    beat_times: np.ndarray,
    sr: int,
    onset_envelope: np.ndarray,
    hop_length: int,
    supported_meters: Sequence[tuple[int, int]] = SUPPORTED_METERS,
) -> tuple[np.ndarray, float, float]:
    """
    Pick the most plausible meter among a small supported set and infer downbeats.
    """
    if len(beat_times) == 0:
        return np.array([], dtype=float), 0.0, 0.0

    scored_candidates: list[tuple[float, int, float, int, int]] = []
    for numerator, denominator in supported_meters:
        score, offset, strength = _candidate_meter_score(
            beat_times=beat_times,
            sr=sr,
            onset_envelope=onset_envelope,
            hop_length=hop_length,
            meter_numerator=numerator,
        )
        scored_candidates.append((score, offset, strength, numerator, denominator))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_offset, best_strength, numerator, _denominator = scored_candidates[0]
    second_score = scored_candidates[1][0] if len(scored_candidates) > 1 else 0.0
    confidence = max(0.0, float(best_score - second_score))

    downbeat_times = infer_downbeats_from_beats(
        beat_times=beat_times,
        meter_numerator=numerator,
        first_downbeat_index=best_offset,
    )
    return downbeat_times, confidence, best_strength


def make_beat_grid(
    path: str | Path,
    sr: int = 22050,
    supported_meters: Sequence[tuple[int, int]] = SUPPORTED_METERS,
    backend: str = "librosa",
) -> BeatGrid:
    """
    Full baseline pipeline:
    - load audio
    - detect beats
    - fix obvious half/double tempo issues
    - infer meter and downbeats
    """
    if backend == "beat_this":
        return detect_beats_beat_this(
            path=path,
            supported_meters=supported_meters,
            device="cpu",
        )

    if backend != "librosa":
        raise ValueError(f"Unsupported backend: {backend}")

    y, sr = load_audio(path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo_bpm, beat_times, onset_envelope, hop_length, beat_strength_mean = detect_beats_librosa(
        y, sr
    )
    tempo_bpm = fix_half_double_tempo(tempo_bpm)
    downbeat_times, meter_confidence, downbeat_strength_mean = infer_meter_and_downbeats(
        beat_times=beat_times,
        sr=sr,
        onset_envelope=onset_envelope,
        hop_length=hop_length,
        supported_meters=supported_meters,
    )

    return BeatGrid(
        tempo_bpm=tempo_bpm,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        beat_confidence=BeatStageConfidence(
            beat_strength_mean=beat_strength_mean,
            meter_confidence=meter_confidence,
            downbeat_strength_mean=downbeat_strength_mean,
        ),
        sr=sr,
        duration=duration,
    )


def snap_time_to_nearest_beat(t: float, beat_times: np.ndarray) -> int:
    """
    Return the index of the nearest beat.
    """
    if len(beat_times) == 0:
        raise ValueError("beat_times is empty")
    return int(np.argmin(np.abs(beat_times - t)))


def snap_time_to_previous_beat(t: float, beat_times: np.ndarray) -> int:
    """
    Return the index of the previous beat.
    Better for chord changes because it avoids pushing them late.
    """
    if len(beat_times) == 0:
        raise ValueError("beat_times is empty")
    idx = int(np.searchsorted(beat_times, t, side="right") - 1)
    return max(0, idx)


def snap_time_to_nearest_bar(t: float, downbeat_times: np.ndarray) -> int:
    """
    Return the index of the nearest downbeat / bar start.
    """
    if len(downbeat_times) == 0:
        raise ValueError("downbeat_times is empty")
    return int(np.argmin(np.abs(downbeat_times - t)))


def load_beat_annotations(path: str | Path) -> np.ndarray:
    """
    Load beat times from a simple text annotation file.

    Supported formats:
    - one timestamp per line
    - whitespace/comma separated rows where the first token is the timestamp
    - common `.lab` format rows such as: `time beat_index`
    """
    beat_times: list[float] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part for part in line.replace(",", " ").split() if part]
        if not parts:
            continue

        try:
            beat_times.append(float(parts[0]))
        except ValueError:
            continue

    return np.asarray(sorted(beat_times), dtype=float)


def _match_events(
    reference_times: np.ndarray,
    estimated_times: np.ndarray,
    tolerance_seconds: float,
) -> tuple[int, list[float]]:
    """
    Greedy 1:1 event matching within a tolerance window.
    """
    ref = np.asarray(reference_times, dtype=float)
    est = np.asarray(estimated_times, dtype=float)
    i = 0
    j = 0
    matched = 0
    errors: list[float] = []

    while i < len(ref) and j < len(est):
        delta = est[j] - ref[i]
        if abs(delta) <= tolerance_seconds:
            matched += 1
            errors.append(abs(delta))
            i += 1
            j += 1
        elif est[j] < ref[i] - tolerance_seconds:
            j += 1
        else:
            i += 1

    return matched, errors


def estimate_tempo_from_beats(beat_times: np.ndarray) -> float:
    if len(beat_times) < 2:
        return 0.0
    ibis = np.diff(beat_times)
    median_ibi = float(np.median(ibis))
    if median_ibi <= 0:
        return 0.0
    return 60.0 / median_ibi


def evaluate_beat_tracking(
    reference_times: np.ndarray,
    estimated_times: np.ndarray,
    tolerance_seconds: float = 0.07,
) -> BeatTrackingMetrics:
    """
    Compute event-based beat tracking accuracy.
    """
    reference_times = np.asarray(sorted(reference_times), dtype=float)
    estimated_times = np.asarray(sorted(estimated_times), dtype=float)

    matched_beats, errors = _match_events(
        reference_times=reference_times,
        estimated_times=estimated_times,
        tolerance_seconds=tolerance_seconds,
    )

    precision = matched_beats / len(estimated_times) if len(estimated_times) > 0 else 0.0
    recall = matched_beats / len(reference_times) if len(reference_times) > 0 else 0.0
    if precision + recall == 0:
        f_measure = 0.0
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    if errors:
        mean_abs_error_ms = float(np.mean(errors) * 1000.0)
        median_abs_error_ms = float(np.median(errors) * 1000.0)
    else:
        mean_abs_error_ms = 0.0
        median_abs_error_ms = 0.0

    tempo_reference_bpm = estimate_tempo_from_beats(reference_times)
    tempo_estimated_bpm = estimate_tempo_from_beats(estimated_times)
    if tempo_reference_bpm <= 0 or tempo_estimated_bpm <= 0:
        tempo_accuracy_8pct = 0.0
    else:
        relative_error = abs(tempo_estimated_bpm - tempo_reference_bpm) / tempo_reference_bpm
        tempo_accuracy_8pct = 1.0 if relative_error <= 0.08 else 0.0

    return BeatTrackingMetrics(
        precision=precision,
        recall=recall,
        f_measure=f_measure,
        matched_beats=matched_beats,
        reference_beats=len(reference_times),
        estimated_beats=len(estimated_times),
        mean_absolute_error_ms=mean_abs_error_ms,
        median_absolute_error_ms=median_abs_error_ms,
        tempo_reference_bpm=tempo_reference_bpm,
        tempo_estimated_bpm=tempo_estimated_bpm,
        tempo_accuracy_8pct=tempo_accuracy_8pct,
    )


def evaluate_audio_beats(
    audio_path: str | Path,
    annotation_path: str | Path,
    sr: int = 22050,
    tolerance_seconds: float = 0.07,
    backend: str = "librosa",
) -> tuple[BeatGrid, BeatTrackingMetrics]:
    grid = make_beat_grid(audio_path, sr=sr, backend=backend)
    reference_times = load_beat_annotations(annotation_path)
    metrics = evaluate_beat_tracking(reference_times, grid.beat_times, tolerance_seconds)
    return grid, metrics


def find_annotation_file(audio_path: str | Path) -> Optional[Path]:
    audio_path = Path(audio_path)
    for extension in ANNOTATION_EXTENSIONS:
        candidate = audio_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def print_grid_summary(grid: BeatGrid, max_rows: int = 20) -> None:
    print(f"Tempo: {grid.tempo_bpm:.2f} BPM")
    print(f"Duration: {grid.duration:.2f} sec")
    print(f"Beats detected: {len(grid.beat_times)}")
    print(f"Downbeats detected: {len(grid.downbeat_times)}")
    print(
        "Confidence:"
        f" beat={grid.beat_confidence.beat_strength_mean:.3f},"
        f" meter={grid.beat_confidence.meter_confidence:.3f},"
        f" downbeat={grid.beat_confidence.downbeat_strength_mean:.3f}"
    )
    print("\nFirst beats:")
    for i, t in enumerate(grid.beat_times):
        if i >= max_rows:
            break
        print(f"beat_idx={i:3d}  time={t:7.3f}s")


def plot_beats(
    path: str | Path,
    grid: BeatGrid,
    max_time: Optional[float] = 30.0,
) -> None:
    """
    Plot waveform with beat and downbeat markers.
    """
    import matplotlib.pyplot as plt

    y, sr = load_audio(path, sr=grid.sr)

    if max_time is not None:
        max_samples = min(len(y), int(max_time * sr))
        y = y[:max_samples]
        end_time = max_samples / sr
    else:
        end_time = len(y) / sr

    times = np.arange(len(y)) / sr

    plt.figure(figsize=(14, 4))
    plt.plot(times, y, linewidth=0.7, alpha=0.8)

    for bt in grid.beat_times:
        if bt <= end_time:
            plt.axvline(bt, linewidth=0.8, alpha=0.5)

    for db in grid.downbeat_times:
        if db <= end_time:
            plt.axvline(db, linewidth=2.0, alpha=0.9)

    plt.title(
        f"Beat grid: {Path(path).name}"
        f" | tempo ≈ {grid.tempo_bpm:.1f} BPM"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def save_grid_json(grid: BeatGrid, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(grid.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    audio_path = "your_song.mp3"

    grid = make_beat_grid(audio_path, sr=22050)
    print_grid_summary(grid)
    plot_beats(audio_path, grid, max_time=30.0)
