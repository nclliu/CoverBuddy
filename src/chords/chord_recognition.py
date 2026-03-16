import os
import json
import autochord


# Chord Recognitions


def recognize_chords(audio_path, output_lab_path=None):
    """
    Run autochord's chord recognition on an audio file.
    Caches results to a JSON file so we don't re-process the same song.

    Parameters:
        audio_path (str):           Path to a .wav or .mp3 file
        output_lab_path (str|None): Where to save the .lab output file

    Returns:
        list of tuples: [(start_time, end_time, chord_label), ...]
    """
    # skip reprocessing if we already have results
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    export_dir = "output"
    os.makedirs(export_dir, exist_ok=True)
    cache_path = os.path.join(export_dir, f"{song_name}_chords.json")

    if os.path.exists(cache_path):
        print(f"   Chord predictions already cached at {cache_path}. Skipping.")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        return [(c["start"], c["end"], c["chord"]) for c in cached]

    # Run autochord
    print(f"   Running chord recognition on: {audio_path}")

    # autochord.recognize() returns list of (start, end, chord) tuples
    if output_lab_path:
        chords = autochord.recognize(audio_path, lab_fn=output_lab_path)
    else:
        chords = autochord.recognize(audio_path)

    print(f"   Found {len(chords)} chord segments.")

    # Save cache for next time
    cache_data = [{"start": s, "end": e, "chord": c} for s, e, c in chords]
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=4)
    print(f"   Cached chord predictions to {cache_path}")

    return chords


def load_lab_file(filepath):
    """
    Parse a .lab file

    Each line is: start_time  end_time  chord_label
    Used by both Isophonics and McGill Billboard datasets.

    Parameters:
        filepath (str): Path to the .lab file

    Returns:
        list of tuples: [(start_time, end_time, chord_label), ...]
    """
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2]
                entries.append((start, end, label))
    return entries


# Normalize these notes cause they sound the same but have different names
ENHARMONIC_MAP = {
    "Db": "C#",
    "Eb": "D#",
    "Fb": "E",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
}


def simplify_chord(chord_label):
    """
    Normalize a chord label to root + major/minor for fair comparison.

    Parameters:
        chord_label (str): Chord in any format

    Returns:
        str: Normalized chord (e.g., "C#", "Am", "N")
    """
    if chord_label in ("N", "X", "silence", ""):
        return "N"

    if ":" in chord_label:
        root, quality = chord_label.split(":", 1)
        root = ENHARMONIC_MAP.get(root, root)
        if quality.startswith("min"):
            return root + "m"
        else:
            return root

    # --- Handle autochord notation (already simple) ---
    # Examples: "Am", "C", "F#m", "Bbm"
    if chord_label.endswith("m") and len(chord_label) > 1:
        root = chord_label[:-1]
        root = ENHARMONIC_MAP.get(root, root)
        return root + "m"
    else:
        root = ENHARMONIC_MAP.get(chord_label, chord_label)
        return root


def evaluate_frame_accuracy(predicted, ground_truth, frame_size=0.1):
    """
    Compare predicted chords against ground truth using frame-level accuracy.

    The MIREX standard evaluation:
      1. Divide the song into small time frames (100ms each)
      2. For each frame, look up what chord the prediction says
         and what the ground truth says
      3. Count matching frames after normalizing labels

    Parameters:
        predicted:    list of (start, end, chord_label) from our model
        ground_truth: list of (start, end, chord_label) from dataset
        frame_size:   seconds per frame (0.1 = 100ms is standard)

    Returns:
        float: Accuracy between 0.0 and 1.0
    """
    pred_end = max(end for _, end, _ in predicted) if predicted else 0
    gt_end = max(end for _, end, _ in ground_truth) if ground_truth else 0
    total_duration = max(pred_end, gt_end)

    if total_duration == 0:
        return 0.0

    total_frames = 0
    correct_frames = 0

    t = 0.0
    while t < total_duration:
        # Look up predicted chord at time t
        pred_chord = "N"
        for start, end, label in predicted:
            if start <= t < end:
                pred_chord = simplify_chord(label)
                break

        # Look up ground truth chord at time t
        gt_chord = "N"
        for start, end, label in ground_truth:
            if start <= t < end:
                gt_chord = simplify_chord(label)
                break

        total_frames += 1
        if pred_chord == gt_chord:
            correct_frames += 1

        t += frame_size

    return correct_frames / total_frames if total_frames > 0 else 0.0
