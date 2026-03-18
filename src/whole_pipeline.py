import sys
import os
import json
import argparse
import numpy as np

# Required for the current autochord/TensorFlow compatibility path.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from beats.beat_track import make_beat_grid, build_bar_grid

from chords.chord_recognition import recognize_chords, simplify_chord

from lyrics.pipeline import extract_vocals, get_timestamped_lyrics

from export.export_sheet import convert_json_to_musicxml, convert_xml_to_pdf_musescore


DEFAULT_BEAT_BACKEND = "beat_this"


def build_output_stem(audio_path, beat_backend):
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    return f"{song_name}_{beat_backend}"


def get_chord_at_time(t, chords):
    """Look up what chord is playing at time t"""
    for start, end, label in chords:
        if start <= t < end:
            return simplify_chord(label)
    return "N"


def get_words_near_time(t, words, tolerance=0.15):
    """Find any words whose start time is closest to this beat"""
    nearby = []
    for word_start, word_end, text in words:
        if abs(word_start - t) < tolerance:
            nearby.append(text)
    return " ".join(nearby)


def extract_word_timestamps(lyrics_data):
    """Pull timestamps from the lyrics"""
    words = []
    for segment in lyrics_data.get("segments", []):
        for w in segment.get("words", []):
            if "start" in w and "end" in w and "word" in w:
                words.append((w["start"], w["end"], w["word"]))
    return words


def generate_chord_chart(audio_path, beat_backend=DEFAULT_BEAT_BACKEND):
    """
    Step 1: Source separation (Demucs) → vocals for lyrics
    Step 2: Beat tracking (Beat This! by default) → beat grid
    Step 3: Chord recognition (autochord) → timestamped chords
    Step 4: Lyrics transcription (WhisperX) → timestamped words
    Step 5: Snap everything to beat grid → chord chart
    """

    print(f"Processing {audio_path}\n")

    print("Separating vocals with Demucs")
    vocal_path = extract_vocals(audio_path)

    print(f"Detecting beats with {beat_backend}")
    beat_grid = make_beat_grid(audio_path, sr=22050, backend=beat_backend)
    beat_times = beat_grid.beat_times
    downbeat_times = beat_grid.downbeat_times

    print("Recognizing chords...")
    chords = recognize_chords(audio_path)

    print("Transcribing lyrics...\n")
    lyrics_data = get_timestamped_lyrics(vocal_path)
    words = extract_word_timestamps(lyrics_data)

    print("Building chord chart...\n")

    if len(downbeat_times) >= 2:
        # Count beats between first two downbeats to get meter
        first_bar_beats = np.sum(
            (beat_times >= downbeat_times[0] - 0.01)
            & (beat_times < downbeat_times[1] - 0.01)
        )
        beats_per_bar = max(int(first_bar_beats), 4)
    else:
        beats_per_bar = 4

    bar_grid = build_bar_grid(beat_times, meter_numerator=beats_per_bar)

    bars = {}
    for i, (bar_num, beat_in_bar) in enumerate(bar_grid):
        if bar_num not in bars:
            bars[bar_num] = []
        chord = get_chord_at_time(beat_times[i], chords)
        nearby_words = get_words_near_time(beat_times[i], words)
        bars[bar_num].append(
            {
                "beat": beat_in_bar,
                "time": float(beat_times[i]),
                "chord": chord,
                "words": nearby_words,
            }
        )

    output = {
        "beat_tracking_backend": beat_backend,
        "tempo_bpm": beat_grid.tempo_bpm,
        "beats_per_bar": beats_per_bar,
        "bars": [],
    }
    for bar_num in sorted(bars.keys()):
        bar_beats = bars[bar_num]
        bar_chord = bar_beats[0]["chord"] if bar_beats else "N"
        bar_words = " ".join(b["words"] for b in bar_beats if b["words"]).strip()
        output["bars"].append(
            {
                "bar": bar_num,
                "chord": bar_chord,
                "lyrics": bar_words,
                "beats": bar_beats,
            }
        )

    output_stem = build_output_stem(audio_path, beat_backend)
    export_dir = "output"
    os.makedirs(export_dir, exist_ok=True)
    output_path = os.path.join(export_dir, f"{output_stem}_chart.json")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nChart saved to: {output_path}")

    print("\nGenerating MusicXML Lead Sheet...")
    xml_file = convert_json_to_musicxml(output, audio_path, output_stem=output_stem)
    _ = convert_xml_to_pdf_musescore(xml_file)

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full CoverBuddy audio-to-lead-sheet pipeline."
    )
    parser.add_argument("audio_path", help="Path to the input audio file.")
    parser.add_argument(
        "--beat-backend",
        choices=("beat_this", "librosa"),
        default=DEFAULT_BEAT_BACKEND,
        help="Beat tracking backend to use for the pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_chord_chart(args.audio_path, beat_backend=args.beat_backend)
