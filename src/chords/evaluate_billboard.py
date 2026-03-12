import os
import json
import argparse
from chord_recognition import recognize_chords, load_lab_file, evaluate_frame_accuracy


AUDIO_DIR = "billboard_audio"
ANNOTATION_DIR = "billboard_annotations"
CACHE_FILE = "test_results/chord_eval_cache.json"

def load_cache():
    """Loads evaluation progress from disk if it exists"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {
        "song_ids": [], 
        "accuracies": [], 
        "total_accuracy": 0.0,
        "songs_processed": 0, 
    }


def save_cache(cache_data):
    """Saves current evaluation progress to disk immediately."""
    os.makedirs(os.path.dirname(CACHE_FILE) or ".", exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=4)


def run_evaluation(audio_dir, annotation_dir):
    """
    Runs chord recognition on all songs that have both audio + ground truth,
    computes frame-level accuracy for each, and generates a summary graph.
    """
    # Make sure directories exist
    if not os.path.isdir(audio_dir) or not os.path.isdir(annotation_dir):
        print("Error: audio or annotation directory not found.")
        return

    # Find all audio files
    audio_files = sorted([
        f for f in os.listdir(audio_dir)
        if f.endswith(('.wav', '.mp3'))
    ])

    if not audio_files:
        print(f"No audio files found in {audio_dir}/")
        print("Add .wav or .mp3 files and try again.")
        return

    # Load checkpoint
    cache = load_cache()

    for audio_filename in audio_files:
        song_id = os.path.splitext(audio_filename)[0]

        if song_id in cache["song_ids"]:
            print(f"\nSkipping {song_id} — already evaluated in cache.")
            continue

        lab_path = os.path.join(annotation_dir, f"{song_id}.lab")
        if not os.path.exists(lab_path):
            print(f"\nSkipping {song_id} — no .lab file found.")
            continue
 
        audio_path = os.path.join(audio_dir, audio_filename)
        print(f"\nProcessing: {song_id}")

        try:
            predicted_chords = recognize_chords(audio_path)
            ground_truth = load_lab_file(lab_path)
            accuracy = evaluate_frame_accuracy(predicted_chords, ground_truth)
            accuracy_pct = accuracy * 100

            print(f"   Frame-level accuracy: {accuracy_pct:.1f}%")

            # Update cache and save immediately
            cache["song_ids"].append(song_id)
            cache["accuracies"].append(accuracy_pct)
            cache["total_accuracy"] += accuracy
            cache["songs_processed"] += 1
            save_cache(cache)

        except Exception as e:
            print(f"   Error processing {song_id}: {e}")


    if cache["songs_processed"] == 0:
        return

    avg_accuracy = (cache["total_accuracy"] / cache["songs_processed"]) * 100
    print(f"Songs evaluated: {cache['songs_processed']}")
    print(f"Average frame-level accuracy: {avg_accuracy:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default=AUDIO_DIR)
    parser.add_argument("--annotation_dir", default=ANNOTATION_DIR)
    args = parser.parse_args()
    run_evaluation(args.audio_dir, args.annotation_dir)
