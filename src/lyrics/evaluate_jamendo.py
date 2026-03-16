import os
import soundfile as sf
import jiwer
import string
import json
from datasets import load_dataset
import matplotlib.pyplot as plt

# Import your existing pipeline functions
from pipeline import extract_vocals, get_timestamped_lyrics

CACHE_FILE = "src/lyrics/test_results/evaluation_cache.json"


def clean_text(text: str) -> str:
    """Removes punctuation and lowercases text for a fair WER comparison."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def load_cache():
    """Loads the progress cache if it exists."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {
        "song_labels": [],
        "wer_percentages": [],
        "total_wer": 0,
        "songs_processed": 0,
    }


def save_cache(cache_data):
    """Saves the current progress to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)


def test_whisperx_with_jamendo():
    print("Downloading Jamendo dataset from Hugging Face...")

    dataset = load_dataset("jamendolyrics/jamendolyrics", "en", split="test")

    os.makedirs("src/lyrics/jamendo_audio", exist_ok=True)
    os.makedirs("src/lyrics/jamendo_truth", exist_ok=True)

    test_subset = dataset

    # Load our checkpoint data
    cache_data = load_cache()

    for i, song in enumerate(test_subset):
        song_id = song.get("track_id", f"song_{i}")

        # --- THE CHECKPOINT CHECK ---
        if song_id in cache_data["song_labels"]:
            print(f"\nSkipping {song_id} - Already evaluated in cache.")
            continue

        print(f"\nProcessing: {song_id}")

        # --- Step 1: Save Audio to Disk ---
        wav_path = os.path.join("src/lyrics/jamendo_audio", f"{song_id}.wav")

        if not os.path.exists(wav_path):
            print("   Saving audio array to disk...")
            audio_data = song["audio"]["array"]
            sr = song["audio"]["sampling_rate"]

            sf.write(wav_path, audio_data, sr)
        else:
            print(f"   Audio file {wav_path} already exists. Skipping write.")

        # --- Step 2: Extract Ground Truth Lyrics ---
        truth_path = os.path.join("src/lyrics/jamendo_truth", f"{song_id}.txt")

        if not os.path.exists(truth_path):
            print("   Saving ground truth lyrics to disk...")
            words = song["words"]
            if isinstance(words, dict) and "text" in words:
                truth_text = " ".join(words["text"])
            else:
                truth_text = " ".join([w["text"] for w in words])

            with open(truth_path, "w", encoding="utf-8") as f:
                f.write(truth_text)
        else:
            print(f"   Truth file {truth_path} already exists. Skipping write.")
            with open(truth_path, "r", encoding="utf-8") as f:
                truth_text = f.read()

        clean_truth = clean_text(truth_text)

        # --- Step 3: Run the Pipeline ---
        try:
            print("   Isolating vocals...")
            vocal_path = extract_vocals(wav_path)

            print("   Running WhisperX...")
            whisper_output = get_timestamped_lyrics(
                vocal_path, model="large-v2", name=f"{song_id}.json"
            )

            predicted_text = " ".join(
                [seg["text"] for seg in whisper_output.get("segments", [])]
            )
            clean_pred = clean_text(predicted_text)

            # --- Step 4: Evaluate ---
            wer_score = jiwer.wer(clean_truth, clean_pred)

            # Update our cache variables
            cache_data["song_labels"].append(song_id)
            cache_data["wer_percentages"].append(wer_score * 100)
            cache_data["total_wer"] += wer_score
            cache_data["songs_processed"] += 1

            # Save progress to the hard drive immediately!
            save_cache(cache_data)

            print(f"WER: {wer_score:.2%}")
            print(f"   Truth: '{clean_truth[:60]}...'")
            print(f"   Pred:  '{clean_pred[:60]}...'")

        except Exception as e:
            print(f"Error processing {song_id}: {e}")

    # --- Step 5: Generate the Graph ---
    if cache_data["songs_processed"] > 0:
        avg_wer = (cache_data["total_wer"] / cache_data["songs_processed"]) * 100
        print(f"\nAverage WER for Jamendo Test: {avg_wer:.2f}%")

        print("\nGenerating Performance Graph...")
        plt.figure(figsize=(14, 6))

        bars = plt.bar(
            cache_data["song_labels"],
            cache_data["wer_percentages"],
            color="#4C72B0",
            edgecolor="black",
        )

        plt.axhline(
            y=avg_wer,
            color="#C44E52",
            linestyle="--",
            linewidth=2,
            label=f"Average WER: {avg_wer:.1f}%",
        )

        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 1,
                f"{yval:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
            )

        plt.ylabel("Word Error Rate (%)", fontsize=12)
        plt.xlabel("Song ID", fontsize=12)
        plt.title(
            "WhisperX Transcription Accuracy on Jamendo Dataset (Lower is Better)",
            fontsize=14,
            pad=15,
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, max(max(cache_data["wer_percentages"]) + 15, 100))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend()

        graph_filename = "whisperx_evaluation_graph.png"
        plt.savefig(graph_filename, dpi=300, bbox_inches="tight")
        print(f"Graph successfully saved to your project folder as: {graph_filename}")


if __name__ == "__main__":
    test_whisperx_with_jamendo()
