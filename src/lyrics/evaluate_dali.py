import os
import string
import jiwer
import DALI as dali_code

# Import your existing pipeline functions
from pipeline import extract_vocals, get_timestamped_lyrics


def clean_text(text: str) -> str:
    """Removes punctuation and lowercases text for a fair WER comparison."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def extract_dali_ground_truth(entry) -> str:
    """Extracts a continuous string of lyrics from a DALI entry."""
    # Ensure annotations are in the horizontal format
    if entry.annotations["type"] == "vertical":
        entry.vertical2horizontal()

    # Access the words level of the horizontal format
    words_annot = entry.annotations["annot"]["words"]

    # Extract the 'text' key from each word dictionary and join them
    text_list = [word_dict["text"] for word_dict in words_annot]
    raw_truth = " ".join(text_list)

    return clean_text(raw_truth)


def extract_whisper_text(whisper_data: dict) -> str:
    """Extracts a continuous string of lyrics from WhisperX output."""
    full_text = " ".join(
        [segment["text"] for segment in whisper_data.get("segments", [])]
    )
    return clean_text(full_text)


def run_evaluation(dali_data_path: str, audio_folder_path: str):
    """Runs the full evaluation pipeline on the DALI dataset."""
    print("📚 Loading DALI Dataset...")
    # Load the DALI dataset
    dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])

    total_wer = 0
    songs_processed = 0

    # Iterate through the dataset
    for dali_id, entry in dali_data.items():
        # DALI stores the audio path in the info dictionary
        # You will need to ensure this path points to your downloaded audio files
        song_audio_path = os.path.join(audio_folder_path, f"{dali_id}.wav")

        if not os.path.exists(song_audio_path):
            print(f"⚠️ Audio for {dali_id} not found. Skipping.")
            continue

        print(f"\n🎧 Processing: {entry.info['artist']} - {entry.info['title']}")

        try:
            # 1. Get DALI Ground Truth
            ground_truth = extract_dali_ground_truth(entry)

            # 2. Run your Demucs + WhisperX pipeline
            vocal_path = extract_vocals(song_audio_path)
            whisper_output = get_timestamped_lyrics(vocal_path)
            predicted_text = extract_whisper_text(whisper_output)

            # 3. Calculate Error Rate
            wer_score = jiwer.wer(ground_truth, predicted_text)
            total_wer += wer_score
            songs_processed += 1

            print(f"📊 Word Error Rate for this song: {wer_score:.2%}")

        except Exception as e:
            print(f"❌ Failed to process {dali_id}: {e}")

    # Final Results
    if songs_processed > 0:
        average_wer = total_wer / songs_processed
        print(f"\n🏆 === FINAL EVALUATION === 🏆")
        print(f"Total Songs Processed: {songs_processed}")
        print(f"Average Word Error Rate (WER): {average_wer:.2%}")


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.getcwd())

    # Builds the absolute paths
    DALI_DATA_PATH = os.path.join(BASE_DIR, "src/lyrics/dali_data")
    AUDIO_FOLDER_PATH = os.path.join(BASE_DIR, "src/lyrics/dali_audio")

    run_evaluation(DALI_DATA_PATH, AUDIO_FOLDER_PATH)
