import os
import subprocess
import torch
import whisperx
import json
from pathlib import Path


def extract_vocals(input_audio_path: str, output_dir: str = "src/lyrics") -> str:
    """
    Runs Demucs to separate vocals from the instrumental track.
    Returns the file path to the extracted vocals.wav.
    """
    song_name = Path(input_audio_path).stem

    vocal_path = os.path.join(output_dir, "htdemucs", song_name, "vocals.wav")
    print(f"   Checking for vocals at: {vocal_path}")

    if os.path.exists(vocal_path):
        print(f"   Vocals already exist at {vocal_path}. Skipping Demucs.")
        return vocal_path

    print(f"Extracting vocals from {input_audio_path} using Demucs...")

    command = [
        "python3",
        "-m",
        "demucs",
        "-n",
        "htdemucs",
        "--two-stems=vocals",
        "--out",
        output_dir,
        input_audio_path,
    ]

    subprocess.run(command, check=True)

    vocal_path = os.path.join(output_dir, "htdemucs", song_name, "vocals.wav")

    if not os.path.exists(vocal_path):
        raise FileNotFoundError(f"Demucs failed to create vocal file at {vocal_path}")

    print(f"Vocals extracted to: {vocal_path}")
    return vocal_path


def get_timestamped_lyrics(
    vocal_audio_path: str, model: str = "base", name: str = "whisperx_output.json"
) -> dict:
    """
    Runs WhisperX on the vocal track to get word-level timestamps.
    Saves the output to a JSON file to skip processing on future runs.
    """
    # Define where we want to save the JSON file (right next to the vocals.wav)
    output_dir = os.path.dirname(vocal_audio_path)
    json_path = os.path.join(output_dir, name)

    # 1. Check if we already transcribed this file
    if os.path.exists(json_path):
        print(f"   Transcription already exists at {json_path}. Skipping WhisperX.")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. If no JSON exists, run the WhisperX pipeline
    print("   Loading WhisperX for transcription and alignment...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = whisperx.load_model(model, device, compute_type=compute_type)
    audio = whisperx.load_audio(vocal_audio_path)

    # Force language="en" to prevent hallucinations on noisy intro audio
    result = model.transcribe(audio, batch_size=8, language="en")

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    print("   Transcription and alignment complete!")

    # 3. Save the result to a JSON file for next time
    print(f"   Saving WhisperX output to {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, indent=4)

    return aligned_result


if __name__ == "__main__":
    sample_song = "jamendo_audio/song_0.wav"

    try:
        # Step 1: Isolate the vocals
        isolated_vocal_path = extract_vocals(sample_song)

        # Step 2: Get the exact timestamps
        lyrics_data = get_timestamped_lyrics(isolated_vocal_path)

        # Print the first few words and their timestamps to verify
        print("\n--- Output Preview ---")
        for segment in lyrics_data["segments"][
            :2
        ]:  # Just looking at the first 2 segments
            for word in segment.get("words", []):
                print(
                    f"Word: '{word.get('word')}' | Start: {word.get('start')}s | End: {word.get('end')}s"
                )

    except Exception as e:
        print(f"An error occurred: {e}")
