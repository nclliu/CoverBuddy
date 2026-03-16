import subprocess
import whisperx
import os


def separate_vocals(input_path):
    # Runs Demucs via system call to extract vocals
    # Returns path to the vocals.wav file
    pass


def transcribe_with_alignment(vocal_path):
    # Loads WhisperX model
    # Processes the vocal_path
    # Returns a JSON-like object with word-level timestamps
    pass


def run_pipeline(song_path):
    vocal_file = separate_vocals(song_path)
    lyrics_data = transcribe_with_alignment(vocal_file)
    return lyrics_data
