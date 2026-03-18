import os
import subprocess
from music21 import stream, meter, tempo, harmony, note, metadata
from fractions import Fraction


def convert_json_to_musicxml(chart_data, audio_path):
    """Converts the CoverBuddy JSON output into a MusicXML lead sheet."""
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    score = stream.Score()

    score.metadata = metadata.Metadata()
    score.metadata.title = song_name

    part = stream.Part()

    # 1. Set Tempo and Time Signature
    bpm = round(chart_data.get("tempo_bpm", 120))
    beats_per_bar = chart_data.get("beats_per_bar", 4)

    part.append(tempo.MetronomeMark(number=bpm))
    part.append(meter.TimeSignature(f"{beats_per_bar}/4"))

    # 2. Build the measures
    for bar_data in chart_data.get("bars", []):
        m = stream.Measure(number=bar_data["bar"])

        # Add the Chord Symbol
        chord_str = bar_data["chord"]
        if chord_str and chord_str != "N":
            try:
                # music21 natively understands standard chord strings (e.g., 'Am', 'C#')
                cs = harmony.ChordSymbol(chord_str)
                m.append(cs)
            except Exception:
                pass  # Skip unparseable chords

        # Add the lyrics
        lyrics_text = bar_data.get("lyrics", "")

        if lyrics_text and lyrics_text.strip():
            # 1. Split the sentence into a list of individual words
            words = lyrics_text.split()

            # 2. Divide the measure's time evenly among the words
            # (e.g., 4 beats / 5 words = 0.8 beats per rest)
            duration_per_word = Fraction(beats_per_bar / len(words))

            for word in words:
                r = note.Rest(quarterLength=duration_per_word)
                r.lyric = word
                r.style.hideObjectOnPrint = True

                m.append(r)
        else:
            # If measure is an instrumental break (no lyrics)
            r = note.Rest(quarterLength=beats_per_bar)
            m.append(r)

        part.append(m)

    score.append(part)

    # 3. Save the MusicXML file
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_xml_path = os.path.join(output_dir, f"{song_name}.musicxml")
    score.write("musicxml", fp=output_xml_path)
    print(f"MusicXML successfully saved to: {output_xml_path}")

    return output_xml_path


def convert_xml_to_pdf_musescore(xml_path):
    """
    Optional: Uses MuseScore's command line interface to convert the XML to PDF.
    """
    output_pdf = os.path.splitext(xml_path)[0] + ".pdf"

    # musescore_exec = "mscore"
    # for testing (Ian) -- using WSL on Windows
    musescore_exec = "/mnt/c/Program Files/MuseScore 4/bin/MuseScore4.exe"

    try:
        print(f"Converting {xml_path} to PDF...")
        subprocess.run([musescore_exec, xml_path, "-o", output_pdf], check=True)
        print(f"PDF generated: {output_pdf}")
    except FileNotFoundError:
        print("MuseScore executable not found in path. Skipping PDF conversion.")
    except subprocess.CalledProcessError as e:
        print(f"Error generating PDF via MuseScore: {e}")
