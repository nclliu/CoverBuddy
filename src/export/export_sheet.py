import os
import subprocess
import shutil
from music21 import stream, meter, tempo, harmony, note, metadata


def convert_json_to_musicxml(chart_data, audio_path, output_stem=None):
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

        # Attach lyrics at the beat level so multiple words that were already
        # snapped onto the same beat stay grouped together in one lyric event.
        beat_entries = bar_data.get("beats", [])
        if beat_entries:
            for beat_data in beat_entries:
                r = note.Rest(quarterLength=1)
                beat_words = beat_data.get("words", "").strip()
                if beat_words:
                    r.lyric = beat_words
                r.style.hideObjectOnPrint = True
                m.append(r)
        else:
            r = note.Rest(quarterLength=beats_per_bar)
            lyrics_text = bar_data.get("lyrics", "").strip()
            if lyrics_text:
                r.lyric = lyrics_text
            r.style.hideObjectOnPrint = True
            m.append(r)

        part.append(m)

    score.append(part)

    # 3. Save the MusicXML file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_name = output_stem or song_name
    output_xml_path = os.path.join(output_dir, f"{output_name}.musicxml")
    score.write("musicxml", fp=output_xml_path)
    print(f"MusicXML successfully saved to: {output_xml_path}")

    return output_xml_path


def convert_xml_to_pdf_musescore(xml_path):
    """
    Optional: Uses MuseScore's command line interface to convert the XML to PDF.
    """
    output_pdf = os.path.splitext(xml_path)[0] + ".pdf"

    candidates = [
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/Applications/MuseScore Studio.app/Contents/MacOS/mscore",
        shutil.which("mscore"),
        shutil.which("MuseScore"),
    ]
    musescore_exec = next((path for path in candidates if path and os.path.exists(path)), None)

    try:
        if musescore_exec is None:
            raise FileNotFoundError
        print(f"Converting {xml_path} to PDF...")
        subprocess.run([musescore_exec, xml_path, "-o", output_pdf], check=True)
        print(f"PDF generated: {output_pdf}")
    except FileNotFoundError:
        print("MuseScore executable not found in path. Skipping PDF conversion.")
    except subprocess.CalledProcessError as e:
        print(f"Error generating PDF via MuseScore: {e}")
