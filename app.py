import streamlit as st
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import tempfile

st.title("🎧 AutoClean Audio – Rauschfreie Audioaufnahmen")

with st.expander("ℹ️ Wie funktioniert AutoClean Audio?"):
    st.markdown("""
**AutoClean Audio** nutzt ein statistisches Modell zur Rauschunterdrückung.  
Es funktioniert so:

1. Aus dem **Anfang deiner Aufnahme** wird automatisch ein kurzes **Rauschprofil** erstellt.
2. Dieses Rauschprofil wird verwendet, um Hintergrundgeräusche im restlichen Signal zu erkennen.
3. Der Algorithmus **subtrahiert diese Störungen** aus der Aufnahme – so bleibt deine Stimme klarer hörbar.

📌 **Wichtig**: Wenn deine Sprache sehr leise oder das Rauschen sehr ähnlich zur Stimme ist, kann es zu Beeinträchtigungen kommen.  
In einem späteren Update bieten wir zusätzliche **KI-gestützte Filtermethoden** für noch bessere Qualität.

Gruß
Luca
""")


uploaded_file = st.file_uploader(
    "Lade eine Audiodatei hoch", 
    type=["wav", "mp3", "flac", "ogg", "acc", "opus", "wma", "aiff", "m4a", "amr", "speex"]
)

preset = st.selectbox ("Noise Reduction Preset" , ["Custom", "(0.4) Light", "(0.7) Balanced", "(1.0) Strong"])

if preset == "Custom":
    prop_decrease = st.slider ("Noise Reduction Strength", min_value=0.0, max_value=1.0, value = 0.85, step = 0.05)

elif preset == "(0.4) Light":
    prop_decrease = 0.4

elif preset == "(0.7) Balanced":
    prop_decrease = 0.7

elif preset == "(1.0) Strong":
    prop_decrease = 1.0

volume_factor = st.slider ("Output Volume", min_value=0.0, max_value=2.0, value = 1.0, step = 0.05)

number_of_passes = st.slider("Durchlauf Anzahl", min_value=1, max_value=5, value = 1, step = 1)

if number_of_passes > 1:
    st.warning("Mehrere Durchläufe können Restrauschen weiter reduzieren, beeinträchtigen aber auch Sprachqualität und klangfarbe.")



if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    default_filename = uploaded_file.name
    for s in [".wav", ".mp3", ".flac", ".ogg", ".acc", ".opus", ".wma", ".aiff", ".m4a", ".amr", ".speex"]:
        if default_filename.endswith(s):
            default_filename = default_filename.removesuffix(s) + "_cleaned"
            break
    output_filename = st.text_input("Dateiname für die bereinigte Datei (ohne Endung):", value=default_filename)

    output_format = st.selectbox(
        "Ausgabeformat",
        ["wav", "mp3", "flac", "ogg", "aiff"]
    )

    if st.button("Audio verarbeiten"):

        with st.spinner("Verarbeite die Datei..."):
            # Temporäre Datei schreiben
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            # Audio laden
            y, sr = librosa.load(temp_path, sr=None)

            # Rauschprofil (einfach: erstes Drittel)
            noise_sample = y[0:int(0.33 * len(y))]

            # Rauschunterdrückung
            reduced = y.copy()
            progress_bar = st.progress(0)
            for i in range(number_of_passes):
                reduced = nr.reduce_noise(y=reduced, sr=sr, y_noise=noise_sample, prop_decrease=prop_decrease)
                progress_bar.progress(int((i+1) / number_of_passes * 100))

            reduced = reduced * volume_factor

            # Ausgabe speichern
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}").name
            sf.write(output_path, reduced, sr, format=output_format.upper())

            st.success("Fertig! Hier ist deine bereinigte Datei:")
            st.audio(output_path, format="audio/{output_format}")

            

            with open(output_path, "rb") as f:
                st.download_button(
                    label="\:floppy_disk: Bereinigte Datei herunterladen",
                    data=f,
                    file_name=f"{output_filename}.{output_format}")