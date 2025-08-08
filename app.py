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


uploaded_file = st.file_uploader("Lade eine Audiodatei hoch (.wav oder .mp3)", type=["wav", "mp3"])

preset = st.selectbox ("Noise Reduction Preset" , ["Custom", "Light", "Balanced", "Strong"])

if preset != "Custom":
    prop_decrease = st.slider ("Noise Reduction Strength", min_value=0.0, max_value=1.0, value = 0.85, step = 0.05)

volume_factor = st.slider ("Output Volume", min_value=0.0, max_value=2.0, value = 1.0, step = 0.05)

if preset == "Light":
    prop_decrease = 0.4

if preset == "Balanced":
    prop_decrease = 0.7

if preset == "Light":
    prop_decrease = 1.0

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Verarbeite die Datei..."):
        # Temporäre Datei schreiben
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Audio laden
        y, sr = librosa.load(temp_path, sr=None)

        # Rauschprofil (einfach: erstes Drittel)
        noise_sample = y[0:int(0.25 * len(y))]

        # Rauschunterdrückung + Lautstärke
        reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=prop_decrease)
        reduced_noise = reduced_noise * volume_factor

        # Ausgabe speichern
        output_path = temp_path + "_clean.wav"
        sf.write(output_path, reduced_noise, sr)

        st.success("Fertig! Hier ist deine bereinigte Datei:")
        st.audio(output_path, format="audio/wav")
        with open(output_path, "rb") as f:
            st.download_button(label="⬇️ Bereinigte Datei herunterladen", data=f, file_name="autoclean_output.wav")
