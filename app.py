import streamlit as st
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import tempfile
from scipy.signal import butter, lfilter

def highpass_filter(data, sr, cutoff=80):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(1, norm_cutoff, btype="high", analog=False)
    return lfilter(b, a, data)

def apply_eq(data, sr):
    # leichte Höhenanhebung zwischen 3kHz und 6kHz
    fft_data = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1/sr)
    boost_band = (freqs >= 3000) & (freqs <= 6000)
    fft_data[boost_band] *= 1.2
    return np.fft.irfft(fft_data)

def normalize_audio(data, target_dBFS=-14):
    rms = np.sqrt(np.mean(data**2))
    if rms > 0:
        scalar = 10**(target_dBFS / 20) / rms
        return data * scalar
    return data

def process_audio(y, sr, prop_decrease, num_of_passes, volume_factor):
    # Stereo-Unterstützung: jeden Kanal einzeln verarbeiten
    if y.ndim == 1:  # Mono
        y = [y]
    else:
        y = [y[0], y[1]]

    processed_channels = []
    for channel in y:
        # Rauschreduzierung
        reduced = channel.copy()
        progress_bar = st.progress(0)
        for i in range(num_of_passes):
            reduced = nr.reduce_noise(y=reduced, sr=sr, y_noise=noise_sample, prop_decrease=prop_decrease)
            progress_bar.progress(int((i+1) / number_of_passes * 100))

        reduced = reduced * volume_factor


        # Sprachverbesserung
        reduced = highpass_filter(reduced, sr)
        reduced = apply_eq(reduced, sr)
        reduced = normalize_audio(reduced)

        max_amp = np.max(np.abs(reduced))
        if max_amp > 1.0:
            reduced = reduced / max_amp * 0.98

        processed_channels.append(reduced)

    # Wenn Stereo, wieder zusammenfügen
    if len(processed_channels) == 2:
        return np.vstack(processed_channels)
    else:
        return processed_channels[0]



# Title of Web-App
st.title("🎧 AutoClean Audio – Rauschunterdrückung für Audioaufnahmen")

# expander to get description
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




# uploader
uploaded_file = st.file_uploader(
    "Lade eine Audiodatei hoch", 
    type=["wav", "mp3", "flac", "ogg", "acc", "opus", "wma", "aiff", "m4a", "amr", "speex"]
)

# drop down to select noise reduction strength
preset = st.selectbox ("Noise Reduction Preset" , ["(1.0) Strong", "(0.7) Balanced", "(0.4) Light", "Custom"])

if preset == "Custom":
    prop_decrease = st.slider (
        "Noise Reduction Strength", min_value=0.0, max_value=1.0, value = 0.85, step = 0.05
    ) # to create slider, if custom is picked

elif preset == "(0.4) Light":
    prop_decrease = 0.4

elif preset == "(0.7) Balanced":
    prop_decrease = 0.7

elif preset == "(1.0) Strong":
    prop_decrease = 1.0

volume_factor = st.slider ("Output Volume", min_value=0.0, max_value=2.0, value = 1.0, step = 0.05)

number_of_passes = st.slider("Durchlauf Anzahl (empfohlen: 1)", min_value=1, max_value=5, value = 1, step = 1)

# warning for higher number_of_passes
if number_of_passes > 1:
    st.warning("Mehrere Durchläufe können Restrauschen weiter reduzieren, beeinträchtigen aber auch Sprachqualität und klangfarbe.")

# uploaded file -> .wav   and set output_filename
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
    # button to start cleaning
    if st.button("Audio verarbeiten"):

        with st.spinner("Verarbeite die Datei..."):
            # write temp data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            # load audio
            y, sr = librosa.load(temp_path, sr=None)

            # nise-profile first 1/3
            noise_sample = y[0:int(0.33 * len(y))]

            # noise reduction with number_of_passes
            #reduced = y.copy()
            #progress_bar = st.progress(0)
            #for i in range(number_of_passes):
            #    reduced = nr.reduce_noise(y=reduced, sr=sr, y_noise=noise_sample, prop_decrease=prop_decrease)
            #    progress_bar.progress(int((i+1) / number_of_passes * 100))

            #reduced = reduced * volume_factor

            reduced = process_audio(y, sr, prop_decrease, number_of_passes, volume_factor)

            # safe output
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}").name
            sf.write(output_path, reduced, sr, format=output_format.upper())

            st.success("Fertig! Hier ist deine bereinigte Datei:")
            st.audio(output_path, format="audio/{output_format}")

            # download
            with open(output_path, "rb") as f:
                st.download_button(
                    label="\:floppy_disk: Bereinigte Datei herunterladen",
                    data=f,
                    file_name=f"{output_filename}.{output_format}")