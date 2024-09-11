import streamlit as st
import numpy as np
import librosa
import os
from scipy.io.wavfile import write
import sounddevice as sd
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.ndimage import (generate_binary_structure, iterate_structure, binary_erosion)
from operator import itemgetter
from typing import List, Tuple
from collections import defaultdict

def fingerprint(audio_data: np.ndarray, sr: int, plot: bool = False) -> List[Tuple[int, int, float]]:
    """
    Generate audio fingerprints using the Shazam algorithm.
    
    :param audio_data: The audio time series
    :param sr: Sampling rate of the audio
    :param plot: Whether to plot the spectrogram and peaks
    :return: List of fingerprints (time, frequency, amplitude)
    """
    # Compute the spectrogram
    spectrogram = librosa.stft(audio_data, n_fft=2048, hop_length=512)
    spectrogram = np.abs(spectrogram)
    
    # Apply logarithmic scaling
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Find local maxima
    neighborhood = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(neighborhood, 20)
    local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
    background = (spectrogram == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    
    # Extract peak coordinates and amplitudes
    amps = spectrogram[detected_peaks]
    freqs, times = np.where(detected_peaks)
    
    # Sort peaks by amplitude and keep top 'k'
    k = int(np.sum(detected_peaks) * 0.1)  # Keep top 10% of peaks
    i = amps.argsort()[::-1][:k]
    amps = amps[i]
    freqs = freqs[i]
    times = times[i]
    
    if plot:
        fig, ax = plt.subplots()
        librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        ax.scatter(times, freqs, color='r', s=1)
        st.pyplot(fig)
    
    return list(zip(times, freqs, amps))

def create_hashes(fingerprints: List[Tuple[int, int, float]], fan_value: int = 15) -> List[Tuple[Tuple[int, int], int]]:
    """
    Create hashes from fingerprints using combinatorial hashing.
    
    :param fingerprints: List of fingerprints (time, frequency, amplitude)
    :param fan_value: Number of following peaks to consider for each anchor point
    :return: List of hashes ((freq1, freq2), time_delta)
    """
    hashes = []
    for i in range(len(fingerprints)):
        for j in range(1, min(fan_value, len(fingerprints) - i)):
            freq1 = fingerprints[i][1]
            freq2 = fingerprints[i + j][1]
            t1 = fingerprints[i][0]
            t2 = fingerprints[i + j][0]
            t_delta = t2 - t1
            if t_delta >= 0 and t_delta < 200:  # Limit time delta to avoid spurious matches
                hashes.append(((freq1, freq2), t_delta))
    return hashes

def load_data():
    """Load songs from the 'data' folder, extract fingerprints, and store them."""
    fingerprints = {}
    for filename in os.listdir('data'):
        if filename.endswith(".mp3"):
            file_path = os.path.join('data', filename)
            audio_data, sr = librosa.load(file_path, sr=None)
            song_fingerprints = fingerprint(audio_data, sr)
            song_hashes = create_hashes(song_fingerprints)
            fingerprints[filename] = song_hashes
    return fingerprints

def match_song(input_hashes: List[Tuple[Tuple[int, int], int]], database: dict) -> str:
    """
    Match input hashes against the database of songs.
    
    :param input_hashes: Hashes of the input audio
    :param database: Dictionary of song names to their hashes
    :return: Name of the matched song or "Unknown song"
    """
    matches = defaultdict(int)
    for song_name, song_hashes in database.items():
        song_hash_dict = defaultdict(list)
        for hash_value, t_delta in song_hashes:
            song_hash_dict[hash_value].append(t_delta)
        
        for input_hash, input_t_delta in input_hashes:
            if input_hash in song_hash_dict:
                for song_t_delta in song_hash_dict[input_hash]:
                    matches[song_name] += 1
    
    if matches:
        best_match = max(matches.items(), key=itemgetter(1))
        if best_match[1] > len(input_hashes) * 0.05:  # At least 5% of hashes should match
            return best_match[0]
    return "Unknown song"

# Streamlit app
st.title("Song Recognition System")

# Load all song fingerprints from data
st.write("Loading songs from 'data' folder...")
song_database = load_data()
st.write(f"Loaded {len(song_database)} songs.")

# Record a song from the user
if st.button("Record a Song"):
    duration = 10
    fs = 44100
    st.write(f"Recording for {duration} seconds...")
    recorded_audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    
    # Save the recorded audio to a file
    recorded_file = "recorded_song.wav"
    write(recorded_file, fs, recorded_audio)
    st.audio(recorded_file)
    
    # Process recorded audio
    recorded_fingerprints = fingerprint(recorded_audio.flatten(), fs, plot=True)
    recorded_hashes = create_hashes(recorded_fingerprints)
    
    # Match the recorded song
    matched_song = match_song(recorded_hashes, song_database)
    st.write(f"Matched song: {matched_song}")

# Upload a song from the user
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
if uploaded_file is not None:
    st.write("Processing uploaded audio file...")
    audio_data, sr = librosa.load(BytesIO(uploaded_file.read()), sr=None)
    
    # Process uploaded audio
    uploaded_fingerprints = fingerprint(audio_data, sr, plot=True)
    uploaded_hashes = create_hashes(uploaded_fingerprints)
    
    # Match the uploaded song
    matched_song = match_song(uploaded_hashes, song_database)
    st.write(f"Matched song: {matched_song}")