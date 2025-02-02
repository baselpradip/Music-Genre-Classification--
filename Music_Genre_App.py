import os
import librosa
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.image import resize

# Set environment to avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the pre-trained model
MODEL_PATH = "Trained_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define the music genres
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Chunk settings
    chunk_duration = 2  # seconds
    overlap_duration = 1  # seconds

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# Model prediction function
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Streamlit UI
st.title("Music Genre Classification App")
st.write("Upload an audio file to predict its genre.")

uploaded_file = st.file_uploader("Choose an audio file (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    # Save uploaded file temporarily
    file_path = "temp_audio_file.mp3"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Process and predict
    st.write("Processing audio and making prediction...")
    try:
        X_test = load_and_preprocess_data(file_path)
        c_index = model_prediction(X_test)
        st.success(f"Model Prediction: Music Genre --> {classes[c_index]}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        os.remove(file_path)
