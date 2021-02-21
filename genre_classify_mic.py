# This file records audio from the microphone and collects data every 3 seconds. 
# These 3 second audio samples are passed through the CNN model to predict the genre of the music which it received.
# Every classification updates a matplotlib plot that shows the model's scores for each genre for that 3 second sample.
# once started, it continues classifying the input sound forever until the script is interrupted.

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

import librosa
import librosa.display

import pyaudio
import time
import wave

#load model
model = tf.keras.models.load_model("model80.h5")

# Audio
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

plt.figure(figsize=(10,5))
#loop for recording 3 second samples from microphone and classifying genre
while(1):
    # print("recording...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    # print("finished recording")

    # classify frames
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    t, rate = librosa.load("./file.wav", duration=3) # returns sample rate and data (sig)
    audio_spect = librosa.feature.melspectrogram(t, rate) # convert to melspectogram

    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'
    genres = genres.split()

    spect = audio_spect.reshape(1,128,130,1) / 255.0
    
    plt.clf()
    sns.barplot(x=genres,y=model.predict(spect)[0])
    plt.title("Model Prediction");
   
    plt.pause(0.05)

 
plt.show()
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
