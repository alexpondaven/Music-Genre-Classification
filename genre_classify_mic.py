# This file records audio from the microphone and collects 3 second audio samples (updating ever 0.5 seconds). 
# These 3 second audio samples are passed through the CNN model to predict the genre of the music which it received.
# Every classification updates a matplotlib plot that shows the model's scores for each genre for that 3 second sample.
# once started, it continues classifying the input sound forever until the script is interrupted.

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display

import pyaudio
import time
import wave
import soundfile as sf

#load model
model = tf.keras.models.load_model("model80.h5")

# Audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.5
WAVE_OUTPUT_FILENAME = "out.wav"

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'
genres = genres.split()
 
audio = pyaudio.PyAudio()
#select input device
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

# Above code outputs the index of all input devices
# Change this parameter to the index of the input device you would like to take the input from
device_index= int(input("Index of input device (e.g. 1,2,...): "))


# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=device_index)




plt.figure(figsize=(10,5))
#loop for recording 3 second samples from microphone and classifying genre
while(1):
    # print("recording...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    # print("finished recording")

    # Put frames into wave file
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    # create new 3 second file using 2.5 seconds of old audio sample and 0.5 seconds of new data
    old_sound, rate = librosa.load("./file.wav", duration=3) # need to have a 3 second wav file in ./file.wav to update
    #print('shape of old_sound ==> ' + str(old_sound.shape))
    recorded_sound, rate = librosa.load("./out.wav", duration=RECORD_SECONDS)
    #print('shape of recorded_sound ==> ' + str(recorded_sound.shape))
    old_sound = old_sound[-int((3-RECORD_SECONDS)*rate):-1]
    new_sound = np.append(old_sound, recorded_sound)
    # extend data to make it 3 seconds
    #print("rate",rate)
    #print(new_sound[0:50])
    new_sound = np.pad(new_sound, (3*rate - len(new_sound),0))
    #print('shape of old_sound+recorded_sound ==> ' + str(new_sound.shape))
    sf.write("./file.wav", new_sound, rate)

    # do classification on file.wav
    #t, rate = librosa.load("./file.wav", duration=3) # returns sample rate and data (sig)
    audio_spect = librosa.feature.melspectrogram(new_sound, rate) # convert to melspectogram

    spect = audio_spect.reshape(1,128,130,1) / 255.0
    
    plt.clf()
    sns.barplot(x=genres,y=model.predict(spect)[0])
    plt.title("Music Genre Classification - CNN Model Prediction")
    plt.xlabel("Genre")
    plt.ylabel("Model Score (on 3 second sample)")
   
    plt.pause(0.05)

 
plt.show()
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
