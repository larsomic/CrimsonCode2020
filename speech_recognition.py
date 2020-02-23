# CrimsonCode Hackathon 2020
# speech_recognition.py
# Date: 02/22/20 - 02/23/20
# Created by: Michael Larson, David Allen, Meghna Dutta, and Peter Wang
# Description: A function that takes a .wav file and sends it to Google API for audio processing.
# This returns string of words that it found and we also used it to get the times of when teh words appeared.
# Based off of Google documentation.
# Link: https://cloud.google.com/speech-to-text/docs

# INcluding all libraries that Google needs for their API to run
from google.cloud import speech_v1p1beta1
from google.cloud.speech_v1p1beta1 import enums
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums

import io

# Defining a user defined called sample_recognize. 
# This function passes a .wav file (infile) to Google and gives all the data it gets to a data base.
def sample_recognize(inString):

    timestamps = []
    words = []
    client = speech_v1.SpeechClient()
    enable_word_time_offsets = True

    # Setting language code to US because we only set it up for english songs.
    language_code = "en-US"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 44100

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "enable_word_time_offsets": enable_word_time_offsets,
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
    }
    with io.open(inString, "rb") as f:
        content = f.read()
    audio = {"content": content}

    response = client.recognize(config, audio)
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(alternative.transcript)
        for word in alternative.words:
            temp = word.start_time.nanos
            time = word.start_time.seconds
            timestamp = str(time)+ '.' + str(temp)
            print(timestamp)
            words.append(word.word)
            timestamps.append(timestamp)
    return [words, timestamps]

if __name__ == "__main__":
    sample_recognize(r'C:\Users\mike2\OneDrive\Desktop\CrimsonCode\Recording10.wav')
