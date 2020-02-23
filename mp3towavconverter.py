# CrimsonCode Hackathon 2020
# mp3towavconverter.py
# Date: 02/22/20 - 02/23/20
# Created by: Michael Larson, David Allen, Meghna Dutta, and Peter Wang
# Description: A function that takes a source string and converts that .mp3 file into a wav file.
# Based off of pydub documentation.
# Link: https://http://pydub.com/

# Including library pydub which allows us to edit audio
from os import path
from pydub import AudioSegment

# Creating a user defined function called mp3ToWav which accepts a source string and a destination string
# This takes the source (.mp3) and converts to destination (.wav)
def mp3ToWav(src, dst):
    # Takes the source .mp3 and stores it in sound
    # https://github.com/jiaaro/pydub (creator of pydubs github)
    sound = AudioSegment.from_mp3(src)

    # Google's API can not listen to stereo audio so we are forced to make it mono chanel
    # This is done with pydub set_channels to 1 channel
    sound = sound.set_channels(1)

    # Makes a new file with the name of the destination input. This file has the mode .wav
    sound.export(dst, format = "wav")