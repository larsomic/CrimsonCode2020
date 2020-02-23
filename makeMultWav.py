# CrimsonCode Hackathon 2020
# makeMultWav.py
# Date: 02/22/20 - 02/23/20
# Created by: Michael Larson, David Allen, Meghna Dutta, and Peter Wang
# Description: A function that divides the long .wav file into smaller files so that they can be sent to Google's API
# Based off of pydub and context documentation.
# Link: https://http://pydub.com/
# Link: https://docs.python.org/3/library/contextlib.html

# including the libraries we need to manipulate audio files
import wave
import contextlib
from pydub import AudioSegment
import generate_video as gv
import speech_recognition as sr

# Defining a user function for splitting the .wav into multiple .wav files to send to google
def makeWavs(fname):
    # Taken from contextlib docs
    with contextlib.closing(wave.open(fname,'r')) as f:
        # Creating list variables
        words = []
        timestamps = []

        # Retreiving data from contextlib
        frames = f.getnframes()        
        rate = f.getframerate()

        # Calculating the duration using frame and frame rate
        duration = frames / float(rate)

        # Finding the length of video in minutes
        mins = (duration / 60)

        n = 0

        # Declaring end, so if video is less than a minute it will have an end because it doesn't enter loop
        end = (n * 1000 * 60) + 60 * 1000

        # Loops for the number of minutes
        while n < mins -1:
            # Declaring start variable to be at n minute marker
            start = n * 1000 * 60 + 1
            # Declaring end variable as the last frame of the minute
            end = (n * 1000 * 60) + 60 * 1000

            # Stores the audio file into a var called audio
            audio = AudioSegment.from_wav(fname)

            # Cuts the audio from start var to end var
            audio = audio[start:end]

            # Exports the shortened .wav and concatanates so naming makes sense for looping
            audio.export('wavFile' + str(n) +'.wav', format="wav")

            #incraments inside loop
            n +=1
        # Gets new audio for the remaining time after the whole minute clips are gone
        audio = AudioSegment.from_wav(fname)

        # Cuts the code from the final minute to the final second
        audio = audio[end:end+((mins-n)*60)*1000]

        # Exports the final seconds of the .wav
        audio.export('wavFile' + str(n) +'.wav', format="wav")
        i = 0

        # Looping for the ammount of minutes
        while i <= n:
            # Storing the location in a str var
            strin = r'C:\Users\mike2\OneDrive\Desktop\CrimsonCode\wavFile'+str(i)+'.wav'

            # Passing the .wav into our function that interacts with Google API
            l = sr.sample_recognize(strin)

            # Adds data to the word list
            words.extend(l[0])

            # Adjusts time stamps for more minute files
            l[1] = [float(x) + i*60 for x in l[1]]

            # Adds timestamp to the time list
            timestamps.extend(l[1])

            # Incraments counter
            i+=1
        # Prints words and time stamp
        print(words, timestamps)

        # Creates a new list  
        clip_list = []  

        # Generating random color
        rgb = gv.random_rgb()

        # Starts a new blank video
        start_clip = gv.generate_blank_video((960,540), rgb, timestamps[0])

        # Adds the clip to the clip_list
        clip_list.append(start_clip)

        # Adds new clip to list
        clip_list.extend(gv.generate_text_clips(words, timestamps))

        # Prints clip list
        print(clip_list)
        video = gv.combine_clips(clip_list)
        video = gv.add_sound('wavFile.wav', video)
        gv.write_vid(video, "Lyric_Video")
        
