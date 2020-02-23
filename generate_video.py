# CrimsonCode Hackathon 2020
# generate_video.py
# Date: 02/22/20 - 02/23/20
# Created by: Michael Larson, David Allen, Meghna Dutta, and Peter Wang
# Description: A .py file that takes the words that are given to it and puts it into a video.
# Link: https://http://pydub.com/

import moviepy.editor as mp
import numpy as np

##################################################
# Generates an empty solid color background video
# Takes 2 arguments: Tuple aspect ratio, rgb, duration 
def generate_blank_video(ar, rgb, dur):
    return mp.ColorClip(size=ar,color=rgb,duration=dur)

##################################################
# Generates a text still                         
# Takes 3 arguments: text, font-size, duration   
def generate_text_clip(text, ftsize, dur):
    txt_clip = mp.TextClip(text,font='Amiri-Bold',fontsize=ftsize,color='white',
                           stroke_color='black')
    txt_clip = txt_clip.set_pos('center').set_duration(dur)
    return txt_clip

###################################################
# Returns an array of text stills from list of text
# Takes 1 argument: array of strings
def generate_text_clips(text_list, times=[]):
    clip_list = []
    for c, text in enumerate(text_list):
        txt_clip = mp.TextClip(text,font='Amiri-Bold',fontsize=45,color='white',stroke_color='black')
        if (times==[]):
            dur = 1
        else:
            try:
                dur = np.ceil(float(times[c+1]) - float(times[c]))
            except IndexError:
                dur = 1
            
        txt_clip = txt_clip.set_pos('center').set_duration(dur)
        rgb = random_rgb()
        clip = generate_blank_video((960,540), rgb, 1)
        vid = mp.CompositeVideoClip([clip, txt_clip])
        clip_list.append(vid)
    return clip_list

def write_vid(video, title):
    video.write_videofile(title+".mp4", fps=25)

def combine_clips(clip_list):    
    video = mp.concatenate(clip_list, method="compose")
    return video

def add_sound(local_sound_path, video):
    audioclip = mp.AudioFileClip(local_sound_path)
    video = video.set_audio(audioclip)
    return video

def random_rgb():
    r = np.random.randint(0,255)
    g = np.random.randint(0,255)
    b = np.random.randint(0,255)
    return [r,g,b]

if __name__ == "__main__":
    # Random color
    rgb = random_rgb()
    audioclip = mp.AudioFileClip("Recording3.wav")
    clip_list = []
    start_clip = generate_blank_video((960,540), rgb, 2)
    clip_list.append(start_clip)
    # Insert array of words for the song
    clip_list.extend(generate_text_clips(["Peter's","New","Song"]))
    clip_list.append(start_clip)
    video = mp.concatenate(clip_list, method="compose")
    video = video.set_audio(audioclip)
    video.write_videofile("lyric_video.mp4", fps=25)
    print ("Mike is very hot")
