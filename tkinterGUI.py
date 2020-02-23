# CrimsonCode Hackathon 2020
# tkinterGui.py
# Date: 02/22/20 - 02/23/20
# Created by: Michael Larson, David Allen, Meghna Dutta, and Peter Wang
# Description: A python file that uses tkinter to create a GUI for our application.
# Based off of a tutorial series on YouTube by freeCodeCamp.org
# Link: youtube.com/watch?v=YXPyB4XeYLA

# Importing libraries to use tkinter GUI and a library for converting mp3 to wav. The last import is for converting a long .wav
# Into .wav files that are a minute long
import tkinter as tkin
import mp3towavconverter as mp
import makeMultWav as mmw

# Making a window storing it in root
root = tkin.Tk()

myLabel = tkin.Label(root, text="Welcome to LYRIC BOX!")
myLabel.pack()

myLabel1 = tkin.Label(root, text="Enter the path to your .MP3 file below!")
myLabel1.pack()

# Making an entry variavle to get user input from window
e = tkin.Entry(root, width=50, bg="grey", fg="blue")

# Placing the user input box on the window
e.pack()

# Gets what the user entered and stores it into e
e.get()

# Defining a function that we will call when the user clicks on the button
def myClick():
    # Calls user defined function mp3ToWav which takes the user input and a destination file and creates a new wav file
    mp.mp3ToWav(e.get(),'wavFile.wav')

    # Calls a user defined function called makeWavs that crates one minute long .wav files from a larger .wav
    mmw.makeWavs('wavFile.wav')

# Creates a new button for the user to click that says "Enter Your .mp3 path" and when it is 
# clicked it calls user created function myClick
myButton = tkin.Button(root, text="Create your Video!", padx=50, pady=20, command=myClick, fg="blue", bg="grey")

# Places the button we created onto the screen
myButton.pack()

# Creating a loop so we will keep running untill something happens.
root.mainloop()