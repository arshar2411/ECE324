from os import path
from pydub import AudioSegment

# files                                                                         
src = "https://sf16-sg.tiktokcdn.com/obj/tiktok-obj/6893870290623810306.mp3"
dst = "test.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")