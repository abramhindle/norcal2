import sys
import cv2
import numpy as np
from numpy import *
import random
import time
import pickle
import os.path
import scipy.io
import scipy.io.wavfile

# run me to generate frames.pkl and audio.pkl
videoname = "vtest.avi"

if len(sys.argv) >= 2:
    videoname = sys.argv[1]

print "Opening %s" % videoname
cap = cv2.VideoCapture(videoname)

running = True

frames = []


SW,SH = (64,64)
old_half_frames = 4
old_frames = [np.zeros( (SW/2) * (SH/2) ).astype(np.float32) for i in range(0,old_half_frames)]

running = True
count = 0
while(running):
    ret, frame = cap.read()
    if (not ret):
        running = False
        continue
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(grey, (SW,SH))
    scaled = scaled.astype(np.float32)
    scaled /= 255.0
    frames.append( np.concatenate(old_frames + [scaled.flatten()]) )
    smaller = cv2.resize(grey, (SW/2,SH/2))
    smaller = smaller.astype(np.float32)
    smaller /= 255.0
    del old_frames[0]
    old_frames.append(smaller.flatten())
    count += 1
        
pickle.dump(np.array(frames), file('frames.pkl','wb'))

# let's do a nasty trick
if not os.path.isfile("vtest.wav"):
    os.system("avconv -i vtest.avi  -acodec pcm_s16le -ar 30720 -ac 1  vtest.wav")
    #os.system("avconv -i vtest.avi  -acodec pcm_s16le -ar 22050 -ac 1  vtest.wav")


samples = 1024
window_size = 2048
swin_size = window_size / 2 + 1
# hamming chosen so can invert later!
windowed = scipy.hamming(window_size)

wav = scipy.io.wavfile.read("vtest.wav")
wavdata = wav[1].astype(np.float32)
norm = (wavdata)/(2.0**15)
# pad norm with zeros
nsamples = int(math.ceil(len(norm)/float(samples)))
norm.resize(samples*nsamples)
# the +1 is because there's no good relationship between samples and
# window_size it'll just add a buncha zeros anyways
norm.resize((window_size+1)*math.ceil(len(norm)/float(window_size)))
# we're dumping phase 
# ffts = np.array([scipy.real(scipy.fft(norm[i*samples:i*samples+window_size] * windowed))[0:swin_size] for i in range(0,nsamples)])
# changed this to magnitude!
ffts = np.array([np.absolute(scipy.fft(norm[i*samples:i*samples+window_size]))[0:swin_size] for i in range(0,nsamples)])

# get all phase
# ffts = np.array([np.angle(scipy.fft(norm[i*samples:i*samples+window_size]))[0:swin_size] for i in range(0,nsamples)])


# keep 0s 0
ffts = ffts/max(fabs(ffts.max()),fabs(ffts.min()))
pickle.dump(ffts, file('stft.pkl','wb'))
