import sys
import cv2
import numpy as np
from numpy import *
import random
import time
import pickle
import os.path
import scipy.io
import theanets
import pickle
import numpy as np
#import sounddevice as sd
import random
#sd.default.samplerate = 44100
#sd.default.channels = 1
#from matplotlib.pyplot import plot, show, imshow

import liblo
import argparse

parser = argparse.ArgumentParser(description='Do deep learning and pump into osc')
parser.add_argument('-c',default=0, help='Camera')
parser.add_argument('-osc1', dest='osc1', default=7770,help="OSC Port")
parser.add_argument('-osc2', dest='osc2', default=7771,help="OSC Port")
parser.add_argument('-a', dest='a', default=1.0,help="A/phi multiplier")
parser.add_argument('-b', dest='b', default=0.0,help="b offset")
parser.add_argument('-z', dest='z', default=0,help="b offset")
args = parser.parse_args()

cap = cv2.VideoCapture(int(args.c))

# if len(sys.argv) < 2:
#     print "Opening vtest.avi"
#     cap = cv2.VideoCapture("vtest.avi")
# else:
#     print "Opening %s" % sys.argv[1]
#     cap = cv2.VideoCapture(sys.argv[1])
# 
running = True

frames = []

# load brain
# cv2.namedWindow("frame", 1)
brain = theanets.feedforward.Regressor.load("stft-theanet.py.net.pkl.cpu")#brain-1438666035")
#brain = theanets.feedforward.Regressor.load("brain-1438666035")
brain._graphs = {} 
brain._functions = {}
osr = 30720

ret, frame = cap.read()

target1 = liblo.Address(args.osc1)
target2 = liblo.Address(args.osc2)

def sendOSC(path,*args):
    liblo.send(target1, path, *args)
    liblo.send(target1, path, *args)



    
def gaussian_noise(inarr,mean=0.0,scale=1.0):    
    noise = np.random.normal(mean,scale,inarr.shape)
    return inarr + noise.reshape(inarr.shape)

outs = []
window_size = 2048
#windowed = scipy.hanning(window_size)
windowed = scipy.hamming(window_size)
swin_size = window_size / 2 + 1
alen = 1024 # audio length
window = np.hanning(alen)
frames = 0
overlapsize = window_size - alen
overlap = np.zeros(overlapsize)

# crazy phase stuff

flat_window = np.ones(window_size)
olaps = int(math.ceil((window_size - alen))) # half
flat_window[0:olaps] = np.arange(0,olaps)
flat_window[olaps:window_size] = np.arange(0,olaps)[::-1]
flat_window /= float(olaps-1)

# debug
# outwav.write_frames(windowed)

last_phase = np.zeros(window_size)
invwindow = 1.0/scipy.hamming(window_size)
amax=0.5


cones = np.zeros(swin_size-1).astype(complex) + complex(0,1)
oldout = np.zeros(swin_size)

phase = np.zeros(window_size) #np.random.normal(0,np.pi/2, window_size)
# phase       = np.random.normal(np.pi/2,np.pi,window_size)
# staticphase = np.random.normal(0,np.pi/2.0,window_size)
staticphase = np.ones(window_size).astype(float32)*np.pi/2.0

#phase = np.zeros(window_size)

SW,SH = (64,64)
old_half_frames = 4
old_frames = [np.zeros( (SW/2) * (SH/2) ) for i in range(0,old_half_frames)]


dooverlaps = True
dowindowing = True
phasei=0

    
out = brain.predict([np.zeros(2*64*64)])[0]

sampled = np.array([1,2,3,4,6,8,10,12,14,16,20,24,32,64,128,256])
lasts = None
#cv2.namedWindow("grey", cv2.cv.CV_WINDOW_NORMAL)
cv2.namedWindow("scaled", cv2.cv.CV_WINDOW_NORMAL)
phi = float(args.a)
b = float(args.b)
zeros = int(args.z)


#fourcc = cv2.cv.FOURCC(*'XVID')
#writer = None 

deep_learnings = True


outi = [8,7,9,6,10,5,11,4,12,3,13,2,14,1,15,0]
outi.reverse()
outi = np.array(outi)
ohori = None
overti = None
while(running):
    ret, frame = cap.read()
    if (not ret):
        running = False
        continue    
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(grey, (SW,SH))

    # stripes
    horiz = cv2.resize(scaled, (16, 1)) 
    verti  = cv2.resize(scaled, (1,16))
    if (ohori == None):
        ohori = horiz
        overti = verti
    ohori = horiz
    overti = overti
    outhoriz = np.abs(horiz-ohori)[0,outi].tolist()
    outverti = np.abs(verti-overti)[outi,0].tolist()
    # zero out values due to zero setting
    for i in range(len(outverti)-zeros,len(outverti)):
        outverti[i] = 0.0
        outhoriz[i] = 0.0
    sendOSC("/webcam/horiz",*outhoriz )
    sendOSC("/webcam/vert", *outverti )

    cv2.imshow("horiz",cv2.resize(horiz,(256,64)))
    cv2.imshow("vert",cv2.resize(verti,(64,256)))


    #cv2.imshow('grey',grey)    
    #if writer == None:
    #    (myw,myh,_) = frame.shape
    #    writer = cv2.VideoWriter("webcam.avi",fourcc,30,(myh,myw),1)
    #writer.write(frame)
    scaled = scaled.astype(np.float32)
    scaled /= 255.0
    #scaled = (1.0/phi)*(scaled/(1.0/1.0))**0.5
    scaled = phi*scaled + b
    scaled[scaled < 0.0] = 0.0
    cv2.imshow('scaled',scaled)
    
    data = np.concatenate(old_frames + [scaled.flatten()]) 
    smaller = cv2.resize(grey, (SW/2,SH/2))
    smaller = smaller.astype(np.float32)
    smaller /= 255.0
    del old_frames[0]
    old_frames.append(smaller.flatten())

    deep_learnings = frames % 3 != 0
    if (deep_learnings):
        out = brain.predict([data])[0]
        
        # out is the guts of a fourier transform
        # inverse fft won't work well
        
        # sample and spam
        sample = out[sampled]
        if lasts == None:
            lasts = sample
        l = np.abs(sample - lasts).tolist()
        #l = sample.tolist()
        #print l
        sendOSC("/fft/sbins", *l)
        print l
        lasts = sample

    ckey = cv2.waitKey(1) & 0xFF
    if ckey==27:
        break
    elif ckey==ord('P'):
        phi *= 1.1
        print phi
    elif ckey==ord('p'):
        phi *= 0.9
        print phi
    elif ckey==ord('Z'):
        zeros += 1
        print zeros
    elif ckey==ord('z'):
        zeros = max(0,zeros - 1)
        print zeros
    elif ckey==ord('L'):
        b += 0.1
        print b
    elif ckey==ord('l'):
        b -= 0.1
        print b

    frames += 1
    if frames % 30 == 0:
        print (frames, amax)

cv2.destroyAllWindows()




