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
import random

import liblo
import argparse

parser = argparse.ArgumentParser(description='Do deep learning and pump into osc')
parser.add_argument('-c',default=0, help='Camera')
parser.add_argument('-i',default=None, help='Input File')
parser.add_argument('-osc1', dest='osc1', default=7770,help="OSC Port")
parser.add_argument('-osc2', dest='osc2', default=7771,help="OSC Port")
parser.add_argument('-a', dest='a', default=1.0,help="A/phi multiplier")
parser.add_argument('-b', dest='b', default=0.0,help="b offset")
parser.add_argument('-z', dest='z', default=0,help="b offset")
args = parser.parse_args()

if args.i == None:
    cap = cv2.VideoCapture(int(args.c))
else:
    cap = cv2.VideoCapture(args.i)

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

target1 = liblo.Address(int(args.osc1))
target2 = liblo.Address(int(args.osc2))

def sendOSC(path,*args):
    liblo.send(target1, path, *args)
    liblo.send(target2, path, *args)



    
def gaussian_noise(inarr,mean=0.0,scale=1.0):    
    noise = np.random.normal(mean,scale,inarr.shape)
    return inarr + noise.reshape(inarr.shape)

outs = []
frames = 0






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
cv2.namedWindow("horiz", cv2.cv.CV_WINDOW_NORMAL)
cv2.namedWindow("vert", cv2.cv.CV_WINDOW_NORMAL)
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
diff = False
while(running):
    ret, frame = cap.read()
    if (not ret):
        running = False
        continue    
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(grey, (SW,SH))

    scaled = scaled.astype(np.float32)
    scaled /= 255.0
    #scaled = (1.0/phi)*(scaled/(1.0/1.0))**0.5
    scaled = phi*scaled + b
    scaled[scaled < 0.0] = 0.0
    cv2.imshow('scaled',scaled)

    # stripes
    horiz = cv2.resize(scaled, (16, 1)) 
    vert  = cv2.resize(scaled, (1,16))
    verti   = vert/(10000.0)
    horizi  = horiz/(10000.0)

    if (ohori == None):
        ohori = horizi
        overti = verti
    #outhoriz = (np.abs(horizi-ohori)[0,outi] ).tolist()
    #outverti = (np.abs(verti-overti)[outi,0]).tolist()
    if not diff:
        outhoriz = (np.abs(horizi)[0,outi] ).tolist()
        outverti = (np.abs(verti)[outi,0]).tolist()
    else:
        outhoriz = (np.abs(horizi - ohori)[0,outi] ).tolist()
        outverti = (np.abs(verti - overti)[outi,0]).tolist()


    ohori = horizi
    overti = verti

    # zero out values due to zero setting
    for i in range(len(outverti)-zeros,len(outverti)):
        outverti[i] = 0.0
        outhoriz[i] = 0.0
    print ("Horiz",outhoriz)
    sendOSC("/webcam/horiz",*outhoriz )
    sendOSC("/webcam/vert", *outverti )

    cv2.imshow("horiz",cv2.resize(horiz,(256,64)))
    cv2.imshow("vert",cv2.resize(vert,(64,256)))


    
    data = np.concatenate(old_frames + [scaled.flatten()]) 
    smaller = cv2.resize(grey, (SW/2,SH/2))
    smaller = smaller.astype(np.float32)
    smaller /= 255.0
    del old_frames[0]
    old_frames.append(smaller.flatten())

    deep_learnings = frames % 5 != 0
    deep_learnings = True
    if (deep_learnings):
        out = brain.predict([data])[0]
        
        # out is the guts of a fourier transform
        # inverse fft won't work well
        
        # sample and spam
        sample = out[sampled]
        if lasts == None:
            lasts = sample
        l = np.abs(sample - lasts).tolist()
        sendOSC("/fft/sbins", *l)
        print sum(l)
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
        zeros = min(16,zeros+1)
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
    elif ckey==ord('D'):
        diff = True
    elif ckey==ord('d'):
        diff = False


    frames += 1
    if frames % 30 == 0:
        print (frames, amax)

cv2.destroyAllWindows()




