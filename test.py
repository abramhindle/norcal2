import pickle
import theanets
import numpy
brain = theanets.feedforward.Regressor.load("stft-theanet.py.net.pkl.cpu")#brain-1438666035")
brain._graphs = {} 
brain._functions = {}
print brain.predict([numpy.zeros(4096*2)])[0]
