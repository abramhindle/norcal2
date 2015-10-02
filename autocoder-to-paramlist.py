import os
os.environ['THEANO_FLAGS']="device=cpu"
import theanets
import theano
import argparse
theano.config.experimental.unpickle_gpu_on_cpu
import numpy as np
from sys import argv
try:
   import cPickle as pickle
except:
   import pickle
import cPickle

parser = argparse.ArgumentParser(description='Convert autocoders to CPU!')
parser.add_argument('-i', help='Input Brain Pickle')
parser.add_argument('-o', help='Output ParamList Pickle')
args = parser.parse_args()


def to_param_list(network):
        newlayers = list()
        for layer in network.layers:
                params = list()
                for param in layer.params:
                        name = str(param)
                        values = param.get_value()
                        params.append((name,values))
                newlayers.append(params)
        return newlayers

oldnet = pickle.load(file(args.i,'rb'))
oldnetp = to_param_list(oldnet)

cPickle.dump(oldnetp, file(args.o,'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
