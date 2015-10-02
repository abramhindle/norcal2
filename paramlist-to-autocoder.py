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


parser = argparse.ArgumentParser(description='Convert Paramlist to Pickle!')
parser.add_argument('-i', help='Input ParamList Pickle')
parser.add_argument('-o', help='Output Pickle')


def set_network(network, param_list):
        for layeri in range(0,len(network.layers)):
                print layeri
                player = param_list[layeri]
                layer  = network.layers[layeri]
                for parami in range(0,len(layer.params)):
                        pparam = player[parami]
                        param  = layer.params[parami]
                        if not str(param) == pparam[0]:
                                raise Exception(" %s != %s ", str(param), pparam[0])
                        print pparam[0]
                        param.set_value(pparam[1])



args = parser.parse_args()
oldnetp = pickle.load(file(args.i,'rb'))

inputs = 4096*2
win_size = 2048
swin_size = win_size / 2 + 1
output_size = swin_size
hidlayersize = win_size
exp = theanets.Experiment(theanets.Regressor,layers=[inputs, inputs/2, inputs/3, inputs/4, output_size])
net = exp.network

set_network(net, oldnetp)
cPickle.dump(net, file(args.o,'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

