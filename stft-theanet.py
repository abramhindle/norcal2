import theanets
import pickle
import numpy as np
import climate
import logging

climate.enable_default_logging()

# input 64*64 grayscale bitmap
# output samples 22050/30
# fft windows of 1024
# cut down to real values
# cut down again
inputs = 4096*2
win_size = 2048
swin_size = win_size / 2 + 1
output_size = swin_size
hidlayersize = win_size
exp = theanets.Experiment(theanets.Regressor,layers=[inputs, inputs/2, inputs/3, inputs/4, output_size])
net = exp.network

logging.info("Read frames.pkl")
frames = pickle.load(file('frames.pkl'))
logging.info("Read stft.pkl")
audio  = pickle.load(file('stft.pkl'))
train = frames
outputs = audio
train = train.astype(np.float32)
outputs = outputs.astype(np.float32)[0:train.shape[0]]
shuffleids = np.arange(train.shape[0])
np.random.shuffle(shuffleids)
testids = shuffleids[0:len(shuffleids)*3/4]
validids = shuffleids[len(shuffleids)*3/4:]
ttrain = train[shuffleids]
toutputs = outputs[shuffleids]
vtrain = train[validids]
voutputs = outputs[validids]

logging.info("Pretraining")
net.train([ttrain, toutputs],
          [vtrain, voutputs],
          algo='layerwise',
          learning_rate=1e-3,
          save_every=25,
          batch_size=1024,
          patience = 1,
          min_improvement = 0.01,
          save_progress="current_pre_brain.pkl",
          momentum=0.9)

logging.info("Finetune Training")
net.train([ttrain, toutputs],
          [vtrain, voutputs],
          algo='rmsprop',
          learning_rate=1e-3,
          save_every=25,
          batch_size=1024,
          patience = 10,
          min_improvement = 0.01,
          save_progress="current_brain.pkl",
          momentum=0.9)

net.save('stft-theanet.py.net.pkl')


