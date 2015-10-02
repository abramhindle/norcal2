#!/usr/bin/env python3

"""Create a JACK client that copies input audio directly to the outputs.

This is somewhat modeled after the "thru_client.c" example of JACK 2:
http://github.com/jackaudio/jack2/blob/master/example-clients/thru_client.c

If you have a microphone and loudspeakers connected, this might cause an
acoustical feedback!

"""
import sys
import signal
import os
import jack
import threading
import numpy
import numpy as np
import collections
import logging

if sys.version_info < (3, 0):
    # In Python 2.x, event.wait() cannot be interrupted with Ctrl+C.
    # Therefore, we disable the whole KeyboardInterrupt mechanism.
    # This will not close the JACK client properly, but at least we can
    # use Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
else:
    # If you use Python 3.x, everything is fine.
    pass

#argv = iter(sys.argv)
# By default, use script name without extension as client name:

class JackFiller(object):
    def __init__(self,**kwargs):
        logging.info("Constructing JackFiller")
        defaultclientname = os.path.splitext(os.path.basename(next(iter(sys.argv))))[0]
        self.clientname = kwargs.get("clientname", defaultclientname)
        self.servername = kwargs.get("servername", None)
        self.client = jack.Client(self.clientname, servername=self.servername)

        if self.client.status.server_started:
            print("JACK server started")
        if self.client.status.name_not_unique:
            print("unique name {0!r} assigned".format(client.name))
        self.event = threading.Event()
        self.channels = kwargs.get("channels",1)
        self.is_done = False
        self.old = None
        def process(frames):
            client = self.client
            assert frames == client.blocksize
            for i in range(0,self.channels):
                o = client.outports[i]
                arr = o.get_array()
                if (self.old == None):
                    self.old = arr.copy()
                n = len(arr)
                j = 0
                while j < n:                    
                    if len(self.queue) == 0:
                        # fill with zeros
                        arr[j:n] = numpy.zeros(n - j)
                        j = n
                        if self.is_done:
                            self.event.set()
                    else:
                        narr = self.queue.popleft()
                        diff = n - j
                        if diff == len(narr):
                            # totally effecient
                            arr[j:n] = narr
                            j = n
                        elif diff < len(narr):
                            arr[j:n] = narr[0:diff]
                            # now add it back ineffeciently
                            narr = narr[diff:len(narr)]
                            self.queue.appendleft(narr)
                            j = n
                        else:
                            # ok so the buffers are smaller
                            arr[j:j+len(narr)] = narr
                            j += len(narr)
                self.old[:] = arr
                


        self.callback = process

        def shutdown(status, reason):
            print("JACK shutdown!")
            print("status:", status)
            print("reason:", reason)
            self.event.set()

        # create two port pairs
        for number in range(1, self.channels+1): 
            logging.info("Channel %s" % number)
            if kwargs.get("inputs",False):
                logging.info("Connecting Inputs")
                self.client.inports.register("input_{0}".format(number))
            if kwargs.get("outputs",True):
                logging.info("Connecting Outputs")
                self.client.outports.register("output_{0}".format(number))

        self.queue = collections.deque()
        self.client.set_process_callback(self.callback)
        self.client.set_shutdown_callback( shutdown )
        logging.info("Done constructing")

        
    def connect_to_physical_ports(self):
        client = self.client
        capture = client.get_ports(is_physical=True, is_output=True)
        if not capture:
            raise RuntimeError("No physical capture ports")
            
        for src, dest in zip(capture, client.inports):
            client.connect(src, dest)

        playback = client.get_ports(is_physical=True, is_input=True)
        if not playback:
            raise RuntimeError("No physical playback ports")
        
        for src, dest in zip(client.outports, playback):
            client.connect(src, dest)

    def activate(self):
        self.client.activate()

    def shutdown(self):
        self.event.set()        

    def wait(self):
        try:
            self.event.wait()
        except KeyboardInterrupt:
            print("\nInterrupted by user")

    def loop(self):
        print("Press Ctrl+C to stop")
        try:
            self.event.wait()
        except KeyboardInterrupt:
            print("\nInterrupted by user")

    def add(self, arr):
        self.queue.append(arr)
    
    def __enter__(self):
        return self.client.__enter__()
    def __exit__(self, type, value, traceback):
        self.client.__exit__(type,value,traceback)

    def done(self):
        self.is_done = True

if __name__ == '__main__':
    # import scikits.audiolab
    sr = 44100.0
    channels = 1
    freq = 440.0
    #outwav = scikits.audiolab.Sndfile("out.wav",mode='w',format=scikits.audiolab.Format(),channels=channels,samplerate=int(sr))

    logging.basicConfig(stream = sys.stderr, level=logging.INFO)
    jf = JackFiller(inputs=False,outputs=True,channels=channels)
    n = 4096
    t = np.arange(float(n))
    
    jf.activate()
    jf.connect_to_physical_ports()
    
    for i in range(1,42):        
        frames = 0.5*np.sin(2 * freq * np.pi * (i*n+t)/sr)
        jf.add(frames)
        #outwav.write_frames(frames)
    jf.done()
    #outwav.sync()
    #del outwav
    jf.wait()
