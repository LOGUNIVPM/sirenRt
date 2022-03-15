import sounddevice as sd
import numpy as np
import threading
from neuralOps import *

# AUDIO-RELATED DEFINES
AUDIOSAMPLERATE = 16000
NEURALNET_BUFFER_S = 0.5 # the neural network requires a buffer of this length is seconds
AUDIODEVICENAME = 'default'

class audioThread(threading.Thread):
    audioDevList = []
    neural = []

    def __init__(self):
        super().__init__()
        self.Fs = AUDIOSAMPLERATE
        self.alarm = False
        self.statusChanged = False
        self.event = threading.Event()
        self.neural = neuralOps(self.Fs)

    def fillAudioDevList(self):
        devs = sd.query_devices()
        for d in devs:
            self.audioDevList.append(d["name"])

    def initAudio(self,audiodevname):
        sd.default.device = audiodevname
        sd.default.samplerate = self.Fs
        sd.default.channels = 1

    def recordOnce(self, instance): # instance is the calling button, unuseful
        # test rec
        try:
            print("Start Recording...")
            self.recordedAudio = sd.rec(int(1 * self.Fs), blocking=True)
            print("Done")
        except Exception as e:
            print(e)
            print("Troubles in recording... Abort") # TODO: Try to do something manually
            exit(-1)

    def playbackOnce(self, instance):
        try:
            print("Start Playback...")
            sd.play(self.recordedAudio, self.Fs, blocking=True)
            print("Done")
        except Exception as e:
            print(e)
            print("Troubles in playback... Abort") # TODO: Try to do something manually
            exit(-1)

    def launch(self):
        try:
            self.stream = sd.InputStream(device=sd.default.device,
                                    channels=int(sd.default.channels[0]),
                                    samplerate=int(sd.default.samplerate),
                                    blocksize=int(self.Fs*NEURALNET_BUFFER_S), # without this the blocksize varies for each callback
                                    callback=self.process)
            with self.stream:
                print('Audio processing started...')
                self.event.clear() # in case it was previously set by terminate()
                self.event.wait()
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))


    def terminate(self):
        self.stream.stop()
        self.event.set() # this tells event to stop waiting

    def process(self, indata, frames, time, status):
        # WARNING: the sounddevice callback cannot be debugged (executed at PortAudio level): do the processing elsewhere and share data using sempahores
        raiseflag = 0
        if status:
            print(status)

        prediction = self.neural.processAudio(indata)
        #avgenv = np.mean(np.abs(indata))
        #print(avgenv)
        if prediction == True:
            raiseflag = 1

        if raiseflag != self.alarm:
            self.alarm = raiseflag
            self.statusChanged = True