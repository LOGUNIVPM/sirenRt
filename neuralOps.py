
import torch
import torchaudio.transforms
from utils import Params
from model import *
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:6")
else:
    device = torch.device("cpu")

class neuralOps():

    def __init__(self, samplerate):
        self.sirenModel = SirenCNN()
        self.samplerate = samplerate

        # parse arguments from file
        params = Params('config.json')
        self.melSpectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.samplerate,
                                                                   n_fft=params.fft_size,
                                                                   win_length=params.window_length,
                                                                   hop_length=params.hop_size,
                                                                   n_mels=params.mels_number)

        # Load the best saved model
        checkpoint = os.path.join("./models/", "model_best.pt")
        saved_model = utils.load_checkpoint(checkpoint, self.sirenModel, optimizer=None, parallel=False)

    def processAudio(self, npVec):
        # check vector shape and adapt
        torchVec = torch.from_numpy(np.transpose(npVec))
        torchVec = torchVec[None, :]

        # transform and send to device
        input = self.melSpectrogram(torchVec)
        input = input.to(device)

        # forward
        #prediction = self.sirenModel.forward(npVec)
        outputs = self.sirenModel(input)
        _, prediction = torch.max(outputs, 1)

        return prediction

