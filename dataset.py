import argparse
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from utils import *


class SirenSynthDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]


if __name__ == "__main__":

    # Link to config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    params = Params(args.config_path)

    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda:6")
        print("Device: {}".format(device))
        print("Device name: {}".format(torch.cuda.get_device_properties(device).name))
    else:
        device = torch.device("cpu")
        print("Device: {}".format(device))

    # Transformation of audio file to mel-spectrograms
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=params.sample_rate,
        n_fft=params.fft_size,
        win_length=params.window_length,
        hop_length=params.hop_size,
        n_mels=params.mels_number
    )

    # Dataset creation
    dataset = SirenSynthDataset(params.train_annotations_file,
                                params.train_audio_dir,
                                mel_spectrogram,
                                params.sample_rate,
                                params.num_samples,
                                device)

    print(f"There are {len(dataset)} samples in the dataset.")
    signal, label = dataset[0]
