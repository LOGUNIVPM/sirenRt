import json
import numpy as np
import os
import random
import shutil
import torch
from torch.utils.data import DataLoader


class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            params = json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class LossRunningAverage:
    def __init__(self):
        self.total = 0
        self.steps = 0

    def update(self, loss):
        self.total += loss
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


class AccRunningAverage:
    def __init__(self):
        self.total = 0
        self.steps = 0

    def update(self, accuracy):
        self.total += accuracy
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_train_dataloader(train_data, train_batch_size, train_sampler):
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, sampler=train_sampler)
    return train_dataloader


def create_valid_dataloader(valid_data, valid_batch_size, valid_sampler):
    valid_dataloader = DataLoader(valid_data, batch_size=valid_batch_size, sampler=valid_sampler)
    return valid_dataloader


def create_test_dataloader(test_data, test_batch_size):
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)
    return test_dataloader


def save_checkpoint(state, is_best, checkpoint):
    filename = os.path.join(checkpoint, "model_last.pt")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist")
        os.mkdir(checkpoint)
        torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint, "model_best.pt"))


def load_checkpoint(checkpoint, model, optimizer=None, parallel=False):
    if not os.path.exists(checkpoint):
        raise("File Not Found Error {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
    # checkpoint = torch.load(checkpoint, map_location=torch.device("cuda:6"))
    if parallel:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    print('BUUUUUURP')

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def seed_all(seed):
    if not seed:
        seed = 10
    print("[ Using Seed: ", seed, "]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
