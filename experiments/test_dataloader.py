import numpy as np
import os

from torch.utils.data import DataLoader
from dataLoader.data_iterator import SequenceDataset
from dataLoader.task_sampler import NShotTaskSampler
from config import PATH

def bla():
    data_path = os.path.join(PATH,"data/LibriSpeech/train-clean-100-v2")
    min_seq = 1
    max_seq = 3
    downsampling = 4
    episodes_per_epoch = 1
    n_train = 5
    k_train = 5
    q_train = 5

    train_data = SequenceDataset(data_path, min_seq, max_seq, downsampling)
    train_taskloader = DataLoader(
        train_data,
        batch_sampler = NShotTaskSampler(train_data, episodes_per_epoch, n_train, k_train, q_train),
        num_workers = 4
    )
        
    for batch_index, batch in enumerate(train_taskloader):
        x, y = batch
        print(y.shape)
        print(x.shape)