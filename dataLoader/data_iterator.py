import random
import os

import numpy as np
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset
from config import DATA_PATH

class SequenceDataset(Dataset):
    def __init__(self, min_seq, max_seq, downsampling, subset, seq=False):
        self.sequence = seq
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.downsampling = downsampling
        
        self.df = pd.DataFrame(self.index_subset(subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        # print(self.datasetid_to_filepath[item])
        sample, samplerate = torchaudio.load(self.datasetid_to_filepath[item])

        #random sequence length between min and max length
        if (int)(sample.shape[-1]/samplerate) >= self.max_seq:
            seq_length = (int)(random.uniform(self.min_seq, self.max_seq)*samplerate)
        
            index = random.randrange(0,sample.shape[-1]-seq_length)
            sample = sample[:,index:index+seq_length]
        
        #add padding at the end to make max length sequence
        padding = torch.zeros(self.max_seq*samplerate - sample.shape[-1]).unsqueeze(0)
        sample = torch.cat((sample,padding),1)

        if self.sequence:
            #downsample kHz
            sample = sample[:,::self.downsampling]
        else:
            sample = torchaudio.transforms.Spectrogram(n_fft=255, hop_length=160)(sample)

        label = self.datasetid_to_class_id[item]
        return(sample,label)
        
    def __len__(self):
        return(len(self.df))

    def num_classes(self):
        return(len(self.df['class_name'].unique()))
    
    @staticmethod
    def index_subset(subset):
        path = os.path.join(DATA_PATH, subset)
        samples = []
        for root, folders, files in os.walk(path):
            if not files:
                continue;
            
            alphabet = root.split('/')[-3]
            class_name = root.split('/')[-2]
            
            for file in files:
                samples.append({
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, file)
                })
                
        return(samples)

class SpectrogramDataset(Dataset):
    def __init__(self, min_seq, max_seq, downsampling, subset):
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.downsampling = downsampling
        
        self.df = pd.DataFrame(self.index_subset(subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']
        
    def __getitem__(self, item):
        # TODO
        return(sample,label)
        
    def __len__(self):
        return(len(self.df))

    def num_classes(self):
        return(len(self.df['class_name'].unique()))
    
    @staticmethod
    def index_subset(subset):
        path = os.path.join(DATA_PATH, subset)
        samples = []
        for root, folders, files in os.walk(path):
            if not files:
                continue;
            
            alphabet = root.split('/')[-3]
            class_name = root.split('/')[-2]
            
            for file in files:
                samples.append({
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, file)
                })
                
        return(samples)