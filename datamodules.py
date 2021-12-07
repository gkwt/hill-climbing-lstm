import os, sys
import multiprocessing

import torch
import pytorch_lightning as pl

import selfies as sf

class SELFIESDataset(torch.utils.data.Dataset):
    def __init__(self, sf_list, vocab, len_molecule = None):
        self.sf_list = sf_list
        self.vocab = vocab
        if len_molecule is None:
            self.len_molecule = max([sf.len_selfies(s) for s in sf_list])
        else:
            self.len_molecule = len_molecule

    def __len__(self):
        return len(self.sf_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        encoding = sf.selfies_to_encoding(
            self.sf_list[idx],
            vocab_stoi = self.vocab,
            pad_to_len = self.len_molecule,
            enc_type = 'one_hot'
        )

        return torch.LongTensor(encoding)

# class SMILESDataset(torch.utils.data.Dataset):
    #TODO

class SELFIESDataModule(pl.LightningDataModule):
    def __init__(self, sf_list, train_ratio = 0.7, batch_size = 128, num_workers = None):
        super().__init__()
        self.sf_list = sf_list
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers

        alphabet = sf.get_alphabet_from_selfies(sf_list)
        alphabet.add("[nop]")
        alphabet = list(sorted(alphabet))
        self.vocab = {s: i for i, s in enumerate(alphabet)}
        self.inv_vocab = {v: k  for k, v in self.vocab.items()}

        self.dataset = SELFIESDataset(self.sf_list, self.vocab)
        self.len_molecule = self.dataset.len_molecule
        self.len_alphabet = len(alphabet)

    def setup(self, stage = None):
        train_size = int(len(self.dataset) * self.train_ratio)
        valid_size = len(self.dataset) - train_size
        self.train_set, self.valid_set = torch.utils.data.random_split(self.dataset, [train_size, valid_size])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set, batch_size = self.batch_size)

# class SMILESDataModule(pl.LightningDataModule):
    #TODO
