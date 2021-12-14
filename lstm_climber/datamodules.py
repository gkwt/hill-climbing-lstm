import os, sys
import multiprocessing

import torch
import pytorch_lightning as pl

import rdkit.Chem as Chem
import selfies as sf

from . import utils

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
            pad_to_len = self.len_molecule + 1,     # add 1 padding for target sequence
            enc_type = 'one_hot'
        )
        feature = torch.LongTensor(encoding[:-1])
        target = torch.LongTensor(encoding[1:])     # targets are shifted by one character

        return feature, target

class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, smi_list, vocab, len_molecule = None):
        self.smi_list = smi_list
        self.vocab = vocab
        if len_molecule is None:
            self.len_molecule = max([len(s) for s in smi_list])
        else:
            self.len_molecule = len_molecule
    
    def __len__(self):
        return len(self.smi_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        encoding = utils.smiles_to_encoding(
            self.smi_list[idx],
            vocab_stoi = self.vocab,
            pad_to_len = self.len_molecule + 1,
            enc_type = 'one_hot'
        )
        feature = torch.LongTensor(encoding[:-1])
        target = torch.LongTensor(encoding[1:])

        return feature, target

class SELFIESDataModule(pl.LightningDataModule):
    def __init__(self, sf_list, train_ratio = 0.8, random_split = False,
            batch_size = 128, num_workers = None):
        super().__init__()
        self.sf_list = sf_list
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.random_split = random_split

        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers

        alphabet = sf.get_alphabet_from_selfies(sf_list)
        alphabet.add("[nop]")
        alphabet = list(sorted(alphabet))
        self.alphabet = alphabet
        self.vocab = {s: i for i, s in enumerate(self.alphabet)}
        self.inv_vocab = {v: k  for k, v in self.vocab.items()}

        self.dataset = SELFIESDataset(self.sf_list, self.vocab)
        self.len_molecule = self.dataset.len_molecule
        self.len_alphabet = len(self.alphabet)

    def setup(self, stage = None):
        train_size = int(len(self.dataset) * self.train_ratio)
        valid_size = len(self.dataset) - train_size
        if self.random_split:
            self.train_set, self.valid_set = torch.utils.data.random_split(self.dataset, [train_size, valid_size])
        else:
            self.train_set = torch.utils.data.Subset(self.dataset, range(0, train_size))
            self.valid_set = torch.utils.data.Subset(self.dataset, range(train_size, len(self.dataset)))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size = self.batch_size, shuffle = True, 
            num_workers = self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set, batch_size = self.batch_size, 
            num_workers = self.num_workers)

    def encode_string(self, inp_str: str):
        ''' Encode an input string into selfies.
        '''
        encoding = sf.selfies_to_encoding(
            inp_str,
            vocab_stoi = self.vocab,
            pad_to_len = sf.len_selfies(inp_str),
            enc_type = 'one_hot'
        )
        return torch.LongTensor(encoding)

    def logits_to_smiles(self, logits: torch.Tensor, return_selfies: bool = False):
        ''' Turns a list of logits into a list of canonical smiles.
        '''
        labels = logits.argmax(dim=-1).numpy()
        smi_list = []
        sfs_list = []

        for enc in labels:
            sfs = sf.encoding_to_selfies(
                enc,
                vocab_itos = self.inv_vocab,
                enc_type = 'label'
            )
            sfs_list.append(sfs)
            smi = utils.sanitize_smiles(sf.decoder(sfs))
            smi_list.append(smi)

        if return_selfies:
            return smi_list, sfs_list
            
        return smi_list

class SMILESDataModule(pl.LightningDataModule):
    def __init__(self, smi_list, train_ratio = 0.8, random_split = False,
            batch_size = 128, num_workers = None):
        super().__init__()
        self.smi_list = smi_list
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.random_split = False

        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers

        # extract alphabet from smi_list
        self.alphabet = utils.get_smiles_alphabet(self.smi_list)
        self.vocab = {s: i for i, s in enumerate(self.alphabet)}
        self.inv_vocab = {v: k  for k, v in self.vocab.items()}

        self.dataset = SMILESDataset(self.smi_list, self.vocab)
        self.len_molecule = self.dataset.len_molecule
        self.len_alphabet = len(self.alphabet)

    def setup(self, stage = None):
        train_size = int(len(self.dataset) * self.train_ratio)
        valid_size = len(self.dataset) - train_size
        if self.random_split:
            self.train_set, self.valid_set = torch.utils.data.random_split(self.dataset, [train_size, valid_size])
        else:
            self.train_set = torch.utils.data.Subset(self.dataset, range(0, train_size))
            self.valid_set = torch.utils.data.Subset(self.dataset, range(train_size, len(self.dataset)))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size = self.batch_size, shuffle = True, 
            num_workers = self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set, batch_size = self.batch_size, 
            num_workers = self.num_workers)

    def encode_string(self, inp_str: str):
        ''' Encode an input string into onehot smiles.
        '''
        encoding = utils.smiles_to_encoding(
            inp_str,
            vocab_stoi = self.vocab,
            pad_to_len = len(inp_str),
            enc_type = 'one_hot'
        )
        return torch.LongTensor(encoding)

    def logits_to_smiles(self, logits: torch.Tensor):
        ''' Turns a list of logits into a list of canonical smiles.
        '''
        labels = logits.argmax(dim=-1).numpy()
        smi_list = []

        for enc in labels:
            smi = utils.encoding_to_smiles(
                enc,
                vocab_itos = self.inv_vocab,
                enc_type = 'label'
            )
            smi = utils.sanitize_smiles(smi)
            smi_list.append(smi)

        return smi_list
