import os, sys
from typing import List, Tuple, Union, Dict

import torch
import torch.nn.functional as F

import selfies as sf
import rdkit.Chem as Chem

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def sanitize_smiles(smi: str) -> str:
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except:
        return None

def get_lists(data_path: str, sep: str, header: int = None, 
        smiles_name: Union[int, str] = 0) -> Tuple[List[str], List[str]]:
    ''' Get smi_list from data specified at data_path.
    May require change by user if format is different.
    '''
    data = pd.read_csv(data_path, sep=sep, header=header) 
    data[smiles_name] = data[smiles_name].apply(sanitize_smiles)
    data = data.dropna()

    smi_list = data[smiles_name].tolist()
    sfs_list = data[smiles_name].apply(sf.encoder).tolist()
    return smi_list, sfs_list

def get_mols(smi_list: List[str]):
    mols = []
    for smi in smi_list:
        mols.append(Chem.MolFromSmiles(smi))
    return mols

def plot_metrics(csv_log_path: str, out_path: str, metrics: List[str] = ['accuracy', 'loss']):
    data = pd.read_csv(csv_log_path)

    cols = [f'train_{m}' for m in metrics]
    cols.append('epoch')
    train_metrics = data[cols].dropna()
    train_metrics = train_metrics.rename({f'train_{m}': m for m in metrics}, axis=1)
    train_metrics['type'] = ['train'] * len(train_metrics)

    cols = [f'val_{m}' for m in metrics]
    cols.append('epoch')
    val_metrics = data[cols].dropna()
    val_metrics = val_metrics.rename({f'val_{m}': m for m in metrics}, axis=1)
    val_metrics['type'] = ['val'] * len(val_metrics)

    all_metrics = pd.concat([train_metrics, val_metrics])

    for m in metrics:
        sns.lineplot(data = all_metrics, x = 'epoch', y = m, hue = 'type')
        plt.savefig(os.path.join(out_path, f'{m}.png'))
        plt.close()

    print('Metrics plotted!')


def get_smiles_alphabet(smi_list: List[str]) -> List[str]:
    smiles_alphabet = list(set(''.join(smi_list)))
    smiles_alphabet = sorted(smiles_alphabet)
    smiles_alphabet.append(' ')
    return smiles_alphabet

def smiles_to_encoding(smi: str, vocab_stoi: Dict, pad_to_len: int, 
        enc_type: str = None):
    # create labels for input, padded to len_molecule
    pad_idx = vocab_stoi[' ']
    labels = torch.LongTensor([vocab_stoi[c] for c in smi])
    pad_len = pad_to_len - len(labels)
    if pad_len > 0:
        labels = torch.cat((labels, torch.LongTensor([pad_idx]*pad_len)), dim=0)
    
    if enc_type == 'label':
        return labels
    elif enc_type == 'one_hot':
        one_hot = F.one_hot(labels, len(vocab_stoi))
        return one_hot
    else:
        one_hot = F.one_hot(labels, len(vocab_stoi))
        return labels, one_hot

def encoding_to_smiles(encoding: Union[List, torch.Tensor], vocab_itos: Dict, enc_type: str):
    if enc_type == 'one_hot':
        if type(encoding) == torch.Tensor:
            labels = encoding.argmax(dim = -1).numpy()
        else: 
            labels = np.argmax(encoding, dim=-1)
    else:
        labels = encoding
    
    smi = ''
    for i in labels:
        smi += vocab_itos[i]
    return smi

def get_nth_char(inp_str: str, n: int, string_type: str):
    if string_type == 'selfies':
        ind = inp_str.replace(']', '~', n-1).find(']')
        return inp_str[:ind + 1]
    elif string_type == 'smiles':
        return inp_str[:n]
    else:
        raise ValueError('No such string representation.')