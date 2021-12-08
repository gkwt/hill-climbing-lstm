from typing import List, Tuple, Union

import selfies as sf
import rdkit.Chem as Chem

import pandas as pd
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

def plot_metrics(csv_log_path: str):
    #TODO
    return