import os
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import rdkit.Chem as Chem
import selfies as sf

from datamodules import SELFIESDataModule
from network import LanguageModel

def sanitize_smiles(smi: str) -> str:
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except:
        return None

def get_lists(data_path: str, sep: str, header: int = None, 
        smiles_name: int = 0) -> Tuple[List[str], List[str]]:
    ''' Get smi_list from data specified at data_path.
    May require change by user if format is different.
    '''
    data = pd.read_csv(data_path, sep=sep, header=header) 
    data[smiles_name] = data[smiles_name].apply(sanitize_smiles)
    data = data.dropna()

    smi_list = data[smiles_name].tolist()
    sfs_list = data[smiles_name].apply(sf.encoder).tolist()
    return smi_list, sfs_list
    

# define parameters
data_path       = 'data/hce.txt'
string_type     = 'selfies'


# get the data
smi_list, sfs_list = get_lists(data_path, sep=' ')
if string_type == 'selfies':
    str_list = sfs_list
    dm = SELFIESDataModule(str_list, train_ratio = 0.7, batch_size = 128, num_workers = None)
elif string_type == 'smiles':
    str_list = smi_list
    # dm = SMILESDataModule(str_list)
else:
    raise ValueError('No such string representation.')

# print some stuff about the data
print(f'You are using the "{string_type}" representation.')
print(f'Number of molecules: {len(str_list)}')
print(f'Length of longest molecule: {dm.len_molecule}')
print(f'Length of alphabet: {dm.len_alphabet}')

# create model
model = LanguageModel(1024, 3, dm.len_alphabet, dm.len_molecule)

# default logger used by trainer
logger = CSVLogger(os.path.join(os.getcwd(), 'trained_models'), name=string_type)

callbacks = [
    ModelCheckpoint(dirpath = logger.log_dir, filename = 'final_model', monitor = 'val_loss', mode = 'min', verbose=True),
    EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=True)
]

trainer = pl.Trainer(
    accelerator = 'auto', 
    logger=logger,
    max_epochs = 10, 
    callbacks = callbacks,
    enable_progress_bar = True
)
trainer.fit(model, dm)