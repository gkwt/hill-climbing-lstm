import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import rdkit.Chem as Chem
import selfies as sf
import pandas as pd

from lstm_climber import SELFIESDataModule, SMILESDataModule, LanguageModel
import lstm_climber.utils as utils

# define parameters
data_path       = 'data/hce.txt'
string_type     = 'selfies' # 'smiles'
num_workers     = 6

# get the data
smi_list, sfs_list = utils.get_lists(data_path, sep=' ')
if string_type == 'selfies':
    str_list = sfs_list
    dm = SELFIESDataModule(str_list, batch_size = 128, num_workers = num_workers)
elif string_type == 'smiles':
    str_list = smi_list
    dm = SMILESDataModule(str_list, batch_size = 128, num_workers = num_workers)
else:
    raise ValueError('No such string representation.')

# print some stuff about the data
print(f'You are using the "{string_type}" representation.')
print(f'Number of molecules: {len(str_list)}')
print(f'Length of longest molecule: {dm.len_molecule}')
print(f'Length of alphabet: {dm.len_alphabet}')
print(f'Alphabet: {dm.alphabet}')

# create model
model = LanguageModel(1024, 3, dm.len_alphabet, dm.len_molecule)

# default logger used by trainer
logger = CSVLogger(os.path.join(os.getcwd(), 'trained_models'), name=string_type)

callbacks = [
    ModelCheckpoint(dirpath = logger.log_dir, filename = 'final_model', monitor = 'val_loss', mode = 'min', verbose=True),
    EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=True)
]

trainer = pl.Trainer(
    accelerator = 'gpu', 
    devices = torch.cuda.device_count(),
    logger=logger,
    max_epochs = 100, 
    callbacks = callbacks,
    enable_progress_bar = False
)
trainer.fit(model, dm)
print('Finished training!')

metrics = utils.plot_metrics(os.path.join(logger.log_dir, 'metrics.csv'), logger.log_dir)

print('Done!')