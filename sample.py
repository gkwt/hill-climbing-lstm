import selfies as sf
import rdkit.Chem as Chem
from rdkit.Chem import Draw

import pytorch_lightning as pl

from lstm_climber import SELFIESDataModule, LanguageModel
import lstm_climber.utils as utils

# define parameters
data_path       = 'data/hce.txt'
string_type     = 'selfies'
model_path      = 'trained_models/selfies/version_0/final_model.ckpt'

smi_list, sfs_list = utils.get_lists(data_path, sep=' ')
if string_type == 'selfies':
    str_list = sfs_list
    dm = SELFIESDataModule(str_list, train_ratio = 0.7, batch_size = 128, num_workers = None)
elif string_type == 'smiles':
    str_list = smi_list
    # dm = SMILESDataModule(str_list)
else:
    raise ValueError('No such string representation.')

model = LanguageModel.load_from_checkpoint(model_path)
seed = '[C][S]'
len_seed = sf.len_selfies(seed)
onehot_seed = dm.encode_string(seed, len_seed)

# sample the molecule
sampled_molecules = model.sample(onehot_seed, 10)
new_smi_list = dm.logits_to_smiles(sampled_molecules)
print(new_smi_list)

mol_list = utils.get_mols(new_smi_list)
fig = Draw.MolsToGridImage(mol_list, molsPerRow = 5)
fig.save('sampled_mols_[C][S].png')







