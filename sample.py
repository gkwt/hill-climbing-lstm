import selfies as sf
import rdkit.Chem as Chem
from rdkit.Chem import Draw

import pytorch_lightning as pl

from lstm_climber import SELFIESDataModule, SMILESDataModule, LanguageModel
import lstm_climber.utils as utils

# define parameters
data_path       = 'data/hce.txt'
string_type     = 'smiles'  # 'selfies'
model_path      = 'trained_models/smiles/version_3/final_model.ckpt'
num_workers     = 6

smi_list, sfs_list = utils.get_lists(data_path, sep=' ')
if string_type == 'selfies':
    str_list = sfs_list
    dm = SELFIESDataModule(str_list, batch_size = 128, num_workers = num_workers)
    seed = '[C]'
elif string_type == 'smiles':
    str_list = smi_list
    dm = SMILESDataModule(str_list, batch_size = 128, num_workers = num_workers)
    seed = 'C'
else:
    raise ValueError('No such string representation.')

model = LanguageModel.load_from_checkpoint(model_path)
onehot_seed = dm.encode_string(seed)

# sample the molecule
sampled_molecules = model.sample(onehot_seed, 1, temperature=1.0)
new_smi_list = dm.logits_to_smiles(sampled_molecules)
print(new_smi_list)

mol_list = utils.get_mols(new_smi_list)
fig = Draw.MolsToGridImage(mol_list, molsPerRow = 5)
fig.save(f'sampled_mols_{seed}.png')







