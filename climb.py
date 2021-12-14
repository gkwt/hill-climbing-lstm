import os, sys
from typing import List

import torch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import selfies as sf
import rdkit.Chem as Chem
from rdkit.Chem import Draw, Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


from lstm_climber import SELFIESDataModule, SMILESDataModule, LanguageModel
import lstm_climber.utils as utils


def fitness_function(smi: str):
    ''' Return a fitness values.
    '''
    #TODO
    # for debugging logP
    mol = Chem.MolFromSmiles(smi)
    log_P = Descriptors.MolLogP(mol)
    return log_P

# define parameters
data_path       = 'data/hce.txt'
string_type     = 'selfies' # smiles
model_path      = f'trained_models/{string_type}/version_0/final_model.ckpt'
num_workers     = 6

num_generations = 20
samps_per_seed  = 10
num_top         = 5
num_seed_chars  = 5
temperature     = 1.0

smi_list, sfs_list = utils.get_lists(data_path, sep=' ')
if string_type == 'selfies':
    str_list = sfs_list
    dm = SELFIESDataModule(str_list, batch_size = 128, num_workers = num_workers)
elif string_type == 'smiles':
    str_list = smi_list
    dm = SMILESDataModule(str_list, batch_size = 128, num_workers = num_workers)
else:
    raise ValueError('No such string representation.')

### Starting algorithm
# Step 1: Gather data, get the top molecules
new_df = pd.DataFrame({'smiles': smi_list})                 # smiles are canonicalized, no duplicates
new_df['fitness'] = new_df['smiles'].apply(fitness_function)
topk = new_df.nlargest(num_top, 'fitness')
collector = pd.DataFrame(columns = ['smiles', 'fitness', 'generation'])
best_collector = pd.DataFrame(columns = ['smiles', 'fitness', 'generation'])

# prepare topk for sampling
topk['selfies'] = topk['smiles'].apply(sf.encoder)
topk = topk.reset_index(drop = True)

# load model
model = LanguageModel.load_from_checkpoint(model_path)
for gen in range(num_generations):

    # get the starting seeds
    topk['seeds'] = topk[string_type].apply(lambda x: utils.get_nth_char(x, num_seed_chars, string_type))
    # add an additional character if the starting seed becomes stale
    # increase temperature for more random sampling
    if topk['seeds'].duplicated().any():
        num_seed_chars += 1
        temperature += 0.05

    print(f'Generation {gen}:')
    cols = ['smiles', 'selfies'] if string_type == 'selfies' else ['smiles']
    new_df = pd.DataFrame(columns=cols)
    for i, row in topk.iterrows():

        # Step 2: Select starting seeds from top molecules for sampling
        top_str = row['seeds']
        print(f'Sampling from {top_str}: {i+1}/{num_top}')
        onehot_seed = dm.encode_string(top_str)

        # Step 3: Sample from seeds, remove duplicates (total samps = samps_per_seed * num_top)
        num_samps = (i+1)*samps_per_seed - len(new_df)
        while num_samps > 0:
            sampled_molecules = model.sample(onehot_seed, num_samps, temperature = temperature)
            if string_type == 'selfies':
                new_smiles, new_selfies = dm.logits_to_smiles(sampled_molecules, return_selfies=True)
                new = pd.DataFrame({'smiles': new_smiles, 'selfies': new_selfies})
            else:
                new_smiles = dm.logits_to_smiles(sampled_molecules)
                new = pd.DataFrame({'smiles': new_smiles})
            new_df = pd.concat([new_df, new])
            new_df = new_df.drop_duplicates('smiles').dropna()        # drop duplicates and nans
            num_samps = (i+1)*samps_per_seed - len(new_df)
        
    new_df['fitness'] = new_df['smiles'].apply(fitness_function)
    new_df['generation'] = [gen] * len(new_df)
    collector = pd.concat([collector, new_df], ignore_index=True)

    best_in_gen = new_df.nlargest(1, 'fitness')

    best_overall = collector.nlargest(1, 'fitness')
    best_overall['generation'] = gen

    # best_collector = pd.concat([best_collector, new_smi_df.nlargest(1, 'fitness')])
    best_collector = pd.concat([best_collector, best_overall] , ignore_index=True)

    print(f'Best in generation:     {best_in_gen["smiles"].iloc[0]}   {best_in_gen["fitness"].iloc[0]}')
    print(f'Best overall:           {best_overall["smiles"].iloc[0]}   {best_overall["fitness"].iloc[0]}')

    # Step 3: Pick out the next seeds
    topk = collector.nlargest(num_top, 'fitness')
    topk = topk.reset_index(drop = True)
    

sns.lineplot(data=best_collector, x='generation', y='fitness')
plt.savefig('best.png')
plt.close()

sns.lineplot(data=collector, x='generation', y='fitness')
plt.savefig('collected.png')
plt.close()

