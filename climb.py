import os, sys
from typing import List
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

import torch
import pytorch_lightning as pl
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
from lstm_climber.datamodules import SMILESDataset, SELFIESDataset
import lstm_climber.utils as utils


def fitness_function(smi: str):
    ''' Return a fitness values.
    '''
    #TODO
    # for debugging logP
    try:
        mol = Chem.MolFromSmiles(smi)
        log_P = Descriptors.MolLogP(mol)
        return log_P
    except:
        return -100.0

# define parameters
data_path       = 'data/hce.txt'
string_type     = 'smiles' # smiles
model_path      = f'trained_models/{string_type}/version_0/final_model.ckpt'
out_path        = f'RESULTS_retrain_{string_type}'
num_workers     = 6

# control sampling
# total number of samples = num_best * samps_per_seed * num_randomize
num_generations = 10
num_best        = 2             # number of top molecules to generate seeds
num_randomize   = 5             # number of randomized smiles
samps_per_seed  = 50            # number of samples for each seed
num_seed_chars  = None          # number of initial characters to sample from
                                # if none, sample characters 1/4 - 3/4 of strings
temperature     = 1.1           # < 1.0 is less random, > 1.0 is more random
retrain         = True          # retrain the network on new molecules

print(f'Total numbers searched per iteration: {num_randomize * num_best * samps_per_seed}')

# get original dataset, with datamodule classes
smi_list, sfs_list = utils.get_lists(data_path, sep=' ')
if string_type == 'selfies':
    str_list = sfs_list
    dm = SELFIESDataModule(str_list, batch_size = 128, num_workers = num_workers)
elif string_type == 'smiles':
    str_list = smi_list
    dm = SMILESDataModule(str_list, batch_size = 128, num_workers = num_workers)
else:
    raise ValueError('No such string representation.')

# create folder for results
if not os.path.isdir(out_path):
    os.mkdir(out_path)

### Starting algorithm
# Step 1: Gather data, get the top molecules
start_df = pd.DataFrame({'smiles': smi_list})        # smiles are canonicalized, no duplicates
start_df['fitness'] = start_df['smiles'].apply(fitness_function)
topk = start_df.nlargest(num_best, 'fitness')
original_best = topk.iloc[0]['fitness']

# load model and collector
model = LanguageModel.load_from_checkpoint(model_path)
collector = pd.DataFrame(columns = ['smiles', 'fitness', 'generation'])
best_collector = pd.DataFrame(columns = ['smiles', 'fitness', 'generation'])

for gen in range(num_generations):

    # load previous model
    if gen > 0 and retrain:
        model = LanguageModel.load_from_checkpoint(os.path.join(out_path, f'{gen-1}_model.ckpt'))

    print(f'Generation {gen}:')
    cols = ['smiles', 'selfies'] if string_type == 'selfies' else ['smiles']
    new_df = pd.DataFrame(columns=cols)

    # Step 1: Create seeds
    rand_string = []
    for i, row in topk.iterrows():      # loop through top molecules
        if string_type == 'selfies':
            # randomize the smiles before encoding to selfies to increase diversity
            # make sure the new selfies can be encoded by the datamodule
            rand_sfs = []
            while len(rand_sfs) < num_randomize:
                smi_list = utils.randomize_smiles(row['smiles'], num_randomize)
                sfs_list = [sf.encoder(s) for s in smi_list]
                try:
                    _ = [dm.encode_string(s) for s in sfs_list]
                    rand_sfs.extend(sfs_list)
                except:
                    print('Error with randomized SELFIES encoding.')
            rand_string.extend(rand_sfs)
        else:
            rand_string.extend([row['smiles']]*num_randomize)

    for i, s in enumerate(rand_string):

        # Step 2: Select starting seeds from top molecules for sampling
        if num_seed_chars is None:
            frac = np.random.rand()/2.0 + 0.25      # from 0.25 to 0.75
            len_fn = sf.len_selfies if string_type == 'selfies' else len
            seed = utils.get_n_char(s, int(len_fn(s) * frac), string_type)
        else:
            seed = utils.get_n_char(s, num_seed_chars, string_type)
        onehot_seed = dm.encode_string(seed)
        print(f'Sampling from {seed}: {i+1}/{len(rand_string)}')

        # Step 3: Sample from seeds, KEEP duplicates
        num_samps = (i+1) * samps_per_seed - len(new_df)
        while num_samps > 0:
            sampled_molecules = model.sample(onehot_seed, num_samps, temperature = temperature)
            if string_type == 'selfies':
                new_smiles, new_selfies = dm.logits_to_smiles(sampled_molecules, return_selfies=True)
                new = pd.DataFrame({'smiles': new_smiles, 'selfies': new_selfies})
            else:
                new_smiles = dm.logits_to_smiles(sampled_molecules, canonicalize=False)
                new = pd.DataFrame({'smiles': new_smiles})
            new_df = pd.concat([new_df, new])
            # drop duplicates within a generation, preserve NaNs/invalid smiles (comment to turn off)
            # new_df = new_df[ (~new_df.duplicated('smiles')) | (new_df['smiles'].isnull())]      
            num_samps = (i+1) * samps_per_seed - len(new_df)
            # import pdb; pdb.set_trace()

    # Step 4: Calculate new fitnesses and gather results
    new_df['fitness'] = new_df['smiles'].apply(fitness_function)
    new_df['generation'] = [gen] * len(new_df)
    collector = pd.concat([collector, new_df], ignore_index=True)

    best_in_gen = new_df.nlargest(1, 'fitness')
    best_overall = collector.nlargest(1, 'fitness')
    best_overall['generation'] = gen
    best_collector = pd.concat([best_collector, best_overall] , ignore_index=True)

    print(f'Best in generation:     {best_in_gen["smiles"].iloc[0]}   {best_in_gen["fitness"].iloc[0]}')
    print(f'Best overall:           {best_overall["smiles"].iloc[0]}   {best_overall["fitness"].iloc[0]}')

    topk = collector.nlargest(num_best, 'fitness')
    topk = topk.reset_index(drop = True)

    # Step 5: Retrain the network on the new molecules if desired
    if retrain:
        # create new dataset depending on string type
        train_strings = new_df[string_type].dropna().tolist()          # remove invalid molecules before training
        if string_type == 'selfies':
            dataset = SELFIESDataset(train_strings, dm.vocab, dm.len_molecule)
        else:
            dataset = SMILESDataset(train_strings, dm.vocab, dm.len_molecule)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size = dm.batch_size)

        # set model parameters
        model.verbose = True
        model.learning_rate = 1e-4          # use a smaller learning rate for fine-tuning
        
        if torch.cuda.is_available():
            trainer = pl.Trainer(
                accelerator = 'gpu',
                devices = torch.cuda.device_count(),
                enable_checkpointing=False,
                logger=False, 
                max_epochs=10, 
                enable_progress_bar=False,
                num_sanity_val_steps = 0
            )
        else:
            trainer = pl.Trainer(
                accelerator = 'cpu',
                enable_checkpointing=False,
                logger=False, 
                max_epochs=10, 
                enable_progress_bar=False,
                num_sanity_val_steps = 0
            )
        
        # fit the model
        trainer.fit(model, train_loader)
        trainer.validate(model, train_loader)
        trainer.save_checkpoint(os.path.join(out_path, f'{gen}_model.ckpt'))

# plot the run
sns.lineplot(data=best_collector, x='generation', y='fitness')
plt.plot([0, num_generations - 1], [original_best, original_best])
plt.xlim([0, num_generations - 1])
plt.savefig(os.path.join(out_path, 'best_trace.png'))
plt.close()

collector.to_csv(os.path.join(out_path, 'all_results.csv'), index=False)

