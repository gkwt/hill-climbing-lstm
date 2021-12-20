# Hill Climbing LSTM

Use LSTM coupled with hill climbing algorithm to optimize a certain property in generated molecules.

This code is part of a benchmarking project for molecular discovery using generative models.

## Training the model

Train the model by running

```
python train.py
```

In the `train.py` file, modify the `data_path`, and `string_type` (selfies or smiles) used. 

The `data_path` can be a csv or plain text file. The data will be read using `utils.get_lists()` function. Specify the 
- `data_path`: path to the data file
- `sep (default= ' ')`: delimiter for columns of data in file
- `header (default=None)`: number of rows used as column names
- `smiles_name (default=0)`: name of column for the molecule smiles (if header=None, use smiles_name=0)


## Climbing

Climb the LSTM using

```
python climb.py
```

In the `climb.py` file, modify the `fitness_function`, which takes in a smiles and returns a target value. Specify the `data_path` (see above), the `model_path`, where the trained LSTM is stored, the `string_type` (selfies or smiles), and the `out_path`, where the climbing results are stored.

You can change the sampling strategies by changing the keywords.
- `num_generation`: how many generations of climbing
- `num_best`: top `k` values used after sorting
- `num_randomize`: number of randomized SELFIES generated from top `k` smiles (SMILES LSTM cannot handle randomized smiles, so the smiles are just duplicated `num_randomize` times)
- `samps_per_seed`: number of times each of the randomized seeds are sampled
- `temperature`: controls randomness of sampling (larger is more random)
- `retrain`: boolean that specifies whether the model is retrained on the new sampled molecules per generation

Note that the total number of sampled molecules is `num_best * num_randomize * samps_per_seed`.



