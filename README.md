# Efficient Discovery of Actual Cause using ML Techniques  

## Requirements

Before running this code, ensure you have installed the required packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```


## Data Generation Command

A Python script `data_gen.py` that allows you to generate synthetic data for testing or development purposes. Below are the instructions on how to use this script.

- `--init_n_samples`: Specifies the number of samples you want to generate.
- `--init_n_features`: Specifies the number of features each sample should have.
- `--init_per`: Defines the distribution for the labels, which should be a value between 0 and 1.



```bash
python data_gen.py --init_n_samples=<number of samples> --init_n_features=<number of features> --init_per=<distribution [0,1]>
```

### Output
The generated data saved in `org_data.csv`. 


## Autoencoder

A Python script `autoencoder.py` allows you to construct an autoencoder for dimension reduction. Below are the instructions on how to use this script.

#### Parameters

- `--init_dim`: The dimension you want to reduce your data to.
- `--init_drop_out`: Dropout rate for the neural network; should be between 0 and 1.
- `--init_lr`: The learning rate for training the autoencoder.

```bash
python autoencoder.py --init_dim=<dimension> --init_drop_out=<drop out [0,1]> --init_lr=<learning rate>
```

### Output

The reduced data will be saved in a file named `data.csv`.

## SMT

A Python script `smt.py` allows you to find in dimension reducted data. Below are the instructions on how to use this script.


```bash
python smt.py 
```

### Output

The set of cause or causes will be printed in the output. 
