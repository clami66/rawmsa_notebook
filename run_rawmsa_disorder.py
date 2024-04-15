#!/usr/bin/env python

# Import packages
import tqdm
import numpy as np
import sys, os
import argparse
from pathlib import Path
from time import gmtime, strftime

from Bio import SeqIO
import tensorflow as tf
import numpy as np
from numpy import newaxis
from tqdm import tqdm
import glob
from utils.NetUtils import CustomMetrics, DataProcessor, a3m2aln, aln2num, pad_or_trim_array

def parse_config(config_file):
    try:
        with open(config_file, 'r') as f:
            print(f'\nFound config file at {config_file}')
            config = {}
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=')
                    config[key.strip()] = value.strip()
                    print('=>', key, value)
                    if not os.path.exists(value):
                        print(f'File not found [{value}]. Exiting...')
                        sys.exit()
            print('Filecheck: OK')
            return config
        
    except FileNotFoundError:
        print(f'\nConfig file not found at {config_file}. Exiting...')
        sys.exit()
# Define command-line arguments
parser = argparse.ArgumentParser(description='Identify disordered regions in a protein sequence.')
parser.add_argument('--config', type=str, default='config_eval.txt', help='Path to the config file.')
args = parser.parse_args()

# Parse config file
config = parse_config(args.config)

if __name__ == "__main__":
    # Load parameters from config file
    input_file = config.get('input_file')
    trained_models = config.get('trained_models')
    msa_path = config.get('msa_path')
    out_path = config.get('out_path')

    # Load input file
    records = list(SeqIO.parse(input_file, "fasta"))
    test_list = len(records)
    
    # Generate filestructure for internal datasets
    internal_datasets = {'aln':'data/aln', 
                         'num':'data/num', 
                         'npy':'data/npy'}
    
    for datasets in internal_datasets.values():
        if not os.path.exists(datasets):
            os.makedirs(datasets)

    # Process msa (.a3m) -> .aln -> .num
    print("\nProcessing msas [.a3m -> .aln -> .num]")
    for record in records:
        protein = record.id
        msa_file_path = f"{msa_path}/{protein}.a3m"   # Read a3m
        if os.path.isfile(msa_file_path):   # If msa available
            a3m2aln(msa_file_path, f"{internal_datasets['aln']}/{protein}.aln")  #a3m to aln
            aln2num(f"{internal_datasets['aln']}/{protein}.aln", f"{internal_datasets['num']}/{protein}.num")   # aln2num

            # NUM to feature .npy
            msa_array = np.loadtxt(f"{internal_datasets['num']}/{protein}.num", dtype=int)
            if len(np.shape(msa_array)) == 1:
                msa_array = msa_array[np.newaxis, :]
            else:
                msa_array_ = pad_or_trim_array(msa_array, 3000).T
                data = {'features': msa_array_}   # Fill feature set
                np.save(f"{internal_datasets['npy']}/{protein}.npy", data)
        else:
            print(f'MSA not present at {msa_file_path}')
            print('Run mmseq2')
    
    # Load model and predict
    for model_path in glob.glob(trained_models+'/*'):
        print('\nFound trained model:', model_path)
        ew_model = tf.keras.models.load_model(model_path, custom_objects={"balanced_acc": CustomMetrics.balanced_acc})
        alignment_max_depth = int(model_path.split('_')[-4]) # Required model_name to be in a specific_format

        print(f'Predicting disorder for {len(records)} proteins')
        for record in tqdm(records):
            protein = record.id
            features = np.load(f"{internal_datasets['npy']}/{protein}.npy", allow_pickle=True).item()['features']
            X = features[:, :alignment_max_depth][np.newaxis, :] #.reshape(length * alignment_max_depth)[np.newaxis, :]

            y = ew_model.predict(X,verbose=0)
            print(y)
            logits_positive = np.squeeze(y)[:,1]
            y_binary = np.argmax(np.squeeze(y), axis=-1)

            # Write results to file
            with open(f'{out_path}/{protein}.caid', 'w') as reader: 
                reader.write(f'>{protein}\n')
                for (i, residue, logit, pred) in zip(range(len(record.seq)), list(record.seq), logits_positive, y_binary):
                    reader.write('{}\t{}\t{:.3f}\t{}\n'.format(i + 1, residue, logit, pred)) 
        break