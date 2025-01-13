#!/usr/bin/env python

# Import packages
import tqdm
import numpy as np
import sys, os
import argparse
from pathlib import Path
from time import gmtime, strftime
import time
import csv

from Bio import SeqIO
import tensorflow as tf
import numpy as np
from numpy import newaxis
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    print(" \n \
                                                _ _                   _           \n \
                                                | (_)                 | |          \n \
    _ __ __ ___      ___ __ ___  ___  __ _   __| |_ ___  ___  _ __ __| | ___ _ __ \n \
    | '__/ _` \ \ /\ / / '_ ` _ \/ __|/ _` | / _` | / __|/ _ \| '__/ _` |/ _ \ '__|\n \
    | | | (_| |\ V  V /| | | | | \__ \ (_| || (_| | \__ \ (_) | | | (_| |  __/ |   \n \
    |_|  \__,_| \_/\_/ |_| |_| |_|___/\__,_| \__,_|_|___/\___/|_|  \__,_|\___|_|   \n \
                                        ______                                    \n \
                                        |______|                                   \n \
                        ")

    # Parse config file
    config = parse_config(args.config)

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
    models = []
    for model_path in glob.glob(trained_models+'/*'):
        alignment_max_depth = int(model_path.split('_')[-4]) # Required model_name to be in a specific_format
        model = tf.keras.models.load_model(model_path, custom_objects={"balanced_acc": CustomMetrics.balanced_acc})
        models.append((model, alignment_max_depth))

    exec_times = []
    print(f'Predicting disorder for {len(records)} proteins')
    for record in records:
        start_time = time.time()
        list_y = []
        for ew_model, alignment_max_depth in models:
            protein = record.id
            features = np.load(f"{internal_datasets['npy']}/{protein}.npy", allow_pickle=True).item()['features']
            X = features[:, :alignment_max_depth][np.newaxis, :] #.reshape(length * alignment_max_depth)[np.newaxis, :]

            y = ew_model.predict(X,verbose=0)
            list_y.append(np.squeeze(y))
            
        mean_y = np.mean(list_y, axis=0)                    # Compute mean logits
        mean_y = tf.nn.softmax(mean_y, axis=1, name=None)   # Apply softmaax on mean logits
        logits_positive = mean_y[:,1]                       # Get positive probs
        y_binary = np.argmax(mean_y, axis=-1)               # Binarize

        # Write results to file
        with open(f'{out_path}/{protein}.caid', 'w') as reader: 
            reader.write(f'>{protein}\n')
            for (i, residue, logit, pred) in zip(range(len(record.seq)), list(record.seq), logits_positive, y_binary):
                reader.write('{}\t{}\t{:.3f}\t{}\n'.format(i + 1, residue, logit, pred)) 
        end_time = time.time()
        exec_times.append(end_time - start_time)
        print(f'{protein} took {end_time - start_time} secs')

        fig = plt.figure(figsize=(10,10))
        #fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(10) 

        # Define subplots with unequal sizes
        ax1 = fig.add_subplot(gs[:8]) 
        ax2 = fig.add_subplot(gs[8]) 
        ax3 = fig.add_subplot(gs[9]) 

        res_1 = np.array(logits_positive).reshape(-1,1)
        res_2 = np.array(y_binary).reshape(-1,1)

        ax1.imshow(X[0].T, aspect='auto', cmap='tab20b')
        ax2.imshow(res_1.T, aspect='auto' , vmin=0, vmax=1)
        ax3.imshow(res_2.T, aspect='auto', cmap='binary')
        plt.tight_layout()
        plt.savefig(f"{out_path}/{protein}.pdf", format="pdf", bbox_inches="tight")

    with open('timings.csv', 'w') as timingfile:
        timingfile.write('sequence,milliseconds\n')
        for record, time_taken in zip(records, exec_times):
            timingfile.write(f'{str(record.id)},{str(int(time_taken * 1000))}\n')  # Convert seconds to milliseconds