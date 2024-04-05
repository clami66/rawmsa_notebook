import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
import pandas as pd

import pandas as pd
from Bio.SeqRecord import SeqRecord
import os
from io import StringIO

from concurrent.futures import ThreadPoolExecutor
from Bio import SeqIO
from Bio import ExPASy
#from tqdm import tqdm

master_df = pd.read_csv('/proj/wallner/users/x_yogka/disorder_prediction/data/DisProt release_2023_12 with_ambiguous_evidences.tsv', sep='\t')
master_df


def pad_or_trim_array(arr, target_shape):
    rows, cols = arr.shape
    if rows < target_shape:
        pad_width = [(0, max(0, target_shape - rows)), (0, 0)]
        return np.pad(arr, pad_width, mode='constant', constant_values=0)
    elif rows > target_shape:
        return arr[:target_shape, :]
    else:
        return arr
        
def read_fasta_file(file_path):
    """Read FASTA file and return sequences."""
    with open(file_path, "r") as file:
        sequences = list(SeqIO.parse(file, "fasta"))
    return sequences


basepath = '/proj/wallner/users/x_yogka/disorder_prediction_v2/data/'
fasta_file_path = f'{basepath}merged_fastas.fasta'
fasta_records = read_fasta_file(fasta_file_path)

bad_proteins = []
master_dataset = {}

num_path = f'{basepath}msas_hhblits/'
master_df = pd.read_csv(f'{basepath}DisProt_release_2023_12_with_ambiguous_evidences.tsv', sep='\t')
protein_group = master_df.groupby(by='acc')
target_shape = 3000

for i, record in tqdm(enumerate(fasta_records)):
    intact=True
    protein = record.name.split('|')[1]

    if protein in protein_
    try:
        protein_record = protein_group.get_group(protein).reset_index()
    except:
        bad_proteins.append((protein, 'not in record'))
        continue

    if len(record.seq)>0:
        sequence = record.seq
        # Primary check for agreement between to distprot data and Uniprot sequence
        disordered_regions = []
        for index, row in protein_record.iterrows():
            try:
                assert sequence[row.start-1:row.end]==row.region_sequence, f'Mismatch: {row.disprot_id}, {protein}, Index:{index}/{len(protein_record)}, Coord:[{row.start},{row.end}]'
                disordered_idxs = np.arange(row.start-1, row.end)
                disordered_regions.append(disordered_idxs)
            except AssertionError as e:
                bad_proteins.append((protein,'Mismatch'))
                print(e)
                intact=False
                break
        
        # Generate feature and labels
        if intact:
            msa_array = np.loadtxt(f'{num_path}{protein}.aln.num', dtype=int)     # Read MSA.num file as array
            if len(np.shape(msa_array))==1:
                msa_array = msa_array[np.newaxis,:]
                print('Short MSA', protein)
            else:
                msa_array_ = pad_or_trim_array(msa_array, target_shape).T
                label_array = np.zeros(msa_array.shape[1], dtype=int)  # Initialize empty label array
                label_array[np.concatenate(disordered_regions)]=1
                master_dataset[protein]={'features': msa_array_, 'labels':label_array}   # Fill feature set
                np.save(f'data/processed_data_hhblits/{protein}.npy', master_dataset[protein])
                # else:
                #     bad_proteins.append((protein,'Short MSA (1)'))

    else:
        print('Bad fasta:', protein)
        bad_proteins.append((protein, 'bad_fasta'))
        continue
print(f'\nFlagged {len(bad_proteins)}/{len(master_df.acc.unique())} proteins')
#protein_slice