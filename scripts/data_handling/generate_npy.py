import argparse
import glob
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed

class ProteinProcessor:
    def __init__(self, master_df, target_shape, path_to_num, outpath):
        self.master_df = master_df
        self.target_shape = target_shape
        self.path_to_num = path_to_num
        self.bad_proteins = []
        self.master_dataset = {}
        self.outpath = outpath

    def pad_or_trim_array(self, arr):
        rows, cols = arr.shape
        if rows < self.target_shape:
            pad_width = [(0, max(0, self.target_shape - rows)), (0, 0)]
            return np.pad(arr, pad_width, mode='constant', constant_values=0)
        elif rows > self.target_shape:
            return arr[:self.target_shape, :]
        else:
            return arr

    def process_protein(self, record):
        intact = True
        protein = record.name.split('|')[1]

        try:
            protein_record = self.master_df.get_group(protein).reset_index()
        except KeyError:
            self.bad_proteins.append((protein, 'not in record'))
            return

        if len(record.seq) > 0:
            sequence = record.seq
            disordered_regions = []
            for index, row in protein_record.iterrows():
                try:
                    assert sequence[row.start-1:row.end] == row.region_sequence, f'Mismatch: {row.disprot_id}, {protein}, Index:{index}/{len(protein_record)}, Coord:[{row.start},{row.end}]'
                    disordered_idxs = np.arange(row.start-1, row.end)
                    disordered_regions.append(disordered_idxs)
                except AssertionError as e:
                    self.bad_proteins.append((protein, 'Mismatch'))
                    print(e)
                    intact = False
                    break

            if intact:
                msa_array = np.loadtxt(f'{self.path_to_num}/{protein}.num', dtype=int)
                if len(np.shape(msa_array)) == 1:
                    msa_array = msa_array[np.newaxis, :]
                    #print('Short MSA', protein)
                else:
                    msa_array_ = self.pad_or_trim_array(msa_array).T
                    label_array = np.zeros(msa_array.shape[1], dtype=int)
                    label_array[np.concatenate(disordered_regions)] = 1
                    self.master_dataset[protein] = {'features': msa_array_, 'labels': label_array}
                    np.save(f'{self.outpath}/{protein}.npy', self.master_dataset[protein])
        else:
            print('Bad fasta:', protein)
            self.bad_proteins.append((protein, 'bad_fasta'))

    def process_proteins_concurrently(self, fasta_records):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_protein, record) for record in fasta_records]
            for future in tqdm(as_completed(futures), total=len(futures)):
                pass

class FastaProcessor:
    def __init__(self, fasta_file_path, path_to_num, disprot_tsv, outpath):
        self.fasta_file_path = fasta_file_path
        self.path_to_num = path_to_num
        self.disprot_tsv = disprot_tsv
        self.outpath = outpath

    def process_fasta(self):
        fasta_records = self.read_fasta_file(self.fasta_file_path)
        master_df = pd.read_csv(self.disprot_tsv, sep='\t')
        protein_group = master_df.groupby(by='acc')
        target_shape = 3000

        processor = ProteinProcessor(master_df=protein_group, target_shape=target_shape, path_to_num=self.path_to_num, outpath=self.outpath)
        print(f'Generating .npy files for {len(fasta_records)} proteins')
        print('Output path:', self.outpath)
        for record in tqdm(fasta_records, ncols=50):
            processor.process_protein(record)

    def read_fasta_file(self, file_path):
        """Read FASTA file and return sequences."""
        with open(file_path, "r") as file:
            sequences = list(SeqIO.parse(file, "fasta"))
        return sequences

def parse_args():
    parser = argparse.ArgumentParser(description="Process protein aln to geenrate .npy files for training")
    parser.add_argument("--fasta_file", type=str, help="Path to the FASTA file.")
    parser.add_argument("--path_to_num", type=str, help="Path to .aln files")
    parser.add_argument("--disprot_tsv", type=str, help="disprot_tsv")
    parser.add_argument("--target_msa_depth", type=int, help="target_msa_depth")
    parser.add_argument("--outpath", type=str, help="path to store .npy files")

    return parser.parse_args()

def main():
    args = parse_args()
    fasta_processor = FastaProcessor(args.fasta_file, args.path_to_num, args.disprot_tsv, args.outpath)
    fasta_processor.process_fasta()

if __name__ == "__main__":
    main()
