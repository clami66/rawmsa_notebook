import unittest
import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from generate_npy import ProteinProcessor

class TestProteinProcessor(unittest.TestCase):
    def setUp(self):
        # Setting up test data
        self.master_df = pd.DataFrame({
            'acc': ['protein1', 'protein2'],
            'start': [1, 3],
            'end': [2, 5],
            'region_sequence': ['LM', 'AC']
        })
        self.fasta_record = SeqRecord("ACHKLMG")
        self.target_shape = 10
        self.path_to_num = "path/to/num"
        self.outpath = "path/to/output"

    def test_pad_or_trim_array(self):
        # Test padding when array is smaller than target_shape
        processor = ProteinProcessor(self.master_df, self.target_shape, self.path_to_num, self.outpath)
        input_array = np.array([[1, 2], [3, 4]])
        padded_array = processor.pad_or_trim_array(input_array)
        expected_padded_array = np.pad(input_array, [(0, 8), (0, 0)], mode='constant', constant_values=0)
        self.assertTrue(np.array_equal(padded_array, expected_padded_array))

        # Test trimming when array is larger than target_shape
        input_array = np.random.rand(15, 5)
        trimmed_array = processor.pad_or_trim_array(input_array)
        expected_trimmed_array = input_array[:10, :]
        self.assertTrue(np.array_equal(trimmed_array, expected_trimmed_array))

        # Test no padding or trimming when array shape matches target_shape
        input_array = np.random.rand(10, 5)
        output_array = processor.pad_or_trim_array(input_array)
        self.assertTrue(np.array_equal(output_array, input_array))

    def test_process_protein(self):
        processor = ProteinProcessor(self.master_df, self.target_shape, self.path_to_num, self.outpath)
        processor.process_protein(self.fasta_record)  # Assuming record processing doesn't raise errors
        # Assert some conditions here based on the logic inside process_protein method

    # Add more test methods for other functionalities as needed

if __name__ == '__main__':
    unittest.main()
