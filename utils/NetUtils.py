import os, sys

import tensorflow as tf
import numpy as np
from pathlib import Path

class CustomMetrics:
    @staticmethod
    def true_positives(y_true, y_pred):
        correct_preds = tf.cast(tf.equal(tf.reshape(y_true, [-1]), tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)), tf.float32)
        true_pos = tf.cast(tf.reduce_sum(correct_preds * tf.reshape(y_true, [-1])), tf.int64)
        return true_pos

    @staticmethod
    def true_negatives(y_true, y_pred):
        correct_preds = tf.cast(tf.equal(tf.reshape(y_true, [-1]), tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)), tf.float32)
        true_neg = tf.cast(tf.reduce_sum(correct_preds * (1 - tf.reshape(y_true, [-1]))), tf.int64)
        return true_neg

    @staticmethod
    def positives(y_true, y_pred):
        pos = tf.cast(tf.reduce_sum(tf.reshape(y_true, [-1])), tf.int64)
        return pos

    @staticmethod
    def negatives(y_true, y_pred):
        neg = tf.cast(tf.reduce_sum(1 - tf.reshape(y_true, [-1])), tf.int64)
        return neg
    
    @staticmethod
    def balanced_acc(y_true, y_pred):
	#q2balanced = (float(tps)/ps + float(tns)/ns)/2
        correct_preds = tf.cast(tf.equal(tf.reshape(y_true, [-1]), tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)), tf.float32)
        true_pos = tf.cast(tf.reduce_sum(correct_preds * tf.reshape(y_true, [-1])), tf.float32)
        true_neg = tf.cast(tf.reduce_sum(correct_preds * (1 - tf.reshape(y_true, [-1]))), tf.float32)
        pos = tf.cast(tf.reduce_sum(tf.reshape(y_true, [-1])), tf.float32)
        neg = tf.cast(tf.reduce_sum(1 - tf.reshape(y_true, [-1])), tf.float32)
        return (tf.math.divide_no_nan(true_pos, pos) + tf.math.divide_no_nan(true_neg, neg))/2

    def true_positives_np(y_true, y_pred):
        # flatten y_true in case it's in shape (num_samples, 1) instead of (num_samples,)
        correct_preds = np.equal(np.squeeze(y_true.astype(np.float32)), np.argmax(y_pred, axis = -1))
        # correct_preds = K.cast(K.equal(K.cast(y_true, 'int64'), K.cast(K.argmax(y_pred, axis=-1), 'int64')), 'int64')
        true_pos = np.sum(correct_preds * np.squeeze(y_true))
        return true_pos

    def true_negatives_np(y_true, y_pred):
        # flatten y_true in case it's in shape (num_samples, 1) instead of (num_samples,)
        correct_preds = np.equal(np.squeeze(y_true.astype(np.float32)), np.argmax(y_pred, axis = -1))
        # correct_preds = K.cast(K.equal(K.cast(y_true, 'int64'), K.cast(K.argmax(y_pred, axis=-1), 'int64')), 'int64')
        true_neg = np.sum(correct_preds * (1 - np.squeeze(y_true)))
        return true_neg

class DataProcessor:
    def __init__(self, config):
        self.alignment_max_depth = int(config.get('alignment_max_depth', 1000))
        self.data_path = str(config.get('data_path'))

    def process_npy(self, data, alignment_max_depth):
        features, labels = data['features'], data['labels']
        # Process X
        length = features.shape[0]
        if length > 3000:
            length = 3000
        X = features[:length, :alignment_max_depth][np.newaxis, :] #.reshape(length * alignment_max_depth)[np.newaxis, :]

        labels = np.reshape(labels[:length], (1, length, 1))
        return X, labels
    
    def count_steps(self, data_list):
        count = 0
        for target in data_list:
            target = target.rstrip()
            if Path(self.data_path, f"{target}.npy").exists():
                count += 1
        return count

    def generate_inputs_onego(self, alignment_max_depth, data_list):
        for target in data_list:
            target = target.rstrip()
            target_path = Path(self.data_path, f'{target}.npy')
            if target_path.exists():
                data = np.load(target_path, allow_pickle=True).item()
            else:
                continue
            X, labels = self.process_npy(data, alignment_max_depth)

            yield X, labels

def aln2num(aln, output):
    letter_to_number = {'P': 1, 'U': 2, 'C': 3, 'A': 4, 'G': 5, 'S': 6, 'N': 7, 'B': 8, 'D': 9, 'E': 10,
                        'Z': 11, 'Q': 12, 'R': 13, 'K': 14, 'H': 15, 'F': 16, 'Y': 17, 'W': 18, 'M': 19,
                        'L': 20, 'I': 21, 'V': 22, 'T': 23, '-': 24, 'X': 25}

    with open(aln) as input_aln_file, open(output, 'w') as output_file:
        count, limit = 0,  3000
        for line in input_aln_file:
            if count < limit:
                transformed_line = ''
                for char in line.rstrip():
                    number = letter_to_number.get(char, 25)
                    transformed_line += str(number) + ' '
                output_file.write(transformed_line+'\n')
                count += 1
    return None

def a3m2aln(msa, outpath):
    cmd = f"egrep -v '^>' {msa} | sed 's/[a-z]//g'>{outpath}"
    os.system(cmd)
    #print(cmd)
    return None

def pad_or_trim_array(arr, target_shape):
    rows, cols = arr.shape
    if rows < target_shape:
        pad_width = [(0, max(0, target_shape - rows)), (0, 0)]
        return np.pad(arr, pad_width, mode='constant', constant_values=0)
    elif rows > target_shape:
        return arr[:target_shape, :]
    else:
        return arr