import argparse

import tensorflow as tf
import numpy as np
from numpy import newaxis
from tqdm import tqdm
import glob
import pickle

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

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

def parse_config(config_file):
    with open(config_file, 'r') as f:
        config = {}
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                config[key.strip()] = value.strip()
        return config

# Define command-line arguments
parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('--config', type=str, default='config_eval.txt', help='Path to the config text file.')
args = parser.parse_args()

# Parse config file
config = parse_config(args.config)
print(config)

if __name__ == "__main__":
    # Load parameters from config file
    test_file = config.get('test_file')
    trained_models = config.get('trained_models')
    data_path = config.get('data_path')

    # Load train, test, and validation data
    test_list = open(test_file).readlines()

    # Initialize result dict
    results={}

    for model_path in glob.glob(trained_models):
        ew_model = tf.keras.models.load_model(model_path, custom_objects={"true_positives": CustomMetrics.true_positives,
                                                                        "true_negatives": CustomMetrics.true_positives,
                                                                        "positives": CustomMetrics.true_positives,
                                                                        "negatives": CustomMetrics.true_positives,
                                                                        "balanced_acc": CustomMetrics.balanced_acc})

        alignment_max_depth = int(model_path.split('depth_')[-1].split('_stage')[0])
        labels_all = []
        y_all = []
        for target in tqdm(test_list):
            target = target.rstrip()
            data = np.load(f'{data_path}{target}.npy', allow_pickle=True).item()
            features, labels = data['features'], data['labels']

            # Process X
            length = features.shape[0]
            X = features[:, :alignment_max_depth].reshape(length * alignment_max_depth)[newaxis, :]

            # Process Y
            labels_ = labels[np.newaxis, :]
            labels_ = np.reshape(labels_, (1, labels_.shape[1], 1))
            y = ew_model.predict(X)

            labels_all.append(labels_.flatten())
            y_all.append(y[0][:,1])

        labels_all_arr = np.concatenate(labels_all)
        y_all_arr = np.concatenate(y_all)

        results[f'rawmsa_{alignment_max_depth}']={'labels':labels_all_arr, 'y_all':y_all_arr}

    with open('filename.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)