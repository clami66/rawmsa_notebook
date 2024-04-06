import argparse
from pathlib import Path

import tensorflow as tf
import numpy as np
from numpy import newaxis
from tqdm import tqdm
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

class DataProcessor:
    @staticmethod
    def process_npy(data, alignment_max_depth):
        features, labels = data['features'], data['labels']
        # Process X
        length = features.shape[0]
        if length > 3000:
            length = 3000
        X = features[:length, :alignment_max_depth][np.newaxis, :] #.reshape(length * alignment_max_depth)[np.newaxis, :]

        labels = np.reshape(labels[:length], (1, length, 1))
        return X, labels
    
    @staticmethod
    def count_steps(data_list):
        count = 0
        for target in data_list:
            target = target.rstrip()
            if Path(data_path, f"{target}.npy").exists():
                count += 1
        return count

    @staticmethod
    def generate_inputs_onego(data_list, alignment_max_depth):
        for target in data_list:
            target = target.rstrip()
            target_path = Path(data_path, f'{target}.npy')
            if target_path.exists():
                data = np.load(target_path, allow_pickle=True).item()
            else:
                continue
            X, labels = DataProcessor.process_npy(data, alignment_max_depth)

            yield X, labels

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
parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument('--config', type=str, default='config_eval.txt', help='Path to the eval config text file.')
args = parser.parse_args()

# Parse config file
config = parse_config(args.config)
print('\nEvaluationg on the following parameters...')
for key, value in config.items():
    print('>>', key, value)

if __name__ == "__main__":
    # Load parameters from config file
    test_file = config.get('test_file')
    trained_models = config.get('trained_models')
    data_path = config.get('data_path')
    out_path = config.get('out_path')

    # Load train, test, and validation data
    test_list = open(test_file).readlines()

    # Initialize result dict
    results={}
    for model_path in glob.glob(trained_models+'/*'):
        if 'mmseqs' in model_path:
            print(model_path)
            continue

        print('\nFound trained model:', model_path)
        print(model_path.split('_'))
        ew_model = tf.keras.models.load_model(model_path, custom_objects={"true_positives": CustomMetrics.true_positives,
                                                                        "true_negatives": CustomMetrics.true_positives,
                                                                        "positives": CustomMetrics.true_positives,
                                                                        "negatives": CustomMetrics.true_positives,
                                                                        "balanced_acc": CustomMetrics.balanced_acc})

        alignment_max_depth = int(model_path.split('_')[-2])    # This has to be decided on
        msa_tool = str(model_path.split('_')[-3])
        labels_all, y_all = [], []
        
        print(f'Testing {len(test_list)} proteins')
        for target in tqdm(test_list, ncols=100):
            target = target.rstrip()
            target_path = Path(data_path, f"{target}.npy")
            if target_path.exists():
                data = np.load(target_path, allow_pickle=True).item()
            else:
                continue

            X, labels = DataProcessor.process_npy(data, alignment_max_depth)
            y = ew_model.predict(X)

            labels_all.append(labels.flatten())
            y_all.append(y[0][:, 1])

        labels_all_arr = np.concatenate(labels_all)
        y_all_arr = np.concatenate(y_all)

        results[f'rawmsa_{alignment_max_depth}']={'labels':labels_all_arr, 'y_all':y_all_arr}

    with open(f'{out_path}/evaluation_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(results.keys())
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    for key in results.keys():
        labels_all_arr, y_all_arr = results[key]['labels'], results[key]['y_all']
        pr, re, _ = precision_recall_curve(labels_all_arr, y_all_arr)
        aupr = np.round(average_precision_score(labels_all_arr, y_all_arr),2)
        fpr, tpr, thresholds = roc_curve(labels_all_arr, y_all_arr, pos_label=1)
        auroc = str(np.round(roc_auc_score(labels_all_arr, y_all_arr),2))

        ax[1].plot(re, pr, label=key.split('/')[-1]+f' ({aupr})')
        ax[0].set_xlabel('FPR')
        ax[0].set_ylabel('TPR')
        ax[0].plot(fpr, tpr, label=key.split('/')[-1]+f' (auroc:{auroc}, aupr: {aupr})')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        #ax[0].legend(fontsize=8)
    #average_precision_score(labels_all_arr, y_all_arr)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.75, 1.15))
    #plt.legend()
    plt.tight_layout()
    fig.savefig(f'{out_path}/evaluation_results.pdf', bbox_inches='tight')