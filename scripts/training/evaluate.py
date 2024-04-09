import sys
import argparse
from pathlib import Path
from time import gmtime, strftime

import tensorflow as tf
import numpy as np
from numpy import newaxis
from tqdm import tqdm
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from utils.NetUtils import CustomMetrics, DataProcessor

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

    data_processor = DataProcessor(config)
    test_steps = data_processor.count_steps(test_list)
    print("Training steps:", test_steps)

    # Initialize result dict
    results={}
    timestr = strftime("%Y%m%d-%H%M%S")
    for model_path in glob.glob(trained_models+'/*'):
        if 'Transformer' in model_path:
            continue

        print('\nFound trained model:', model_path)
        ew_model = tf.keras.models.load_model(model_path, custom_objects={"true_positives": CustomMetrics.true_positives,
                                                                        "true_negatives": CustomMetrics.true_positives,
                                                                        "positives": CustomMetrics.true_positives,
                                                                        "negatives": CustomMetrics.true_positives,
                                                                        "balanced_acc": CustomMetrics.balanced_acc})
        print(model_path.split('_'), model_path.split('_')[-4])
        try:
            alignment_max_depth = int(model_path.split('_')[-4])    # This has to be decided on
        except:
            alignment_max_depth = int(model_path.split('_')[7])  
        # msa_tool = str(model_path.split('_')[-4])
        labels_all, y_all = [], []
        
        print(f'Testing {len(test_list)} proteins')
        for target in tqdm(test_list, ncols=100):
            target = target.rstrip()
            target_path = Path(data_path, f"{target}.npy")
            if target_path.exists():
                data = np.load(target_path, allow_pickle=True).item()
            else:
                continue

            X, labels = data_processor.process_npy(data, alignment_max_depth)
            y = ew_model.predict(X,verbose=0)

            labels_all.append(labels.flatten())
            y_all.append(y[0][:, 1])

        labels_all_arr = np.concatenate(labels_all)
        y_all_arr = np.concatenate(y_all)

        results[f"{model_path.split('/')[-1]}"]={'labels':labels_all_arr, 'y_all':y_all_arr}

        del ew_model

    with open(f'{out_path}/{timestr}_evaluation_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(results.keys())
    timestr = strftime("%Y%m%d-%H%M%S")
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
    fig.legend(handles, labels, bbox_to_anchor=(0.75, 0))
    #plt.legend()
    plt.tight_layout()
    fig.savefig(f'{out_path}/{timestr}_evaluation_results.pdf', bbox_inches='tight')