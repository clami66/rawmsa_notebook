# Import libraries
import os
import sys
import math
import h5py
import argparse
from pathlib import Path
from time import gmtime, strftime

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn import metrics #import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, balanced_accuracy_score

from models import ConvLSTM, Attention
from utils.NetUtils import CustomMetrics, DataProcessor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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
parser.add_argument('--config', type=str, default='config.txt', help='Path to the config text file.')
args = parser.parse_args()

# Parse config file
config = parse_config(args.config)
print(config)
    
if __name__ == "__main__":
    # PARAMS
    # Load parameters from config file
    model_type = config.get("model_type")
    train_file = config.get('train_file')
    test_file = config.get('test_file') # unused
    validation_file = config.get('validation_file')
    msa_tool = config.get('msa_tool')
    data_path = config.get('data_path')
    log_dir = config.get('log_path')
    model_dir = config.get("model_path")
    alignment_max_depth = int(config.get('alignment_max_depth'))
    num_epochs = int(config.get('num_epochs', 100))
    num_cpu = int(config.get('num_cpu', 30)) # unused

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # Load train, test, and validation data
    train_list = open(train_file).readlines()
    validate_list = open(validation_file).readlines()

    # INITIALIZE MODELS
    input_shape = (None, alignment_max_depth)
    if model_type == "ConvLSTM":
        model = ConvLSTM.Model(config)
    elif model_type == "Attention":
        model = Attention.Model(config)
    # elif model_type =='Transformer':
    #     model = Transformer.Model(config)
    # elif model_type =='ConvLSTM_v2':
    #     model = ConvLSTM_v2.Model(config)
    else:
        print(f"Model type {model_type} doesn't exist")
        sys.exit(1)
    model.compile_model()
    model.model.summary()

    # TRAINING
    best_aupr, best_model_path = 0, 'dummy'

    Path(log_dir).mkdir(parents=True, exist_ok=True)    # Make log dir
    timestr = strftime("%Y%m%d-%H%M%S")
    log_file = f'{log_dir}/{timestr}_{msa_tool}_full_{alignment_max_depth}_{model_type}'
    with open(log_file, mode='w') as f:
        f.write(f'epoch, msa_depth, auroc, aupr\n')

    data_processor = DataProcessor(config)
    train_steps = data_processor.count_steps(train_list)
    print("Training steps:", train_steps)
    cce = tf.keras.losses.SparseCategoricalCrossentropy()

    for e in range(num_epochs):
        print('Fit, epoch ' + str(e) + ":")
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        history = model.model.fit(data_processor.generate_inputs_onego(alignment_max_depth, train_list),
                                steps_per_epoch=train_steps,
                                epochs=1,
                                callbacks=[],
                                use_multiprocessing=False)
        
        print(f'Testing model on {validation_file}, {len(validate_list)} proteins ...')
        labels_all_test = []
        y_all_test = []
        ps = ns = tps = tns = 0

        for target in tqdm(validate_list):
            target = target.rstrip()
            target_path = Path(data_path, f"{target}.npy")
            if target_path.exists():
                data = np.load(target_path, allow_pickle=True).item()
            else:
                continue

            X, labels = DataProcessor.process_npy(data, alignment_max_depth)
            y = model.model.predict(X)

            ps += np.sum(np.squeeze(labels))
            ns += np.sum(1 - np.squeeze(labels))
            tps += CustomMetrics.true_positives_np(labels, y)
            tns += CustomMetrics.true_negatives_np(labels, y)

            labels_all_test.append(labels.flatten())
            y_all_test.append(y[0])

        print(y)
        labels_all_test_arr = np.concatenate(labels_all_test)
        y_all_test_arr = np.concatenate(y_all_test)

        #pr, re, _ = metrics.precision_recall_curve(labels_all_test_arr, y_all_test_arr[:, 1])
        aupr = metrics.average_precision_score(labels_all_test_arr, y_all_test_arr[:, 1])
        #fpr, tpr, thresholds = metrics.roc_curve(labels_all_test_arr, y_all_test_arr[:, 1], pos_label=1)
        auroc = metrics.roc_auc_score(labels_all_test_arr, y_all_test_arr[:, 1])
        balanced_accuracy = (tps/ps + tns/ns) / 2
        val_loss = cce(labels_all_test_arr, y_all_test_arr)
        print(f'Epoch {e}, Depth: {alignment_max_depth:.3f}, loss: {val_loss:.3f}, auroc: {auroc:.3f}, aupr: {aupr:.3f}, balanced acc: {balanced_accuracy:.3f}')

        # Print test metrics
        with open(log_file, mode='a') as f:
            f.write(f'{e},{alignment_max_depth},{auroc},{aupr}\n')
        f.close()

        # Calculate AUPR and compare with best AUPR
        if aupr > best_aupr:
            # Save the model
            savepath = Path(model_dir, f"{timestr}_best_model_{msa_tool}_full_{alignment_max_depth}_{model_type}_ep{e}_{np.round(aupr,2)}.h5")
            model.model.save(savepath)
            print(f'aupr improved from {best_aupr:.4f} to {aupr:.4f}, saving model')
            
            # delete previious best save
            if os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_aupr = aupr
            best_model_path = savepath

    print(f'>>Training finished ({num_epochs} epochs, {alignment_max_depth} depth, {msa_tool} msatool)')
    print('>>Best validation aupr:', best_aupr)
    print('>>Trained model saved at:', best_model_path)


    
