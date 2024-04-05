# Import libraries
import os
import math
import h5py
import argparse
from pathlib import Path
from time import gmtime, strftime

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from models import ConvLSTM
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


class DataProcessor:
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
                features, labels = data['features'], data['labels']
            else:
                continue

            # Process X
            length = features.shape[0]
            X_batch = features[:, :alignment_max_depth][np.newaxis, :] #.reshape(length * alignment_max_depth)[np.newaxis, :]

            labels = np.reshape(labels, (1, labels.shape[0], 1))
            yield(X_batch, labels)
    

if __name__ == "__main__":
    # PARAMS
    # Load parameters from config file
    train_file = config.get('train_file')
    test_file = config.get('test_file')
    validation_file = config.get('validation_file')
    msa_tool = config.get('msa_tool')
    data_path = config.get('data_path')
    log_dir = config.get('log_path')
    model_dir = config.get("model_path")
    alignment_max_depth = int(config.get('alignment_max_depth', 1000))
    embed_size = int(config.get('embed_size', 16))
    stage1_depth = int(config.get('stage1_depth', 2))
    conv_depth = int(config.get('conv_depth', 10))
    n_filters = int(config.get('n_filters', 16))
    pool_depth = int(config.get('pool_depth', 10))
    bidir_size = int(config.get('bidir_size', 50))
    stage2_depth = int(config.get('stage2_depth', 2))
    dropfrac = float(config.get('dropfrac', 0.5))
    batch_size = int(config.get('batch_size', 4))
    num_epochs = int(config.get('num_epochs', 100))
    num_cpu = int(config.get('num_cpu', 30))

    # Load train, test, and validation data
    train_list = open(train_file).readlines()
    validate_list = open(validation_file).readlines()

    # INITIALIZE GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU') # unhide potentially hidden GPU
    tf.config.get_visible_devices()

    # INITIALIZE MODELS
    input_shape = (None, alignment_max_depth)
    model = ConvLSTM.Model(input_shape, alignment_max_depth, embed_size, stage1_depth, conv_depth, n_filters, pool_depth, bidir_size, stage2_depth, dropfrac)
    model.compile_model()
    print(model.model.summary())
    #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # TRAINING
    track_history = []
    best_aupr = 0

    Path(log_dir).mkdir(parents=True, exist_ok=True)    # Make log dir
    timestr = strftime("%Y%m%d-%H%M%S")
    with open(f'{log_dir}{timestr}_{msa_tool}_full_{alignment_max_depth}', mode='w') as f:
        f.write(f'epoch, msa_depth, auroc, aupr\n')

    train_steps = DataProcessor.count_steps(train_list)
    print("Training steps:", train_steps)

    for e in range(num_epochs):
        print('Fit, epoch ' + str(e) + ":")
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        history = model.model.fit(DataProcessor.generate_inputs_onego(train_list,alignment_max_depth),
                                steps_per_epoch=train_steps,
                                epochs=1,
                                callbacks=[],
                                use_multiprocessing=False)
        
        print(f'Testing model on {validation_file}, {len(validate_list)} proteins ...')
        labels_all_test = []
        y_all_test = []
        for target in tqdm(validate_list):
            target = target.rstrip()
            data = np.load(f'{data_path}{target}.npy', allow_pickle=True).item()
            features, labels = data['features'], data['labels']

            # Process X
            length = features.shape[0]
            X = features[:, :alignment_max_depth].reshape(length * alignment_max_depth)[np.newaxis, :]

            # Process Y
            labels_ = labels[np.newaxis, :]
            labels_ = np.reshape(labels_, (1, labels_.shape[1], 1))
            y = model.model.predict(X)

            labels_all_test.append(labels_.flatten())
            y_all_test.append(y[0][:,1])

        labels_all_test_arr = np.concatenate(labels_all_test)
        y_all_test_arr = np.concatenate(y_all_test)

        pr, re, _ = precision_recall_curve(labels_all_test_arr, y_all_test_arr)
        aupr = average_precision_score(labels_all_test_arr, y_all_test_arr)
        fpr, tpr, thresholds = roc_curve(labels_all_test_arr, y_all_test_arr, pos_label=1)
        auroc = roc_auc_score(labels_all_test_arr, y_all_test_arr)

        print(f'Epoch {e}, Depth: {alignment_max_depth}, auroc: {auroc}, aupr: {aupr}')

        # Print test metrics
        with open(f'{log_dir}{timestr}_{msa_tool}_full_{alignment_max_depth}', mode='a') as f:
            f.write(f'{e},{alignment_max_depth},{auroc},{aupr}\n')
        f.close()

        # Calculate AUPR and compare with best AUPR
        if aupr > best_aupr:
            # Save the model
            savepath = Path(model_dir, f"best_model_{msa_tool}_full_{alignment_max_depth}_{np.round(aupr,2)}.h5")
            model.model.save(savepath)
            print(f'aupr improved from {best_aupr} to {aupr}, saving model')
            best_aupr = aupr


    
