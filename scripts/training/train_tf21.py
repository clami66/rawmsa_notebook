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
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

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

class CustomModel:
    def __init__(self, input_shape, alignment_max_depth, embed_size, stage1_depth, conv_depth, n_filters, pool_depth, bidir_size, stage2_depth, dropfrac):
        self.input_shape = input_shape
        self.alignment_max_depth = alignment_max_depth
        self.embed_size = embed_size
        self.stage1_depth = stage1_depth
        self.conv_depth = conv_depth
        self.n_filters = n_filters
        self.pool_depth = pool_depth
        self.bidir_size = bidir_size
        self.stage2_depth = stage2_depth
        self.dropfrac = dropfrac
        self.model = self.build_model()

    def convnet(self, input_layer, conv_window, filter_size, pool_window, norm):
        stage1 = layers.Conv2D(filter_size, (1, conv_window), activation=None, padding='same')(input_layer)
        stage1 = layers.Activation('relu')(stage1)
        if norm:
            stage1 = layers.BatchNormalization()(stage1)
        output = layers.MaxPooling2D(pool_size=(1, pool_window))(stage1)
        return output

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        embedded = layers.Embedding(26, self.embed_size)(inputs)
        reshaped = layers.Reshape((-1, self.alignment_max_depth, self.embed_size))(embedded)

        convoluted = reshaped
        norm = False
        for i in range(self.stage1_depth):
            convoluted = self.convnet(convoluted, self.conv_depth, self.n_filters, self.pool_depth, norm)

        reshaped_2 = layers.Reshape((-1, self.n_filters*int(self.alignment_max_depth/(self.pool_depth**self.stage1_depth))))(convoluted)

        bidir_output = reshaped_2
        for i in range(self.stage2_depth):
            bidir_output = layers.Bidirectional(layers.LSTM(self.bidir_size, return_sequences=True), merge_mode='ave')(bidir_output)
            bidir_output = layers.Dropout(self.dropfrac)(bidir_output)

        #predictions_ = layers.Bidirectional(layers.LSTM(2, return_sequences=True, activation='tanh'), merge_mode='ave')(bidir_output)
        predictions=Dense(2, activation='softmax')(bidir_output)
        predictions = layers.Activation('softmax')(predictions_)

        model = Model(inputs=inputs, outputs=predictions)
        return model

    def compile_model(self):
        print('Compiling the model...')
        self.model.compile(optimizer='rmsprop',
                           loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy', CustomMetrics.balanced_acc])

class DataProcessor:
    @staticmethod
    def count_steps(data_list):
        count = 0
        for target in data_list:
            target = target.rstrip()
            if Path(f'{data_path}/{target}.npy').exists():
                count += 1

        return count


    @staticmethod
    def generate_inputs_onego(data_list, alignment_max_depth):
        for target in data_list:
            target = target.rstrip()
            try:
                data = np.load(f'{data_path}/{target}.npy', allow_pickle=True).item()
                features, labels = data['features'], data['labels']
            except:
                pass

            # Process X
            length = features.shape[0]
            X_batch = features[:, :alignment_max_depth].reshape(length * alignment_max_depth)[np.newaxis, :]

            # Process Y
            labels_ = labels[np.newaxis, :]
            labels_ = np.reshape(labels_, (1, labels_.shape[1], 1))
            yield(X_batch, labels_)
    

if __name__ == "__main__":
    # PARAMS
    # Load parameters from config file
    train_file = config.get('train_file')
    test_file = config.get('test_file')
    validation_file = config.get('validation_file')
    msa_tool = config.get('msa_tool')
    data_path = config.get('data_path')
    log_dir = config.get('log_path')
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
    input_shape = (None,)
    model = CustomModel(input_shape, alignment_max_depth, embed_size, stage1_depth, conv_depth, n_filters, pool_depth, bidir_size, stage2_depth, dropfrac)
    model.compile_model()
    print(model.model.summary())
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

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
                                callbacks=[tensorboard_callback],
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

        # labels_all_arr, y_all_arr = results[key]['labels'], results[key]['y_all']
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
            model.model.save(f'trained_models/best_model_{msa_tool}_full_{alignment_max_depth}_{np.round(aupr,2)}.h5')
            print(f'aupr improved from {best_aupr} to {aupr}, saving model')
            best_aupr = aupr
        
        # log_dir = f'training_logs/mmseq_msa_all/'
        # Path(log_dir).mkdir(parents=True, exist_ok=True)    # Make log dir

        # with open(f'{log_dir}', mode='w') as f:
        #     f.write(s)
                
        # # Evaluate the model on test data
        # test_metrics = model.model.evaluate(DataProcessor.generate_inputs_onego(test_list, alignment_max_depth),
        #                                      steps=len(test_list),
        #                                      use_multiprocessing=False)
        
        # # Print test metrics
        # print("Test Metrics:")
        # for metric_name, metric_value in zip(model.model.metrics_names, test_metrics):
        #     print(f"{metric_name}: {metric_value}")
    
        #if e%20==0:
    #     np.save(f'results/training_history_legacy_{alignment_max_depth}_{e}.npy', history.history)
    #     track_history.append(np.array(list(history.history.values())).flatten())
    # history_df = pd.DataFrame(np.array(track_history), columns=history.history.keys())
    # history_df.to_csv(f'results/history_df_legacy_{alignment_max_depth}_{num_epochs}.csv')

    
