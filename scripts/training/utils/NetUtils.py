import tensorflow as tf
import numpy as np

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
