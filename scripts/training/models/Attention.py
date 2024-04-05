import tensorflow as tf
from tensorflow.keras import layers, models
from utils.NetUtils import CustomMetrics


class GlobalMaxPoolAcrossTime(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalMaxPoolAcrossTime, self).__init__(**kwargs)

    # (Batch_size, time, y, x, channels) -> (Batch_size, time, x, channels)
    def call(self, inputs):
        return K.max(inputs, axis=2, keepdims=False)

class GetTargetSeq(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GetTargetSeq, self).__init__(**kwargs)
        
    def call(self, inputs):
        return tf.slice(inputs, begin=[0, 0, 0], size=[-1, -1, 1])
    
class Tile(tf.keras.layers.Layer):
    def __init__(self, depth=100, **kwargs):
        self.depth=depth
        super(Tile, self).__init__(**kwargs)
        
    def call(self, inputs):
        return tf.tile(inputs, multiples=[1, 1, self.depth])
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
        })
        return config


class Model:
    def __init__(self, config):
        self.alignment_max_depth = int(config.get('alignment_max_depth', 1000))
        self.input_shape = (None, self.alignment_max_depth)
        self.embed_size = int(config.get('embed_size', 16))
        self.rnn_size = int(config.get('bidir_size', 50))
        self.dropfrac = float(config.get('dropfrac', 0.5))
        self.model = self.build_model()

    def encoder(self, inputs):
        # inputs shape should be (batch, length, depth)
        
        # embed target sequence alone
        target = GetTargetSeq()(inputs)
        tiled_target = Tile(depth=self.alignment_max_depth)(target)
        embedded_target = layers.Embedding(26, self.embed_size)(tiled_target)
        embedded_target = layers.Reshape((-1, self.alignment_max_depth*self.embed_size))(embedded_target)

        # embed input MSA separately
        embedded = layers.Embedding(26, self.embed_size)(inputs)
        embedded = layers.Reshape((-1, self.alignment_max_depth*self.embed_size))(embedded)

        embedded = layers.Add()([embedded, embedded_target])
        rnn = layers.Bidirectional(layers.LSTM(self.rnn_size, return_sequences=True), merge_mode="ave")(embedded)
        att = layers.Attention()([rnn, rnn])
        return att

    def decoder(self, inputs):
        rnn = layers.Bidirectional(layers.LSTM(self.rnn_size, return_sequences=True), merge_mode="ave")(inputs)
        att = layers.AdditiveAttention()([inputs, rnn])
        return att

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        output = layers.Dense(2, activation="softmax")(decoded)

        model = models.Model(inputs=inputs, outputs=output)
        return model

    def compile_model(self):
        print('Compiling the model...')
        self.model.compile(optimizer='rmsprop',
                           loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy', CustomMetrics.true_positives, CustomMetrics.positives, CustomMetrics.balanced_acc])
