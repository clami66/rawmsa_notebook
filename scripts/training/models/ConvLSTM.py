from tensorflow.keras import layers, models
from utils.NetUtils import CustomMetrics

class Model:
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

    def convnet(self, input_layer, conv_window, filter_size, pool_window, norm=False):
        stage1 = layers.Conv2D(filter_size, (1, conv_window), activation=None, padding='same')(input_layer)
        stage1 = layers.Activation('relu')(stage1)
        if norm:
            stage1 = layers.BatchNormalization()(stage1)
        output = layers.MaxPooling2D(pool_size=(1, pool_window))(stage1)
        return output

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        embedded = layers.Embedding(26, self.embed_size)(inputs)

        convoluted = embedded
        for i in range(self.stage1_depth):
            convoluted = self.convnet(convoluted, self.conv_depth, self.n_filters, self.pool_depth)

        reshaped_2 = layers.Reshape((-1, self.n_filters*int(self.alignment_max_depth/(self.pool_depth**self.stage1_depth))))(convoluted)

        bidir_output = reshaped_2
        for i in range(self.stage2_depth):
            bidir_output = layers.Bidirectional(layers.LSTM(self.bidir_size, return_sequences=True), merge_mode='ave')(bidir_output)
            bidir_output = layers.Dropout(self.dropfrac)(bidir_output)

        #predictions_ = layers.Bidirectional(layers.LSTM(2, return_sequences=True, activation='tanh'), merge_mode='ave')(bidir_output)
        #predictions = layers.Activation('softmax')(predictions)

        predictions = layers.Dense(2, activation='softmax')(bidir_output)

        model = models.Model(inputs=inputs, outputs=predictions)
        return model

    def compile_model(self):
        print('Compiling the model...')
        self.model.compile(optimizer='rmsprop',
                           loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy', CustomMetrics.balanced_acc])
