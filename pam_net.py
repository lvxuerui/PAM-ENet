import tensorflow as tf
import os
import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Flatten, Dense, Input, LSTM, Reshape, Conv2D, GRU
from keras import backend as K
import numpy as np
from keras.utils import np_utils


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W_b = self.add_weight(name='W_b',
                                   shape=tf.TensorShape(1),
                                   initializer='zeros',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        out1 = Reshape((-1, 1))(inputs[0])
        out2 = Reshape((-1, 1))(inputs[1])
        print('out12: ', out2.shape, out1.shape)
        # out1 = keras.layers.Lambda(lambda out1: self.W_a * out1)(out1)
        # out2 = keras.layers.Lambda(lambda out2: (1 - self.W_a) * out2)(out2)
        out3 = keras.layers.concatenate([out1, out2], axis=1)

        out3_r = keras.layers.Reshape((1, -1))(out3)  # 1*32
        out3_d1 = keras.layers.Dot(axes=[2, 1])([out3, out3_r])
        out3_S1 = keras.layers.Activation('softmax')(out3_d1)
        out3_S1 = keras.layers.Dot(axes=[2, 1])([out3_S1, out3])

        out3 = keras.layers.Lambda(lambda out3: self.W_b * out3)(out3)
        out3_S1 = keras.layers.Lambda(lambda out3_S1: (1 - self.W_b) * out3_S1)(out3_S1)

        out3 = keras.layers.Lambda(lambda A: (A[0] + A[1]))([out3, out3_S1])
        print('123: ', out3.shape)
        return out3

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return (input_shape[0][0], self.output_dim, 1)


def gru_cnn_attention(L, I, S, k1, k2, number_of_lstm, alpha):
    # cnn_gru
    input_lmd = Input(shape=(k1, L, 1), name='input_lmd')
    input_tid = Input(shape=(k2, I, 1), name='input_tid')
    input_tsd = Input(shape=(k2, S, 1), name='input_tsd')

    # upsampling
    upsampling1 = keras.layers.UpSampling2D(size=(2, 2))(input_lmd)
    # print(upsampling1.shape)
    # build the convolutional block
    conv_first1 = Conv2D(32, (1, alpha*8), strides=(1, 8))(upsampling1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (8, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (8, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.MaxPool2D((2, 1), (2, 1))(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.MaxPool2D(2, 2)(conv_first1)
    beta = int((8*(10-alpha))/8) + 1
    print("alpha, beta = ", alpha, ", ", beta)
    conv_first1 = Conv2D(32, (3, int(beta/2)))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Dropout(0.5)(conv_first1)

    conv_flatten = Flatten()(conv_first1)
    out1 = Dense(number_of_lstm, activation='relu')(conv_flatten)
    out1 = Reshape((-1, 1))(out1)

    gru1 = Reshape((k2, -1))(input_tsd)
    gru1 = GRU(number_of_lstm, activation='relu')(gru1)

    gru2 = Reshape((k2, -1))(input_tid)
    gru2 = GRU(number_of_lstm, activation='relu')(gru2)

    out2 = AttentionLayer(gru1.shape[1]+gru2.shape[1])([gru1, gru2])
    out2 = Flatten()(out2)
    out2 = Dense(16, activation='relu')(out2)

    out3 = AttentionLayer(number_of_lstm+number_of_lstm)([out1, out2])

    out3 = Flatten()(out3)
    out3 = Dense(3, activation='softmax')(out3)
    model = Model(inputs=[input_lmd, input_tid, input_tsd], outputs=out3)

    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model


def gru_40(L, n_past):
    # cnn_gru
    input_lmd = Input(shape=(n_past, L, 1), name='input_lmd')
    gru2 = Reshape((n_past, -1))(input_lmd)
    gru2 = GRU(16, activation='relu')(gru2)
    out3 = Dense(3, activation='softmax')(gru2)
    model = Model(inputs=input_lmd, outputs=out3)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model


def prepare_x(data):
    # 获取已标准化的限价委托单, 前40个特征
    df1 = data[:40, :].T
    return np.array(df1)


def prepare_x2(data):
    # 获取时间不敏感特征和时间敏感特征,后104个特征
    df1 = data[40:144, :].T
    print(len(df1))
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].T
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY


from keras.models import load_model

pam_enet = gru_40(40, n_past)

# local path
data_path = '../BenchmarkDatasets/NoAuction/'
print('setup:0.9, 0.1')
dec_train = np.loadtxt(data_path + '1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_9.txt')
dec_test3 = np.loadtxt(data_path + '1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')



# extract limit order book data from the FI-2010 dataset for 40 features
train_40_lob = prepare_x(dec_train)[:train_num]
if setup != 1:
    test_40_lob = prepare_x(dec_test)[:]
test_40_lob2 = prepare_x(dec_test3)[:]


train_104_lob = prepare_x2(dec_train)[n_past-n_past_gru:n_past-n_past_gru+100]
if setup != 1:
    test_104_lob = prepare_x2(dec_test)[n_past-n_past_gru:n_past-n_past_gru+100]
test_104_lob2 = prepare_x2(dec_test3)[n_past-n_past_gru:n_past-n_past_gru+100]

# extract label from the FI-2010 dataset
train_label = get_label(dec_train)[:train_num]
test_label2 = get_label(dec_test3)[:]
print(train_40_lob.shape, train_104_lob.shape, train_label.shape)
dec_train = []
dec_test = []
dec_test3 = []


# prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon.
trainX_40_CNN, trainY_CNN = data_classification(train_40_lob, train_label, T=n_past)
trainX_104_CNN, trainY_CNN2 = data_classification(train_104_lob, train_label, T=n_past_gru)
trainY_CNN = trainY_CNN[:, n_future-1] - 1
trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)
print(trainX_40_CNN.shape, trainX_104_CNN.shape, trainY_CNN.shape)
trainX_CNN1 = trainX_40_CNN
trainX_CNN2 = trainX_104_CNN
trainX_40_CNN = []
trainX_104_CNN = []


testX_40_CNN2, testY_CNN2 = data_classification(test_40_lob2, test_label2, T=n_past)
testX_104_CNN2, testY_CNN22 = data_classification(test_104_lob2, test_label2, T=n_past_gru)
testY_CNN22=[]
testY_CNN2 = testY_CNN2[:, n_future-1] - 1
testY_CNN2 = np_utils.to_categorical(testY_CNN2, 3)
testX_CNN2_1 = testX_40_CNN2
testX_CNN2_2 = testX_104_CNN2
testX_40_CNN2 = []
testX_104_CNN2 = []
print(testX_CNN2_2.shape, testY_CNN2.shape)


import argparse
from keras.callbacks import ModelCheckpoint

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="./weights")
args = vars(ap.parse_args())

checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss", mode="min",
                             save_best_only=True, verbose=1)
callbacks = [checkpoint]

print("[INFO] training our network")

pam_enet.fit(x={'input_lmd': trainX_CNN1,
                'input_tid': trainX_CNN2[:, :, :48],
                'input_tsd': trainX_CNN2[:, :, 48:]},
            y=trainY_CNN, batch_size=batch_size, epochs=epochs, class_weight='auto',
            validation_split=0.1, callbacks=callbacks, verbose=1, shuffle=True)

model = load_model(model_name+'_' + str(n_future) + '0_set' + str(setup) + '.h5',
                   custom_objects={'AttentionLayer': AttentionLayer})


print('choose best')
pred = model.predict(x={'input_lmd': testX_CNN2_1,
                        'input_tid': testX_CNN2_2[:, :, :48],
                        'input_tsd': testX_CNN2_2[:, :, 48:]})


