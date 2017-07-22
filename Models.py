import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, LSTM, Convolution2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras import backend as K

def r2(y_true, y_pred):
    SSE = K.sum(K.square(y_true-y_pred))
    SST = K.sum(K.square(y_true-K.mean(y_true)))
    return 1-SSE/SST
    
def fullyconnected_multiple_ELUs():
    input_1 = Input(batch_shape = (None, 60))

    layer1 = Dense(2048, activation='elu')(input_1)
    layer2 = Dense(1024, activation='elu')(layer1)

    layer3 = Dense(512, activation='elu')(layer2)
    layer4 = Dense(256, activation='elu')(layer3)

    layer5 = Dense(128, activation='elu')(layer4)
    layer6 = Dense(64, activation='elu')(layer5)

    layer7 = Dense(32, activation='elu')(layer5)
    last_layer = Dense(16, activation='elu')(layer7)

    output1 = Dense(1, name="sigma")(last_layer)
    output2 = Dense(1, name="mu")(last_layer)
    output3 = Dense(1, name="jump_sigma")(last_layer)
    output4 = Dense(1, name="jump_mu")(last_layer)
    output5 = Dense(1, name="lambda")(last_layer)
    
    feedforward = Model(input = input_1, output=[output1, output2, output3, output4, output5])
    
    feedforward.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2, 'mean_absolute_percentage_error'])
    
    return feedforward

def covnet_multiple_ELUs_8_layers():

    input_1 = Input(shape = (40, 50, 1))

    layer1 = Convolution2D(32, 12, 12, border_mode='same', activation='elu')(input_1)
    layer2 = Convolution2D(32, 12, 12, border_mode='same', activation='elu')(layer1)
    layer3 = MaxPooling2D(pool_size=(2,2))(layer2)

    layer4 = Convolution2D(64, 6, 6, border_mode='same', activation='elu')(layer3)
    layer5 = Convolution2D(64, 6, 6, border_mode='same', activation='elu')(layer4)
    layer6 = MaxPooling2D(pool_size=(2,2))(layer5)

    layer7 = Convolution2D(128, 3, 3, border_mode='same', activation='elu')(layer6)
    layer8 = Convolution2D(128, 3, 3, border_mode='same', activation='elu')(layer7)
    layer9 = MaxPooling2D(pool_size=(2,2))(layer8)

    flatten = Flatten()(layer9)
    last_layer = Dense(256, activation='elu')(flatten)
    output1 = Dense(1, name="sigma")(last_layer)
    output2 = Dense(1, name="mu")(last_layer)
    output3 = Dense(1, name="jump_sigma")(last_layer)
    output4 = Dense(1, name="jump_mu")(last_layer)
    output5 = Dense(1, name="lambda")(last_layer)

    convnet_mo_elu = Model(input = input_1, output=[output1, output2, output3, output4, output5])

    convnet_mo_elu.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error', r2])
    
    return convnet_mo_elu

def covnet_single_ReLUs_6_layers():
    convnet_lambda = Sequential()
    convnet_lambda.add(Conv2D(32, (12, 12), input_shape=(40, 50, 1), padding='same', activation='relu'))
    convnet_lambda.add(Conv2D(32, (12, 12), padding='same', activation='elu'))
    convnet_lambda.add(MaxPooling2D(pool_size=(2,2)))

    convnet_lambda.add(Conv2D(64, (6, 6), padding='same', activation='elu'))
    convnet_lambda.add(Conv2D(64, (6, 6), padding='same', activation='elu'))
    convnet_lambda.add(MaxPooling2D(pool_size=(2,2)))

    convnet_lambda.add(Flatten())
    convnet_lambda.add(Dense(512, activation='relu'))
    convnet_lambda.add(Dense(1))
    
    convnet_lambda.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error', r2])
    
    return convnet_lambda