import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, LSTM, MaxPooling2D, Input, merge
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras import backend as K

def r2(y_true, y_pred):
    """
    returns the correlation coefficient of y_pred against y_true.

    :param y_true: the true values (independent variable)
    :param y_pred: the predicted values (dependent variable)
    """
    
    SSE = K.sum(K.square(y_true-y_pred))
    SST = K.sum(K.square(y_true-K.mean(y_true)))
    
    return 1-SSE/SST
    
def fullyconnected_multiple_ELUs():
    """
    returns a 9-layer fully connected architecture (with ELU activation units) as a Keras Model, with outputs for all the parameters of the Merton Jump Diffusion stochastic process (sigma, mu, jump_sigma, jump_mu, lambda).
    
    .. note:: be sure to order the array of desired output parameters as follows: sigma, mu, jump_sigma, jump_mu, lambda.  Each their own array of desired values for every simulated sample path.
    """
    
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

def fullyconnected_multiple_ReLUs():
    """
    returns a 9-layer fully connected architecture (with ReLU activation units) as a Keras Model, with outputs for all the parameters of the Merton Jump Diffusion stochastic process (sigma, mu, jump_sigma, jump_mu, lambda).
    
    .. note:: be sure to order the array of desired output parameters as follows: sigma, mu, jump_sigma, jump_mu, lambda.  Each their own array of desired values for every simulated sample path.
    """
    
    input_1 = Input(batch_shape = (None, 60))

    layer1 = Dense(2048, activation='relu')(input_1)
    layer2 = Dense(1024, activation='relu')(layer1)

    layer3 = Dense(512, activation='relu')(layer2)
    layer4 = Dense(256, activation='relu')(layer3)

    layer5 = Dense(128, activation='relu')(layer4)
    layer6 = Dense(64, activation='relu')(layer5)

    layer7 = Dense(32, activation='relu')(layer5)
    last_layer = Dense(16, activation='relu')(layer7)

    output1 = Dense(1, name="sigma")(last_layer)
    output2 = Dense(1, name="mu")(last_layer)
    output3 = Dense(1, name="jump_sigma")(last_layer)
    output4 = Dense(1, name="jump_mu")(last_layer)
    output5 = Dense(1, name="lambda")(last_layer)
    
    feedforward = Model(input = input_1, output=[output1, output2, output3, output4, output5])
    
    feedforward.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2, 'mean_absolute_percentage_error'])
    
    return feedforward

def covnet_multiple_ELUs_8_layers(r_squared = True):
    """
    returns an 8-layer convolutional architecture (with ELU activation units) as a Keras Model, with outputs for all the parameters of the Merton Jump Diffusion stochastic process (sigma, mu, jump_sigma, jump_mu, lambda).
    
    .. note:: be sure to order the array of desired output parameters as follows: sigma, mu, jump_sigma, jump_mu, lambda.  Each their own array of desired values for every simulated sample path.
    """
    
    input_1 = Input(shape = (40, 50, 1))

    layer1 = Conv2D(32, (12, 12), padding='same', activation='elu')(input_1)
    layer2 = Conv2D(32, (12, 12), padding='same', activation='elu')(layer1)
    layer3 = MaxPooling2D(pool_size=(2,2))(layer2)

    layer4 = Conv2D(64, (6, 6), padding='same', activation='elu')(layer3)
    layer5 = Conv2D(64, (6, 6), padding='same', activation='elu')(layer4)
    layer6 = MaxPooling2D(pool_size=(2,2))(layer5)

    layer7 = Conv2D(128, (3, 3), padding='same', activation='elu')(layer6)
    layer8 = Conv2D(128, (3, 3), padding='same', activation='elu')(layer7)
    layer9 = MaxPooling2D(pool_size=(2,2))(layer8)

    flatten = Flatten()(layer9)
    last_layer = Dense(256, activation='elu')(flatten)
    output1 = Dense(1, name="sigma")(last_layer)
    output2 = Dense(1, name="mu")(last_layer)
    output3 = Dense(1, name="jump_sigma")(last_layer)
    output4 = Dense(1, name="jump_mu")(last_layer)
    output5 = Dense(1, name="lambda")(last_layer)

    convnet_mo_elu = Model(input = input_1, output=[output1, output2, output3, output4, output5])

    if r_squared:
        convnet_mo_elu.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error', r2])
    else:
        convnet_mo_elu.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error'])
    
    return convnet_mo_elu

def covnet_multiple_ELUs_10_layers(r_squared = True):
    """
    returns an 8-layer convolutional architecture (with ELU activation units) as a Keras Model, with outputs for all the parameters of the Merton Jump Diffusion stochastic process (sigma, mu, jump_sigma, jump_mu, lambda).
    
    .. note:: be sure to order the array of desired output parameters as follows: sigma, mu, jump_sigma, jump_mu, lambda.  Each their own array of desired values for every simulated sample path.
    """
    
    input_1 = Input(shape = (40, 50, 1))

    layer1 = Conv2D(32, (12, 12), padding='same', activation='elu')(input_1)
    layer2 = Conv2D(32, (12, 12), padding='same', activation='elu')(layer1)
    layer3 = MaxPooling2D(pool_size=(2,2))(layer2)

    layer4 = Conv2D(64, (6, 6), padding='same', activation='elu')(layer3)
    layer5 = Conv2D(64, (6, 6), padding='same', activation='elu')(layer4)
    layer6 = MaxPooling2D(pool_size=(2,2))(layer5)

    layer7 = Conv2D(128, (3, 3), padding='same', activation='elu')(layer6)
    layer8 = Conv2D(128, (3, 3), padding='same', activation='elu')(layer7)
    layer9 = MaxPooling2D(pool_size=(2,2))(layer8)

    flatten = Flatten()(layer9)
    
    output1_layer1 = Dense(32, activation='elu')(flatten)
    output1_layer2 = Dense(16, activation='elu')(output1_layer1)
    output1 = Dense(1, name="sigma")(output1_layer2)
    
    output2_layer1 = Dense(32, activation='elu')(flatten)
    output2_layer2 = Dense(16, activation='elu')(output2_layer1)
    output2 = Dense(1, name="mu")(output2_layer2)
    
    output3_layer1 = Dense(32, activation='elu')(flatten)
    output3_layer2 = Dense(16, activation='elu')(output3_layer1)
    output3 = Dense(1, name="jump_sigma")(output3_layer2)
    
    output4_layer1 = Dense(32, activation='elu')(flatten)
    output4_layer2 = Dense(16, activation='elu')(output4_layer1)
    output4 = Dense(1, name="jump_mu")(output4_layer2)
    
    output5_layer1 = Dense(32, activation='elu')(flatten)
    output5_layer2 = Dense(16, activation='elu')(output5_layer1)
    output5 = Dense(1, name="lambda")(output5_layer2)

    convnet_mo_elu = Model(input = input_1, output=[output1, output2, output3, output4, output5])

    if r_squared:
        convnet_mo_elu.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error', r2])
    else:
        convnet_mo_elu.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error'])
    
    return convnet_mo_elu



def covnet_multiple_ELUs_10_layers_dilated(r_squared = True):
    """
    returns an 8-layer convolutional architecture (with ELU activation units) as a Keras Model, with outputs for all the parameters of the Merton Jump Diffusion stochastic process (sigma, mu, jump_sigma, jump_mu, lambda).
    
    .. note:: be sure to order the array of desired output parameters as follows: sigma, mu, jump_sigma, jump_mu, lambda.  Each their own array of desired values for every simulated sample path.
    """
    
    input_1 = Input(shape = (40, 50, 1))

    layer1 = Conv2D(32, (4, 4), padding='same', activation='elu')(input_1)
    dilated_layer = Conv2D(32, (4, 4), padding='same', dilation_rate=(3, 3), activation='elu')(input_1)
    
    merged = merge([layer1, dilated_layer], mode='concat',concat_axis=1) #or 'concat' or 'mul'
    
    layer2 = Conv2D(32, (12, 12), padding='same', activation='elu')(merged)
    layer3 = MaxPooling2D(pool_size=(2,2))(layer2)

    layer4 = Conv2D(64, (6, 6), padding='same', activation='elu')(layer3)
    layer5 = Conv2D(64, (6, 6), padding='same', activation='elu')(layer4)
    layer6 = MaxPooling2D(pool_size=(2,2))(layer5)

    layer7 = Conv2D(128, (3, 3), padding='same', activation='elu')(layer6)
    layer8 = Conv2D(128, (3, 3), padding='same', activation='elu')(layer7)
    layer9 = MaxPooling2D(pool_size=(2,2))(layer8)

    flatten = Flatten()(layer9)
    
    output1_layer1 = Dense(32, activation='elu')(flatten)
    output1_layer2 = Dense(16, activation='elu')(output1_layer1)
    output1 = Dense(1, name="sigma")(output1_layer2)
    
    output2_layer1 = Dense(32, activation='elu')(flatten)
    output2_layer2 = Dense(16, activation='elu')(output2_layer1)
    output2 = Dense(1, name="mu")(output2_layer2)
    
    output3_layer1 = Dense(32, activation='elu')(flatten)
    output3_layer2 = Dense(16, activation='elu')(output3_layer1)
    output3 = Dense(1, name="jump_sigma")(output3_layer2)
    
    output4_layer1 = Dense(32, activation='elu')(flatten)
    output4_layer2 = Dense(16, activation='elu')(output4_layer1)
    output4 = Dense(1, name="jump_mu")(output4_layer2)
    
    output5_layer1 = Dense(32, activation='elu')(flatten)
    output5_layer2 = Dense(16, activation='elu')(output5_layer1)
    output5 = Dense(1, name="lambda")(output5_layer2)

    convnet_mo_elu = Model(input = input_1, output=[output1, output2, output3, output4, output5])

    if r_squared:
        convnet_mo_elu.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error', r2])
    else:
        convnet_mo_elu.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error'])
    
    return convnet_mo_elu

def covnet_single_ReLUs_6_layers():
    """
    returns a 6-layer convolutional architecture (with ReLU activation units) as a Keras Model, with outputs for only a single parameter.
    """
    
    convnet_lambda = Sequential()
    convnet_lambda.add(Conv2D(32, (12, 12), input_shape=(40, 50, 1), padding='same', activation='relu'))
    convnet_lambda.add(Conv2D(32, (12, 12), padding='same', activation='relu'))
    convnet_lambda.add(MaxPooling2D(pool_size=(2,2)))

    convnet_lambda.add(Conv2D(64, (6, 6), padding='same', activation='relu'))
    convnet_lambda.add(Conv2D(64, (6, 6), padding='same', activation='relu'))
    convnet_lambda.add(MaxPooling2D(pool_size=(2,2)))

    convnet_lambda.add(Flatten())
    convnet_lambda.add(Dense(512, activation='relu'))
    convnet_lambda.add(Dense(1))
    
    convnet_lambda.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error', r2])
    
    return convnet_lambda