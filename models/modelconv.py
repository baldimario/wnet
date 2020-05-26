import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Multiply, Activation, Dense, Flatten, Add, Conv2D, Conv1D, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model, Sequential


def get_model(input_shape=8000, channels=64, n_layers=15, n_dilation_depth=9, n_filters=32, residual=True, filter_width=2, dilation_factor=2):
    net_input = Input([input_shape, channels])
    net_first = Conv1D(n_filters, filter_width, padding='same', name='first_scale')(net_input)

    skips = []
    net_layer_input = net_first
    for i in range(1, n_layers+1):
        d = dilation_factor**(i % n_dilation_depth)
        net_tanh = Conv1D(n_filters, filter_width, padding='same', activation='tanh', dilation_rate=d, name='tanh_layer_'+str(i))(net_layer_input)
        net_sigmoid = Conv1D(n_filters, filter_width, padding='same', activation='sigmoid', dilation_rate=d, name='sigmoid_layer_'+str(i))(net_layer_input)
        net_gated = Multiply(name='multiply_layer_'+str(i))([net_tanh, net_sigmoid])
        net_skip = Conv1D(1, 1, name='scale_'+str(i))(net_gated)

        skips.append(net_skip)

        net_residual = Add(name='residual_'+str(i))([net_skip, net_layer_input]) if residual else net_layer_input
        net_layer_input = net_residual

    net_sum = Add(name='sum')(skips)
    net_all = Activation('relu', name='all')(net_sum)
    net_scale1 = Conv1D(1, 1, activation='relu', name='scale_relu')(net_all)
    net_scale2 = Conv1D(1, 1, name='scale')(net_scale1)
    net_flatten = Flatten(name='flatten')(net_scale2)
    net_output = Dense(channels, activation='softmax', name='output')(net_flatten)

    model = Model(net_input, net_output)

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    return model

def get_model_(input_shape=8000, channels=64, n_layers=15, n_dilation_depth=9, n_filters=32, residual=True, filter_width=2):
    net_input = Input([input_shape, channels])
    net_first = Conv1D(n_filters, filter_width, padding='causal', activation='linear', name='first_scale')(net_input)

    skips = []
    net_layer_input = net_first

    for i in range(1, n_layers+1):
        d = 2**(i%(n_dilation_depth))
        #print('layer_{} dilation_{}'.format(i, d))

        net_tanh = Conv1D(n_filters, filter_width, padding='causal', activation='tanh', dilation_rate=d, name='tanh_layer_'+str(i))(net_layer_input)
        net_sigmoid = Conv1D(n_filters, filter_width, padding='causal', activation='sigmoid', dilation_rate=d, name='sigmoid_layer_'+str(i))(net_layer_input)
        net_gated = Multiply(name='multiply_layer_'+str(i))([net_tanh, net_sigmoid])

        #net_slice = Lambda(lambda x: x[:, -1:, :], name='slice_layer'+str(i))(net_gated)
        #net_skip = Conv1D(1, 1, activation='linear', name='skip_layer_'+str(i))(net_slice)
        net_skip = Conv1D(n_filters, 1, activation='linear', name='skip_layer_' + str(i))(net_gated)

        if residual:
            net_forward = Add()([net_skip, net_layer_input])
        else:
            net_forward = net_skip

        skips.append(net_skip)
        net_layer_input = net_forward

    net_sum = Add(name='global_add')(skips)
    net_skips = Activation('relu')(net_sum)
    net_scale1 = Conv1D(1, 1, padding='same', activation='relu', name='last_scale1')(net_sum)
    #net_scale2 = Conv1D(channels, 1, padding='same', activation='relu', name='last_scale2')(net_scale1)

    net_output = Conv1D(channels, 1, padding='same', activation='softmax', name='output')(net_scale1)
    #net_flatten = Flatten(name='flatten')(net_scale2)
    #net_output = Activation('softmax')(net_flatten) #Dense(channels, activation='softmax', name='output')(net_flatten)

    model = Model(net_input, net_output)

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model
