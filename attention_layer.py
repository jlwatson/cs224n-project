import numpy as np

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Wrapper, Recurrent

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        super(Attention, self).build(input_shape)

    def call(self, x):
        shape = K.shape(x)
        samples, time_steps, embedding_dim = shape[0], shape[1], shape[2]
        prev_hidden_states = x[:, :time_steps-1, :]
        final_hidden_state = x[:, time_steps-1, :]

        scores = K.sum(prev_hidden_states * K.expand_dims(final_hidden_state, axis=1), axis=2)
        weights = K.softmax(scores)
        attention = K.sum(K.expand_dims(weights, axis=2) *  prev_hidden_states, axis=1)

        return K.concatenate([final_hidden_state, attention], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] * 2)
